"""
jetson_combined.py
------------------
- Streams webcam (index 0) and RealSense over ZMQ (ports 5556, 5557)
- Reads gamepad and sends differential drive commands to Arduino over serial
- Follower arm mirrors leader arm always
- Records LeRobot episodes with follower arm + cameras

Run on Jetson:
    python jetson_combined.py
    python jetson_combined.py --record --task "pick up the cube" --num-episodes 10 --repo-id local/my-dataset

TODO (SLAM additions):
    - RGB-D ZMQ bridge (ports 5559/5560) → ros2_scan_bridge.py
    - IMU streaming from BNO08x over I2C → ZMQ port 5563
    - Foxglove WebSocket bridge
"""

import os
import sys
import time
import threading
import argparse
import select
import json
import serial

import cv2
import zmq
import pyrealsense2 as rs
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["SDL_JOYSTICK_HIDAPI"] = "0"

import pygame

# ── Config ────────────────────────────────────────────────────────────────────

SERIAL_PORT     = "/dev/ttyACM0"
BAUD_RATE       = 115200

AXIS_MAX        = 0.6
TOLERANCE       = 0.01
DEADZONE        = 0.08
LEFT_AXIS       = 1
RIGHT_AXIS      = 3
SEND_HZ         = 10
WATCHDOG_HZ     = 4

WEBCAM_INDEX    = 6
QUALITY         = 50
WEBCAM_PORT     = 5556
REALSENSE_PORT  = 5557

FOLLOWER_PORT        = "/dev/ttyACM1"
FOLLOWER_ID          = "my_awesome_follower_arm"
LEADER_IP            = "10.0.0.53"
LEADER_ZMQ_PORT      = 5555
FOLLOWER_RECONNECT_S = 3.0
FOLLOWER_RECV_MS     = 500   # ZMQ timeout; triggers keepalive when leader is quiet

# ── Shared state ──────────────────────────────────────────────────────────────

latest_frames    = {"webcam": None, "realsense": None}
frame_locks      = {"webcam": threading.Lock(), "realsense": threading.Lock()}

follower_instance = None
follower_lock     = threading.Lock()
latest_action     = None
action_lock       = threading.Lock()

stop_event = threading.Event()

# ── ZMQ context ───────────────────────────────────────────────────────────────

zmq_ctx = zmq.Context()

def make_pub(port):
    sock = zmq_ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.bind(f"tcp://*:{port}")
    return sock

# ── Camera threads ────────────────────────────────────────────────────────────
        ret, frame = cap.read()
        if ret:
            with frame_locks["webcam"]:
                latest_frames["webcam"] = frame.copy()
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
            sock.send(buf.tobytes())
    cap.release()

def capture_realsense():
    sock = make_pub(REALSENSE_PORT)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    print("[realsense] started")
    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            frame = frames.get_color_frame()
            if not frame:
                continue
            img = np.asanyarray(frame.get_data())
            with frame_locks["realsense"]:
                latest_frames["realsense"] = img.copy()
            _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
            sock.send(buf.tobytes())
    finally:
        pipeline.stop()
        print("[realsense] stopped")

# ── Serial helpers ────────────────────────────────────────────────────────────

def normalise(raw: float) -> float:
    v = -max(-1.0, min(1.0, raw / AXIS_MAX))
    return 0.0 if abs(v) < DEADZONE else round(v, 3)

def build_packet(left: float, right: float) -> bytes:
    return f"L:{left:+.3f},R:{right:+.3f}\n".encode()

# ── Joystick/serial thread ────────────────────────────────────────────────────

def drive_loop():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("[joystick] no joystick detected —
    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"[joystick] {joy.get_name()}")

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"[serial] {SERIAL_PORT} @ {BAUD_RATE} baud")

    prev_left = prev_right = None
    min_interval = 1.0 / SEND_HZ if SEND_HZ > 0 else 0
    last_send = last_keepalive = 0.0

    try:
        while not stop_event.is_set():
            pygame.event.pump()
            left  = normalise(joy.get_axis(LEFT_AXIS))
            right = normalise(joy.get_axis(RIGHT_AXIS))
            now   = time.monotonic()

            changed = (
                prev_left  is None or
                prev_right is None or
                abs(left  - prev_left)  > TOLERANCE or
                abs(right - prev_right) > TOLERANCE
            )
            ready = (now - last_send) >= min_interval

            if changed and ready:
                pkt = build_packet(-left, -right)
                ser.write(pkt)
                print(f"[tx] {pkt.decode().strip()}")
                prev_left = left
                prev_right = right
                last_send = last_keepalive = now
            elif (now - last_keepalive) >= (1.0 / WATCHDOG_HZ):
                pkt = build_packet(-prev_left or 0.0, -prev_right or 0.0)
                ser.write(pkt)
                ser.flushInput()
                last_keepalive = now

            time.sleep(0.005)

    except Exception as e:
        print(f"[drive] error: {e}")
    finally:
        ser.write(build_packet(0.0, 0.0))
        ser.flushInput()
        time.sleep(0.1)
        ser.close()
        pygame.quit()
        print("[drive] stopped")

# ── Follower helpers ──────────────────────────────────────────────────────────
def _force_release_port(port: str) -> None:
    """
    Evict any stale file descriptor on the serial port by briefly opening it
    with exclusive access (TIOCEXCL), which causes the OS to revoke any other
    holder, then immediately closing it.

    This works regardless of which internal registry the Dynamixel SDK uses
    and handles the case where ModemManager or brltty briefly probed the port
    and left it locked.

    If the port doesn't exist yet (USB not enumerated) this is a no-op.
    """
    if not os.path.exists(port):
        return
    try:
        import serial as _serial
        s = _serial.Serial(port, baudrate=1000000, exclusive=True, timeout=0)
        s.close()
        print(f"[follower] force-released {port}")
    except Exception as e:
        # Don't let a failed eviction prevent a connect attempt.
        # The SDK open may still succeed if nothing was actually holding it.
        print(f"[follower] port release warning: {e}")


def _port_alive(port: str) -> bool:
    """True while the device node exists (USB still enumerated)."""
    return os.path.exists(port)


# ── Follower loop ─────────────────────────────────────────────────────────────
    while not stop_event.is_set():
        follower    = None
        leader_sock = None
        last_action = None

        try:
            # Evict any stale fd before the SDK opens the port.
            _force_release_port(FOLLOWER_PORT)

            cfg      = SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID)
            follower = SO101Follower(cfg)
            follower.connect()
            print(f"[follower] connected on {FOLLOWER_PORT}")

            # Expose only after connect() fully returns.
            with follower_lock:
                follower_instance = follower

            # Subscribe to leader.
            leader_sock = zmq_ctx.socket(zmq.SUB)
            leader_sock.setsockopt(zmq.RCVHWM, 1)
            leader_sock.setsockopt(zmq.CONFLATE, 1)
            # Non-blocking poll so we can send keepalives and check port health.
            leader_sock.setsockopt(zmq.RCVTIMEO, FOLLOWER_RECV_MS)
            leader_sock.connect(f"tcp://{LEADER_IP}:{LEADER_ZMQ_PORT}")
            leader_sock.setsockopt(zmq.SUBSCRIBE, b"")
            print(f"[follower] subscribed to leader at {LEADER_IP}:{LEADER_ZMQ_PORT}")

            while not stop_event.is_set():

                # Bail early if USB disappeared — avoids a noisy SDK write failure.
                if not _port_alive(FOLLOWER_PORT):
                    print(f"[follower] {FOLLOWER_PORT} disappeared — reconnecting")
                    break

                # Receive command, or use last known position as keepalive.
                action = None
                try:
                    msg    = leader_sock.recv_string()
                    action = json.loads(msg)
                except zmq.Again:
                    action = last_action  # keepalive: hold last position

                if action is None:
                    # No message received yet since (re)connect.
                    continue

                try:
                    with follower_lock:
                        follower.send_action(action)
                    last_action = action
                    with action_lock:
                        latest_action = action

                except Exception as e:
                    err = str(e)
                    print(f"[follower] send_action failed: {err}")
                    # Hard SDK port errors → full reconnect (evicts stale fd).
                    # Transient per-servo errors (CRC, single timeout) → continue.
                    if "Port is in use" in err or "TxRxResult" in err:
                        print("[follower] SDK port error — triggering reconnect")
                        break

        except KeyboardInterrupt:
            stop_event.set()
            break
        except Exception as e:
            print(f"[follower] error: {e}")
        finally:
            # Hide instance before disconnecting so other threads don't
            # call send_action on a closing handle.
            with follower_lock:
                follower_instance = None
            if leader_sock is not None:
                try:
                    leader_sock.close()
                except Exception:
                    pass
            if follower is not None:
                try:
                    follower.disconnect()
                except Exception:
                    pass
            print("[follower] stopped")

        if not stop_event.is_set():
            print(f"[follower] reconnecting in {FOLLOWER_RECONNECT_S}s…")
            time.sleep(FOLLOWER_RECONNECT_S)

    print("[follower] thread exiting")

# ── LeRobot recording ─────────────────────────────────────────────────────────
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                      "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        features=features,
        robot_type="so101",
        use_videos=True,
    )
    print(f"[record] dataset created: {repo_id}")

    try:
        for episode_idx in range(num_episodes):
            input(f"\n[record] Press Enter to start episode {episode_idx + 1}/{num_episodes}...")
            dataset.clear_episode_buffer()
            print("[record] Recording — pre
                    action_vec = state.copy()

                with frame_locks["webcam"]:
                    webcam_frame = latest_frames["webcam"]
                with frame_locks["realsense"]:
                    realsense_frame = latest_frames["realsense"]

                if webcam_frame is not None and realsense_frame is not None:
                    dataset.add_frame({
                        "observation.state": state,
                        "observation.images.webcam": webcam_frame,
                        "observation.images.wrist": realsense_frame,
                        "action": action_vec,
                        "task": task,
                    })

                if wait_for_key():
                    save = wait_for_enter_or_discard()
                    if save:
                        dataset.save_episode()
                        print(f"[record] Episode {episode_idx + 1} saved "
                              f"({dataset.num_frames} frames total)")
                    else:
                        dataset.clear_episode_buffer()
                        print(f"[record] Episode {episode_idx + 1} discarded")
                    break

                time.sleep(1 / 30)

    except KeyboardInterrupt:
        print("\n[record] interrupted — finalizing")
    finally:
        dataset.finalize()
        print(f"[record] dataset finalized: {repo_id}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",         type=str, default="robot task")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--repo-id",      type=str, default="local/robot-dataset")
    parser.add_argument("--record",       action="store_true")
    args = parser.parse_args()

    threading.Thread(target=capture_webcam,    daemon=True).start()
    threading.Thread(target=capture_realsense, daemon=True).start()
    threading.Thread(target=drive_loop,        daemon=True).start()
    threading.Thread(target=follower_loop,     daemon=True).start()

    if args.record:
        record_loop(args.task, args.num_episodes, args.repo_id)
    else:
        print("Running. Ctrl-C to stop.")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            stop_event.set()
            time.sleep(1)
            print("Shutdown complete.")

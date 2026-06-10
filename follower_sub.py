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

WEBCAM_INDEX    = 0
QUALITY         = 50
WEBCAM_PORT     = 5556
REALSENSE_PORT  = 5557

FOLLOWER_PORT   = "/dev/ttyACM1"
FOLLOWER_ID     = "my_awesome_follower_arm"
LEADER_IP       = "10.0.0.10"
LEADER_ZMQ_PORT = 5555

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

def capture_webcam():
    sock = make_pub(WEBCAM_PORT)
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"[webcam] opened: {cap.isOpened()}")
    while not stop_event.is_set():
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
        print("[joystick] no joystick detected — drive loop exiting")
        return

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

# ── Follower loop (always runs) ───────────────────────────────────────────────

def follower_loop():
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    global follower_instance

    cfg = SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID)
    follower = SO101Follower(cfg)
    follower.connect()
    print(f"[follower] connected on {FOLLOWER_PORT}")

    with follower_lock:
        follower_instance = follower

    leader_sock = zmq_ctx.socket(zmq.SUB)
    leader_sock.setsockopt(zmq.RCVHWM, 1)
    leader_sock.setsockopt(zmq.CONFLATE, 1)
    leader_sock.connect(f"tcp://{LEADER_IP}:{LEADER_ZMQ_PORT}")
    leader_sock.setsockopt(zmq.SUBSCRIBE, b"")
    print(f"[follower] subscribed to leader at {LEADER_IP}:{LEADER_ZMQ_PORT}")

    try:
        while not stop_event.is_set():
            msg = leader_sock.recv_string()
            action = json.loads(msg)
            with follower_lock:
                follower.send_action(action)
            with action_lock:
                global latest_action
                latest_action = action
    except Exception as e:
        print(f"[follower] error: {e}")
    finally:
        follower.disconnect()
        leader_sock.close()
        print("[follower] stopped")

# ── LeRobot recording ─────────────────────────────────────────────────────────

def wait_for_key():
    return select.select([sys.stdin], [], [], 0)[0]

def wait_for_enter_or_discard():
    while True:
        line = sys.stdin.readline().strip().lower()
        if line == 'd':
            return False
        return True

def record_loop(task: str, num_episodes: int, repo_id: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Wait for follower to connect
    print("[record] waiting for follower arm...")
    while True:
        with follower_lock:
            if follower_instance is not None:
                break
        time.sleep(0.1)
    print("[record] follower ready")

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                      "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"],
        },
        "observation.images.webcam": {
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
            print("[record] Recording — press Enter to save, D+Enter to discard")

            while True:
                with follower_lock:
                    obs = follower_instance.get_observation()

                state = np.array([
                    obs["shoulder_pan.pos"],
                    obs["shoulder_lift.pos"],
                    obs["elbow_flex.pos"],
                    obs["wrist_flex.pos"],
                    obs["wrist_roll.pos"],
                    obs["gripper.pos"],
                ], dtype=np.float32)

                with action_lock:
                    act = latest_action

                if act is not None:
                    action_vec = np.array([
                        act["shoulder_pan.pos"],
                        act["shoulder_lift.pos"],
                        act["elbow_flex.pos"],
                        act["wrist_flex.pos"],
                        act["wrist_roll.pos"],
                        act["gripper.pos"],
                    ], dtype=np.float32)
                else:
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

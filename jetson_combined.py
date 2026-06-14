"""
    python jetson_combined.py --record --task "pick up the cube" --num-episodes 10 --repo-id local/my-dataset
    python jetson_combined.py --foxglove                     # stream to Foxglove Studio (port 8765)

Install foxglove bridge dep:
    pip install foxglove-websocket

Install IMU dep:
    pip install adafruit-circuitpython-bno08x
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

# RTAB-Map / ZMQ bridge ports
RGB_BRIDGE_PORT   = 5559  # raw BGR image bytes → ros2_scan_bridge.py
DEPTH_BRIDGE_PORT = 5560  # raw uint16 depth bytes → ros2_scan_bridge.py
IMU_BRIDGE_PORT   = 5563  # JSON accel/gyro/quat → ros2_scan_bridge.py

FOLLOWER_PORT        = "/dev/ttyACM1"
FOLLOWER_ID          = "my_awesome_follower_arm"
LEADER_IP            = "10.0.0.53"
LEADER_ZMQ_PORT      = 5555
FOLLOWER_RECONNECT_S = 3.0
FOLLOWER_RECV_MS     = 500   # ZMQ timeout; triggers keepalive when leader is quiet

# ── Robot geometry ────────────────────────────────────────────────────────────

WHEEL_BASE      = 0.20   # distance between left and right wheels (metres) — tune to your car
WHEEL_RADIUS    = 0.033  # driven wheel radius (metres) — tune to your car
MAX_WHEEL_SPEED = 1.5    # rad/s at full command (|cmd| == 1.0) — tune to your car

# ── Foxglove WebSocket bridge ─────────────────────────────────────────────────

FOXGLOVE_PORT    = 8765
FOXGLOVE_HZ_CAM  = 15
FOXGLOVE_HZ_SLOW = 10

# ── IMU (BNO08x over I2C) ────────────────────────────────────────────────────

IMU_STARTUP_DELAY = 5.0   # seconds — let RealSense claim USB before I2C init
IMU_HZ            = 100   # sensor report rate (Hz)
IMU_I2C_ADDR      = 0x4B

# ── Shared state ──────────────────────────────────────────────────────────────

latest_frames    = {"webcam": None, "realsense": None, "depth": None}
frame_locks      = {"webcam": threading.Lock(), "realsense": threading.Lock(), "depth": threading.Lock()}

follower_instance = None
follower_lock     = threading.Lock()
latest_action     = None
action_lock       = threading.Lock()

drive_lock = threading.Lock()
drive_cmd  = {"left": 0.0, "right": 0.0}  # normalised [-1, 1]

imu_lock = threading.Lock()
latest_imu = None  # dict: {"accel": (x,y,z), "gyro": (x,y,z), "quat": (i,j,k,w)}

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

def _rs_hardware_reset():
    """Issue a USB hardware reset to all connected RealSense devices."""
    try:
        ctx = rs.context()
        for dev in ctx.query_devices():
            serial = dev.get_info(rs.camera_info.serial_number)
            print(f"[realsense] hardware reset → {serial}")
            dev.hardware_reset()
        time.sleep(2.0)  # device needs time to re-enumerate on USB
    except Exception as e:
        print(f"[realsense] hardware reset failed: {e}")


def _disable_usb_autosuspend():
    """Disable USB autosuspend to prevent the kernel from powering down the camera."""
    try:
        import glob
        for path in glob.glob("/sys/bus/usb/devices/*/power/autosuspend_delay_ms"):
            with open(path, "w") as f:
                f.write("-1")
        print("[realsense] USB autosuspend disabled")
    except Exception as e:
        print(f"[realsense] could not disable USB autosuspend: {e}")


def capture_realsense():
    """
    Capture loop with depth alignment and automatic reconnection.

    Publishes:
      - JPEG colour on REALSENSE_PORT (Foxglove / recording)
      - Raw BGR + depth on RGB_BRIDGE_PORT / DEPTH_BRIDGE_PORT (ros2_scan_bridge → RTAB-Map)

    Uses try_wait_for_frames (non-throwing) instead of wait_for_frames to avoid
    hanging on USB stalls.  Issues a hardware reset before each reconnect attempt.
    """
    _disable_usb_autosuspend()

    sock       = make_pub(REALSENSE_PORT)
    rgb_sock   = make_pub(RGB_BRIDGE_PORT)
    depth_sock = make_pub(DEPTH_BRIDGE_PORT)

    TIMEOUT_MS       = 2000
    CONSECUTIVE_MAX  = 5
    STALE_TIMEOUT    = 5.0    # seconds with no new frames → force reconnect
    RECONNECT_DELAY  = 3.0

    while not stop_event.is_set():
        pipeline = None
        try:
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            pipeline.start(cfg)
            #align = rs.align(rs.stream.color)
            print("[realsense] started (colour + depth)")
            consecutive_timeouts = 0
            last_good_frame = time.monotonic()

            while not stop_event.is_set():
                # Non-throwing: returns (success, frameset) instead of
                # hanging indefinitely on USB stalls.
                success, raw_frames = pipeline.try_wait_for_frames(TIMEOUT_MS)

                if not success:
                    consecutive_timeouts += 1
                    print(f"[realsense] frame timeout #{consecutive_timeouts}")
                    if consecutive_timeouts >= CONSECUTIVE_MAX:
                        print("[realsense] too many timeouts — reconnecting")
                        break
                    continue

                # Staleness check: if we got a frameset but haven't had a
                # usable colour frame in STALE_TIMEOUT seconds, force restart.
                consecutive_timeouts = 0

                #frames = align.process(raw_frames)
                frames = raw_frames
                color  = frames.get_color_frame()
                depth  = frames.get_depth_frame()
                if not color:
                    if (time.monotonic() - last_good_frame) > STALE_TIMEOUT:
                        print("[realsense] no colour frames — stale, reconnecting")
                        break
                    continue

                last_good_frame = time.monotonic()
                img = np.asanyarray(color.get_data())

                with frame_locks["realsense"]:
                    latest_frames["realsense"] = img.copy()
                if depth:
                    with frame_locks["depth"]:
                        latest_frames["depth"] = np.asanyarray(depth.get_data()).copy()

                # JPEG for Foxglove / recording
                _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                sock.send(buf.tobytes())

                # Raw frames for ros2_scan_bridge → RTAB-Map
                h, w = img.shape[:2]
                rgb_sock.send(np.array([h, w], dtype=np.int32).tobytes() + img.tobytes())
                if depth:
                    d_arr = np.asanyarray(depth.get_data())
                    depth_sock.send(np.array([*d_arr.shape], dtype=np.int32).tobytes() + d_arr.tobytes())

        except Exception as e:
            print(f"[realsense] error: {e}")
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            print("[realsense] pipeline stopped")

        if not stop_event.is_set():
            print(f"[realsense] hardware reset + reconnect in {RECONNECT_DELAY}s…")
            _rs_hardware_reset()
            time.sleep(RECONNECT_DELAY)

    print("[realsense] thread exiting")

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
            ready = (now - last_send) >= min_interval

            if changed and ready:
                pkt = build_packet(-left, -right)
                ser.write(pkt)
                print(f"[tx] {pkt.decode().strip()}")
                prev_left = left
                prev_right = right
                last_send = last_keepalive = now
                with drive_lock:
                    drive_cmd["left"]  = -left
                    drive_cmd["right"] = -right
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
        print(f"[follower] port release warning: {e}")


def _port_alive(port: str) -> bool:
    """True while the device node exists (USB still enumerated)."""
    return os.path.exists(port)


# ── Follower loop ─────────────────────────────────────────────────────────────
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    global follower_instance, latest_action

    while not stop_event.is_set():
        follower    = None
        leader_sock = None
        last_action = None

        try:
            _force_release_port(FOLLOWER_PORT)

            cfg      = SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID)
            follower = SO101Follower(cfg)
            follower.connect()
            print(f"[follower] connected on {FOLLOWER_PORT}")

            with follower_lock:
                follower_instance = follower

            leader_sock = zmq_ctx.socket(zmq.SUB)
            leader_sock.setsockopt(zmq.RCVHWM, 200)  # buffer commands; CONFLATE must be OFF for arm
            leader_sock.setsockopt(zmq.CONFLATE, 0)   # deliver every joint command in order
            leader_sock.setsockopt(zmq.RCVTIMEO, FOLLOWER_RECV_MS)
            leader_sock.connect(f"tcp://{LEADER_IP}:{LEADER_ZMQ_PORT}")
            leader_sock.setsockopt(zmq.SUBSCRIBE, b"")
            print(f"[follower] subscribed to leader at {LEADER_IP}:{LEADER_ZMQ_PORT}")

            while not stop_event.is_set():

                if not _port_alive(FOLLOWER_PORT):
                    print(f"[follower] {FOLLOWER_PORT} disappeared — reconnecting")
                    break

                action = None
                try:
                    msg    = leader_sock.recv_string()
                    action = json.loads(msg)
                except zmq.Again:
                    action = last_action  # keepalive: hold last position

                if action is None:
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
                    if "Port is in use" in err or "TxRxResult" in err:
                        print("[follower] SDK port error — triggering reconnect")
                        break

        except KeyboardInterrupt:
            stop_event.set()
            break
        except Exception as e:
            print(f"[follower] error: {e}")
        finally:
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


# ── IMU thread (BNO08x) ──────────────────────────────────────────────────────

def imu_loop():
    """
    Read BNO08x over I2C (bus 7 on Yahboom Jetson carrier) and publish
    accel/gyro/quaternion as JSON over ZMQ.  Delays startup to avoid
    contention with RealSense USB init.
    """
    global latest_imu

    print(f"[imu] waiting {IMU_STARTUP_DELAY}s for RealSense to initialise…")
    time.sleep(IMU_STARTUP_DELAY)

    imu_sock = make_pub(IMU_BRIDGE_PORT)

    try:
        import board
        import busio
        from adafruit_bno08x.i2c import BNO08X_I2C
        from adafruit_bno08x import (
            BNO_REPORT_ACCELEROMETER,
            BNO_REPORT_GYROSCOPE,
            BNO_REPORT_ROTATION_VECTOR,
        )
    except ImportError as e:
        print(f"[imu] missing dependency: {e}")
        print("[imu] pip install adafruit-circuitpython-bno08x")
        return

    RECONNECT_DELAY = 3.0
    report_interval_us = int(1e6 / IMU_HZ)  # microseconds

    while not stop_event.is_set():
        try:
            # Blinka maps board.SCL/SDA → I2C bus 7 on Yahboom carrier
            i2c = busio.I2C(board.SCL, board.SDA)
            bno = BNO08X_I2C(i2c, address=IMU_I2C_ADDR)

            bno.enable_feature(BNO_REPORT_ACCELEROMETER,   report_interval_us)
            bno.enable_feature(BNO_REPORT_GYROSCOPE,       report_interval_us)
            bno.enable_feature(BNO_REPORT_ROTATION_VECTOR, report_interval_us)
            print(f"[imu] BNO08x ready at 0x{IMU_I2C_ADDR:02X}, {IMU_HZ}Hz")

            interval = 1.0 / IMU_HZ

            while not stop_event.is_set():
                t0 = time.monotonic()

                accel = bno.acceleration      # (ax, ay, az) m/s²
                gyro  = bno.gyro              # (gx, gy, gz) rad/s
                quat  = bno.quaternion        # (i, j, k, w)

                if accel is None or gyro is None or quat is None:
                    time.sleep(0.01)
                    continue

                imu_data = {
                    "accel": list(accel),
                    "gyro":  list(gyro),
                    "quat":  list(quat),       # [i, j, k, w]
                    "t":     time.time(),
                }

                with imu_lock:
                    latest_imu = imu_data

                imu_sock.send_string(json.dumps(imu_data))

                elapsed = time.monotonic() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)

        except Exception as e:
            print(f"[imu] error: {e}")

        if not stop_event.is_set():
            print(f"[imu] reconnecting in {RECONNECT_DELAY}s…")
            time.sleep(RECONNECT_DELAY)

    print("[imu] thread exiting")


# ── Foxglove WebSocket bridge ─────────────────────────────────────────────────
#
# Uses the official `foxglove-websocket` Python library (no ROS2 required).
# Publishes these channels to Foxglove Studio:
#
#   /camera/webcam          foxglove.CompressedImage
#   /camera/realsense       foxglove.CompressedImage
#   /odom                   foxglove.PosesInFrame   (identity — SLAM pose comes from ROS2 TF)
#   /drive_cmd              foxglove.Twist
#   /arm/joint_states       foxglove.JointState
#   /imu                    foxglove.Imu            (BNO08x accel/gyro/quaternion)
#
# Connect from Foxglove Studio: File → Open connection → WebSocket → ws://<ip>:8765

def _foxglove_schema(name: str) -> dict:
    schemas = {
        "foxglove.CompressedImage": {
            "title": "CompressedImage",
            "type": "object",
            "properties": {
                "timestamp":  {"type": "object",
                               "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}},

                                         "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
                "frame_id":             {"type": "string"},
                "orientation":          {"type": "object",
                                         "properties": {"x": {"type": "number"}, "y": {"type": "number"},
                                                        "z": {"type": "number"}, "w": {"type": "number"}}},
                "angular_velocity":     {"type": "object",
                                         "properties": {"x": {"type": "number"}, "y": {"type": "number"},
                                                        "z": {"type": "number"}}},
                "linear_acceleration":  {"type": "object",
                                         "properties": {"x": {"type": "number"}, "y": {"type": "number"},
                                                        "z": {"type": "number"}}},
            },
        },
    }
    return schemas[name]


def _ts(t: float) -> dict:
    sec  = int(t)
    nsec = int((t - sec) * 1e9)
    return {"sec": sec, "nsec": nsec}


def foxglove_bridge():
    import asyncio
    import base64

    try:
        from foxglove_websocket.server import FoxgloveServer
    except ImportError:
        print("[foxglove] 'foxglove-websocket' not installed — skipping bridge")
        print("[foxglove] Install with: pip install foxglove-websocket")
        return

    ARM_JOINT_NAMES = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",

                "encoding":   "json",
                "schemaName": "foxglove.CompressedImage",
                "schema":     json.dumps(_foxglove_schema("foxglove.CompressedImage")),
            })
            ch_odom = await server.add_channel({
                "topic":      "/odom",
                "encoding":   "json",
                "schemaName": "foxglove.PosesInFrame",
                "schema":     json.dumps(_foxglove_schema("foxglove.PosesInFrame")),
            })
            ch_drive = await server.add_channel({
                "topic":      "/drive_cmd",
                "encoding":   "json",
                "schemaName": "foxglove.Twist",
                "schema":     json.dumps(_foxglove_schema("foxglove.Twist")),
            })
            ch_joints = await server.add_channel({
                "topic":      "/arm/joint_states",
                "encoding":   "json",
                "schemaName": "foxglove.JointState",
                "schema":     json.dumps(_foxglove_schema("foxglove.JointState")),
            })
            ch_imu = await server.add_channel({
                "topic":      "/imu",
                "encoding":   "json",
                "schemaName": "foxglove.Imu",
                "schema":     json.dumps(_foxglove_schema("foxglove.Imu")),
            })

            print(f"[foxglove] server live on ws://0.0.0.0:{FOXGLOVE_PORT}")
            print(f"[foxglove] connect Foxglove Studio → ws://<jetson-ip>:{FOXGLOVE_PORT}")

            cam_interval  = 1.0 / FOXGLOVE_HZ_CAM
            slow_interval = 1.0 / FOXGLOVE_HZ_SLOW
            last_cam  = 0.0
            last_slow = 0.0

            while not stop_event.is_set():
                now = time.monotonic()
                ts  = _ts(time.time())

                # ── Cameras ───────────────────────────────────────────────
                if now - last_cam >= cam_interval:
                    last_cam = now

                    with frame_locks["webcam"]:
                        wf = latest_frames.get("webcam")
                    if wf is not None:
                        _, buf = cv2.imencode(".jpg", wf, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                        await server.send_message(
                            ch_webcam,
                            int(time.time() * 1e9),
                            json.dumps({
                                "timestamp": ts,
                                "frame_id":  "webcam",
                                "format":    "jpeg",
                                "data":      base64.b64encode(buf.tobytes()).decode(),
                            }).encode(),
                        )

                    with frame_locks["realsense"]:
                        rf = latest_frames.get("realsense")
                    if rf is not None:
                        _, buf = cv2.imencode(".jpg", rf, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                        await server.send_message(
                            ch_rs,
                            int(time.time() * 1e9),
                            json.dumps({
                                "timestamp": ts,
                                "frame_id":  "realsense_color",
                                "format":    "jpeg",
                                "data":      base64.b64encode(buf.tobytes()).decode(),
                            }).encode(),
                        )

                # ── Slow channels ─────────────────────────────────────────
                if now - last_slow >= slow_interval:
                    last_slow = now

                    # Identity pose — real SLAM pose comes from ROS2 TF if slam_launch.py is running
                    await server.send_message(
                        ch_odom,
                        int(time.time() * 1e9),
                        json.dumps({
                            "timestamp": ts,
                            "frame_id":  "map",
                            "poses": [{
                                "position":    {"x": 0.0, "y": 0.0, "z": 0.0},
                                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                            }],
                        }).encode(),
                    )

                    with drive_lock:
                        l_cmd = drive_cmd["left"]
                        r_cmd = drive_cmd["right"]
                    lin_x = (l_cmd + r_cmd) / 2.0 * MAX_WHEEL_SPEED * WHEEL_RADIUS
                    ang_z = (r_cmd - l_cmd) / WHEEL_BASE * WHEEL_RADIUS
                    await server.send_message(
                        ch_drive,
                        int(time.time() * 1e9),
                        json.dumps({
                            "linear":  {"x": lin_x, "y": 0.0, "z": 0.0},
                            "angular": {"x": 0.0,   "y": 0.0, "z": ang_z},
                        }).encode(),
                    )

                    # Read last known joint positions from shared state instead
                    # of calling get_observation() —
                                "position":  positions,
                                "velocity":  [0.0] * 6,
                                "effort":    [0.0] * 6,
                            }).encode(),
                        )

                    # IMU
                    with imu_lock:
                        imu = latest_imu
                    if imu is not None:
                        q = imu["quat"]   # [i, j, k, w]
                        g = imu["gyro"]   # [x, y, z]
                        a = imu["accel"]  # [x, y, z]
                        await server.send_message(
                            ch_imu,
                            int(time.time() * 1e9),
                            json.dumps({
                                "timestamp":           ts,
                                "frame_id":            "imu_link",
                                "orientation":         {"x": q[0], "y": q[1], "z": q[2], "w": q[3]},
                                "angular_velocity":    {"x": g[0], "y": g[1], "z": g[2]},
                                "linear_acceleration": {"x": a[0], "y": a[1], "z": a[2]},
                            }).encode(),
                        )

                await asyncio.sleep(0.01)

    asyncio.run(run())


# ── LeRobot recording ─────────────────────────────────────────────────────────

def wait_for_key():
    return select.select([sys.s
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
            print("[record] Recording —
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
    parser.add_argument("--foxglove",     action="store_true",
                        help=f"Start Foxglove WebSocket bridge on port {FOXGLOVE_PORT}")
    args = parser.parse_args()

    threading.Thread(target=capture_webcam,    daemon=True).start()
    threading.Thread(target=capture_realsense, daemon=True).start()
    threading.Thread(target=drive_loop,        daemon=True).start()
    threading.Thread(target=follower_loop,     daemon=True).start()
    threading.Thread(target=imu_loop,          daemon=True).start()

    if args.foxglove:
        threading.Thread(target=foxglove_bridge, daemon=True).start()

    if args.record:
        record_loop(args.task, args.num_episodes, args.repo_id)
    else:
        print("Running. Ctrl-C to stop.")
        print("Tip: add --foxglove to stream cameras/drive/arm to Foxglove Studio")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            stop_event.set()
            time.sleep(1)
            print("Shutdown complete.")
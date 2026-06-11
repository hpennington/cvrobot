"""
jetson_combined.py
------------------
- Streams webcam (index 0) and RealSense over ZMQ (ports 5556, 5557)
- Reads gamepad and sends differential drive commands to Arduino over serial
- Follower arm mirrors leader arm always
- Records LeRobot episodes with follower arm + cameras
- RGB-D streaming for RTAB-Map SLAM (run slam_launch.py separately)
- Foxglove WebSocket bridge for live visualisation in Foxglove Studio

Run on Jetson:
    python jetson_combined.py
    python jetson_combined.py --record --task "pick up the cube" --num-episodes 10 --repo-id local/my-dataset
    python jetson_combined.py --slam                         # enable mapping
    python jetson_combined.py --slam --save-map /tmp/my_map  # save map on exit
    python jetson_combined.py --foxglove                     # stream to Foxglove Studio (port 8765)
    python jetson_combined.py --slam --foxglove              # mapping + live visualisation

Install foxglove bridge dep:
    pip install foxglove-websocket
"""

import os
import sys
import time
import math
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

# ── Robot geometry ───────────────────────────────────────────────────────────

WHEEL_BASE      = 0.20   # distance between left and right wheels (metres) — tune to your car
WHEEL_RADIUS    = 0.033  # driven wheel radius (metres) — tune to your car
MAX_WHEEL_SPEED = 1.5    # rad/s at full command (|cmd| == 1.0) — tune to your car

# ── RTAB-Map / ZMQ bridge ports ───────────────────────────────────────────────

RGB_BRIDGE_PORT   = 5559  # raw BGR image bytes → ros2_scan_bridge.py
DEPTH_BRIDGE_PORT = 5560  # raw uint16 depth bytes → ros2_scan_bridge.py
IR_LEFT_PORT      = 5561  # raw Y8 IR left  → isaac_ros_bridge.py
IR_RIGHT_PORT     = 5562  # raw Y8 IR right → isaac_ros_bridge.py

# Foxglove WebSocket bridge
FOXGLOVE_PORT    = 8765   # connect Foxglove Studio to ws://<jetson-ip>:8765
FOXGLOVE_HZ_CAM  = 15    # image publish rate (per camera)
FOXGLOVE_HZ_SLOW = 10    # scan / odom / joints publish rate


# ── Shared state ──────────────────────────────────────────────────────────────

latest_frames    = {"webcam": None, "realsense": None, "depth": None}
frame_locks      = {"webcam": threading.Lock(), "realsense": threading.Lock(), "depth": threading.Lock()}

follower_instance = None
follower_lock     = threading.Lock()
latest_action     = None
action_lock       = threading.Lock()

# Drive command state — written by drive_loop, read by Foxglove bridge for /drive_cmd
drive_lock    = threading.Lock()
drive_cmd     = {"left": 0.0, "right": 0.0}  # normalised [-1, 1]

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

def _make_rs_pipeline(enable_ir: bool = False) -> tuple:
    """Create, configure, and start a RealSense pipeline.
    Colour at 30 fps, depth at 15 fps. IR stereo at 30 fps when enable_ir=True.
    Returns (pipeline, align).
    """
    pipeline = rs.pipeline()
    cfg = rs.config()
    if enable_ir:
        # IR-only mode: drop colour to free USB bandwidth for full-res 30fps IR stereo
        cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        cfg.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
        cfg.enable_stream(rs.stream.depth,    640, 480, rs.format.z16, 30)
        pipeline.start(cfg)
        return pipeline, None   # no align without colour
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    pipeline.start(cfg)
    return pipeline, rs.align(rs.stream.color)


def capture_realsense():
    """
    Capture loop with automatic reconnection.

    If wait_for_frames() times out (USB stall, power-state glitch) or any
    other RuntimeError occurs the pipeline is torn down, we wait briefly,
    then try to reopen the device.  The thread never exits on its own —
    only stop_event terminates it.
    """
    sock       = make_pub(REALSENSE_PORT)
    rgb_sock      = make_pub(RGB_BRIDGE_PORT)
    depth_sock    = make_pub(DEPTH_BRIDGE_PORT)
    ir_left_sock  = make_pub(IR_LEFT_PORT)
    ir_right_sock = make_pub(IR_RIGHT_PORT)

    TIMEOUT_MS       = 2000   # generous timeout per frame
    CONSECUTIVE_MAX  = 5      # restart after this many consecutive timeouts
    RECONNECT_DELAY  = 3.0    # seconds to wait before reopening device

    while not stop_event.is_set():
        pipeline = None
        try:
            pipeline, align = _make_rs_pipeline(enable_ir=False)
            print("[realsense] started")
            consecutive_timeouts = 0

            while not stop_event.is_set():
                try:
                    raw_frames = pipeline.wait_for_frames(timeout_ms=TIMEOUT_MS)
                    consecutive_timeouts = 0
                except RuntimeError as e:
                    consecutive_timeouts += 1
                    print(f"[realsense] frame timeout #{consecutive_timeouts}: {e}")
                    if consecutive_timeouts >= CONSECUTIVE_MAX:
                        print("[realsense] too many timeouts — reconnecting")
                        break
                    continue

                # ── Colour + depth mode (RTAB-Map) ───────────────────────────
                if align is not None:
                    frames = align.process(raw_frames)
                    color  = frames.get_color_frame()
                    depth  = frames.get_depth_frame()
                    if not color:
                        continue
                    img = np.asanyarray(color.get_data())
                    with frame_locks["realsense"]:
                        latest_frames["realsense"] = img.copy()
                        if depth:
                            latest_frames["depth"] = np.asanyarray(depth.get_data()).copy()
                    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                    sock.send(buf.tobytes())
                    h, w = img.shape[:2]
                    rgb_sock.send(np.array([h, w], dtype=np.int32).tobytes() + img.tobytes())
                    if depth:
                        d_arr = np.asanyarray(depth.get_data())
                        depth_sock.send(np.array([*d_arr.shape], dtype=np.int32).tobytes() + d_arr.tobytes())
                else:
                    # ── IR-only mode (Isaac ROS cuVSLAM) ─────────────────────
                    depth = raw_frames.get_depth_frame()
                    if depth:
                        d_arr = np.asanyarray(depth.get_data())
                        latest_frames["depth"] = d_arr.copy()

                # IR stereo for Isaac ROS cuVSLAM
                ir_left  = raw_frames.get_infrared_frame(1)
                ir_right = raw_frames.get_infrared_frame(2)
                if ir_left:
                    ir_l = np.asanyarray(ir_left.get_data())
                    ir_left_sock.send(np.array([*ir_l.shape], dtype=np.int32).tobytes() + ir_l.tobytes())
                if ir_right:
                    ir_r = np.asanyarray(ir_right.get_data())
                    ir_right_sock.send(np.array([*ir_r.shape], dtype=np.int32).tobytes() + ir_r.tobytes())

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
            print(f"[realsense] reconnecting in {RECONNECT_DELAY}s…")
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


# ── Foxglove WebSocket bridge ─────────────────────────────────────────────────
#
# Uses the official `foxglove-websocket` Python library (no ROS2 required).
# Publishes these channels to Foxglove Studio:
#
#   /camera/webcam          foxglove.CompressedImage
#   /camera/realsense       foxglove.CompressedImage
#   /scan                   foxglove.LaserScan
#   /odom                   foxglove.PosesInFrame   (single pose)
#   /drive_cmd              foxglove.Twist (linear.x = fwd, angular.z = yaw rate)
#   /arm/joint_states       foxglove.FrameTransform (6-DOF joint positions as floats)
#
# Connect from Foxglove Studio: File → Open connection → WebSocket → ws://<ip>:8765

def _foxglove_schema(name: str) -> dict:
    """Return the JSON schema dict for a given foxglove well-known type."""
    schemas = {
        "foxglove.CompressedImage": {
            "title": "CompressedImage",
            "type": "object",
            "properties": {
                "timestamp":  {"type": "object",
                               "properties": {"sec": {"type":"integer"}, "nsec": {"type":"integer"}}},
                "frame_id":   {"type": "string"},
                "data":       {"type": "string", "contentEncoding": "base64"},
                "format":     {"type": "string"},
            },
        },
        "foxglove.LaserScan": {
            "title": "LaserScan",
            "type": "object",
            "properties": {
                "timestamp":        {"type": "object",
                                     "properties": {"sec": {"type":"integer"}, "nsec": {"type":"integer"}}},
                "frame_id":         {"type": "string"},
                "pose":             {"type": "object"},
                "start_angle":      {"type": "number"},
                "end_angle":        {"type": "number"},
                "ranges":           {"type": "array", "items": {"type": "number"}},
                "intensities":      {"type": "array", "items": {"type": "number"}},
            },
        },
        "foxglove.PosesInFrame": {
            "title": "PosesInFrame",
            "type": "object",
            "properties": {
                "timestamp": {"type": "object",
                              "properties": {"sec": {"type":"integer"}, "nsec": {"type":"integer"}}},
                "frame_id":  {"type": "string"},
                "poses":     {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "position":    {"type": "object",
                                           "properties": {"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"}}},
                            "orientation": {"type": "object",
                                           "properties": {"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"},"w":{"type":"number"}}},
                        },
                    },
                },
            },
        },
        "foxglove.Twist": {
            "title": "Twist",
            "type": "object",
            "properties": {
                "linear":  {"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"}}},
                "angular": {"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"}}},
            },
        },
        "foxglove.JointState": {
            "title": "JointState",
            "type": "object",
            "properties": {
                "timestamp":  {"type": "object",
                               "properties": {"sec": {"type":"integer"}, "nsec": {"type":"integer"}}},
                "frame_id":   {"type": "string"},
                "name":       {"type": "array", "items": {"type": "string"}},
                "position":   {"type": "array", "items": {"type": "number"}},
                "velocity":   {"type": "array", "items": {"type": "number"}},
                "effort":     {"type": "array", "items": {"type": "number"}},
            },
        },
    }
    return schemas[name]


def _ts(t: float) -> dict:
    """Float seconds → {sec, nsec} timestamp dict."""
    sec  = int(t)
    nsec = int((t - sec) * 1e9)
    return {"sec": sec, "nsec": nsec}


def foxglove_bridge(_odom=None):
    """
    Runs an asyncio event loop in a dedicated thread that hosts the Foxglove
    WebSocket server and pushes channel data at configured rates.
    """
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
        "wrist_flex", "wrist_roll", "gripper",
    ]

    async def run():
        async with FoxgloveServer(
            host="0.0.0.0",
            port=FOXGLOVE_PORT,
            name="jetson_robot",
        ) as server:

            # ── Register channels ──────────────────────────────────────────
            ch_webcam = await server.add_channel({
                "topic":    "/camera/webcam",
                "encoding": "json",
                "schemaName": "foxglove.CompressedImage",
                "schema":   json.dumps(_foxglove_schema("foxglove.CompressedImage")),
            })
            ch_rs = await server.add_channel({
                "topic":    "/camera/realsense",
                "encoding": "json",
                "schemaName": "foxglove.CompressedImage",
                "schema":   json.dumps(_foxglove_schema("foxglove.CompressedImage")),
            })
            ch_scan = await server.add_channel({
                "topic":    "/scan",
                "encoding": "json",
                "schemaName": "foxglove.LaserScan",
                "schema":   json.dumps(_foxglove_schema("foxglove.LaserScan")),
            })
            ch_odom = await server.add_channel({
                "topic":    "/odom",
                "encoding": "json",
                "schemaName": "foxglove.PosesInFrame",
                "schema":   json.dumps(_foxglove_schema("foxglove.PosesInFrame")),
            })
            ch_drive = await server.add_channel({
                "topic":    "/drive_cmd",
                "encoding": "json",
                "schemaName": "foxglove.Twist",
                "schema":   json.dumps(_foxglove_schema("foxglove.Twist")),
            })
            ch_joints = await server.add_channel({
                "topic":    "/arm/joint_states",
                "encoding": "json",
                "schemaName": "foxglove.JointState",
                "schema":   json.dumps(_foxglove_schema("foxglove.JointState")),
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

                # ── Cameras (rate-limited) ─────────────────────────────────
                if now - last_cam >= cam_interval:
                    last_cam = now

                    with frame_locks["webcam"]:
                        wf = latest_frames.get("webcam")
                    if wf is not None:
                        _, buf = cv2.imencode(
                            ".jpg", wf,
                            [cv2.IMWRITE_JPEG_QUALITY, QUALITY]
                        )
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
                        _, buf = cv2.imencode(
                            ".jpg", rf,
                            [cv2.IMWRITE_JPEG_QUALITY, QUALITY]
                        )
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

                # ── Slow channels: scan / odom / drive / joints ────────────
                if now - last_slow >= slow_interval:
                    last_slow = now

                    # LaserScan
                    with frame_locks["depth"]:
                        depth_img = latest_frames.get("depth")
                    # /scan not published in RTAB-Map mode — skip

                    # Pose: identity — actual pose comes from RTAB-Map /map→base_link TF.
                    # We publish a static origin marker so the channel exists in Foxglove;
                    # add a /tf or /map subscription in Studio to see the real slam pose.
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

                    # Drive command as Twist
                    with drive_lock:
                        l_cmd = drive_cmd["left"]
                        r_cmd = drive_cmd["right"]
                    lin_x  = (l_cmd + r_cmd) / 2.0 * MAX_WHEEL_SPEED * WHEEL_RADIUS
                    ang_z  = (r_cmd - l_cmd) / WHEEL_BASE * WHEEL_RADIUS
                    await server.send_message(
                        ch_drive,
                        int(time.time() * 1e9),
                        json.dumps({
                            "linear":  {"x": lin_x, "y": 0.0, "z": 0.0},
                            "angular": {"x": 0.0,   "y": 0.0, "z": ang_z},
                        }).encode(),
                    )

                    # Arm joint states
                    with follower_lock:
                        fi = follower_instance
                    if fi is not None:
                        try:
                            obs = fi.get_observation()
                            positions = [
                                obs.get(f"{name}.pos", 0.0)
                                for name in ARM_JOINT_NAMES
                            ]
                            await server.send_message(
                                ch_joints,
                                int(time.time() * 1e9),
                                json.dumps({
                                    "timestamp": ts,
                                    "frame_id":  "base_link",
                                    "name":      ARM_JOINT_NAMES,
                                    "position":  positions,
                                    "velocity":  [0.0] * 6,
                                    "effort":    [0.0] * 6,
                                }).encode(),
                            )
                        except Exception:
                            pass

                await asyncio.sleep(0.01)

    asyncio.run(run())


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
    parser.add_argument("--foxglove",     action="store_true",
                        help=f"Start Foxglove WebSocket bridge on port {FOXGLOVE_PORT}")
    args = parser.parse_args()

    threading.Thread(target=capture_webcam,    daemon=True).start()
    threading.Thread(target=capture_realsense, daemon=True).start()
    threading.Thread(target=drive_loop,        daemon=True).start()
    threading.Thread(target=follower_loop,     daemon=True).start()

    if args.foxglove:
        threading.Thread(target=foxglove_bridge, args=(None,), daemon=True).start()

    if args.record:
        record_loop(args.task, args.num_episodes, args.repo_id)
    else:
        print("Running. Ctrl-C to stop.")
        print("For RTAB-Map SLAM run: ros2 launch ~/ros2_ws/slam_launch.py")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            stop_event.set()
            time.sleep(1)
            print("Shutdown complete.")

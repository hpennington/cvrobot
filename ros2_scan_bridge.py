#!/usr/bin/env python3
"""
ros2_scan_bridge.py
-------------------
Receives raw BGR + uint16 depth frames AND packed IMU data from
jetson_combined.py over ZMQ and republishes them as ROS 2 topics for RTAB-Map.

Publishes:
  /camera/color/image_raw        sensor_msgs/Image      (BGR8)
  /camera/color/camera_info      sensor_msgs/CameraInfo
  /camera/depth/image_rect_raw   sensor_msgs/Image      (16UC1, millimetres)
  /camera/depth/camera_info      sensor_msgs/CameraInfo
  /imu/data                      sensor_msgs/Imu        (100 Hz, BNO08x)
  /odom                          nav_msgs/Odometry      (identity + max covariance)
  /tf                            static: odom→base_link, base_link→camera_link,
                                          base_link→imu_link

Run in a sourced ROS 2 terminal (system Python, not conda):

    source /opt/ros/jazzy/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    python3 ~/ros2_ws/ros2_scan_bridge.py
"""

import sys
import time
import struct

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo, Imu
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import TransformStamped
    import tf2_ros
except ImportError:
    sys.exit(
        "ERROR: rclpy not found.\n"
        "Source ROS 2 before running:\n"
        "  source /opt/ros/jazzy/setup.bash"
    )

try:
    import zmq
except ImportError:
    sys.exit("ERROR: pyzmq not found — pip3 install pyzmq")


# ── Config ────────────────────────────────────────────────────────────────────

JETSON_IP         = "127.0.0.1"
RGB_BRIDGE_PORT   = 5559
DEPTH_BRIDGE_PORT = 5560
IMU_BRIDGE_PORT   = 5563

# RealSense D415 intrinsics at 640×480 (colour stream)
FX, FY = 601.023, 601.023
CX, CY = 320.797, 242.064
DISTORTION = [0.0, 0.0, 0.0, 0.0, 0.0]

# Publish at half resolution so rgbd_odometry runs faster. Set to 1 to disable.
DOWNSAMPLE = 2

# camera_link quaternion relative to base_link.
# RealSense D415, lens forward / USB port down, standard ROS optical convention:
# -90° around X then -90° around Z → qx=-0.5, qy=0.5, qz=-0.5, qw=0.5
CAMERA_QX, CAMERA_QY, CAMERA_QZ, CAMERA_QW = -0.5, 0.5, -0.5, 0.5

# ── IMU extrinsic (base_link → imu_link) ─────────────────────────────────────
# Translation: physical offset of IMU from base_link origin in metres.
# Update XYZ to your actual measured offsets.
IMU_TF_X, IMU_TF_Y, IMU_TF_Z = 0.0, 0.0, 0.05

# Rotation: corrects for the BNO08x mounting orientation.
#
# Measured with robot flat and stationary:
#   raw accel  x≈+0.13  y≈+9.80  z≈-1.42
#
# This tells us:
#   • IMU Y axis points DOWN  (y ≈ +9.8 ≈ +g)
#   • IMU is tilted ~8.4° off horizontal  (arcsin(1.42/9.8))
#   • Net rotation around IMU X axis: 90° + 8.4° = 98.4°
#
# Quaternion for pure X-axis rotation by 98.4°:
#   qx = sin(98.4° / 2) = sin(49.2°) ≈ 0.757
#   qw = cos(98.4° / 2) = cos(49.2°) ≈ 0.653
#
# With this TF, RTAB-Map receives gravity along +Z of base_link (ROS convention)
# and gyro axes correctly mapped to the robot frame.
IMU_QX, IMU_QY, IMU_QZ, IMU_QW = 0.757, 0.0, 0.0, 0.653

CONNECT_TIMEOUT = 10.0   # warn if no camera frames within this many seconds

# ── IMU wire format (must match jetson_combined.py) ───────────────────────────
# 11 × float64 little-endian = 88 bytes
#   [0]  timestamp  (time.time())
#   [1-3]  accel x,y,z   m/s² raw (includes gravity)
#   [4-6]  gyro  x,y,z   rad/s
#   [7-10] quaternion i,j,k,real  (adafruit) → ROS x,y,z,w
IMU_PACK_FMT  = "<11d"
IMU_PACK_SIZE = struct.calcsize(IMU_PACK_FMT)

# ── IMU covariance matrices ───────────────────────────────────────────────────
# Row-major 3×3, diagonal = variance (stddev²).
# BNO08x ARVR-stabilised rotation vector: ~1–2° RMS ≈ 0.017–0.035 rad stddev
# Tune if RTAB-Map drift is excessive.
ORIENTATION_COVARIANCE = [
    0.0012, 0.0,    0.0,
    0.0,    0.0012, 0.0,
    0.0,    0.0,    0.0012,
]
ANGULAR_VELOCITY_COVARIANCE = [
    0.0001, 0.0,    0.0,
    0.0,    0.0001, 0.0,
    0.0,    0.0,    0.0001,
]
# Raw accel including gravity — RTAB-Map imu_topic expects this, not linear accel
LINEAR_ACCELERATION_COVARIANCE = [
    0.0025, 0.0,    0.0,
    0.0,    0.0025, 0.0,
    0.0,    0.0,    0.0025,
]


# ── ZMQ helpers ───────────────────────────────────────────────────────────────

def make_sub_socket(ctx: zmq.Context, ip: str, port: int,
                    timeout_ms: int = 50) -> zmq.Socket:
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM,   2)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.connect(f"tcp://{ip}:{port}")
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    return sock


def recv_frame(sock: zmq.Socket) -> np.ndarray | None:
    """
    Receive a shape-prefixed raw frame from ZMQ.
    Wire format: int32[2] (h,w) then raw pixels (uint8 BGR or uint16 depth).
    Returns ndarray or None on timeout.
    """
    try:
        data = sock.recv()
    except zmq.Again:
        return None
    h, w    = np.frombuffer(data[:8], dtype=np.int32)
    payload = data[8:]
    if len(payload) == h * w * 3:
        return np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 3)
    elif len(payload) == h * w * 2:
        return np.frombuffer(payload, dtype=np.uint16).reshape(h, w)
    return None


# ── ROS 2 message factories ───────────────────────────────────────────────────

def make_camera_info(stamp, frame_id, w, h, fx, fy, cx, cy) -> CameraInfo:
    ci = CameraInfo()
    ci.header.stamp    = stamp
    ci.header.frame_id = frame_id
    ci.width, ci.height = w, h
    ci.distortion_model = "plumb_bob"
    ci.d = DISTORTION
    ci.k = [fx,  0.0, cx,  0.0, fy,  cy,  0.0, 0.0, 1.0]
    ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ci.p = [fx,  0.0, cx,  0.0, 0.0, fy,  cy,  0.0, 0.0, 0.0, 1.0, 0.0]
    return ci


def make_image(stamp, frame_id, arr, encoding, step_mult) -> Image:
    msg = Image()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.step     = arr.shape[1] * step_mult
    msg.data     = arr.tobytes()
    return msg


def make_tf(stamp, parent, child,
            tx=0.0, ty=0.0, tz=0.0,
            qx=0.0, qy=0.0, qz=0.0, qw=1.0) -> TransformStamped:
    tf = TransformStamped()
    tf.header.stamp      = stamp
    tf.header.frame_id   = parent
    tf.child_frame_id    = child
    tf.transform.translation.x = tx
    tf.transform.translation.y = ty
    tf.transform.translation.z = tz
    tf.transform.rotation.x = qx
    tf.transform.rotation.y = qy
    tf.transform.rotation.z = qz
    tf.transform.rotation.w = qw
    return tf


# ── Bridge node ───────────────────────────────────────────────────────────────

class RGBDBridgeNode(Node):
    """
    Polls ZMQ for BGR frames, depth frames, and IMU packets;
    republishes all three to ROS 2.

    TF tree published here:
        odom (static identity, taken over by rgbd_odometry once tracking starts)
          └─ base_link
               ├─ camera_link   (fixed extrinsic — camera mount)
               └─ imu_link      (fixed extrinsic — IMU mount, corrected for tilt)
    """

    _ODOM_COV = [0.0] * 36
    for _i in (0, 7, 14):    _ODOM_COV[_i] = 1.0   # position ~1m stddev
    for _i in (21, 28, 35):  _ODOM_COV[_i] = 1.0   # rotation ~57° stddev

    def __init__(self, zmq_ctx: zmq.Context):
        super().__init__("ros2_scan_bridge")

        # ── Static transforms ─────────────────────────────────────────────────
        static_tf = tf2_ros.StaticTransformBroadcaster(self)
        init_stamp = self.get_clock().now().to_msg()
        static_tf.sendTransform([
            # odom → base_link: identity seed (rgbd_odometry takes over)
            make_tf(init_stamp, "odom",      "base_link"),
            # base_link → camera_link: camera extrinsic
            make_tf(init_stamp, "base_link", "camera_link",
                    qx=CAMERA_QX, qy=CAMERA_QY, qz=CAMERA_QZ, qw=CAMERA_QW),
            # base_link → imu_link: corrected for Y-down mounting + 8.4° tilt
            # qx=0.757, qw=0.653 = 98.4° rotation around X axis
            make_tf(init_stamp, "base_link", "imu_link",
                    tx=IMU_TF_X, ty=IMU_TF_Y, tz=IMU_TF_Z,
                    qx=IMU_QX,   qy=IMU_QY,   qz=IMU_QZ,   qw=IMU_QW),
        ])

        # ── Publishers ────────────────────────────────────────────────────────
        self.rgb_pub      = self.create_publisher(Image,      "/camera/color/image_raw",     10)
        self.rgb_info_pub = self.create_publisher(CameraInfo, "/camera/color/camera_info",    10)
        self.dep_pub      = self.create_publisher(Image,      "/camera/depth/image_rect_raw", 10)
        self.dep_info_pub = self.create_publisher(CameraInfo, "/camera/depth/camera_info",    10)
        self.imu_pub      = self.create_publisher(Imu,        "/imu/data",                   10)
        self.odom_pub     = self.create_publisher(Odometry,   "/odom",                       10)
        self.tf_broad     = tf2_ros.TransformBroadcaster(self)

        # ── ZMQ sockets ───────────────────────────────────────────────────────
        self.rgb_sub = make_sub_socket(zmq_ctx, JETSON_IP, RGB_BRIDGE_PORT)
        self.dep_sub = make_sub_socket(zmq_ctx, JETSON_IP, DEPTH_BRIDGE_PORT)
        # IMU: timeout_ms=1 (near non-blocking) — drain queue each spin
        # without holding up camera frames.
        self.imu_sub = make_sub_socket(zmq_ctx, JETSON_IP, IMU_BRIDGE_PORT, timeout_ms=1)

        # Intrinsics scaled for the published resolution
        s = 1.0 / max(1, DOWNSAMPLE)
        self._fx, self._fy = FX * s, FY * s
        self._cx, self._cy = CX * s, CY * s

        self._frames     = 0
        self._imu_frames = 0
        self._t_warn     = time.monotonic()

        self.get_logger().info(
            f"Bridge ready — rgb:{RGB_BRIDGE_PORT} depth:{DEPTH_BRIDGE_PORT} "
            f"imu:{IMU_BRIDGE_PORT} downsample:{DOWNSAMPLE}x\n"
            f"IMU TF: pos=({IMU_TF_X}, {IMU_TF_Y}, {IMU_TF_Z}) "
            f"quat=({IMU_QX:.3f}, {IMU_QY:.3f}, {IMU_QZ:.3f}, {IMU_QW:.3f})"
        )

    # ── IMU ───────────────────────────────────────────────────────────────────

    def _publish_imu(self) -> None:
        """
        Drain the latest IMU packet from ZMQ and publish it.
        CONFLATE=1 on the ZMQ socket keeps only the newest packet, so we get
        one per poll at ~30 Hz. This is fine for RTAB-Map; remove CONFLATE
        and raise RCVHWM if you need every 100 Hz sample for an EKF.
        """
        try:
            raw = self.imu_sub.recv()
        except zmq.Again:
            return
        except Exception as e:
            self.get_logger().warn(f"IMU ZMQ error: {e}", throttle_duration_sec=2.0)
            return

        if len(raw) != IMU_PACK_SIZE:
            self.get_logger().warn(
                f"Bad IMU packet: {len(raw)}B (expected {IMU_PACK_SIZE}B)",
                throttle_duration_sec=5.0,
            )
            return

        (_, ax, ay, az, gx, gy, gz, qi, qj, qk, qr) = struct.unpack(IMU_PACK_FMT, raw)

        msg = Imu()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"

        # Quaternion: adafruit order (i, j, k, real) → ROS order (x, y, z, w)
        msg.orientation.x = qi
        msg.orientation.y = qj
        msg.orientation.z = qk
        msg.orientation.w = qr
        msg.orientation_covariance = ORIENTATION_COVARIANCE

        msg.angular_velocity.x = gx
        msg.angular_velocity.y = gy
        msg.angular_velocity.z = gz
        msg.angular_velocity_covariance = ANGULAR_VELOCITY_COVARIANCE

        # Raw accel including gravity — RTAB-Map expects this (not gravity-free)
        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az
        msg.linear_acceleration_covariance = LINEAR_ACCELERATION_COVARIANCE

        self.imu_pub.publish(msg)
        self._imu_frames += 1

        if self._imu_frames == 1:
            self.get_logger().info("First IMU sample received — /imu/data publishing")

    # ── Odometry + dynamic TF ─────────────────────────────────────────────────

    def _publish_odom(self, stamp) -> None:
        # Stub /odom so rtabmap's approx_sync has something to pair with
        # camera frames before rgbd_odometry starts publishing real odometry.
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id  = "base_link"
        odom.pose.pose.orientation.w = 1.0
        odom.pose.covariance  = list(self._ODOM_COV)
        odom.twist.covariance = list(self._ODOM_COV)
        self.odom_pub.publish(odom)

    # ── Camera ────────────────────────────────────────────────────────────────

    def _publish_cameras(self) -> None:
        rgb = recv_frame(self.rgb_sub)
        dep = recv_frame(self.dep_sub)

        # Single clock call — identical stamp for RGB and depth so
        # rgbd_odometry's approx_sync pairs them with zero interval.
        stamp = self.get_clock().now().to_msg()

        if rgb is not None:
            if DOWNSAMPLE > 1:
                rgb = cv2.resize(rgb, (rgb.shape[1] // DOWNSAMPLE,
                                       rgb.shape[0] // DOWNSAMPLE))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            self.rgb_pub.publish(make_image(stamp, "camera_link", rgb, "rgb8", 3))
            self.rgb_info_pub.publish(make_camera_info(
                stamp, "camera_link",
                w=rgb.shape[1], h=rgb.shape[0],
                fx=self._fx, fy=self._fy, cx=self._cx, cy=self._cy,
            ))
            self._frames += 1
            if self._frames == 1:
                self.get_logger().info(f"First colour frame: {rgb.shape[1]}×{rgb.shape[0]}")

        if dep is not None:
            if DOWNSAMPLE > 1:
                dep = cv2.resize(dep,
                                 (dep.shape[1] // DOWNSAMPLE,
                                  dep.shape[0] // DOWNSAMPLE),
                                 interpolation=cv2.INTER_NEAREST)
            self.dep_pub.publish(make_image(stamp, "camera_link", dep, "16UC1", 2))
            self.dep_info_pub.publish(make_camera_info(
                stamp, "camera_link",
                w=dep.shape[1], h=dep.shape[0],
                fx=self._fx, fy=self._fy, cx=self._cx, cy=self._cy,
            ))

    # ── Main spin ─────────────────────────────────────────────────────────────

    def spin_once(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self._publish_imu()        # drain IMU queue first (fastest source)
        self._publish_odom(stamp)
        self._publish_cameras()

        now = time.monotonic()
        if self._frames == 0 and (now - self._t_warn) > CONNECT_TIMEOUT:
            self.get_logger().warn(
                f"No camera frames in {CONNECT_TIMEOUT}s — is jetson_combined.py running?"
            )
            self._t_warn = now

        if self._imu_frames == 0 and (now - self._t_warn) > CONNECT_TIMEOUT:
            self.get_logger().warn(
                "No IMU frames received — is BNO08x wired and adafruit_bno08x installed?"
            )
            self._t_warn = now

    def destroy(self) -> None:
        self.rgb_sub.close()
        self.dep_sub.close()
        self.imu_sub.close()
        super().destroy_node()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    rclpy.init()
    ctx  = zmq.Context()
    node = RGBDBridgeNode(ctx)
    try:
        while rclpy.ok():
            node.spin_once()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"Stopped — {node._frames} colour frames, {node._imu_frames} IMU samples"
        )
        node.destroy()
        rclpy.shutdown()
        ctx.term()


if __name__ == "__main__":
    main()

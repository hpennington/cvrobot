#!/usr/bin/env python3
"""
ros2_scan_bridge.py
-------------------
Receives raw BGR + uint16 depth frames from jetson_combined.py over ZMQ
and republishes them as ROS 2 topics for RTAB-Map:

  /camera/color/image_raw        sensor_msgs/Image   (BGR8)
  /camera/color/camera_info      sensor_msgs/CameraInfo
  /camera/depth/image_rect_raw   sensor_msgs/Image   (16UC1, millimetres)
  /camera/depth/camera_info      sensor_msgs/CameraInfo
  /odom                          nav_msgs/Odometry   (identity + max covariance)
  /tf                            odom → base_link → camera_link

RGB and depth frames are stamped identically so RTAB-Map's approx_sync
pairs them without timestamp drift errors.

Run in a sourced ROS 2 terminal (system Python, not conda):

    source /opt/ros/jazzy/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    python3 ~/ros2_ws/ros2_scan_bridge.py

Then launch RTAB-Map:

    ros2 launch rtabmap_launch rtabmap.launch.py \
      rtabmap_args:="--delete_db_on_start" \
      rgb_topic:=/camera/color/image_raw \
      depth_topic:=/camera/depth/image_rect_raw \
      camera_info_topic:=/camera/color/camera_info \
      approx_sync:=true \
      frame_id:=base_link \
      visual_odometry:=true \
      rtabmap_viz:=false
"""

import sys
import time

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
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

# RealSense D415 exact intrinsics at 640×480 from rs-enumerate-devices -c
# These are the colour camera values (used for RGB-D SLAM with RTAB-Map)
FX, FY = 601.023, 601.023
CX, CY = 320.797, 242.064
DISTORTION = [0.0, 0.0, 0.0, 0.0, 0.0]   # D415 colour stream is near-zero

# Publish at half resolution (320×240) so rgbd_odometry runs at full frame rate.
# Set to 1 to disable.
DOWNSAMPLE = 2

# camera_link orientation relative to base_link.
# RealSense D415 standard mounting: lens forward, USB port down.
# The optical frame has Z forward, X right, Y down.
# To convert to ROS convention (X forward, Y left, Z up) we apply:
# -90° around X then -90° around Z  →  qx=-0.5, qy=0.5, qz=-0.5, qw=0.5
# Adjust if your camera is mounted differently.
CAMERA_QX, CAMERA_QY, CAMERA_QZ, CAMERA_QW = -0.5, 0.5, -0.5, 0.5

# Warn if no frames arrive within this many seconds of startup
CONNECT_TIMEOUT = 10.0


# ── ZMQ helpers ───────────────────────────────────────────────────────────────

def make_sub_socket(ctx: zmq.Context, ip: str, port: int) -> zmq.Socket:
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM,   2)    # drop old frames if consumer is slow
    sock.setsockopt(zmq.CONFLATE, 1)    # keep only the latest message
    sock.setsockopt(zmq.RCVTIMEO, 50)  # ms — non-blocking poll
    sock.connect(f"tcp://{ip}:{port}")
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    return sock


def recv_frame(sock: zmq.Socket) -> np.ndarray | None:
    """
    Receive a shape-prefixed raw frame from ZMQ.

    Wire format (from jetson_combined.py):
      bytes 0-7   : int32[2]  →  [height, width]
      bytes 8-end : raw pixel data
        BGR  frames: uint8,  h×w×3 bytes
        depth frames: uint16, h×w×2 bytes

    Returns an ndarray or None on timeout.
    """
    try:
        data = sock.recv()
    except zmq.Again:
        return None

    h, w = np.frombuffer(data[:8], dtype=np.int32)
    payload = data[8:]

    if len(payload) == h * w * 3:                          # BGR
        return np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 3)
    elif len(payload) == h * w * 2:                        # depth uint16
        return np.frombuffer(payload, dtype=np.uint16).reshape(h, w)
    return None


# ── ROS 2 message factories ───────────────────────────────────────────────────

def make_camera_info(stamp, frame_id: str, w: int, h: int,
                     fx: float, fy: float, cx: float, cy: float) -> CameraInfo:
    ci = CameraInfo()
    ci.header.stamp    = stamp
    ci.header.frame_id = frame_id
    ci.width           = w
    ci.height          = h
    ci.distortion_model = "plumb_bob"
    ci.d = DISTORTION
    ci.k = [fx,  0.0, cx,
            0.0, fy,  cy,
            0.0, 0.0, 1.0]
    ci.r = [1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0]
    ci.p = [fx,  0.0, cx,  0.0,
            0.0, fy,  cy,  0.0,
            0.0, 0.0, 1.0, 0.0]
    return ci


def make_image(stamp, frame_id: str, arr: np.ndarray,
               encoding: str, step_multiplier: int) -> Image:
    msg = Image()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.height   = arr.shape[0]
    msg.width    = arr.shape[1]
    msg.encoding = encoding
    msg.step     = arr.shape[1] * step_multiplier
    msg.data     = arr.tobytes()
    return msg


def make_tf(stamp, parent: str, child: str,
            qx=0.0, qy=0.0, qz=0.0, qw=1.0) -> TransformStamped:
    tf = TransformStamped()
    tf.header.stamp    = stamp
    tf.header.frame_id = parent
    tf.child_frame_id  = child
    tf.transform.rotation.x = qx
    tf.transform.rotation.y = qy
    tf.transform.rotation.z = qz
    tf.transform.rotation.w = qw
    return tf


# ── Bridge node ───────────────────────────────────────────────────────────────

class RGBDBridgeNode(Node):
    """
    Polls ZMQ for BGR + depth frames and republishes to ROS 2.

    Odometry is published as an identity pose with maximum covariance,
    signalling to RTAB-Map that wheel odometry is unavailable and it
    should use its own visual odometry (rgbd_odometry) for motion estimation.
    """

    # Large but finite covariance — signals "odometry unknown" to RTAB-Map
    # without triggering its sanity check (which fires above ~1000).
    # Diagonal indices: 0=x, 7=y, 14=z, 21=roll, 28=pitch, 35=yaw
    _ODOM_COV = [0.0] * 36
    for _i in (0, 7, 14):       # position: 1m std dev
        _ODOM_COV[_i] = 1.0
    for _i in (21, 28, 35):     # rotation: ~57° std dev
        _ODOM_COV[_i] = 1.0

    def __init__(self, zmq_ctx: zmq.Context):
        super().__init__("ros2_scan_bridge")

        # Publish odom→base_link once as static identity at startup.
        # rgbd_odometry will immediately take over this TF once it starts tracking.
        # This prevents "could not find transform" errors during the first few frames.
        static_tf_broad = tf2_ros.StaticTransformBroadcaster(self)
        from builtin_interfaces.msg import Time as _Time
        _init_stamp = self.get_clock().now().to_msg()
        static_tf_broad.sendTransform([
            make_tf(_init_stamp, "odom", "base_link"),
        ])

        self.rgb_pub      = self.create_publisher(Image,      "/camera/color/image_raw",      10)
        self.rgb_info_pub = self.create_publisher(CameraInfo, "/camera/color/camera_info",     10)
        self.dep_pub      = self.create_publisher(Image,      "/camera/depth/image_rect_raw",  10)
        self.dep_info_pub = self.create_publisher(CameraInfo, "/camera/depth/camera_info",     10)
        self.odom_pub     = self.create_publisher(Odometry,   "/odom",                         10)
        self.tf_broad     = tf2_ros.TransformBroadcaster(self)

        self.rgb_sub = make_sub_socket(zmq_ctx, JETSON_IP, RGB_BRIDGE_PORT)
        self.dep_sub = make_sub_socket(zmq_ctx, JETSON_IP, DEPTH_BRIDGE_PORT)

        # Intrinsics scaled for the published resolution
        self._scale = 1.0 / max(1, DOWNSAMPLE)
        self._fx = FX * self._scale
        self._fy = FY * self._scale
        self._cx = CX * self._scale
        self._cy = CY * self._scale

        self._frames = 0
        self._t_warn = time.monotonic()

        self.get_logger().info(
            f"Bridge ready — rgb:{RGB_BRIDGE_PORT} depth:{DEPTH_BRIDGE_PORT} "
            f"downsample:{DOWNSAMPLE}x"
        )

    def _publish_tf_and_odom(self, stamp) -> None:
        # Only publish base_link→camera_link here.
        # odom→base_link is owned by rgbd_odometry — publishing it here
        # too causes a TF conflict that produces impossible pose guesses
        # when rgbd_odometry resets after tracking loss.
        self.tf_broad.sendTransform([
            make_tf(stamp, "base_link", "camera_link",
                    qx=CAMERA_QX, qy=CAMERA_QY,
                    qz=CAMERA_QZ, qw=CAMERA_QW),
        ])
        # Publish a stub /odom message so rtabmap's approx_sync has something
        # to pair with camera frames. rgbd_odometry will override this with
        # its own real odometry on the same topic.
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id  = "base_link"
        odom.pose.pose.orientation.w = 1.0
        odom.pose.covariance  = list(self._ODOM_COV)
        odom.twist.covariance = list(self._ODOM_COV)
        self.odom_pub.publish(odom)

    def _publish_rgb_and_depth(self, stamp) -> None:
        """
        Receive RGB and depth frames and publish them with the SAME timestamp.

        Both frames are stamped with a single clock.now() call so that
        rgbd_odometry's exact-sync (approx_sync=false) can pair them,
        and approx_sync=true pairs them immediately with zero interval.
        """
        rgb = recv_frame(self.rgb_sub)
        dep = recv_frame(self.dep_sub)

        # Shared stamp — one call, same object for both messages
        frame_stamp = self.get_clock().now().to_msg()

        if rgb is not None:
            if DOWNSAMPLE > 1:
                rgb = cv2.resize(rgb, (rgb.shape[1] // DOWNSAMPLE,
                                       rgb.shape[0] // DOWNSAMPLE))
            # Convert BGR (OpenCV default) to RGB (ROS/Foxglove convention)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            self.rgb_pub.publish(make_image(frame_stamp, "camera_link", rgb, "rgb8", 3))
            self.rgb_info_pub.publish(make_camera_info(
                frame_stamp, "camera_link",
                w=rgb.shape[1], h=rgb.shape[0],
                fx=self._fx, fy=self._fy,
                cx=self._cx, cy=self._cy,
            ))
            self._frames += 1
            if self._frames == 1:
                self.get_logger().info(
                    f"First colour frame: {rgb.shape[1]}×{rgb.shape[0]}"
                )

        if dep is not None:
            if DOWNSAMPLE > 1:
                dep = cv2.resize(dep,
                                 (dep.shape[1] // DOWNSAMPLE,
                                  dep.shape[0] // DOWNSAMPLE),
                                 interpolation=cv2.INTER_NEAREST)
            self.dep_pub.publish(make_image(frame_stamp, "camera_link", dep, "16UC1", 2))
            self.dep_info_pub.publish(make_camera_info(
                frame_stamp, "camera_link",
                w=dep.shape[1], h=dep.shape[0],
                fx=self._fx, fy=self._fy,
                cx=self._cx, cy=self._cy,
            ))

    def spin_once(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self._publish_tf_and_odom(stamp)
        self._publish_rgb_and_depth(stamp)

        if self._frames == 0 and (time.monotonic() - self._t_warn) > CONNECT_TIMEOUT:
            self.get_logger().warn(
                f"No frames received in {CONNECT_TIMEOUT}s — "
                "is jetson_combined.py running?"
            )
            self._t_warn = time.monotonic()

    def destroy(self) -> None:
        self.rgb_sub.close()
        self.dep_sub.close()
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
        node.get_logger().info(f"Stopped after {node._frames} colour frames")
        node.destroy()
        rclpy.shutdown()
        ctx.term()


if __name__ == "__main__":
    main()

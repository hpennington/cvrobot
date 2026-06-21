#!/usr/bin/env python3
"""
ros2_scan_bridge.py
-------------------
Reads BNO08x IMU directly over I2C in a background thread and publishes
/imu/data, /odom (identity), and static TF for RTAB-Map. Color/depth now
come straight from the realsense2_camera node (see slam_launch.py) instead
of being forwarded here — the old ZMQ image-forwarding path was capped at
~14fps by per-frame Python message construction, so it was dropped in
favor of the official C++ camera driver.

Publishes:
  /imu/data    sensor_msgs/Imu        (IMU_HZ, BNO08x)
  /odom        nav_msgs/Odometry      (identity + max covariance)
  /tf          static: odom->base_link, base_link->camera_link,
                        base_link->imu_link

RTAB-Map IMU integration (slam_launch.py):
    rgbd_odometry remappings:  ('imu', '/imu/data')
    rgbd_odometry params:      'Odom/GuessMotion': 'true'
                               'Odom/GuessIMU':    'true'
    rtabmap params:            'Optimizer/GravitySigma': '0.3'

Run in a sourced ROS 2 terminal (system Python, not conda):

    source /opt/ros/jazzy/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    python3 ~/ros2_ws/ros2_scan_bridge.py
"""

import sys
import time
import threading

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Imu
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import TransformStamped
    import tf2_ros
except ImportError:
    sys.exit(
        "ERROR: rclpy not found.\n"
        "Source ROS 2 before running:\n"
        "  source /opt/ros/humble/setup.bash"
    )


# ── Config ────────────────────────────────────────────────────────────────────

# camera_link quaternion relative to base_link — physical mounting rotation
# of the D415 on the robot body. realsense2_camera publishes its own
# camera_link -> camera_color_optical_frame transform (REP103 body->optical
# conversion), so this should NOT also bake in that optical-frame rotation —
# verify orientation in RViz/Foxglove after wiring up and adjust if rotated.
CAMERA_QX, CAMERA_QY, CAMERA_QZ, CAMERA_QW = 0.0, 0.0, 0.0, 1.0

ODOM_HZ = 30.0   # /odom publish rate

# ── IMU config ────────────────────────────────────────────────────────────────

IMU_HZ       = 200    # BNO08x report rate — must match interval_us below
IMU_I2C_ADDR = 0x4B   # default BNO08x address on Yahboom carrier

# ── IMU extrinsic (base_link → imu_link) ─────────────────────────────────────
# Translation: physical offset of IMU from base_link origin in metres.
IMU_TF_X, IMU_TF_Y, IMU_TF_Z = 0.05, 0.0, 0.0

# Rotation: corrects for the BNO08x mounting orientation.
#
# Measured with robot flat and stationary:
#   raw accel  x≈+0.13  y≈+9.80  z≈-1.42
#
# IMU Y axis points DOWN (y ≈ +g), tilted ~8.4° off horizontal.
# Net rotation around IMU X axis: 90° + 8.4° = 98.4°
#
# Quaternion: qx = sin(49.2°) ≈ 0.757, qw = cos(49.2°) ≈ 0.653
IMU_QX, IMU_QY, IMU_QZ, IMU_QW = 0.757, 0.0, 0.0, 0.653

CONNECT_TIMEOUT = 10.0   # warn if no IMU frames within this many seconds

# ── IMU covariance matrices ───────────────────────────────────────────────────
# Row-major 3×3, diagonal = variance (stddev²).
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
# Raw accel including gravity — RTAB-Map imu_topic expects this, not gravity-free
LINEAR_ACCELERATION_COVARIANCE = [
    0.0025, 0.0,    0.0,
    0.0,    0.0025, 0.0,
    0.0,    0.0,    0.0025,
]

# ── Odometry covariance ───────────────────────────────────────────────────────
# Defined at module level to avoid the class-scope for-loop leaking _i into
# the class namespace, which causes _ODOM_COV to silently fail as an attribute.
_ODOM_COV = [0.0] * 36
for _cov_i in (0, 7, 14):    _ODOM_COV[_cov_i] = 1.0   # position ~1m stddev
for _cov_i in (21, 28, 35):  _ODOM_COV[_cov_i] = 1.0   # rotation ~57° stddev


# ── ROS 2 message factories ───────────────────────────────────────────────────

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
    Reads BNO08x directly over I2C in a daemon thread (_imu_loop) and
    publishes /imu/data, /odom, and static TF.

    TF tree:
        odom
          └─ base_link
               ├─ camera_link   (realsense2_camera owns frames below this)
               └─ imu_link
    """

    def __init__(self):
        super().__init__("ros2_scan_bridge")

        # ── Static transforms ─────────────────────────────────────────────────
        static_tf  = tf2_ros.StaticTransformBroadcaster(self)
        init_stamp = self.get_clock().now().to_msg()
        static_tf.sendTransform([
            make_tf(init_stamp, "odom",      "base_link"),
            make_tf(init_stamp, "base_link", "camera_link",
                    qx=CAMERA_QX, qy=CAMERA_QY, qz=CAMERA_QZ, qw=CAMERA_QW),
            make_tf(init_stamp, "base_link", "imu_link",
                    tx=IMU_TF_X, ty=IMU_TF_Y, tz=IMU_TF_Z,
                    qx=IMU_QX,   qy=IMU_QY,   qz=IMU_QZ,   qw=IMU_QW),
        ])

        # ── Publishers ────────────────────────────────────────────────────────
        self.imu_pub  = self.create_publisher(Imu,      "/imu/data", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom",     10)
        self.tf_broad = tf2_ros.TransformBroadcaster(self)

        self._imu_frames = 0
        self._t_warn      = time.monotonic()
        self._running     = True

        # ── Odometry timer ────────────────────────────────────────────────────
        self.create_timer(1.0 / ODOM_HZ, self._publish_odom)

        # ── IMU thread ────────────────────────────────────────────────────────
        self._imu_thread = threading.Thread(
            target=self._imu_loop, daemon=True, name="imu_loop"
        )
        self._imu_thread.start()

        # ── Connect-timeout watchdog ──────────────────────────────────────────
        self.create_timer(1.0, self._check_imu_connected)

        self.get_logger().info(
            f"Bridge ready — imu:direct@{IMU_HZ}Hz odom:{ODOM_HZ}Hz\n"
            f"IMU TF: pos=({IMU_TF_X}, {IMU_TF_Y}, {IMU_TF_Z}) "
            f"quat=({IMU_QX:.3f}, {IMU_QY:.3f}, {IMU_QZ:.3f}, {IMU_QW:.3f})"
        )

    # ── IMU thread ────────────────────────────────────────────────────────────

    def _imu_loop(self) -> None:
        try:
            import board
            import busio
            from adafruit_bno08x.i2c import BNO08X_I2C
            from adafruit_bno08x import (
                BNO_REPORT_GYROSCOPE,
                BNO_REPORT_ROTATION_VECTOR,
            )
        except ImportError as e:
            self.get_logger().error(
                f"IMU dependencies missing: {e}\n"
                "Install with: pip install adafruit-circuitpython-bno08x"
            )
            return

        interval_us     = 10_000
        period          = 1.0 / IMU_HZ
        reconnect_delay = 3.0

        try:
            import os as _os
            param = _os.sched_param(_os.sched_get_priority_max(_os.SCHED_FIFO) - 1)
            _os.sched_setscheduler(0, _os.SCHED_FIFO, param)
            self.get_logger().info("IMU thread: SCHED_FIFO priority set")
        except Exception:
            pass

        # ── Pre-allocate message — reuse every iteration ──────────────────
        msg = Imu()
        msg.header.frame_id = "imu_link"
        msg.orientation_covariance        = ORIENTATION_COVARIANCE
        msg.angular_velocity_covariance   = ANGULAR_VELOCITY_COVARIANCE
        msg.linear_acceleration.x         = 0.0
        msg.linear_acceleration.y         = 0.0
        msg.linear_acceleration.z         = 0.0
        msg.linear_acceleration_covariance = [
            9999.0, 0.0, 0.0,  0.0, 9999.0, 0.0,  0.0, 0.0, 9999.0,
        ]

        # ── Cache attribute lookups to locals ─────────────────────────────
        stamp   = msg.header.stamp
        orient  = msg.orientation
        ang_vel = msg.angular_velocity
        publish = self.imu_pub.publish
        log     = self.get_logger()
        _time_ns   = time.time_ns
        _monotonic = time.monotonic
        _sleep     = time.sleep
        _NS        = 1_000_000_000

        while self._running:
            try:
                i2c = busio.I2C(board.SCL, board.SDA, frequency=200_000)
                bno = BNO08X_I2C(i2c, address=IMU_I2C_ADDR)

                bno.enable_feature(BNO_REPORT_GYROSCOPE,       interval_us)
                bno.enable_feature(BNO_REPORT_ROTATION_VECTOR, interval_us)

                log.info(f"BNO08x ready at 0x{IMU_I2C_ADDR:02X}, {IMU_HZ} Hz")

                next_deadline = _monotonic()
                count = 0

                while self._running:
                    gyro = bno.gyro
                    quat = bno.quaternion

                    if gyro is None or quat is None:
                        next_deadline += period
                        rem = next_deadline - _monotonic()
                        if rem > 0:
                            _sleep(rem)
                        continue

                    # Direct timestamp — avoids get_clock().now().to_msg()
                    # object chain (Time + builtin_interfaces.msg.Time allocs)
                    t_ns          = _time_ns()
                    stamp.sec     = t_ns // _NS
                    stamp.nanosec = t_ns % _NS

                    orient.x  = quat[0]
                    orient.y  = quat[1]
                    orient.z  = quat[2]
                    orient.w  = quat[3]

                    ang_vel.x = gyro[0]
                    ang_vel.y = gyro[1]
                    ang_vel.z = gyro[2]

                    publish(msg)
                    count += 1

                    if count == 1:
                        log.info("First IMU sample — /imu/data publishing")

                    self._imu_frames = count

                    next_deadline += period
                    rem = next_deadline - _monotonic()
                    if rem > 0:
                        _sleep(rem)
                    elif rem < -period:
                        next_deadline = _monotonic()

            except Exception as e:
                log.warn(
                    f"IMU error: {e} — reconnecting in {reconnect_delay}s",
                    throttle_duration_sec=5.0,
                )
                _sleep(reconnect_delay)

    # ── Odometry ──────────────────────────────────────────────────────────────

    def _publish_odom(self) -> None:
        odom = Odometry()
        odom.header.stamp    = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id  = "base_link"
        odom.pose.pose.orientation.w = 1.0
        odom.pose.covariance  = list(_ODOM_COV)
        odom.twist.covariance = list(_ODOM_COV)
        self.odom_pub.publish(odom)

    # ── Watchdog ──────────────────────────────────────────────────────────────

    def _check_imu_connected(self) -> None:
        now = time.monotonic()
        if self._imu_frames == 0 and (now - self._t_warn) > CONNECT_TIMEOUT:
            self.get_logger().warn(
                "No IMU frames — check BNO08x wiring and adafruit-circuitpython-bno08x install"
            )
            self._t_warn = now

    def destroy(self) -> None:
        self._running = False
        super().destroy_node()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    rclpy.init()
    node = RGBDBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Stopped — {node._imu_frames} IMU samples")
        node.destroy()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

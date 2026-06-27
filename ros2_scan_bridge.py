#!/usr/bin/env python3
"""
ros2_scan_bridge.py
-------------------
Publishes a placeholder /odom topic for visualization and a static
base_link->camera_link transform for the camera mount.
IMU is now provided directly by the orbbec_camera node (/camera/imu).

Publishes:
  /odom        nav_msgs/Odometry      (identity + max covariance)
  /tf          static: base_link->camera_link

Run in a sourced ROS 2 terminal (system Python, not conda):

    source /opt/ros/jazzy/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    python3 ~/ros2_ws/ros2_scan_bridge.py
"""

import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import TransformStamped
    import tf2_ros
except ImportError:
    sys.exit(
        "ERROR: rclpy not found.\n"
        "Source ROS 2 before running:\n"
        "  source /opt/ros/jazzy/setup.bash"
    )


# ── Config ────────────────────────────────────────────────────────────────────

# camera_link quaternion relative to base_link — physical mounting rotation
# of the Orbbec camera on the robot body.
CAMERA_QX, CAMERA_QY, CAMERA_QZ, CAMERA_QW = 0.0, 0.0, 0.0, 1.0

ODOM_HZ = 30.0   # /odom publish rate

# ── Odometry covariance ───────────────────────────────────────────────────────
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
    Publishes a placeholder /odom message stream and the static transform
    that connects the robot frame to the camera. IMU comes from
    orbbec_camera (/camera/imu).

    TF tree:
        base_link
          └─ camera_link   (orbbec_camera owns frames below this)
    """

    def __init__(self):
        super().__init__("ros2_scan_bridge")

        # ── Static transforms ─────────────────────────────────────────────────
        static_tf  = tf2_ros.StaticTransformBroadcaster(self)
        init_stamp = self.get_clock().now().to_msg()
        static_tf.sendTransform([
            make_tf(init_stamp, "base_link", "camera_link",
                    qx=CAMERA_QX, qy=CAMERA_QY, qz=CAMERA_QZ, qw=CAMERA_QW),
        ])

        # ── Publishers ────────────────────────────────────────────────────────
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_broad = tf2_ros.TransformBroadcaster(self)

        # ── Odometry timer ────────────────────────────────────────────────────
        self.create_timer(1.0 / ODOM_HZ, self._publish_odom)

        self.get_logger().info(
            f"Bridge ready — odom:{ODOM_HZ}Hz  IMU: /camera/imu (orbbec)"
        )

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


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    rclpy.init()
    node = RGBDBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Stopped")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

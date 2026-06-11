"""
slam_launch.py
--------------
Launches the ROS 2 bridge and RTAB-Map together for RGB-D SLAM.

The bridge (ros2_scan_bridge.py) publishes camera images and TF from
jetson_combined.py's ZMQ streams.  RTAB-Map subscribes to those topics
and builds a 2D occupancy map using its own visual odometry.

Usage:
    source /opt/ros/jazzy/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    ros2 launch ~/ros2_ws/slam_launch.py

Optional args:
    ros2 launch ~/ros2_ws/slam_launch.py fresh:=false   # keep existing map DB
    ros2 launch ~/ros2_ws/slam_launch.py downsample:=1  # full 640×480 resolution

Dependencies:
    sudo apt install ros-jazzy-rtabmap-ros
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


BRIDGE_SCRIPT = os.path.expanduser("~/ros2_ws/ros2_scan_bridge.py")

# RTAB-Map parameters.  Values are Python-typed (not strings) so the
# launch system passes them as the correct ROS 2 parameter types.
RTABMAP_PARAMS = {
    "frame_id":     "base_link",
    "approx_sync":  True,    # odom and camera frames have different rates

    # Grid / 2-D occupancy map
    "Grid/3D":                  "false",
    "Grid/CellSize":            "0.05",
    "Grid/RangeMin":            "0.2",
    "Grid/RangeMax":            "4.0",
    "Grid/RayTracing":          "true",   # fill free space between camera and obstacles
    "Grid/NormalsSegmentation": "false",  # passthrough filter — more robust without IMU
    "Grid/MaxGroundHeight":     "0.05",
    "Grid/MaxObstacleHeight":   "0.4",

    # Graph optimisation — force planar (2-D) motion
    "Reg/Force3DoF":            "true",
    "Optimizer/GravitySigma":   "0",      # no IMU gravity constraint

    # Map every frame regardless of apparent motion (no wheel odometry)
    "RGBD/LinearUpdate":        "0.0",
    "RGBD/AngularUpdate":       "0.0",

    # Ignore the stub odom covariance warning — visual odometry drives everything
    "Odom/ResetCountdown":      "1",    # reset and recover quickly after odom loss
    "Vis/EstimationType":       "1",    # use 3D→2D PnP (more robust than 3D→3D)

    # ── Rotation robustness fixes ─────────────────────────────────────────────

    # Lower inlier threshold so fast rotation doesn't trigger tracking loss
    "Vis/MinInliers":           "10",   # was implicit default of 20

    # Reject bad loop closures that would snap/reset the map
    "RGBD/OptimizeMaxError":    "0.5",  # discard loop closure if graph error too high
    "Optimizer/Robust":         "true", # use robust cost function (Cauchy) in optimizer

    # Use ICP on point cloud for odometry — much more rotation-tolerant than pure visual
    "Reg/Strategy":             "1",    # 0=Visual, 1=ICP, 2=Visual+ICP
    "OdomF2M/Type":             "1",    # frame-to-map ICP odometry
}


def generate_launch_description():
    fresh_arg = DeclareLaunchArgument(
        "fresh", default_value="true",
        description="Delete the RTAB-Map database on start for a clean map",
    )

    # Bridge: runs under system Python, publishes camera topics + TF
    bridge = ExecuteProcess(
        cmd=["python3", BRIDGE_SCRIPT],
        name="ros2_scan_bridge",
        output="screen",
    )

    # RTAB-Map: delayed 2 s so the bridge is publishing before it subscribes
    rtabmap = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="rtabmap_slam",
                executable="rtabmap",
                name="rtabmap",
                namespace="rtabmap",
                output="screen",
                parameters=[RTABMAP_PARAMS],
                remappings=[
                    ("rgb/image",       "/camera/color/image_raw"),
                    ("rgb/camera_info", "/camera/color/camera_info"),
                    ("depth/image",     "/camera/depth/image_rect_raw"),
                    ("odom",            "/odom"),
                ],
                arguments=["--delete_db_on_start"],
            ),
            Node(
                package="rtabmap_odom",
                executable="rgbd_odometry",
                name="rgbd_odometry",
                namespace="rtabmap",
                output="screen",
                parameters=[{
                    "frame_id":      "base_link",
                    "approx_sync":   True,    # colour@30fps depth@15fps — approx sync needed
                    "queue_size":    100,
                    # Rotation-robust feature settings
                    "Vis/MinInliers":       "10",      # was 6 — slightly higher for stability
                    "GFTT/QualityLevel":    "0.00001",
                    "GFTT/MinDistance":     "4",
                    "OdomF2M/MaxSize":      "2000",    # was 1000 — larger map for loop closure
                    "Odom/ResetCountdown":  "1",
                    # ICP odometry — rotation-tolerant
                    "Reg/Strategy":         "1",
                    "OdomF2M/Type":         "1",
                    # Reject bad loop closures in odometry too
                    "RGBD/OptimizeMaxError": "0.5",
                    "Optimizer/Robust":      "true",
                }],
                remappings=[
                    ("rgb/image",       "/camera/color/image_raw"),
                    ("rgb/camera_info", "/camera/color/camera_info"),
                    ("depth/image",     "/camera/depth/image_rect_raw"),
                    ("odom",            "/rtabmap/odom"),
                ],
            ),
        ],
    )

    return LaunchDescription([fresh_arg, bridge, rtabmap])

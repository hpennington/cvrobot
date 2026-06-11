"""
slam_launch.py
--------------
Launches the ROS 2 bridge and RTAB-Map together for RGB-D SLAM.

Usage:
    source /opt/ros/jazzy/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    ros2 launch ~/ros2_ws/slam_launch.py

Optional args:
    ros2 launch ~/ros2_ws/slam_launch.py fresh:=false   # keep existing map DB
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


BRIDGE_SCRIPT = os.path.expanduser("~/ros2_ws/ros2_scan_bridge.py")

RTABMAP_PARAMS = {
    "frame_id":    "base_link",
    "approx_sync": True,

    # 2-D occupancy grid
    "Grid/3D":                  "false",
    "Grid/CellSize":            "0.05",
    "Grid/RangeMin":            "0.2",
    "Grid/RangeMax":            "4.0",
    "Grid/RayTracing":          "true",
    "Grid/NormalsSegmentation": "false",
    "Grid/MaxGroundHeight":     "0.05",
    "Grid/MaxObstacleHeight":   "0.4",

    # Planar (2-D) motion constraint
    "Reg/Force3DoF":          "true",
    "Optimizer/GravitySigma": "0",

    # Map every frame (no wheel odometry to gate on)
    "RGBD/LinearUpdate":  "0.0",
    "RGBD/AngularUpdate": "0.0",

    # ── Odom-loss recovery (the main fix) ────────────────────────────────────
    # When odom resets, merge back via loop closure instead of starting a new map
    "Rtabmap/StartNewMapOnLoopClosure": "true",

    # Graph optimizer: disable error threshold so recovered loop closures
    # aren't rejected due to accumulated drift after an odom loss event
    "RGBD/OptimizeMaxError": "0",

    # Robust cost function in optimizer (handles outlier loop closures gracefully)
    "Optimizer/Robust": "true",

    # Visual odometry settings
    "Vis/EstimationType": "1",   # PnP (3D→2D), more robust than 3D→3D
    "Vis/MinInliers":     "10",  # default is 20; lower = more tolerant of fast motion
}

RGBD_ODOM_PARAMS = {
    "frame_id":    "base_link",
    "approx_sync": True,
    "queue_size":  100,

    # ── Odom-loss recovery ────────────────────────────────────────────────────
    # After 1 consecutive failure, reset odom frame and attempt relocalization
    "Odom/ResetCountdown": "1",

    # Visual feature settings — tuned for robustness over speed
    "Vis/MinInliers":    "10",
    "GFTT/QualityLevel": "0.00001",
    "GFTT/MinDistance":  "4",
    "OdomF2M/MaxSize":   "2000",  # larger feature map → better loop closure recall

    # Use visual odometry (not ICP) — ICP needs dense lidar geometry
    "Reg/Strategy": "0",
}

def generate_launch_description():
    fresh_arg = DeclareLaunchArgument(
        "fresh", default_value="true",
        description="Delete the RTAB-Map database on start for a clean map",
    )

    bridge = ExecuteProcess(
        cmd=["python3", BRIDGE_SCRIPT],
        name="ros2_scan_bridge",
        output="screen",
    )

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
                    ("odom",            "/rtabmap/odom"),
                ],
                arguments=["--delete_db_on_start"],
            ),
            Node(
                package="rtabmap_odom",
                executable="rgbd_odometry",
                name="rgbd_odometry",
                namespace="rtabmap",
                output="screen",
                parameters=[RGBD_ODOM_PARAMS],
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

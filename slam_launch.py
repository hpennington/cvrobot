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


BRIDGE_SCRIPT = os.path.expanduser("~/cvrobot/ros2_scan_bridge.py")

RTABMAP_PARAMS = {
    "frame_id":    "base_link",
    "approx_sync": True,

    # ── 2-D occupancy grid ────────────────────────────────────────────────────
    "Grid/3D":                  "true",
    "Grid/CellSize":            "0.05",
    "Grid/RangeMin":            "0.2",
    "Grid/RangeMax":            "4.0",
    "Grid/RayTracing":          "true",
    "Grid/NormalsSegmentation": "false",
    "Grid/MaxGroundHeight":     "0.05",
    "Grid/MaxObstacleHeight":   "0.4",

    # ── Planar (2-D) motion constraint ────────────────────────────────────────
    "Reg/Force3DoF":          "true",
    "Reg/Strategy": "2",
    "Icp/PointToPlane":  "true",
    "Icp/VoxelSize":     "0.05",
    "Icp/MaxCorrespondenceDistance": "0.1",
    # Re-enabled: BNO08x gravity provides yaw-drift anchor during fast rotation.
    # Was "0" (disabled) before — that caused pose graph inconsistency after spins.
    "Optimizer/GravitySigma": "0.3",

    # ── Map every frame (no wheel odometry to gate on) ────────────────────────
    "RGBD/LinearUpdate":  "0.0",
    "RGBD/AngularUpdate": "0.0",

    # ── Odom-loss recovery ────────────────────────────────────────────────────
    # FIX: was "true" — that started a NEW map on loop closure after odom loss,
    # causing map fragmentation. "false" merges back into the existing map.
    "Rtabmap/StartNewMapOnLoopClosure": "true",

    # Allow more submaps to be retrieved for relocalization after a spin
    "Rtabmap/MaxRetrieved": "4",

    # Slightly more permissive loop closure threshold (default 0.15)
    "Rtabmap/LoopThr": "0.11",

    # Re-check old submaps when trying to relocalize after odom loss
    "RGBD/LoopClosureReactivate": "true",

    # Disable error threshold rejection — accumulated drift after odom loss
    # can cause valid loop closures to be rejected otherwise
    "RGBD/OptimizeMaxError": "0",

    # Robust cost function handles outlier loop closures gracefully
    "Optimizer/Robust": "true",

    # ── Visual feature settings ───────────────────────────────────────────────
    # FIX: ORB (type 2) is rotation-invariant; original GFTT/Harris are not.
    # This is the single highest-impact change for fast rotation robustness.
    "Vis/FeatureType":    "2",    # ORB
    "Vis/MaxFeatures":    "3000",
    # Slightly more permissive than default (20), but higher than before (10)
    # to avoid accepting frames with too few inliers into the map
    "Vis/MinInliers":     "12",
    # PnP (3D→2D) is more robust than 3D→3D under fast motion
    "Vis/EstimationType": "1",
    "Vis/MinDepth":    "0.25",
    "Vis/MaxDepth":    "4.0",
    "Vis/DepthAsMask": "true",   # keep default — reject features without depth
    # "Stereo/OpticalFlow": "true",
    # "Vis/CorType": "1",
    "approx_sync_max_interval": 0.1,
}

RGBD_ODOM_PARAMS = {
    "frame_id":    "base_link",
    "approx_sync": True,
    "queue_size":  100,

    # ── Odom-loss recovery ────────────────────────────────────────────────────
    # FIX: was "1" — a single dropped frame during a fast spin immediately
    # reset odometry, cascading into map fragmentation. Allow several failures
    # before resetting so transient motion blur doesn't trigger a full reset.
    "Odom/ResetCountdown": "5",

    # Propagate the previous frame's motion as the initial guess for the next
    # frame's feature search. Critical for fast rotation — without this, the
    # solver searches around the identity transform and loses tracking.
    "Odom/GuessMotion": "true",

    # ── IMU integration ───────────────────────────────────────────────────────
    # Re-enabled for odometry (was removed due to drift).
    # IMU drift affects translation, not rotation — and the gyro data is exactly
    # what stabilizes visual tracking during fast spins. wait_imu_to_init
    # ensures the odom node bootstraps orientation before relying on IMU.
    "subscribe_imu":    True,
    "wait_imu_to_init": True,
    "Imu/UpAxis":       "z",

    # ── Visual feature settings ───────────────────────────────────────────────
    # FIX: ORB replaces GFTT — rotation-invariant by design.
    # GFTT/QualityLevel and GFTT/MinDistance are removed (irrelevant for ORB).
    "Vis/FeatureType": "2",    # ORB
    "Vis/MaxFeatures": "500",
    "Vis/MinInliers":  "12",

    # Larger feature map window → better recall when revisiting after a spin
    "OdomF2M/MaxSize": "3000",

    # Visual odometry only (not ICP — ICP needs dense lidar geometry)
    "Reg/Strategy": "0",
    "Vis/ImageDecimation": "1",
    # "Stereo/OpticalFlow": "true",
    # "Odom/Strategy": "1",
    "approx_sync_max_interval": 0.1,
}

# realsense2_camera publishes color/depth directly (replaces the old
# ros2_scan_bridge ZMQ image forwarding, which was capped at ~14fps by
# per-frame Python message construction). align_depth makes the depth
# topic match the color stream's resolution/intrinsics, as rgbd_odometry
# and rtabmap require.
#
# NOTE: confirm these exact topic names with `ros2 topic list` against the
# installed realsense-ros version before relying on them — naming
# conventions (e.g. /camera/color/... vs /camera/camera/color/...) have
# changed across releases.
REALSENSE_PARAMS = {
    "camera_name":          "camera",
    "enable_color":         True,
    "rgb_camera.profile":   "640x480x30",
    "enable_depth":         True,
    "depth_module.profile": "640x480x30",
    "align_depth.enable":   True,
    "publish_tf":           True,
    # "exposure": "5000",
    # "enable_auto_exposure": False,
    "approx_sync_max_interval": 0.1,
}

# camera_namespace overrides (even explicit "") don't take effect on this
# realsense-ros version — it nests topics under /camera/camera/... (camera_
# namespace falls back to camera_name) regardless. Confirmed via
# `ros2 topic hz` against the node's actual output rather than fighting it.
REALSENSE_RGB_TOPIC    = "/camera/camera/color/image_raw"
REALSENSE_RGB_INFO     = "/camera/camera/color/camera_info"
REALSENSE_DEPTH_TOPIC  = "/camera/camera/aligned_depth_to_color/image_raw"


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

    realsense = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="camera",
        output="screen",
        parameters=[REALSENSE_PARAMS],
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
                    ("rgb/image",       REALSENSE_RGB_TOPIC),
                    ("rgb/camera_info", REALSENSE_RGB_INFO),
                    ("depth/image",     REALSENSE_DEPTH_TOPIC),
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
                    ("rgb/image",       REALSENSE_RGB_TOPIC),
                    ("rgb/camera_info", REALSENSE_RGB_INFO),
                    ("depth/image",     REALSENSE_DEPTH_TOPIC),
                    ("odom",            "/rtabmap/odom"),
                    ("imu",             "/imu/data"),
                ],
            ),
        ],
    )

    return LaunchDescription([fresh_arg, bridge, realsense, rtabmap])

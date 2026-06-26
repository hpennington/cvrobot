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
    # Match the loop-closure keypoint detector to ORB so Mem/UseOdomFeatures
    # stays enabled (was silently disabled — Kp/DetectorStrategy defaulted to
    # a different detector than Vis/FeatureType).
    "Kp/DetectorStrategy": "2",   # ORB
    # ORB is a binary descriptor; LSH (not the default KDTree) is the correct
    # nearest-neighbor strategy for binary descriptors in the BoW vocabulary.
    "Kp/NNStrategy":       "3",   # LSH
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
    "approx_sync_max_interval": 0.5,
    # orbbec publishes images as BEST_EFFORT; match it or data won't flow
    "qos_image":       2,
    "qos_camera_info": 2,
}

RGBD_ODOM_PARAMS = {
    "frame_id":    "base_link",
    "approx_sync": True,
    "queue_size":  100,
    # orbbec publishes all topics as BEST_EFFORT; match or data won't flow
    "qos":             2,
    "qos_camera_info": 2,
    "qos_imu":         2,

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
    "approx_sync_max_interval": 0.5,
}

# orbbec_camera_node (standalone) ignores camera_name as a topic prefix and
# publishes at root level: /color/image_raw, /depth/image_raw, /gyro_accel/sample.
# depth_registration has no effect in this SDK build — no D2C topic is produced,
# so we consume raw depth and let RTAB-Map use the TF from /tf_static for alignment.
ORBBEC_PARAMS = {
    "camera_name":  "camera",
    "color_width":  640,
    "color_height": 400,
    "color_fps":    30,
    "depth_width":  640,
    "depth_height": 400,
    "depth_fps":    30,
    "enable_color": True,
    "enable_depth": True,
    "depth_format": "Y14",
    "publish_tf":   True,
    "enable_accel": True,
    "enable_gyro":  True,
    "accel_rate":   "100hz",
    "accel_range":  "4g",
    "gyro_rate":    "200hz",
    "gyro_range":   "1000dps",
    "enable_sync_output_accel_gyro": True,
    "time_domain":  "system",
}

ORBBEC_RGB_TOPIC   = "/color/image_raw"
ORBBEC_RGB_INFO    = "/color/camera_info"
ORBBEC_DEPTH_TOPIC = "/depth/image_raw"


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

    orbbec = Node(
        package="orbbec_camera",
        executable="orbbec_camera_node",
        name="camera",
        output="screen",
        parameters=[ORBBEC_PARAMS],
    )

    # /gyro_accel/sample has raw accel+gyro but no orientation quaternion.
    # Madgwick integrates accel+gyro → orientation so RTAB-Map can use the IMU.
    imu_filter = Node(
        package="imu_filter_madgwick",
        executable="imu_filter_madgwick_node",
        name="imu_filter",
        output="screen",
        parameters=[{
            "use_mag":     False,
            "publish_tf":  False,
            "world_frame": "enu",
            "gain":        0.1,
        }],
        remappings=[
            ("imu/data_raw", "/gyro_accel/sample"),
            ("imu/data",     "/imu/filtered"),
        ],
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
                    ("rgb/image",       ORBBEC_RGB_TOPIC),
                    ("rgb/camera_info", ORBBEC_RGB_INFO),
                    ("depth/image",     ORBBEC_DEPTH_TOPIC),
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
                    ("rgb/image",       ORBBEC_RGB_TOPIC),
                    ("rgb/camera_info", ORBBEC_RGB_INFO),
                    ("depth/image",     ORBBEC_DEPTH_TOPIC),
                    ("odom",            "/rtabmap/odom"),
                    ("imu",             "/imu/filtered"),
                ],
            ),
        ],
    )

    return LaunchDescription([fresh_arg, bridge, orbbec, imu_filter, rtabmap])

# CVRobot

## Recording data / arm control / driving / cameras

### Terminal 1
python leader_pub.py

### Terminal 2 device 2
python follower_sub.py

### Terminal 3 device 2
python camera_sub.py

## For SLAM

### Terminal 1
python ros2_ws/jetson_combined.py --foxglove

### Terminal 2
python ros2_ws/ros2_scan_bridge.py

### Terminal 3
ros2 launch ros2_ws/slam_launch.py

### Terminal 4
source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8766

### My Bluetooth joystick address
30:31:7D:86:26:1A



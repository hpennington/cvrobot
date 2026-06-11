# CVRobot
This is a tracked robot chassis with a so 101 robotic arm attached to the top

## Instructions

### Dependencies

librealsense
ros2
pygame
zmq

## Recording data / arm control / driving / cameras

### On mac
```bash
# Terminal 1
python leader_pub.py
```

### On jetson orin nano
```bash
# Terminal 1
python follower_sub.py
```

```bash
# Terminal 2
python camera_sub.py
```

## For SLAM

```bash
# Terminal 1
python ros2_ws/jetson_combined.py --foxglove
```

```bash
# Terminal 2
python ros2_ws/ros2_scan_bridge.py
```

```bash
# Terminal 3
ros2 launch ros2_ws/slam_launch.py
```

```bash
# Terminal 4
source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8766
```

### My Bluetooth joystick address
30:31:7D:86:26:1A



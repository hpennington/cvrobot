# CVRobot
This is a tracked robot chassis with a so 101 robotic arm attached to the top

There is a Realsense D415 attached to the chassis
The arduino is being used for the PWM drivers to the H bridges

## Instructions

### Dependencies

librealsense

ros2

pygame

zmq

lerobot

## BOM

## Circuits

## Robot features
Move an object from point a to point b.
- Static object type: e.g. a mug
- Lerobot to grip the object
- SLAM to first map the scene
- SLAM to move object from A to point B
- YOLO to identify the object and track

## Recording data / arm control / driving / cameras

### On mac

#### Terminal 1

```bash
python leader_pub.py
```

### On jetson orin nano

#### Terminal 1

```bash
python follower_sub.py
```

#### Terminal 2

```bash
python camera_sub.py
```

## For SLAM

#### Terminal 1

```bash
python jetson_combined.py --foxglove
```

#### Terminal 2

```bash
python ros2_scan_bridge.py
```

#### Terminal 3

```bash
ros2 launch slam_launch.py
```

#### Terminal 4

```bash
source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8766
```

### My Bluetooth joystick address
30:31:7D:86:26:1A



# Go2 ROS2 Workspace

This workspace brings together several repositories and packages to control the Unitree Go2 robot via ROS2, Gazebo simulation, and WiFi connections. The projects included focus on navigation using a local cost map and control through a `followPath` action.

## Repositories

- **Robot Core Repository**  
  The main repository for working with the Unitree Go2 robot is available at:  
  [Unitree-Go2-Robot/go2_robot](https://github.com/Unitree-Go2-Robot/go2_robot)

- **Gazebo & Nav2 Configurations**  
  This repository provides configurations for Gazebo simulation and the Nav2 stack. It can serve as a replacement for `go2_driver` (untested):  
  [anujjain-dev/unitree-go2-ros2](https://github.com/anujjain-dev/unitree-go2-ros2)

- **WiFi Control via ROS2**  
  To control the robot over WiFi, use the following repository (which leverages the `go2_webrtc` project):  
  [abizovnuralem/go2_ros2_sdk](https://github.com/abizovnuralem/go2_ros2_sdk)

## Packages

The custom packages in this workspace are located in the Path Traker. They implement:
- A working Nav2 setup using only the local cost map.
- Control logic utilizing the `followPath` action.

---


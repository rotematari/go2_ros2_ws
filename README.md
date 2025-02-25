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

## Run Full Experiment
## Hardware Setup & Safety Instructions

1. **Connect the Robot:**  
  Power on the Go2 robot and ensure it is connected via Ethernet to a machine. The machine should have a static IP that matches the robot’s IP (e.g., go2_ip = 192.168.123.18).

2. **Activate the Joystick:**  
  Press and hold the Y button on the joystick for 3 seconds to turn it on. after that press start

3. **Emergency Stop:**  
  In case of any unexpected behavior, press L2 and A simultaneously on the joystick to immediately stop the robot.

### Terminal 1: Prepare Environment and Launch Robot
1. Open a terminal and navigate to the workspace:
  ```
  cd ~/go2_ros2_ws/go2_ws
  ```
2. Source the setup file:
  ```
  source install/setup.bash
  ```
4. Run the launch script:
  ```
  ./scripts/launch_go2.sh
  ```

### Terminal 2: Launch Model Node
1. Use this terminal to launch the model node with a specified argument for the path publication mode. Set `path` to either `true` to publish the full path, or `false` to publish only the goal pose (letting the path planner calculate the path):
  ```
  ros2 launch model_node model.launch.py path:=<true/false>
  ```

### Terminal 3: Publish Prompts
1. In a new terminal, start the prompt publisher:
  ```
  ros2 run model_node prompt_pub
  ```
2. To publish a prompt, simply type the text in the terminal and press Enter.

### Terminal 4: Launch Navigation
1. In a fourth terminal, launch the navigation stack for the path tracker:
  ```
  ros2 launch path_traker nav2_real_launch.launch.py
  ```

Follow these steps in the given order to run the complete experiment.



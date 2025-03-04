#!/bin/bash

# Script: launch_go2.sh
# Description: Launch script for Go2 Robot
# Author: $(whoami)
# Date: $(date +%Y-%m-%d)

source install/setup.bash
export ROBOT_IP="192.168.123.18" #for muliple robots, just split by ,
export CONN_TYPE="webrtc"
export ROBOT_TOKEN="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIiwidWlkIjoxMTk1LCJjdCI6MTczODg2MjI0NiwiaXNzIjoidW5pdHJlZV9yb2JvdCIsInR5cGUiOiJhY2Nlc3NfdG9rZW4iLCJleHAiOjE3NDE0NTQyNDZ9.3pVKZ6EXYhdJofjBvSy2yyRt5Oogvn8nH77hHVLVnd0"
ros2 launch go2_bringup go2.launch.py
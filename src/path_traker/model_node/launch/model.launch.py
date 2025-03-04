from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node

def generate_launch_description():
    # Declare the 'path' argument (true or false)
    path_arg = LaunchConfiguration('path')
    declare_path_arg = DeclareLaunchArgument(
        'path',
        default_value='false',
        description='If true, launch model_path_pub node; else launch model_goal_pose_pub and path_planner_node'
    )

    # Node to be launched when path == true
    model_path_pub_node = Node(
        package='model_node',
        executable='model_path_pub',
        name='model_path_pub',
        output='screen',
        condition=IfCondition(path_arg)
    )

    # Nodes to be launched when path == false
    model_goal_pose_pub_node = Node(
        package='model_node',
        executable='model_goal_pose_pub',
        name='model_goal_pose_pub',
        output='screen',
        condition=UnlessCondition(path_arg)
    )
    path_planner_node = Node(
        package='path_planner_node',
        executable='path_planner_node',
        name='path_planner_node',
        output='screen',
        condition=UnlessCondition(path_arg)
    )

    return LaunchDescription([
        declare_path_arg,
        model_path_pub_node,
        model_goal_pose_pub_node,
        path_planner_node,
    ])
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    pkg_share = get_package_share_directory('path_traker')
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time if true'
    )

    robot_localization_node = Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_node',
            output='screen',
            parameters=[os.path.join(pkg_share, 'configs','ekf.yaml'), {'use_sim_time': LaunchConfiguration('use_sim_time')}]
        )
    
    
    controller_server =Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[os.path.join(pkg_share, 'configs', 'controller_server.yaml')],
        )
    
    path_traker_node =Node(
            package='path_traker',
            executable='path_traker_node',
            name='path_traker_node',
            output='screen',
        )
    lifecycle_manager = Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{'use_sim_time': True},
                        {'autostart': True},
                        {'node_names': ['controller_server']}])

    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'params_file': os.path.join(pkg_share, 'configs', 'nav2_params.yaml')
        }.items()
    )
    return LaunchDescription([
        use_sim_time_arg,
        robot_localization_node,
        nav2_launch,
        # controller_server,
        # lifecycle_manager,
        # path_traker_node,
    ])

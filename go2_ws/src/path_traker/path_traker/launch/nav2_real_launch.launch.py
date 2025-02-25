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
    
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    # params_file = LaunchConfiguration('params_file')
    use_respawn = LaunchConfiguration('use_respawn')
    log_level = LaunchConfiguration('log_level')
    
    remappings = [('/odom', '/footprint_odom')]
    # remappings = []
    
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='False',
        description='Use simulation time if true'
    )
    # declare_params_file_cmd = DeclareLaunchArgument(
    #     'params_file',
    #     default_value=os.path.join(pkg_share, 'configs', 'nav2_params.yaml'),
    #     description='Full path to the ROS2 parameters file to use for all launched nodes',
    # )
    params_file = os.path.join(
        pkg_share,
        'configs',
        'navigation_real.yaml'
    )
    # Ensure the parameter file exists
    if not os.path.isfile(params_file):
        raise FileNotFoundError(f"Parameter file '{params_file}' not found")
    
    
    declare_use_respawn_cmd = DeclareLaunchArgument(
        'use_respawn',
        default_value='True',
        description='Whether to respawn if a node crashes. Applied when composition is disabled.',
    )
    declare_log_level_cmd = DeclareLaunchArgument(
        'log_level', default_value='info', description='log level'
    )
    robot_localization_node = Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_node',
            output='screen',
            parameters=[os.path.join(pkg_share, 'configs','ekf_real.yaml'), {'use_sim_time': use_sim_time}]
        )
    
    
    controller_server =Node(
                package='nav2_controller',
                executable='controller_server',
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[params_file],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings,
            )
    smoother_server = Node(
                package='nav2_smoother',
                executable='smoother_server',
                name='smoother_server',
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[params_file],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings,
            )
    behavior_server = Node(
                package='nav2_behaviors',
                executable='behavior_server',
                name='behavior_server',
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[params_file],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings ,
            )
    bt_navigator = Node(
                package='nav2_bt_navigator',
                executable='bt_navigator',
                name='bt_navigator',
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[params_file],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings,
            )
    velocity_smoother = Node(
                package='nav2_velocity_smoother',
                executable='velocity_smoother',
                name='velocity_smoother',
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[params_file],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings,
            )
    planner_server = Node(
                package='nav2_planner',
                executable='planner_server',
                name='planner_server',
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[params_file],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings,
            )
    
    path_traker_node =Node(
            package='path_traker',
            executable='path_traker_node',
            name='path_traker_node',
            output='screen',
        )
    lifecycle_nodes = [
        'controller_server',
        'smoother_server',
        'behavior_server',
        # 'planner_server',
        'velocity_smoother',
        # 'bt_navigator',
    ]
    lifecycle_manager = Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time},
                        {'autostart': True},
                        {'node_names': lifecycle_nodes }])

    # nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    # nav2_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
    #     ),
    #     launch_arguments={
    #         'use_sim_time': 'true',
    #         'params_file': os.path.join(pkg_share, 'configs', 'nav2_params.yaml')
    #     }.items()
    # )
    return LaunchDescription([
        use_sim_time_arg,
        # declare_params_file_cmd,
        declare_use_respawn_cmd,
        declare_log_level_cmd,
        
        robot_localization_node,
        controller_server,
        smoother_server,
        behavior_server,
        # planner_server,
        velocity_smoother,
        # bt_navigator,
        
        lifecycle_manager,
        # path_traker_node,
    ])

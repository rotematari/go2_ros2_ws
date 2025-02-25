from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'model_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/Speak2Act',
         [f for f in glob('Speak2Act/**/*', recursive=True) if os.path.isfile(f)]),
        ('share/' + package_name + '/launch',
         [f for f in glob('launch/**/*', recursive=True) if os.path.isfile(f)]),
        ('share/' + package_name + '/config',
         [f for f in glob('config/**/*', recursive=True) if os.path.isfile(f)]),
        ('share/' + package_name + '/model',
         [f for f in glob('model/**/*', recursive=True) if os.path.isfile(f)]),
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='admina',
    maintainer_email='rotem.atri@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'prompt_pub = model_node.prompt_pub:main',
            'model_path_pub = model_node.model_path_pub:main',
            'model_goal_pose_pub = model_node.model_goal_pose_pub:main',
        ],
    },
)

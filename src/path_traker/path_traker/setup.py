from setuptools import find_packages, setup
from glob import glob

package_name = 'path_traker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/configs', glob('configs/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*')),
    ],
    install_requires=[
        'setuptools',
        'transformers',
        ],
    zip_safe=True,
    maintainer='admina',
    maintainer_email='rotem.atri@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_traker_node = path_traker.path_traker_node_2:main',
            'path_pub = path_traker.path_pub:main',
            'path_pub_real = path_traker.path_pub_real:main',
            
        ],
    },
)

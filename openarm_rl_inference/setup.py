from setuptools import setup

package_name = 'openarm_rl_inference'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/rl_inference.yaml']),
        ('share/' + package_name + '/launch', ['launch/rl_inference.launch.py']),
        ('lib/' + package_name, [
            'scripts/openarm_rl_inference',
            'scripts/publish_pose_demo.py',
        ]),
    ],
    install_requires=['setuptools', 'numpy', 'torch'],
    zip_safe=False,
    maintainer='Enactic, Inc.',
    maintainer_email='openarm_dev@enactic.ai',
    description='RL inference bridge for the OpenARM left arm.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'openarm_rl_inference = openarm_rl_inference.rl_inference_node:main',
        ],
    },
)

from setuptools import setup, find_packages

setup(
    name='dynamicraft',
    version='1.1',
    include_package_data=True,
    package_dir={"": "src"},
    packages=find_packages(where='src', include=['dynamicraft', 'dynamicraft.*']),
    package_data={'': ['*.txt', '*.md', '*.gif', '*.mp4', '*.py', '*.jpg', '*.png'],},
    # package_data={'motionctrl': ['assets/*', 'configs/*', 'examples/*', 'gradio_utils/*', 'lvdm/*', 'main/*', 'motionctrl/*', 'utils/*']},
    author='Tencent',
    description='A Python package for motion control and object tracking',
    license='MIT',
    keywords='motion control tracking',
    url='https://github.com/SyedSherjeel/motionctrl',
)

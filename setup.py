from setuptools import setup

setup(
    name='bink',
    version='0.1.3',
    packages=['bink', 'bink.metrics', 'bink.callbacks', 'tests', 'tests.metrics', 'tests.callbacks'],
    url='https://github.com/ecs-vlc/PyBink',
    download_url='https://github.com/ecs-vlc/PyBink/archive/v0.1.3.tar.gz',
    license='GPL-3.0',
    author='Matt Painter',
    author_email='mp2u16@ecs.soton.ac.uk',
    description='A model training library for pytorch',
    install_requires=['numpy', 'torch>=0.4', 'torchvision', 'scipy', 'scikit-learn', 'tqdm', 'tensorboardX>=1.2'],
    python_requires='>=3',
)

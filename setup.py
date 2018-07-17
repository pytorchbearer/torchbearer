from setuptools import setup

setup(
    name='sconce',
    version='0.1.3',
    packages=['sconce', 'sconce.metrics', 'sconce.callbacks', 'tests', 'tests.metrics', 'tests.callbacks'],
    url='https://github.com/ecs-vlc/PySconce',
    download_url='https://github.com/ecs-vlc/PySconce/archive/v0.1.3.tar.gz',
    license='GPL-3.0',
    author='Matt Painter',
    author_email='mp2u16@ecs.soton.ac.uk',
    description='A model training library for pytorch',
    install_requires=['numpy', 'torch>=0.4', 'scikit-learn', 'tqdm', 'tensorboardX>=1.2'],
    python_requires='>=3',
)

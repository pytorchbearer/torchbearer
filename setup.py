from setuptools import setup

setup(
    name='bink',
    version='0.1.0',
    packages=['bink', 'bink.metrics', 'bink.runners', 'bink.callbacks', 'tests', 'tests.metrics', 'tests.callbacks'],
    url='https://github.com/MattPainter01/PyTorch-bink',
    download_url='https://github.com/MattPainter01/PyTorch-bink/archive/v0.1.0.tar.gz',
    license='GPL-3.0',
    author='Matt Painter',
    author_email='mp2u16@ecs.soton.ac.uk',
    description='A model training library for pytorch',
    install_requires=['numpy', 'torch>=0.4', 'scikit-learn', 'tqdm', 'tensorboardX>=1.2'],
    python_requires='>=3',
)

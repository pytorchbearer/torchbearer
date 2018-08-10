from setuptools import setup

version_dict = {}
exec(open("./torchbearer/version.py").read(), version_dict)

setup(
    name='torchbearer',
    version=version_dict['__version__'],
    packages=['torchbearer', 'torchbearer.metrics', 'torchbearer.callbacks', 'tests', 'tests.metrics', 'tests.callbacks'],
    url='https://github.com/ecs-vlc/torchbearer',
    download_url='https://github.com/ecs-vlc/torchbearer/archive/' + version_dict['__version__'] + '.tar.gz',
    license='GPL-3.0',
    author='Matt Painter',
    author_email='mp2u16@ecs.soton.ac.uk',
    description='A model training library for pytorch',
    install_requires=['numpy', 'torch>=0.4', 'torchvision', 'scipy', 'scikit-learn', 'tqdm', 'tensorboardX>=1.2'],
    python_requires='>=3',
)

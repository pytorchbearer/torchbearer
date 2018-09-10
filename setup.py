from setuptools import setup

version_dict = {}
exec(open("./torchbearer/version.py").read(), version_dict)

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

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
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['torch>=0.4', 'torchvision', 'tqdm'],
    python_requires='>=3',
)

# torchbearer

[![PyPI version](https://badge.fury.io/py/torchbearer.svg)](https://badge.fury.io/py/torchbearer) [![Build Status](https://travis-ci.com/ecs-vlc/torchbearer.svg?branch=master)](https://travis-ci.com/ecs-vlc/torchbearer) [![codecov](https://codecov.io/gh/ecs-vlc/torchbearer/branch/master/graph/badge.svg)](https://codecov.io/gh/ecs-vlc/torchbearer) [![Documentation Status](https://readthedocs.org/projects/torchbearer/badge/?version=latest)](https://torchbearer.readthedocs.io/en/latest/?badge=latest)

torchbearer: A model training library for researchers using PyTorch
## Contents
- [About](#about)
- [Installation](#installation)
- [Documentation](#docs)
- [Other Libraries](#others)

<a name="about"/>

## About

Torchbearer is a PyTorch model training library designed by researchers, for researchers. Specifically, if you occasionally want to perform advanced custom operations but generally don't want to write hundreds of lines of untested code then this is the library for you. Our design decisions are geared towards flexibility and customisability whilst trying to maintain the simplest possible API.

<a name="installation"/>

## Installation

The easiest way to install torchbearer is with pip:

`pip install torchbearer`

<a name="docs"/>

## Documentation

Our documentation containing the API reference, examples and some notes can be found at [https://torchbearer.readthedocs.io](torchbearer.readthedocs.io)

<a name="others"/>

## Other Libraries

Torchbearer isn't the only library for training PyTorch models. Here are a few others that might better suit your needs:
- [PyToune](https://github.com/GRAAL-Research/pytoune), simple Keras style API
- [ignite](https://github.com/pytorch/ignite), advanced model training from the makers of PyTorch, can need a lot of code for advanced functions (e.g. Tensorboard)
- [TorchNetTwo (TNT)](https://github.com/pytorch/tnt), can be complex to use but well established, somewhat replaced by ignite

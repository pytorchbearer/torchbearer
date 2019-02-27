# \[WIP\] torchbearer.variational
A Variational Auto-Encoder library for PyTorch

## Contents
- [About](#about)
- [Installation](#installation)
- [Goals](#goals)

<a name="about"/>

## About

Torchbearer.variational is a sub-package of [torchbearer](https://github.com/ecs-vlc/torchbearer) which is intended to
re-implement state of the art models and practices relating to the world of Variational Auto-Encoders (VAEs). The goal
is to provide everything from useful abstractions to complete re-implementations of papers. This is in order to support
both research and teaching / learning regarding VAEs.

<a name="installation"/>

## Installation

_variational_ comes packaged with torchbearer and so can be installed by following the instructions [here](https://github.com/ecs-vlc/torchbearer).

<a name="goals"/>

## Goals

Currently, _variational_ only includes abstractions for simple VAEs and some accompaniments, the next steps are as follows:

- Construct some separate part of the docs for the _variational_ content
- Implement a series of standard models with associated notes pages and example usages
- Implement other divergences not in PyTorch such as MMD, Jensen-Shannon, etc.
- Implement and document tools for sampling the latent spaces of models and producing figures
- Implement other dataloaders not in torchvision and add associated docs
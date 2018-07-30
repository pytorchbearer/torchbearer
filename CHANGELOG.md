# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
- Added a on_validation_criterion callback hook
- Added a DatasetValidationSplitter which can be used to create a validation split if required for datasets like Cifar10 or MNIST
- Added simple timer callback
### Changed
### Deprecated
### Removed
### Fixed
- Fixed a bug where checkpointers would not save the model in some cases

## [0.1.4] - 2018-07-23
### Added
- Added a decorator API for metrics which allows decorators to be used for metric construction
- Added a default_for_key decorator which can be used to associate a string with a given metric in metric lists
- Added a decorator API for callbacks which allows decorators to be used for simple callback construction
- Added a add_to_loss callback decorator which allows quicker constructions of callbacks that add values to the loss
### Changed
- Changed the API for running metrics and aggregators to no longer wrap a metric but instead receive input
### Deprecated
### Removed
### Fixed

## [0.1.3] - 2018-07-17
### Added
- Added a flag (step_on_batch) to the LR Scheduler callbacks which allows for step() to be called on each iteration instead of each epoch
- Added on_sample_validation and on_forward_validation calls for validation callbacks
- Added GradientClipping callback which simply clips the absolute gradient of the model parameters
### Changed
- Changed the order of the arguments to the lambda function in the EpochLambda metric for consistency with pytorch and other metrics
- Checkpointers now create directory to savepath if it doesn't exist
- Changed the 'on_forward_criterion' callback method to 'on_criterion'
- Changed epoch number in printer callbacks to be consistent with the rest of torchbearer
### Deprecated
### Removed
### Fixed
- Fixed tests which were failing as of version 0.1.2
- Fixed validation_steps not being added to state
- Fixed checkpointer bug when path contained only filename and no directory path
- Fixed console printer bug not printing validation statistics
- Fixed console printer bug calling final_metrics before they existed in state

## [0.1.2] - 2018-06-08
### Added
- Added support for tuple outputs from generators, torchbearer expects output to be length 2. Specifically, x, y = next() is possible, where x and y can be tuples of arbitrary size or depth
- Added support for torch dtypes in torchbearer Model.to(...)
- Added pickle_module and pickle_protocol to checkpointers for consistency with torch.save
### Changed
- Changed the learning rate scheduler callbacks to no longer require an optimizer and to have the proper arguments
### Deprecated
### Removed
### Fixed
- Fixed an issue in GradientNormClipping which raised a warning in PyTorch >= 0.4

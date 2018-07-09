# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
- Added a flag (step_on_batch) to the LR Scheduler callbacks which allows for step() to be called on each iteration instead of each epoch
- Added on_sample_validation and on_forward_validation calls for validation callbacks
- Added GradientClipping callback which simply clips the absolute gradient of the model parameters
### Changed
- Changed the order of the arguments to the lambda function in the EpochLambda metric for consistency with pytorch and other metrics
- Checkpointers now create directory to savepath if it doesn't exist 
### Deprecated
### Removed
### Fixed
- Fixed tests which were failing as of version 0.1.2
- Fixed validation_steps not being added to state

## [0.1.2] - 2018-06-08
### Added
- Added support for tuple outputs from generators, bink expects output to be length 2. Specifically, x, y = next() is possible, where x and y can be tuples of arbitrary size or depth
- Added support for torch dtypes in bink Model.to(...)
- Added pickle_module and pickle_protocol to checkpointers for consistency with torch.save
### Changed
- Changed the learning rate scheduler callbacks to no longer require an optimizer and to have the proper arguments
### Deprecated
### Removed
### Fixed
- Fixed an issue in GradientNormClipping which raised a warning in PyTorch >= 0.4

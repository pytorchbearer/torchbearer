# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
### Changed
### Deprecated
### Removed
### Fixed
- Model class setting generator incorrectly leading to stop iterations. 

## [0.2.0] - 2018-08-21
### Added
- Added the ability to pass custom arguments to the tqdm callback
- Added an ignore_index flag to the categorical accuracy metric, similar to nn.CrossEntropyLoss. Usage: ``metrics=[CategoricalAccuracyFactory(ignore_index=0)]``
- Added TopKCategoricalAccuracy metric (default for key: top\_5\_acc)
- Added BinaryAccuracy metric (default for key: binary\_acc)
- Added MeanSquaredError metric (default for key: mse)
- Added DefaultAccuracy metric (use with 'acc' or 'accuracy') - infers accuracy from the criterion
- New Trial api ``torchbearer.Trial`` to replace the Model api. Trial api is more atomic and uses the fluent pattern to allow chaining of methods.
- ``torchbearer.Trial`` has with_x_generator and with_x_data methods to add training/validation/testing generators to the trial. There is a with_generators method to allow passing of all generators in one call.
- ``torchbearer.Trial`` has for_x_steps and for_steps to allow running of trails without explicit generators or data tensors
- ``torchbearer.Trial`` keeps a history of run calls which tracks number of epochs ran and the final metrics at each epoch. This allows seamless resuming of trial running.
- ``torchbearer.Trial.state_dict`` now returns the trial history and callback list state allowing for full resuming of trials
- ``torchbearer.Trial`` has a replay method that can replay training (with callbacks and display) from the history. This is useful when loading trials from state.
- The backward call can now be passed args by setting ``state[torchbearer.BACKWARD_ARGS]``
- ``torchbearer.Trial`` implements the forward pass, loss calculation and backward call as a optimizer closure
- Metrics are now explicitly calculated with no gradient
### Changed
- Callback decorators can now be chained to allow construction with multiple methods filled
- Callbacks can now implement ``state_dict`` and ``load_state_dict` to allow callbacks to resume with state
- State dictionary is now accepts StateKey objects which are unique and generated through ``torchbearer.state.get_state``
- State dictionary now warns when accessed with strings as this allows for collisions
- Checkpointer callbacks will now resume from a state dict when resume=True in Trial
### Deprecated
- ``torchbearer.Model`` has been deprecated in favour of the new ``torchbearer.Trial`` api
### Removed
- Removed the MetricFactory class. Decorators still work in the same way but the Factory is no longer needed.
### Fixed

## [0.1.7] - 2018-08-14
### Added
- Added visdom logging support to tensorbard callbacks
- Added option to choose tqdm module (tqdm, tqdm_notebook, ...) to Tqdm callback
- Added some new decorators to simplify custom callbacks that must only run under certain conditions (or even just once).
### Changed
- Instantiation of Model will now trigger a warning pending the new Trial API in the next version
- TensorboardX dependancy now version 1.4
### Deprecated
### Removed
### Fixed
- Mean and standard deviation calculations now work correctly for network outputs with many dimensions
- Callback list no longer shared between fit calls, now a new copy is made each fit

## [0.1.6] - 2018-08-10
### Added
- Added a verbose level (options are now 0,1,2) which will print progress for the entire fit call, updating every epoch. Useful when doing dynamic programming with little data.
- Added support for dictionary outputs of dataloader
- Added abstract superclass for building TensorBoardX based callbacks
### Changed
- Timer callback can now also be used as a metric which allows display of specified timings to printers and has been moved to metrics.
- The loss_criterion is renamed to criterion in `torchbearer.Model` arguments.
- The criterion in `torchbearer.Model` is now optional and will provide a zero loss tensor if it is not given.
- TensorBoard callbacks refactored to be based on a common super class
- TensorBoard callbacks refactored to use a common `SummaryWriter` for each log directory
### Deprecated
### Removed
### Fixed
- Standard deviation calculation now returns 0 instead of complex value when given very close samples

## [0.1.5] - 2018-07-30
### Added
- Added a on_validation_criterion callback hook
- Added a DatasetValidationSplitter which can be used to create a validation split if required for datasets like Cifar10 or MNIST
- Added simple timer callback
### Changed
### Deprecated
### Removed
### Fixed
- Fixed a bug where checkpointers would not save the model in some cases
- Fixed a bug with the ROC metric causing it to not work

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

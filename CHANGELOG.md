# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [Unreleased]
### Added
### Changed
### Deprecated
### Removed
### Fixed

## [0.5.5] - 2023-12-01
### Added
### Changed
### Deprecated
### Removed
### Fixed
- Fixed versions in setup.py

## [0.5.4] - 2023-11-13
### Added
- Allow imaging callback's `to_file` to use state information in the file name
### Changed
### Deprecated
### Removed
### Fixed
- Fixed warnings about the `epoch` argument in schedulers in newer versions of pytorch
- Fixed a bug in access metrics function and callbacks that use it
- Fixed bug where schedulers were called before optimisers with newer versions of pytorch
- Fixed a bug where the csv logger closed the file too early
- Fixed compat with pytorch > 1.1.0 versioning
- Fixed typos in doc strings
- Fixes for tests where pytorch >2 Tensors were causing issues with mocks
- Fix bug in gradient clipping where the parameter generator was consumed on the first pass

## [0.5.3] - 2020-01-31
### Added
- Method in bases to access metrics
### Changed
### Deprecated
### Removed
### Fixed
- Metric access bugs in various callbacks

## [0.5.2] - 2020-01-28
### Added
- Added option to use mixup loss with cutmix
- Support for PyTorch 1.4.0
### Changed
- Changed PyCM save methods to use `*args` and `**kwargs`
### Deprecated
### Removed
### Fixed
- Fixed a bug where the PyCM callback would fail when saving

## [0.5.1] - 2019-11-06
### Added
- Added BCPlus callback for between-class learning
- Added support for PyTorch 1.3
- Added a show flag to the `ImagingCallback.to_pyplot` method, set to false to stop it from calling `plt.show`
- Added manifold mixup
### Changed
- Changed the default behaviour of `ImagingCallback.to_pyplot` to turn off the axis
### Deprecated
### Removed
### Fixed
- Fixed a bug when resuming an old state dict with tqdm enabled
- Fixed a bug in imaging where passing a title to `to_pyplot` was not possible

## [0.5.0] - 2019-09-17
### Added
- Added PyTorch CyclicLR scheduler
### Changed
- Torchbearer now supports Modules with multiple inputs and multiple outputs
### Deprecated
### Removed
- Cyclic LR callback in favour of torch cyclic lr scheduler
- Removed support for PyTorch 0.4.x
### Fixed
- Fixed bug where aggregate predictions couldn't handle empty list
- Fixed a bug where Runtime Errors on forward weren't handled properly
- Fixed a bug where exceptions on forward wouldn't print the traceback properly
- Fixed a documentation mistake whereby ReduceLROnPlateau was said to increase learning rate

## [0.4.0] - 2019-07-05
### Added
- Added ``with_loader`` trial method that allows running of custom batch loaders
- Added a Mock Model which is set when None is passed as the model to a Trial. Mock Model always returns None. 
- Added `__call__(state)` to `StateKey` so that they can now be used as losses
- Added a callback to do cutout regularisation
- Added a `with_data` trial method that allows passing of train, val and test data in one call
- Added the missing on_init callback decorator
- Added a `step_on_batch` flag to the early stopping callback
- Added multi image support to `imaging`
- Added a callback to unpack state into torchbearer.X at sample time for specified keys and update state after the forward pass based on model outputs. This is useful for using DataParallel which pass the main state dict directly. 
- Added callback for generating confusion matrices with PyCM
- Added a mixup callback with associated loss
- Added Label Smoothing Regularisation (LSR) callback
- Added CutMix regularisation
- Added default metric from paper for when Mixup loss is used
### Changed
- Changed history to now just be a list of records
- Categorical Accuracy metric now also accepts tensors of size (B, C) and gets the max over C for the taget class
### Deprecated
### Removed
- Removed the variational sub-package, this will now be packaged separately
- Removed `verbose` argument from the early stopping callback
### Fixed
- Fixed a bug where list or dictionary metrics would cause the tensorboard callback to error
- Fixed a bug where running a trial without training steps would error
- Fixed a bug where the caching imaging callback didn't reset data so couldn't be run in multiple trials
- Fixed a bug in the `ClassAppearanceModel` callback
- Fixed a bug where the state given to predict was not a State object
- Fixed a bug with Cutout on gpu
- Fixed a bug where MakeGrid callback wasn't passing all arguments correctly
- Fixed a bug in `ImagingCallback` that would sometimes cause `make_grid` to throw an error
- Fixed a bug where the verbose argument would not work unless given as a keyword argument
- Fixed a bug where the data_key argument would sometimes not work as expected
- Fixed a bug where cutout required a seed
- Fixed a bug where cutmix wasn't sendign the beta distribution sample to the device

## [0.3.2] - 2019-05-28
### Added
### Changed
### Deprecated
### Removed
### Fixed
- Fixed a bug where for_steps would sometimes not work as expected if called in the wrong order
- Fixed a bug where torchbearer installed via pip would crash on import

## [0.3.1] - 2019-05-24
### Added
- Added cyclic learning rate finder
- Added on_init callback hook to run at the end of trial init
- Added callbacks for weight initialisation in ``torchbearer.callbacks.init``
- Added ``with_closure`` trial method that allows running of custom closures 
- Added ``base_closure`` function to bases that allows creation of standard training loop closures
- Added ``ImagingCallback`` class for callbacks which produce images that can be sent to tensorboard, visdom or a file
- Added ``CachingImagingCallback`` and ``MakeGrid`` callback to make a grid of images
- Added the option to give the ``only_if`` callback decorator a function of self and state rather than just state
- Added Layer-sequential unit-variance (LSUV) initialization
- Added ClassAppearanceModel callback and example page for visualising CNNs
- Added on_checkpoint callback decorator
- Added support for PyTorch 1.1.0
### Changed
- `No_grad` and `enable_grad` decorators are now also context managers
### Deprecated
### Removed
- Removed the fluent decorator, just use return self
- Removed install dependency on `torchvision`, still required for some functionality
### Fixed
- Fixed bug where replay errored when train or val steps were None
- Fixed a bug where mock optimser wouldn't call it's closure
- Fixed a bug where the notebook check raised ModuleNotFoundError when IPython not installed
- Fixed a memory leak with metrics that causes issues with very long epochs
- Fixed a bug with the once and once_per_epoch decorators
- Fixed a bug where the test criterion wouldn't accept a function of state
- Fixed a bug where type inference would not work correctly when chaining ``Trial`` methods
- Fixed a bug where checkpointers would error when they couldn't find the old checkpoint to overwrite
- Fixed a bug where the 'test' label would sometimes not populate correctly in the default accuracy metric

## [0.3.0] - 2019-02-28
### Added
- Added torchbearer.variational, a sub-package for implementations of state of the art variational auto-encoders
- Added SimpleUniform and SimpleExponential distributions
- Added a decorator which can be used to cite a research article as part of a doc string
- Added an optional dimension argument to the mean, std and running_mean metric aggregators
- Added a var metric and decorator which can be used to calculate the variance of a metric
- Added an unbiased flag to the std and var metrics to optionally not apply Bessel's correction (consistent with torch.std / torch.var)
- Added support for rounding 1D lists to the Tqdm callback
- Added SimpleWeibull distribution
- Added support for Python 2.7
- Added SimpleWeibullSimpleWeibullKL
- Added SimpleExponentialSimpleExponentialKL
- Added the option for model parameters only saving to Checkpointers.
- Added documentation about serialization.
- Added support for indefinite data loading. Iterators can now be run until complete independent of epochs or iterators can be refreshed during an epoch if complete. 
- Added support for batch intervals in interval checkpointer
- Added line magic ``%torchbearer notebook``
- Added 'accuracy' variants of 'acc' default metrics
### Changed
- Changed the default behaviour of the std metric to compute the sample std, in line with torch.std
- Tqdm precision argument now rounds to decimal places rather than significant figures
- Trial will now simply infer if the model has an argument called 'state'
- Torchbearer now infers if inside a notebook and will use the appropriate tqdm module if not set
### Deprecated
### Removed
- Removed the old Model API (deprecated since version 0.2.0)
- Removed the 'pass_state' argument from Trial, this will now be inferred
- Removed the 'std' decorator from the default metrics
### Fixed
- Fixed a bug in the weight decay callback which would result in potentially negative decay (now just uses torch.norm)
- Fixed a bug in the cite decorator causing the citation to not show up correctly
- Fixed a memory leak in the mse primitive metric

## [0.2.6.1] - 2019-02-25
### Added
### Changed
### Deprecated
### Removed
### Fixed
- Fixed a bug where predictions would multiply when predict was called more than once

## [0.2.6] - 2018-12-19
### Added
### Changed
- Y_PRED, Y_TRUE and X can now equivalently be accessed as PREDICTION, TARGET and INPUT respectively
### Deprecated
### Removed
### Fixed
- Fixed a bug where the LiveLossPlot callback would trigger an error if run and evaluate were called separately
- Fixed a bug where state key errors would report to the wrong stack level
- Fixed a bug where the user would wrongly get a state key error in some cases

## [0.2.5] - 2018-12-19
### Added
- Added flag to replay to replay only a single batch per epoch
- Added support for PyTorch 1.0.0 and Python 3.7
- MetricTree can now unpack dictionaries from root, this is useful if you want to get a mean of a metric. However, this should be used with caution as it extracts only the first value in the dict and ignores the rest.
- Added a callback for the livelossplot visualisation tool for notebooks
### Changed
- All error / accuracy metrics can now optionally take state keys for predictions and targets as arguments
### Deprecated
### Removed
### Fixed
- Fixed a bug with the EpochLambda metric which required y_true / y_pred to have specific forms

## [0.2.4] - 2018-11-16
### Added
- Added metric functionality to state keys so that they can be used as metrics if desired
- Added customizable precision to the printer callbacks
- Added threshold to binary accuracy. Now it will appropriately handle any values in \[0, 1\]
### Changed
- Changed the default printer precision to 4s.f.
- Tqdm on_epoch now shows metrics immediately when resuming
### Deprecated
### Removed
### Fixed
- Fixed a bug which would incorrectly trigger version warnings when loading in models
- Fixed bugs where the Trial would not fail gracefully if required objects were not in state
- Fixed a bug where none criterion didn't work with the add_to_loss callback
- Fixed a bug where tqdm on_epoch always started at 0

## [0.2.3] - 2018-10-12
### Added
- Added string representation of Trial to give summary
- Added option to log Trial summary to TensorboardText
- Added a callback point ('on_checkpoint') which can be used for model checkpointing after the history ios updated
### Changed
- When resuming training checkpointers no longer delete the state file the trial was loaded from
- Changed the metric eval to include a data_key which tells us what data we are evaluating on
### Deprecated
### Removed
### Fixed
- Fixed a bug where callbacks weren't handled correctly in the predict and evaluate methods of Trial
- Fixed a bug where the history wasn't updated when new metrics were calculated with the evaluate method of Trial
- Fixed a bug where tensorboard writers couldn't be reused 
- Fixed a bug where the none criterion didn't require gradient
- Fix bug where tqdm wouldn't get correct iterator length when evaluating on test generator
- Fixed a bug where evaluating before training tried to update history before it existed
- Fixed a bug where the metrics would output 'val_acc' even if evaluating on test or train data
- Fixed a bug where roc metric didn't detach y_pred before sending to numpy
- Fixed a bug where resuming from a checkpoint saved with one of the callbacks didn't populate the epoch number correctly

## [0.2.2] - 2018-09-18
### Added
- The default_for_key metric decorator can now be used to pass arguments to the init of the inner metric
- The default metric for the key 'top_10_acc' is now the TopKCategoricalAccuracy metric with k set to 10
- Added global verbose flag for trial that can be overridden by run, evaluate, predict
- Added an LR metric which retrieves the current learning rate from the optimizer, default for key 'lr'
### Changed
### Deprecated
### Removed
### Fixed
- Fixed a bug where the DefaultAccuracy metric would not put the inner metric in eval mode if the first call to reset was after the call to eval
- Fixed a bug where trying to load a state dict in a different session to where it was saved didn't work properly
- Fixed a bug where the empty criterion would trigger an error if no Y_TRUE was put in state

## [0.2.1] - 2018-09-11
### Added
- Evaluation and prediction can now be done on any data using data_key keywork arg
- Text tensorboard/visdom logger that writes epoch/batch metrics to text
### Changed
- TensorboardX, Numpy, Scikit-learn and Scipy are no longer dependancies and only required if using the tensorboard callbacks or roc metric
### Deprecated
### Removed
### Fixed
- Model class setting generator incorrectly leading to stop iterations. 
- Argument ordering is consistent in `Trial.with_generators` and `Trial.__init__`
- Added a state dict for the early stopping callback
- Fixed visdom parameters not getting set in some cases

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

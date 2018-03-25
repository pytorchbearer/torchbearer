from torch.utils.data import DataLoader, TensorDataset
from utils import get_train_valid_sets
from bink.callbacks.callbacks import CallbackList
from bink.callbacks.printer import Tqdm
from torch.autograd import Variable
from bink import metrics as bink_metrics


class Model:
    def __init__(self, model, optimizer, loss_criterion, metrics=[]):
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._criterion = loss_criterion
        self._metrics = bink_metrics.MetricList(metrics)
        self._use_cuda = False
        self._sample_device_function = self._cpu_sample

    def fit(self, x, y, batch_size=None, epochs=1, verbose=1, callbacks=[], validation_split=0.0,
            validation_data=None, shuffle=True, class_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, workers=1):
        
        trainset, valset = get_train_valid_sets(x, y, validation_data, validation_split, shuffle=shuffle)
        trainloader = DataLoader(trainset, batch_size, shuffle=shuffle, num_workers=workers)

        if valset is not None:
            valloader = DataLoader(valset, batch_size, shuffle=shuffle, num_workers=workers)
        else:
            valloader = None

        return self.fit_generator(trainloader, train_steps=steps_per_epoch, epochs=epochs, verbose=verbose,
                                  callbacks=callbacks, validation_generator=valloader, validation_steps=validation_steps,
                                  class_weight=class_weight, initial_epoch=initial_epoch)

    def fit_generator(self, generator, train_steps=None, epochs=1, verbose=1, callbacks=[],
                      validation_generator=None, validation_steps=None, class_weight=None, initial_epoch=0):
        self.stop_training = False
        history = None

        if verbose == 1:
            callbacks = [Tqdm()] + callbacks

        _callbacks = CallbackList(callbacks)

        if validation_steps is None and validation_generator is not None:
            validation_steps = len(validation_generator)

        if train_steps is None:
            train_steps = len(generator)

        state = {
            'model': self._model,
            'criterion': self._criterion,
            'optimizer': self._optimizer,
            'max_epochs': epochs,
            'train_steps': train_steps,
            'validation_steps': validation_steps,
            't': 0,
            'generator': generator,
            'use_cuda': self._use_cuda
        }

        if self._use_cuda:
            self._sample_device_function = self._cuda_sample
        else:
            self._sample_device_function = self._cpu_sample

        _callbacks.on_start(state)

        self._model.train()
        for state['epoch'] in range(initial_epoch, epochs):
            if self.stop_training:
                break

            _callbacks.on_start_epoch(state)
            self._metrics.reset(state)

            train_iterator = iter(state['generator'])
            state['train_iterator'] = train_iterator
            for state['t'] in range(0, train_steps):
                data = next(train_iterator)

                # Extract batch
                x, y_true = data
                x, y_true = Variable(x), Variable(y_true)
                x, y_true = self._sample_device_function(x, y_true)
                _callbacks.on_sample(state)

                # Zero grads
                self._optimizer.zero_grad()

                # Forward pass
                y_pred = self._model(x)
                state['y_pred'] = y_pred.data
                state['y_true'] = y_true.data
                _callbacks.on_forward(state)

                # Loss Calculation
                loss = self._criterion(y_pred, y_true)
                state['loss'] = loss.data
                _callbacks.on_forward_criterion(state)
                state['metrics'] = self._metrics.train_dict(state)

                # Backwards pass
                loss.backward()
                _callbacks.on_backward(state)

                # Update parameters
                self._optimizer.step()
                _callbacks.on_update_parameters(state)

            state['final_metrics'] = self._metrics.final_train_dict(state)

            # Validate
            state['validation_generator'] = validation_generator
            if validation_generator is not None:
                self._metrics.reset(state)
                self._model.eval()
                self._validate(validation_generator, validation_steps, state)
                state['final_metrics'].update(self._metrics.final_validate_dict(state))

            _callbacks.on_end_epoch(state)

        _callbacks.on_end(state)

        return history

    def _validate(self, validation_generator, num_validation_steps, state):
        self._model.eval()
        validation_iterator = iter(validation_generator)
        for step in range(num_validation_steps):

            # Load batch
            x, y_true = next(validation_iterator)
            x, y_true = Variable(x, volatile=True), Variable(y_true, volatile=True)
            x, y_true = self._sample_device_function(x, y_true)

            # Forward pass
            y_pred = self._model(x)
            state['y_pred'] = y_pred.data
            state['y_true'] = y_true.data

            # Loss and metrics
            loss = self._criterion(y_pred, y_true)
            state['loss'] = loss.data
            state['metrics'] = self._metrics.validate_dict(state)

    def cuda(self):
        self._model.cuda()
        self._use_cuda = True
        self._sample_device_function = self._cuda_sample
        return self

    def cpu(self):
        self._model.cpu()
        self._use_cuda = False
        self._sample_device_function = self._cpu_sample
        return self

    @staticmethod
    def _cuda_sample(x, y):
        return x.cuda(), y.cuda()

    @staticmethod
    def _cpu_sample(x, y):
        return x, y


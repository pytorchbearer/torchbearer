import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import get_train_valid_sets
from bink.callbacks.callbacks import CallbackList
from bink.callbacks.printer import Tqdm
from torch.autograd import Variable
from bink import metrics as bink_metrics


class Model:
    def __init__(self, model, optimizer, loss_criterion, metrics=[]):
        super().__init__()
        self.main_state = {
            'model': model,
            'criterion': loss_criterion,
            'optimizer': optimizer,
            'use_cuda': False,
            'metric_list': bink_metrics.MetricList(metrics)
        }
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
        history = None

        if verbose == 1:
            callbacks = [Tqdm()] + callbacks
        _callbacks = CallbackList(callbacks)

        # Get train and validation steps
        if validation_steps is None and validation_generator is not None:
            validation_steps = len(validation_generator)
        if train_steps is None:
            train_steps = len(generator)

        # Init state
        state = {
            'max_epochs': epochs,
            'train_steps': train_steps,
            'validation_steps': validation_steps,
            't': 0,
            'generator': generator,
            'stop_training': False
        }
        state.update(self.main_state)

        # Set cuda
        if state['use_cuda']:
            self._sample_device_function = self._cuda_sample
        else:
            self._sample_device_function = self._cpu_sample

        _callbacks.on_start(state)

        self.train()
        for state['epoch'] in range(initial_epoch, epochs):
            if state['stop_training']:
                break

            _callbacks.on_start_epoch(state)
            state['metric_list'].reset(state)

            # Init iterator
            train_iterator = iter(state['generator'])
            state['train_iterator'] = train_iterator
            for state['t'] in range(0, train_steps):
                data = next(train_iterator)

                # Extract batch
                x, y_true = data
                x, y_true = Variable(x), Variable(y_true)
                x, y_true = self._sample_device_function(x), self._sample_device_function(y_true)
                _callbacks.on_sample(state)

                # Zero grads
                state['optimizer'].zero_grad()

                # Forward pass
                y_pred = state['model'](x)
                state['y_pred'] = y_pred.data
                state['y_true'] = y_true.data
                _callbacks.on_forward(state)

                # Loss Calculation
                loss = state['criterion'](y_pred, y_true)
                state['loss'] = loss.data
                _callbacks.on_forward_criterion(state)
                state['metrics'] = state['metric_list'].evaluate_dict(state)

                # Backwards pass
                loss.backward()
                _callbacks.on_backward(state)

                # Update parameters
                state['optimizer'].step()
                _callbacks.on_update_parameters(state)

            state['final_metrics'] = state['metric_list'].evaluate_final_dict(state)

            # Validate
            state['validation_generator'] = validation_generator
            if validation_generator is not None:
                state['metric_list'].reset(state)
                self.eval()
                self._validate(validation_generator, validation_steps, state)
                state['final_metrics'].update(state['metric_list'].evaluate_final_dict(state))

            _callbacks.on_end_epoch(state)

        _callbacks.on_end(state)

        return history

    def _validate(self, validation_generator, num_validation_steps, state):
        self.eval()
        validation_iterator = iter(validation_generator)
        for step in range(num_validation_steps):

            # Load batch
            x, y_true = next(validation_iterator)
            x, y_true = Variable(x, volatile=True), Variable(y_true, volatile=True)
            x, y_true = self._sample_device_function(x), self._sample_device_function(y_true)

            # Forward pass
            y_pred = state['model'](x)
            state['y_pred'] = y_pred.data
            state['y_true'] = y_true.data

            # Loss and metrics
            loss = state['criterion'](y_pred, y_true)
            state['loss'] = loss.data
            state['metrics'] = state['metric_list'].evaluate_dict(state)

    # TODO: num workers?
    def evaluate(self, x=None, y=None, batch_size=32, verbose=1, steps=None):
        trainset = DataLoader(TensorDataset(x, y), batch_size, steps)

        return self.evaluate_generator(trainset, verbose)

    def evaluate_generator(self, generator, verbose=1, steps=None):
        bar = CallbackList([])
        if verbose == 1:
            bar = Tqdm()

        if steps is None:
            steps = len(generator)

        # Init state
        state = {
            'epoch': 0,
            'max_epochs': 1,
            'evaluation_steps': steps,
            't': 0,
            'generator': generator
            }
        state.update(self.main_state)

        state['metric_list'].reset(state)
        state['model'].eval()

        self.eval()
        bar.on_start_epoch(state)
        loader = iter(state['generator'])
        for state['t'] in range(steps):

            # Load batch
            x, y_true = next(loader)
            x, y_true = Variable(x, volatile=True), Variable(y_true, volatile=True)
            x, y_true = self._sample_device_function(x), self._sample_device_function(y_true)

            # Forward pass
            y_pred = state['model'](x)
            state['y_pred'] = y_pred.data
            state['y_true'] = y_true.data

            # Loss and metrics
            loss = state['criterion'](y_pred, y_true)
            state['loss'] = loss.data
            state['metrics'] = state['metric_list'].evaluate_dict(state)
            bar.on_update_parameters(state)

        state['final_metrics'] = state['metric_list'].evaluate_final_dict(state)
        bar.on_end_epoch(state)

        return state['final_metrics']

    def predict(self, x=None, batch_size=None, verbose=0, steps=None):
        pred_set = DataLoader(TensorDataset(x, None), batch_size, steps)

        return self.predict_generator(pred_set, verbose)

    def predict_generator(self, generator, verbose=0, steps=None):
        bar = CallbackList([])
        if verbose == 1:
            bar = Tqdm()

        if steps is None:
            steps = len(generator)

        # Init state
        state = {
            'epoch': 0,
            'max_epochs': 1,
            'generator': generator,
            'prediction_steps': steps,
            'metrics': {},
            'final_metrics': {}
        }
        state.update(self.main_state)

        self.eval()
        bar.on_start_epoch(state)

        loader = iter(state['generator'])
        predictions_list = []
        for state['t'] in range(steps):
            # Load batch
            x, _ = next(loader)
            x = Variable(x, volatile=True)
            x = self._sample_device_function(x)

            # Forward pass
            y_pred = state['model'](x)
            predictions_list.append(y_pred)
            bar.on_update_parameters(state)

        # Aggregate Predictions
        bar.on_end_epoch(state)
        predictions = torch.cat(predictions_list, 0)

        return predictions

    def train(self):
        self.main_state['model'].train()
        self.main_state['metric_list'].train()

    def eval(self):
        self.main_state['model'].eval()
        self.main_state['metric_list'].eval()

    def cuda(self):
        self.main_state['model'].cuda()
        self._sample_device_function = self._cuda_sample
        self.main_state['use_cuda'] = True
        return self

    def cpu(self):
        self.main_state['model'].cpu()
        self._sample_device_function = self._cpu_sample
        self.main_state['use_cuda'] = False
        return self

    @staticmethod
    def _cuda_sample(x):
        return x.cuda()

    @staticmethod
    def _cpu_sample(x):
        return x.cpu()


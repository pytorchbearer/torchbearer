import torch
from torch.utils.data import DataLoader, TensorDataset
from bink.cv_utils import get_train_valid_sets
from bink.callbacks.callbacks import CallbackList
from bink.callbacks.printer import Tqdm
from bink.callbacks.aggregate_predictions import AggregatePredictions
from bink import metrics as bink_metrics


class Model:
    def __init__(self, model, optimizer, loss_criterion, metrics=[]):
        super().__init__()
        self.main_state = {
            'model': model,
            'criterion': loss_criterion,
            'optimizer': optimizer,
            'device': 'cpu',
            'dtype': torch.float32,
            'metric_list': bink_metrics.MetricList(metrics),
            'self': self
        }

    def fit(self, x, y, batch_size=None, epochs=1, verbose=1, callbacks=[], validation_split=0.0,
            validation_data=None, shuffle=True, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, workers=1, pass_state=False):
        
        trainset, valset = get_train_valid_sets(x, y, validation_data, validation_split, shuffle=shuffle)
        trainloader = DataLoader(trainset, batch_size, shuffle=shuffle, num_workers=workers)

        if valset is not None:
            valloader = DataLoader(valset, batch_size, shuffle=shuffle, num_workers=workers)
        else:
            valloader = None

        return self.fit_generator(trainloader, train_steps=steps_per_epoch, epochs=epochs, verbose=verbose,
                                  callbacks=callbacks, validation_generator=valloader, validation_steps=validation_steps,
                                  initial_epoch=initial_epoch, pass_state=pass_state)

    def fit_generator(self, generator, train_steps=None, epochs=1, verbose=1, callbacks=[],
                      validation_generator=None, validation_steps=None, initial_epoch=0, pass_state=False):
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
            't': 0,
            'generator': generator,
            'stop_training': False
        }
        state.update(self.main_state)

        _callbacks.on_start(state)

        for state['epoch'] in range(initial_epoch, epochs):
            _callbacks.on_start_epoch(state)

            state['train_iterator'] = iter(state['generator'])
            self.train()

            _callbacks.on_start_training(state)
            state['metric_list'].reset(state)

            for state['t'] in range(0, state['train_steps']):
                # Extract batch
                self._load_batch_standard('train', state)
                _callbacks.on_sample(state)

                # Zero grads
                state['optimizer'].zero_grad()

                # Forward pass
                if pass_state:
                    state['y_pred'] = state['model'](state['x'], state=state)
                else:
                    state['y_pred'] = state['model'](state['x'])
                _callbacks.on_forward(state)

                # Loss Calculation
                state['loss'] = state['criterion'](state['y_pred'], state['y_true'])

                _callbacks.on_forward_criterion(state)
                state['metrics'] = state['metric_list'].process(state)

                # Backwards pass
                state['loss'].backward()
                _callbacks.on_backward(state)

                # Update parameters
                state['optimizer'].step()
                _callbacks.on_step_training(state)

                if state['stop_training']:
                    break

            state['metrics'].update(state['metric_list'].process_final(state))
            final_metrics = state['metrics']

            _callbacks.on_end_training(state)

            # Validate
            if validation_generator is not None:
                state['validation_generator'] = validation_generator
                state['validation_steps'] = validation_steps
                self.eval()
                self._validate(state, _callbacks, pass_state)

            final_metrics.update(state['metrics'])
            state['metrics'] = final_metrics
            _callbacks.on_end_epoch(state)

            if state['stop_training']:
                break
        _callbacks.on_end(state)

        return state

    def _test_loop(self, state, callbacks, pass_state, batch_loader, num_steps=None):
        with torch.no_grad():
            state['metric_list'].reset(state)
            state['metrics'] = {}

            if num_steps is None:
                num_steps = len(state['validation_generator'])

            state['validation_iterator'] = iter(state['validation_generator'])

            callbacks.on_start_validation(state)

            for state['t'] in range(num_steps):
                # Load batch
                batch_loader('validation', state)

                # Forward pass
                if pass_state:
                    state['y_pred'] = state['model'](state['x'], state=state)
                else:
                    state['y_pred'] = state['model'](state['x'])

                # Loss and metrics
                if 'y_true' in state:
                    state['loss'] = state['criterion'](state['y_pred'], state['y_true'])
                    state['metrics'] = state['metric_list'].process(state)

                callbacks.on_step_validation(state)
                if state['stop_training']:
                    break

            if 'y_true' in state:
                state['metrics'].update(state['metric_list'].process_final(state))
            callbacks.on_end_validation(state)

    def _validate(self, state, _callbacks, pass_state):
        self._test_loop(state, _callbacks, pass_state, self._load_batch_standard, state['validation_steps'])

    def evaluate(self, x=None, y=None, batch_size=32, verbose=1, steps=None):
        trainset = DataLoader(TensorDataset(x, y), batch_size, steps)
        return self.evaluate_generator(trainset, verbose)

    def evaluate_generator(self, generator, verbose=1, steps=None, pass_state=False):
        state = {'epoch': 0, 'max_epochs': 1, 'stop_training': False, 'validation_generator': generator}
        state.update(self.main_state)

        _callbacks = []
        if verbose == 1:
            _callbacks.append(Tqdm('e'))
        self._test_loop(state, CallbackList(_callbacks), pass_state, self._load_batch_standard, steps)

        return state['metrics']

    def predict(self, x=None, batch_size=None, verbose=1, steps=None):
        pred_set = DataLoader(TensorDataset(x, None), batch_size, steps)
        return self.predict_generator(pred_set, verbose)

    def predict_generator(self, generator, verbose=1, steps=None, pass_state=False):
        state = {'epoch': 0, 'max_epochs': 1, 'stop_training': False, 'validation_generator': generator}
        state.update(self.main_state)

        _callbacks = [AggregatePredictions()]
        if verbose == 1:
            _callbacks.append(Tqdm('p'))
        self._test_loop(state, CallbackList(_callbacks), pass_state, self._load_batch_predict, steps)

        return state['final_predictions']

    def train(self):
        self.main_state['model'].train()
        self.main_state['metric_list'].train()

    def eval(self):
        self.main_state['model'].eval()
        self.main_state['metric_list'].eval()

    def to(self, *args, **kwargs):
        self.main_state['model'].to(*args, **kwargs)

        for state in self.main_state['optimizer'].state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(*args, **kwargs)

        device = None
        dtype = None
        for key, val in kwargs.items():
            if key == 'dtype':
                dtype = kwargs['dtype']
            elif 'device' in kwargs:
                device = kwargs['device']
        for arg in args:
            if isinstance(arg, torch.dtype):
                dtype = arg
            else:
                device = arg

        self.main_state['device'] = device
        self.main_state['dtype'] = dtype

        return self

    def cuda(self, device=torch.cuda.current_device()):
        return self.to('cuda:' + str(device))

    def cpu(self):
        return self.to('cpu')

    def load_state_dict(self, state_dict):
        self.main_state['model'].load_state_dict(state_dict['model'])
        self.main_state['optimizer'].load_state_dict(state_dict['optimizer'])

    def state_dict(self):
        state_dict = {
            'model': self.main_state['model'].state_dict(),
            'optimizer': self.main_state['optimizer'].state_dict()
        }
        return state_dict

    @staticmethod
    def _deep_to(batch, device, dtype):
        is_tuple = isinstance(batch, tuple)

        if isinstance(batch, list) or isinstance(batch, tuple):
            batch = list(batch)
            for i in range(len(batch)):
                batch[i] = Model._deep_to(batch[i], device, dtype)
            batch = tuple(batch) if is_tuple else batch
        else:
            if batch.dtype.is_floating_point:
                batch = batch.to(device, dtype)
            else:
                batch = batch.to(device)

        return batch

    @staticmethod
    def _load_batch_standard(iterator, state):
        state['x'], state['y_true'] = Model._deep_to(next(state[iterator + '_iterator']), state['device'], state['dtype'])

    @staticmethod
    def _load_batch_predict(iterator, state):
        data = Model._deep_to(next(state[iterator + '_iterator']), state['device'], state['dtype'])
        if isinstance(data, list) or isinstance(data, tuple):
            state['x'], state['y_true'] = data
        else:
            state['x'] = data

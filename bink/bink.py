import torch
from torch.utils.data import DataLoader, TensorDataset
from bink.utils import get_train_valid_sets
from bink.callbacks.callbacks import CallbackList
from bink.callbacks.printer import Tqdm
from bink.callbacks.aggregate_predictions import AggregatePredictions
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
            'metric_list': bink_metrics.MetricList(metrics),
            'self': self
        }
        self._sample_device_function = self._cpu_sample

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

        # Set cuda
        if state['use_cuda']:
            self._sample_device_function = self._cuda_sample
        else:
            self._sample_device_function = self._cpu_sample

        _callbacks.on_start(state)

        for state['epoch'] in range(initial_epoch, epochs):
            _callbacks.on_start_epoch(state)

            self.train()
            _callbacks.on_start_training(state)
            state['metric_list'].reset(state)

            # Init iterator
            train_iterator = iter(state['generator'])
            state['train_iterator'] = train_iterator
            for state['t'] in range(0, state['train_steps']):
                data = next(train_iterator)

                # Extract batch
                state['x'], state['y_true'] = data
                state['x'], state['y_true'] = Variable(state['x']), Variable(state['y_true'])
                state['x'], state['y_true'] = self._sample_device_function(state['x']), self._sample_device_function(state['y_true'])
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

    def _test_loop(self, state, _callbacks, pass_state, num_steps=None):
        self.eval()
        state['metric_list'].reset(state)

        if num_steps is None:
            num_steps = len(state['validation_generator'])

        _callbacks.on_start_validation(state)
        test_iterator = iter(state['validation_generator'])
        for state['t'] in range(num_steps):
            # Load batch
            state['x'], state['y_true'] = next(test_iterator)
            state['x'], state['y_true'] = Variable(state['x'], volatile=True), Variable(state['y_true'], volatile=True)
            state['x'], state['y_true'] = self._sample_device_function(state['x']), self._sample_device_function(state['y_true'])

            # Forward pass
            if pass_state:
                state['y_pred'] = state['model'](state['x'], state=state)
            else:
                state['y_pred'] = state['model'](state['x'])

            # Loss and metrics
            loss = state['criterion'](state['y_pred'], state['y_true'])
            state['loss'] = loss

            state['metrics'] = state['metric_list'].process(state)
            _callbacks.on_step_validation(state)
            if state['stop_training']:
                break

        state['metrics'].update(state['metric_list'].process_final(state))
        _callbacks.on_end_validation(state)

    def _validate(self, state, _callbacks, pass_state):
        self._test_loop(state, _callbacks, pass_state, state['validation_steps'])

    def evaluate(self, x=None, y=None, batch_size=32, verbose=1, steps=None):
        trainset = DataLoader(TensorDataset(x, y), batch_size, steps)
        return self.evaluate_generator(trainset, verbose)

    def evaluate_generator(self, generator, verbose=1, steps=None, pass_state=False):
        state = {'epoch': 0, 'max_epochs': 1, 'stop_training': False, 'validation_generator': generator}
        state.update(self.main_state)

        _callbacks = []
        if verbose == 1:
            _callbacks.append(Tqdm('e'))
        self._test_loop(state, CallbackList(_callbacks), pass_state, steps)

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
        self._test_loop(state, CallbackList(_callbacks), pass_state, steps)

        return state['final_predictions']

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

    def save(self, model_file, state_file, save_state_dict=False, save_keys=['optimizer', 'criterion', 'use_cuda']):
        '''
        Saves a binkmodel to a torch model file and a bink state file

        :param binkmodel: bink Model to be saved
        :param model_file: File string which torch model will be saved under
        :param state_file: File string which bink state will be sved under
        :param save_state_dict: Whether to save torch model as state_dict
        :param save_keys: Keys of binkmodel state objects to be saved
        '''
        try:

            if save_state_dict:
                torch.save(self.main_state['model'].state_dict(), model_file)
            else:
                torch.save(self.main_state['model'], model_file)

            bink_state = _build_bink_state_object(self.main_state, save_keys)
            torch.save(bink_state, state_file)

        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))

    def load(self, model_file, state_file, torchmodel=None):
        '''
        Loads a bink model saved by Model.save()

        :param model_file: File string to saved torch model
        :param state_file: File string to saved bink state
        :return:
        '''
        try:

            if torchmodel is None:
                torchmodel = torch.load(model_file)
            else:
                state_dict = torch.load(model_file)
                torchmodel.load_state_dict(state_dict)

            bink_state_obj = torch.load(state_file)
            bink_state_obj['model'] = torchmodel

            return _restore_bink_from_state(self, bink_state_obj)

        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))


def _build_bink_state_object(state, save_keys):
    bink_state = {key: state[key] for key in save_keys}
    return bink_state


def _restore_bink_from_state(binkmodel, bink_state_obj):
    binkmodel.main_state.update(bink_state_obj)

    if 'use_cuda' in bink_state_obj and bink_state_obj['use_cuda']:
        binkmodel.cuda()

    return binkmodel

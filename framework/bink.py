from torch.utils.data import DataLoader, TensorDataset
from utils import get_train_valid_sets
from framework.callbacks.callbacks import CallbackList
from torch.autograd import Variable
from framework.metrics.metrics import MetricList


class Model:
    def __init__(self, model, optimizer, loss_criterion, metrics=[]):
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._criterion = loss_criterion
        self._metrics = MetricList(metrics)
        self._use_cuda = False

    def fit(self, x, y, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0,
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

    def fit_generator(self, generator, train_steps=None, epochs=1, verbose=1, callbacks=None,
                      validation_generator=None, validation_steps=None, class_weight=None, initial_epoch=0):
        self.stop_training = False
        history = None
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
            't': 0
        }

        _callbacks.on_start(state)

        self._model.train()
        for epoch in range(initial_epoch, epochs):
            _callbacks.on_start_epoch(state)
            state['epoch'] = epoch

            train_iterator = iter(generator)
            state['train_iterator'] = train_iterator
            for i in range(1, train_steps + 1):
                data = next(train_iterator)

                # Extract batch
                x, y_true = data
                x, y_true = Variable(x), Variable(y_true)
                if self._use_cuda:
                    x, y_true = x.cuda(), y_true.cuda()
                _callbacks.on_sample(state)

                # Zero grads
                self._optimizer.zero_grad()

                # Forward pass
                y_pred = self._model(x)
                state['y_pred'] = y_pred.data
                state['y_true'] = y_true.data
                state['t'] = state['t'] + 1
                _callbacks.on_forward(state)

                # Loss Calculation
                loss = self._criterion(y_pred, y_true)
                state['loss'] = loss.data[0]
                _callbacks.on_forward_criterion(state)
                state['metrics'] = self._metrics.train_dict(state)

                # Backwards pass
                loss.backward()
                _callbacks.on_backward(state)

                # Update parameters
                self._optimizer.step()
                _callbacks.on_update_parameters(state)

            metrics = self._metrics.final_train_dict(state)
            state['metrics'].update(metrics)

            # Validate
            state['validation_generator'] = validation_generator
            if validation_generator is not None:
                self._model.eval()
                self._validate(validation_generator, validation_steps, state)
                metrics = self._metrics.final_validate_dict(state)
                state['metrics'].update(metrics)

            _callbacks.on_end_epoch(state)
            self._metrics.reset()

        _callbacks.on_end(state)

        return history

    def _validate(self, validation_generator, num_validation_steps, state):
        validation_iterator = iter(validation_generator)
        for step in range(num_validation_steps):

            # Load batch
            x, y_true = next(validation_iterator)
            x, y_true = Variable(x, volatile=True), Variable(y_true, volatile=True)
            if self._use_cuda:
                x, y_true = x.cuda(), y_true.cuda()

            # Forward pass
            y_pred = self._model(x)
            state['y_pred'] = y_pred.data
            state['y_true'] = y_true.data

            # Loss and metrics
            loss = self._criterion(y_pred, y_true)
            state['loss'] = loss.data[0]
            metrics = self._metrics.validate_dict(state)
            state['metrics'].update(metrics)

    def cuda(self):
        self._model.cuda()
        self._use_cuda = True
        return self

    def cpu(self):
        self._model.cpu()
        self._use_cuda = False
        return self

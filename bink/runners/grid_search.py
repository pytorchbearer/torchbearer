from sklearn.model_selection import ParameterGrid
from torch import nn

from bink.callbacks import CallbackList
from bink.bink import Model

class GridSearchRunner:
    def __init__(self, trainloader, torch_model_type, optimizer, model_init_arg_dict, optimizer_param_dict,
                 criterion_list, fit_param_dict, metrics=['loss'], monitor='val_loss', return_all_metrics=False,
                 checkpointers=[]):
        self.trainloader = trainloader
        self.torch_model_type = torch_model_type
        self.optimizer = optimizer
        self.model_init_arg_dict = model_init_arg_dict
        self.optimizer_param_dict = optimizer_param_dict
        self.criterion_list = criterion_list
        self.fit_param_dict = fit_param_dict
        self.metrics = metrics
        self.return_all_metrics = return_all_metrics
        self.monitor = monitor
        self.use_cuda = False
        if checkpointers is not []:
            self.check = CallbackList(checkpointers)
        else:
            self.check = None

    def _send_to_list(self, args):
        if not isinstance(args, list):
            return [args]
        else:
            return args

    def _get_param_iterator(self):

        full_dict = {}

        for key, opt in self.optimizer_param_dict.items():
            opt = self._send_to_list(opt)
            if len(opt) > 0:
                full_dict['opt_'+key] = opt

        for key, fit in self.fit_param_dict.items():
            fit = self._send_to_list(fit)
            if len(fit) > 0:
                full_dict['fit_'+key] = fit

        for key, model in self.model_init_arg_dict.items():
            model = self._send_to_list(model)
            if len(model) > 0:
                full_dict['model_'+key] = model

        self.criterion_list = self._send_to_list(self.criterion_list)
        if len(self.criterion_list) > 0:
            full_dict['crit'] = self.criterion_list

        return ParameterGrid(full_dict)

    def run(self):
        self.param_iter = self._get_param_iterator()
        param_iter = self.param_iter
        results = {}
        grid_point = 1

        if self.check is not None:
            self.check.on_start(None)

        for i, params in enumerate(param_iter):
            print('Running Grid Point: {:d}/{:d}'.format(grid_point, param_iter.__len__()))

            model = {key[6:]: value for key,value in params.items() if 'model_' in key}
            opt = {key[4:]: value for key, value in params.items() if 'opt_' in key}
            fit = {key[4:]: value for key, value in params.items() if 'fit_' in key}
            crit = params['crit'] if 'crit' in params.keys() else nn.CrossEntropyLoss()

            torchmodel = self.torch_model_type(**model)
            optim = self.optimizer(torchmodel.parameters(), **opt)
            binkmodel = Model(torchmodel, optim, crit, self.metrics)

            if self.use_cuda:
                binkmodel.cuda()

            state = binkmodel.fit_generator(self.trainloader, **fit)

            if self.check is not None:
                self.check.on_end_epoch(state)

            if self.return_all_metrics:
                results[i] = state['metrics']
            else:
                results[i] = state['metrics'][self.monitor]

            grid_point += 1

        return results

    def describe_grid_point(self, index):
        return self.param_iter[index]

    def cuda(self):
        self.use_cuda = True

    def cpu(self):
        self.use_cuda = False

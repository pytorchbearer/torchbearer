from sklearn.model_selection import ParameterGrid
from bink import Model

class GridSearchRunner:
    def __init__(self, trainloader, torch_model_type, optimizer, model_init_arg_dict, optimizer_param_dict,
                 criterion_list, fit_param_dict, metrics, monitor='val_loss', return_all_metrics=False):
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

    def args_to_format(self):
        pass

    def get_param_iterator(self):

        full_dict = {}

        for key, opt in self.optimizer_param_dict.items():
            if not isinstance(opt, list):
                opt = [opt]
            full_dict['opt_'+key] = opt

        for key, fit in self.fit_param_dict.items():
            if not isinstance(fit, list):
                fit = [fit]
            full_dict['fit_'+key] = fit

        for key, model in self.model_init_arg_dict.items():
            if not isinstance(model, list):
                model = [model]
            full_dict['model_'+key] = model

        if not isinstance(self.criterion_list, list):
            self.criterion_list = [self.criterion_list]
        full_dict['crit'] = self.criterion_list

        return ParameterGrid(full_dict)

    def run(self):
        param_iter = self.get_param_iterator()
        results = {}
        grid_point = 1

        # Renew iterators?
        for i, params in enumerate(param_iter):
            print('Running Grid Point: {:d}/{:d}'.format(grid_point, param_iter.__len__()))

            model = {key[6:]: value for key,value in params.items() if 'model_' in key}
            opt = {key[4:]: value for key, value in params.items() if 'opt_' in key}
            fit = {key[4:]: value for key, value in params.items() if 'fit_' in key}
            crit = params['crit']

            torchmodel = self.torch_model_type(**model)
            optim = self.optimizer(torchmodel.parameters(), **opt)
            binkmodel = Model(torchmodel, optim, crit, self.metrics)

            if self.use_cuda:
                binkmodel.cuda()

            state = binkmodel.fit_generator(self.trainloader, **fit)

            if self.return_all_metrics:
                results[i] = state['metrics']
            else:
                results[i] = state['metrics'][self.monitor]

            grid_point += 1

        return results

    def cuda(self):
        self.use_cuda = True

    def cpu(self):
        self.use_cuda = False
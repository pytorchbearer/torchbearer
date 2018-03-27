import numpy as np
import torch.utils.data as data
from utils import DatasetCrossValidationIter
import copy


class CrossValidationRunner:
    def __init__(self, binkmodel, dataset, batch_size=32, num_folds=2, valid_split=0.1, shuffle=False):
        super().__init__()
        self.binkmodel = binkmodel
        self.num_folds = num_folds
        self.validation_dataset = DatasetCrossValidationIter(dataset, num_folds, valid_split, shuffle=shuffle)
        self.batch_size = batch_size

    def run(self, epochs=1, train_steps=None, verbose=1, callbacks=[]):

        models = []
        metric_aggregate = None

        for i in range(self.num_folds):
            model = copy.deepcopy(self.binkmodel)
            trainset, valset = next(self.validation_dataset)

            trainloader, valloader = data.DataLoader(trainset, self.batch_size), data.DataLoader(valset, self.batch_size)
            state = model.fit_generator(trainloader, validation_generator=valloader, train_steps=train_steps,
                                        epochs=epochs, verbose=verbose, callbacks=callbacks)
            models.append(model)
            
            if metric_aggregate is None:
                metric_aggregate = state['final_metrics']
                for metric in metric_aggregate:
                    metric_aggregate[metric] = [metric_aggregate[metric]]
            else:
                for metric in metric_aggregate:
                    metric_aggregate[metric].append(state['final_metrics'][metric])

        metric_mean = {}
        metric_std = {}
        for metric in metric_aggregate:
            metric_mean[metric] = np.mean(metric_aggregate[metric])
            metric_std[metric] = np.std(metric_aggregate[metric])

        metric_strings = []
        for (key, value) in metric_mean.items():
            std = metric_std[key]
            metric_strings.append('{0}:{1:.03g}(+/-{2:.03f})'.format(key, value, std))

        valid_str = ", ".join(metric_strings)
        print('\nValidation Results:')
        print(valid_str)

        return models
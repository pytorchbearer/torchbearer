import numpy as np
import torch.utils.data as data
from bink.cv_utils import ShuffleSplitCVIter, KFoldCVIter, LeavePOutCVIter
import copy


# TODO: Allow merging of separate validation dataset? 
class CrossValidationRunner:
    def __init__(self, binkmodel, dataset, batch_size=32, num_folds=2, valid_split=0.1, splitter='shufflesplit'):
        """
        Creates a cross validation runner for bink models
        
        :param binkmodel: binkmodel object 
        :param dataset: PyTorch dataset object
        :param batch_size: Batch size for model
        :param num_folds: Number of folds to run
        :param valid_split: Amount of data to use in validation (for shufflesplit and leave_p_out)
        :param splitter: Defines the type of sklearn model selection splitter to use. One of ['shufflesplit', 'kfold', 'leave_p_out']
        """
        super().__init__()
        self.binkmodel = binkmodel
        self.num_folds = num_folds
        
        if splitter == 'shufflesplit':
            self.validation_dataset = ShuffleSplitCVIter(dataset, num_folds, valid_split)
        elif splitter == 'kfold':
            self.validation_dataset = KFoldCVIter(dataset, num_folds)
        elif splitter == 'leave_p_out':
            n = int(len(dataset)*valid_split)
            self.validation_dataset = LeavePOutCVIter(dataset, num_folds, n)
        else: 
            self.validation_dataset = ShuffleSplitCVIter(dataset, num_folds, valid_split)

        self.batch_size = batch_size

    def run(self, epochs=1, train_steps=None, verbose=1, callbacks=[]):

        models = []
        metric_aggregate = None

        for i in range(self.num_folds):
            print('Running Fold: {:d}/{:d}'.format(i+1, self.num_folds))
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
            metric_strings.append('{0}:{1:.03g}(+/-{2:.1E})'.format(key, value, std))

        valid_str = ", ".join(metric_strings)
        print('\nValidation Results:')
        print(valid_str)

        return models
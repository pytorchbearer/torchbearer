import torch.utils.data as data
from utils import DatasetCrossValidation
import copy

class CrossValidationRunner:
    def __init__(self, binkmodel, dataset, batch_size=32, num_folds=2, valid_split=0.1, shuffle=False):
        super().__init__()
        self.binkmodel = binkmodel
        self.num_folds = num_folds
        self.validation_dataset = DatasetCrossValidation(dataset, num_folds, valid_split, shuffle=shuffle)
        self.batch_size = batch_size

    def run(self, train_steps=None, epochs=1, verbose=1, callbacks=[]):

        models = []
        for i in range(self.num_folds):
            model = copy.deepcopy(self.binkmodel)
            trainset, validset = next(self.validation_dataset)

            trainloader, validloader = data.DataLoader(trainset, self.batch_size), data.DataLoader(validset, self.batch_size)
            model.fit_generator(trainloader, validation_generator=validloader, train_steps=train_steps, epochs=epochs, verbose=verbose, callbacks=callbacks)
            models.append(model)

        return models
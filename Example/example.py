import os

import torch.nn as nn
import torch.optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Example import inception_network as nm
from bink.bink import Model
from bink.metrics import RocAucScore


####### Paths #######
dataset_path = '/home/matt/datasets'
model_load_path = ''  # Path to a saved model which is loaded if "newmodel" is true
logfile = 'logs.log'
folderName = 'test'


####### Datasets #######
batch_size = 128
num_workers = 8

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(dataset_path, train=True, transform=transform)
testset =  torchvision.datasets.MNIST(dataset_path, train=False, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


####### Parameters #######
modelType = nm.InceptionSmall
modelName = modelType.name
modelPath = os.getcwd() + '/' + modelName + '/' + folderName + '/'

model = nm.InceptionSmall()

####### Trainer #######

model = Model(model, torch.optim.SGD(model.parameters(), 0.001), nn.CrossEntropyLoss(), metrics=[RocAucScore(), 'acc', 'loss']).cuda()
model.fit_generator(trainloader, validation_generator=testloader, callbacks=[], epochs=10)

import os

import torch.nn as nn
import torch.optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Example import inception_network as nm
from bink import Model
from bink.metrics import RocAucScore
from bink.callbacks import EarlyStopping, TensorBoard, L2WeightDecay, L1WeightDecay

####### Paths #######
dataset_path = '/home/matt/datasets/cifar'
model_load_path = ''  # Path to a saved model which is loaded if "newmodel" is true
logfile = 'logs.log'
folderName = 'test'


####### Datasets #######
batch_size = 128
num_workers = 8

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(dataset_path, train=True, transform=transform)
testset =  torchvision.datasets.CIFAR10(dataset_path, train=False, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


####### Parameters #######
modelType = nm.InceptionSmall
modelName = modelType.name
modelPath = os.getcwd() + '/' + modelName + '/' + folderName + '/'



my_model = nm.InceptionSmall(channels=3)

####### Trainer #######

from datetime import datetime
current_time = datetime.now().strftime('%b%d_%H-%M-%S')

model = Model(my_model, torch.optim.SGD(my_model.parameters(), 0.001), nn.CrossEntropyLoss(), metrics=['acc', 'loss']).cuda()
model.fit_generator(trainloader, validation_generator=testloader, callbacks=[L1WeightDecay(), L2WeightDecay()], epochs=100)

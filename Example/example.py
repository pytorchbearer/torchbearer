import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from framework.bink import Model
from Example import inception_network as nm
import torch.optim

import torch.nn as nn

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

####### Paths #######
dataset_path = '/home/matt/datasets/mnist'
model_load_path = ''  # Path to a saved model which is loaded if "newmodel" is true
logfile = 'logs.log'
folderName = 'test'


####### Datasets #######
batch_size = 32
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
modelArgs = None
modelPath = os.getcwd() + '/' + modelName + '/' + folderName + '/'

# Create new model or load old one?
model = nm.InceptionSmall()

####### Trainer #######

model = Model(model, torch.optim.SGD(model.parameters(), 0.001), nn.CrossEntropyLoss())
model.fit_generator(trainloader)



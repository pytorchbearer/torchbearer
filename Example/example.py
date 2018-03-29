import os

import torch.nn as nn
import torch.optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Example import inception_network as nm
from bink.bink import Model
from bink.metrics import RocAucScore
from bink.callbacks import EarlyStopping, TensorBoard, TensorBoardImageVis


####### Paths #######
dataset_path = '/home/ethan/datasets'
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



model = nm.InceptionSmall(channels=3)

####### Trainer #######

from datetime import datetime
current_time = datetime.now().strftime('%b%d_%H-%M-%S')

model = Model(model, torch.optim.SGD(model.parameters(), 0.001), nn.CrossEntropyLoss(), metrics=['acc', 'loss']).cuda()
model.fit_generator(trainloader, validation_generator=testloader, callbacks=[TensorBoardImageVis(num_images=250, comment=current_time), TensorBoard(comment=current_time, write_batch_metrics=True, write_graph=True)], epochs=100)

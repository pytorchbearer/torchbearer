import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionSmall(nn.Module):
    name = 'Inception'

    def __init__(self, channels=1):
        super(InceptionSmall, self).__init__()
        self.conv1 = ConvModule(channels, 96, 3, 1)
        self.incep = InceptionModule(96, 32, 32)
        self.incep2 = InceptionModule(64, 32, 48)
        self.downsamp = DownSampleModule(80, 80)

        self.incep3 = InceptionModule(160, 112, 48)
        self.incep4 = InceptionModule(160, 96, 64)
        self.incep5 = InceptionModule(160, 80, 80)
        self.incep6 = InceptionModule(160, 48, 96)
        self.downsamp2 = DownSampleModule(144, 96)

        self.incep7 = InceptionModule(240, 176, 160)
        self.incep8 = InceptionModule(336, 176, 160)
        self.meanPool = nn.AvgPool2d(5)

        self.fc1 = nn.Linear(336, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.incep(x)
        x = self.incep2(x)
        x = self.downsamp(x)

        x = self.incep3(x)
        x = self.incep4(x)
        x = self.incep5(x)
        x = self.incep6(x)
        x = self.downsamp2(x)

        x = self.incep7(x)
        x = self.incep8(x)
        x = self.meanPool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        return x

class ConvModule(nn.Module):
    def __init__(self, I, C, K, S, padding=(0, 0)):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(I, C, kernel_size=K, stride=S, padding=padding)
        self.batchNorm = nn.BatchNorm2d(C, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        return F.relu(x, inplace=True)

class InceptionModule(nn.Module):
    def __init__(self, I, C1, C3):
        super(InceptionModule, self).__init__()
        self.conv1 = ConvModule(I, C1, 1, 1)
        self.conv2 = ConvModule(I, C3, 3, 1, padding=(1, 1))

    def forward(self, x):
        y = self.conv1(x)
        z = self.conv2(x)

        outputs = [y, z]
        return torch.cat(outputs, 1)

class DownSampleModule(nn.Module):
    def __init__(self, I, C3):
        super(DownSampleModule, self).__init__()
        self.conv1 = ConvModule(I, C3, 3, 2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        y = self.conv1(x)
        z = self.pool(x)

        outputs = [y, z]
        return torch.cat(outputs, 1)
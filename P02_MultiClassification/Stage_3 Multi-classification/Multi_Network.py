import torch.nn as nn
import torchvision
import torch.nn.functional as F

"""
input_image format should be N x C x H x W
input_image = torch.FloatTensor(1, 28, 28)
input_image = Variable(input_image)
input_image = input_image.unsqueeze(0) # 1 x 1 x 28 x 28
here, input_image = batch (128) x 3 (RGB) x 500 x 500
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() # must run the parent constructor
        self.conv1 = nn.Conv2d(3, 3, 3) # in_channels, out_channels, kernel_size, stride=1, padding=0
        # 3 in_channels(RGB image), 3 kernels(3 3x3x3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(3, 6, 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(6 * 123 *123, 150)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc2_classes = nn.Linear(150, 2)
        self.fc2_species = nn.Linear(150, 3)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        x = x.view(-1, 6 * 123 * 123)
        x = self.fc1(x)
        x = self.relu3(x)
        x = F.dropout(x, training=self.training) #random dropout for training phase, static for testing

        x_classes = self.fc2_classes(x)
        # x_classes = self.softmax(x_classes)
        x_species = self.fc2_species(x)
        # x_species = self.softmax(x_species)

        return x_classes, x_species
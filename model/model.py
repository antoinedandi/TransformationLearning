import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class LocalizationModel(BaseModel):
    def __init__(self):
        super(LocalizationModel,self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=3, out_channels=16,kernel_size=3,
                    stride=1,padding=1),
                nn.ReLU(inplace=True)
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=3,
                    stride=1,padding=1),
                nn.ReLU(inplace=True)
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(
                    in_channels=16, out_channels=32, kernel_size=3,
                    stride=1,padding=1),
                nn.ReLU(inplace=True)
                )
        self.pooling1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv4 = nn.Sequential(
                nn.Conv2d(
                    in_channels=32,out_channels=16,kernel_size=3,
                    stride=1,padding=1),
                nn.ReLU(inplace=True)
                )
        self.conv5 = nn.Sequential(
                nn.Conv2d(
                    in_channels=16,out_channels=32,kernel_size=3,
                    stride=1,padding=1),
                nn.ReLU(inplace=True)
                )
        self.pooling2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.dropout = nn.Dropout2d(0.5)
        self.linear = nn.Sequential(nn.Linear(16*16*32,4))
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pooling1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pooling2(x)
        x = self.dropout(x)
        x = x.view(-1,16*16*32)
        x = self.linear(x)
        return x

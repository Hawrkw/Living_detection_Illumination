import torch
import torch.nn as nn

def conv3x3(in_channels,out_channels,stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)

class efficient_net(nn.Module):
    def __init__(self):
        super(efficient_net,self).__init__()
        self.conv1 = conv3x3(7,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(16,16)
        self.bn2 = nn.BatchNorm2d(16)
        self.maxpool  = nn.MaxPool2d(kernel_size=2,stride=2)
        self.drop1 = nn.Dropout(0.25)
        self.conv3 = conv3x3(16,32)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = conv3x3(32,32)
        self.bn4 = nn.BatchNorm2d(32)
        self.layer1 = nn.Linear(111,64)
        self.drop2 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(64,2)

    def forward(self,img):
        out = self.conv1(img)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.maxpool(out)
        out = self.drop1(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.bn4(out)
        out = self.maxpool(out)
        out = self.drop1(out)
        #深度克隆，不然out改变，featuremap也会改变
        featuremap = out.clone()
        batch = out.size(0)
        #数据转换为batch行
        out = out.reshape(batch,-1)

        out = self.layer1(out)
        out = self.drop2(out)
        out = self.layer2(out)
        return featuremap,out
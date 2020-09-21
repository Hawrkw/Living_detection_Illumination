import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
class Net(nn.Module):
    def __init__(self,input_channel,output_channle):
        super(Net,self).__init__()
        self.hidden = nn.Linear(input_channel,3)
        self.predict = nn.Linear(3,output_channle)

    def forward(self,data):
        output = F.relu(self.hidden(data),inplace=True)
        output = self.predict(output)
        return output

epoch = 13000

#应该输入人脸框大小-宽高，输出距离
net = Net(2,1)
loss_func = nn.MSELoss()
#loss_func = nn.L1Loss()
optimizer = optim.Adam(net.parameters(),0.01)

data_path = 'data/data_w.txt'
data = []
with open(data_path, 'r') as f:
    for line in f.readlines():
        item = list(map(float,line.split(" ")))
        data.append(item)
data = np.array(data)
input = torch.tensor(data[:,0:2], dtype=torch.float)
label = torch.tensor(data[:,2], dtype=torch.float)
# input = torch.range(1,8).reshape(4,2)
# label = [3,7,11,15]
# label = torch.tensor(label,dtype=torch.float)
net.train()
for i in range(epoch):
    if i == 3000:
        print(1)
    losses = []
    predict = net(input)
    # loss = loss_func(predict,label)
    # loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()
    for j in range(input.shape[0]):

        predict1 = net(input[j])
        loss = loss_func(predict1, label[j])
        losses.append(loss)
        #print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # loss_mean = 0
    # for item in losses:
    #     loss_mean += item
    # loss_mean /= len(losses)
    #print(loss1)
    #print(loss_mean)

from matplotlib import pyplot as plt
plt.plot(data[:,0], data[:,2], marker='o', mec='r', mfc='w',label='true')

predict = predict.detach().numpy()
xiang = [34.41,41.34,49.61,59.59,71.59,86.02,86.02,102.56,124.03,122.01,148.15,148.15,148.15]
#xiang = [34.41,49.61,71.59,86.02,86.02,102.56,102.56,122.61,122.61,148.15,148.5]
xiang = np.array(xiang)
plt.plot(data[:,0], predict, marker='*', mec='r', mfc='w',label='MLP')
plt.plot(data[:,0], xiang, marker='*', mec='blue', mfc='w',label='Similar triangles')
plt.legend()
plt.savefig('res1.png')
plt.show()


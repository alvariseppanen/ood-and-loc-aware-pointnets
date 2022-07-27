from __future__ import print_function
from dis import dis
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

g = 32

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc11 = nn.Linear(1024+g, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn11 = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x, voxel):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0] # bs, 1024, 1
        x = x.view(-1, 1024) # bs, 1024

        # add PVLE feature
        x = torch.cat([x, voxel],1) # bs, 1024+g
        x = F.relu(self.bn11(self.fc11(x))) # bs, 1024

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.global_feat = global_feat
        self.feature_transform = feature_transform 

    def forward(self, x, voxel):
        #n_pts = x.size()[2]
        
        trans = self.stn(x, voxel)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))

        avg = torch.mean(x, 2, keepdim=True)[0]
        avg = avg.view(-1, 128)
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 128)

        return x, avg, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc11 = nn.Linear(1, 16)
        self.fc22 = nn.Linear(16, 3)
        self.fc1 = nn.Linear(128+g, 128)
        self.fc3 = nn.Linear(128, k)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        
        self.conv11 = torch.nn.Conv1d(3, 64, 1)
        self.bn11 = nn.BatchNorm1d(64)
        self.fc33 = nn.Linear(64, g)
        self.bn33 = nn.BatchNorm1d(g)

    def forward(self, x, voxel):
        # encode world dist to origin
        #dist = F.relu(self.fc11(dist))
        #dist = F.relu(self.fc22(dist)) # (bs, 3)

        # encode voxel position
        voxel = F.relu(self.bn11(self.conv11(voxel)))
        voxel = voxel.view(-1, 64)
        voxel = F.relu(self.bn33(self.fc33(voxel)))
        #print(voxel.shape)
        #voxel = torch.zeros((x.size()[0],g)).cuda()
        #voxel = torch.zeros((x.size()[0],g)).cpu()

        x, avg, trans_feat = self.feat(x, voxel)
        global_feat = x

        # add PVLE feature
        x = torch.cat((x, voxel), 1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1), global_feat, avg, x

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())
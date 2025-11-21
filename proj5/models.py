import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        self.fc1=nn.Linear(3,64)
        self.fc2=nn.Linear(64,128)
        self.fc3=nn.Linear(128, 1024)
        self.fc4=nn.Linear(1024, 256)
        self.fc5=nn.Linear(256, 64)
        self.fc6=nn.Linear(64, num_classes)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        out=F.relu(self.fc1(points))     # B X N X 64
        out=F.relu(self.fc2(out))        # B X N X 128
        out=F.relu(self.fc3(out))        # B X N X 1024
        
        out=torch.max(out, dim=1).values      # B X 1024

        out=F.relu(self.fc4(out))     # B X 256
        out=F.relu(self.fc5(out))     # B X 64
        out=F.relu(self.fc6(out))     # B X num_classes

        return out

# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        pass

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        pass




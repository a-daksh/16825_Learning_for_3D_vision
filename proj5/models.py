import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        self.fc1=nn.Linear(3,64)
        self.bn1=nn.BatchNorm1d(64)

        self.fc2=nn.Linear(64,128)
        self.bn2=nn.BatchNorm1d(128)
        
        self.fc3=nn.Linear(128, 1024)
        self.bn3=nn.BatchNorm1d(1024)
        
        self.fc4=nn.Linear(1024, 256)
        self.bn4=nn.BatchNorm1d(256)
        
        self.fc5=nn.Linear(256, 64)
        self.bn5=nn.BatchNorm1d(64)
        
        self.fc6=nn.Linear(64, num_classes)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        def apply_bn(x, bn):
            return bn(x.permute(0,2,1)).permute(0,2,1)
        
        out=F.relu(self.fc1(points))     # B X N X 64
        out=apply_bn(out,self.bn1)

        out=F.relu(self.fc2(out))        # B X N X 128
        out=apply_bn(out,self.bn2)
        
        out=F.relu(self.fc3(out))        # B X N X 1024
        out=apply_bn(out,self.bn3)
        
        out=torch.max(out, dim=1).values      # B X 1024

        out=F.relu(self.fc4(out))     # B X 256
        out=self.bn4(out)
        
        out=F.relu(self.fc5(out))     # B X 64
        out=self.bn5(out)
        
        out=self.fc6(out)             # B X num_classes

        return out

# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        self.fc1=nn.Linear(3,64)
        self.bn1=nn.BatchNorm1d(64)

        self.fc2=nn.Linear(64,128)
        self.bn2=nn.BatchNorm1d(128)
        
        self.fc3=nn.Linear(128, 1024)
        self.bn3=nn.BatchNorm1d(1024)

        self.fc4=nn.Linear(1088, 512)
        self.bn4=nn.BatchNorm1d(512)
        
        self.fc5=nn.Linear(512, 256)
        self.bn5=nn.BatchNorm1d(256)
        
        self.fc6=nn.Linear(256, 128)
        self.bn6=nn.BatchNorm1d(128)

        self.fc7=nn.Linear(128, num_seg_classes)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        def apply_bn(x, bn):
            return bn(x.permute(0,2,1)).permute(0,2,1)
        
        out=F.relu(self.fc1(points))     # B X N X 64
        out1=apply_bn(out,self.bn1)

        out=F.relu(self.fc2(out1))        # B X N X 128
        out=apply_bn(out,self.bn2)
        
        out=F.relu(self.fc3(out))        # B X N X 1024
        out=apply_bn(out,self.bn3)
        
        out=torch.max(out, dim=1).values      # B X 1024
        out=(out.unsqueeze(1)).repeat(1,points.shape[1],1)      # B X N X 1024

        out=torch.cat((out1, out), dim=-1)      # B X N X 1088

        out=F.relu(self.fc4(out))       # B X N X 512
        out=apply_bn(out,self.bn4)

        out=F.relu(self.fc5(out))       # B X N X 256
        out=apply_bn(out,self.bn5)

        out=F.relu(self.fc6(out))       # B X N X 128
        out=apply_bn(out,self.bn6)

        out=self.fc7(out)               # B X N X n

        return out


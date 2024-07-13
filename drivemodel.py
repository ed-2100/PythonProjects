import torch.nn as nn

delta_t = 1 / 30

class MecanumDriveModel(nn.Module):
    def __init__(self,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MecanumDriveModel, self).__init__()

        self.fc1 = nn.Linear( 6, 48, False, **factory_kwargs)
        self.fc2 = nn.Linear(48, 24, False, **factory_kwargs)
        self.fc3 = nn.Linear(24, 12, False, **factory_kwargs)
        self.fc4 = nn.Linear(12,  6, True, **factory_kwargs)

        self.bn1 = nn.BatchNorm1d(48, **factory_kwargs)
        self.bn2 = nn.BatchNorm1d(24, **factory_kwargs)
        self.bn3 = nn.BatchNorm1d(12, **factory_kwargs)

        self.relu = nn.LeakyReLU(negative_slope = 0.01)
    
    def forward(self, x):
        a = self.relu(self.bn1(self.fc1(x)))
        b = self.relu(self.bn2(self.fc2(a)))
        c = self.relu(self.bn3(self.fc3(b)))
        d = self.fc4(c)
        return d

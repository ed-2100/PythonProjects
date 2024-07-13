import torch
import torch.nn as nn

delta_t = 1 / 30

class MecanumDriveModel(nn.Module):
    def __init__(self,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MecanumDriveModel, self).__init__()

        input_dim = 6
        output_dim = 6
        hidden_dim = 128

        self.fc1 = nn.Linear(input_dim,       hidden_dim, False,      **factory_kwargs)
        self.fc2 = nn.Linear(hidden_dim,      hidden_dim // 2, False, **factory_kwargs)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4, False, **factory_kwargs)
        self.fc4 = nn.Linear(hidden_dim // 4, output_dim, False,      **factory_kwargs)

        torch.nn.init.kaiming_uniform_(self.fc1.weight, a = 0.01, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight, a = 0.01, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc3.weight, a = 0.01, nonlinearity='leaky_relu')
        torch.nn.init.xavier_uniform_(self.fc4.weight)

        self.bn1 = nn.BatchNorm1d(hidden_dim, **factory_kwargs)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2, **factory_kwargs)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4, **factory_kwargs)

        self.do = nn.Dropout()
        self.relu = nn.LeakyReLU(negative_slope = 0.01)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        a = self.relu(self.bn1(self.fc1(x)))
        b = self.relu(self.bn2(self.fc2(a)))
        c = self.relu(self.bn3(self.fc3(b)))
        d = self.fc4(c)
        return d

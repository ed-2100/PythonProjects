import numpy as np
import scipy.integrate
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import torch.utils.data as data
from collections.abc import Iterator
import time
from sklearn import preprocessing

torch.set_num_threads(15)

global_seed = 123456789

rng = np.random.Generator(np.random.SFC64(np.random.SeedSequence(global_seed)))
torch.manual_seed(global_seed)

delta_t = 1 / 30

T_stall = 5.4 * (1 / 100) # N * m
w_free = 1620 * (1 / 60) * 2 * np.pi # rad / s
r = 4 * 0.0254 # m
mass = 6 # kg

# Percent of maximum motor torque.
Q = 0.65

inertia = mass

# The moment of inertia of a 1 ft x 1 ft square.
moment_of_inertia = mass * (12 * 0.0254) ** 2 / 6

# ##MMMMMMMM##
# ##M      M## |
#   M      M   + l_y
#   M      M   |
# ##M      M## |
# ##MMMMMMMM##
#  ---l_x----

l_x = 0.129907
l_y = 0.095724
a = l_x + l_y

xyt_to_tttt = np.array([[1, -1, -a],
                        [1,  1,  a],
                        [1,  1, -a],
                        [1, -1,  a]]) / r

tttt_to_xyt = np.array([[     1,      1,      1,      1],
                        [    -1,      1,      1,     -1],
                        [-1 / a,  1 / a, -1 / a,  1 / a]]) * (r / 4)

ctl_to_mot_pow = np.array([[1, -1, -1],
                           [1,  1,  1],
                           [1,  1, -1],
                           [1, -1,  1]])

def equations_of_motion(t: np.float32, state: np.ndarray, control: np.ndarray):
    x, y, theta, vx, vy, omega = state

    velocity = np.array([vx, vy, omega])

    dx_dt = velocity

    TTTT_to_axayat = xyt_to_tttt.T / np.array([inertia, inertia, moment_of_inertia])[:, np.newaxis]
    
    w = xyt_to_tttt @ velocity

    motor_powers = ctl_to_mot_pow @ control
    is_same_direction = ((np.sign(motor_powers * w) + 1) / 2)
    motor_torques = T_stall * (1 - np.abs(w) * is_same_direction / w_free) * motor_powers

    relative_acceleration = TTTT_to_axayat @ motor_torques

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s,  0],
                  [s,  c,  0],
                  [0,  0,  1]])

    dv_dt = np.dot(R, relative_acceleration)

    return np.concatenate([dx_dt, dv_dt])

def generate_data():
    position = np.array([0, 0, rng.uniform(-np.pi, np.pi)])

    r = rng.uniform(-1, 1, 3) / 3
    wheel_velocity = ctl_to_mot_pow @ r * w_free
    velocity = tttt_to_xyt @ wheel_velocity

    state = np.concatenate([position, velocity])

    control = rng.uniform(-1, 1, 3) / 3

    solution = scipy.integrate.solve_ivp(lambda t, y: equations_of_motion(t, y, control), (0, delta_t), state, t_eval=np.linspace(0, delta_t, 7))
    
    return np.concatenate([state[2:], control]), solution.y[:, -1] - state

class MecanumDriveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MecanumDriveModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        a = functional.relu(self.bn1(self.fc1(x)))
        b = functional.relu(self.bn2(self.fc2(a)))
        c = functional.relu(self.bn3(self.fc3(b)))
        d = functional.relu(self.bn4(self.fc4(c)))
        e = self.fc5(d)
        return e

class SampleGenerator(Iterator):
    def __init__(self, num_samples, batch_size = 1):
        self._batch = 0
        self.batch_size = batch_size
        self.num_samples = num_samples

        self._inputs = np.empty([num_samples, 4 + 3])
        self._targets = np.empty([num_samples, 6])
        for i in range(num_samples):
            self._inputs[i], self._targets[i] = generate_data()
        
        self._inputs_normalizer = preprocessing.StandardScaler()
        self._inputs_normalizer.fit(self._inputs)
        self._inputs = np.float32(self._inputs_normalizer.transform(self._inputs))
        self._targets_normalizer = preprocessing.StandardScaler()
        self._targets_normalizer.fit(self._targets)
        self._targets = np.float32(self._targets_normalizer.transform(self._targets))

    def __len__(self):
        return -(self.num_samples // -self.batch_size)

    def __next__(self):
        if self._batch * self.batch_size < self.num_samples:
            if self.batch_size == self.num_samples:
                temp = (torch.from_numpy(self._inputs),
                        torch.from_numpy(self._targets))
            elif self.batch_size > 1:
                temp = (torch.from_numpy(self._inputs[self._batch:self._batch+self.batch_size]),
                        torch.from_numpy(self._targets[self._batch:self._batch+self.batch_size]))
            else: # self.batch_size == 1
                temp = (torch.from_numpy(self._inputs[self._batch]),
                        torch.from_numpy(self._targets[self._batch]))
            self._batch += 1
            return temp
        else:
            self._batch = 0
            raise StopIteration

def train(model: nn.Module,
          criterion: nn.Module,
          optimizer: optim.Optimizer,
          dataset: Iterator):
    model.train(mode=True)

    num_epochs = 200000
    for epoch in range(1, num_epochs + 1):
        loss_sum = 0
        error_sum = 0
        
        start = time.time_ns()
        for i, (inputs, targets) in enumerate(dataset):
            outputs = model(inputs)

            error = torch.mean(torch.abs(outputs - targets) / (torch.abs(targets) + torch.finfo(targets.dtype).eps))
            error_sum += error.item()
            
            loss: torch.Tensor = criterion(outputs, targets)
            loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            now = time.time_ns()
            if now - start > 1E9 * 0.05:
                print(f'\r\033[KBatch [{i:0{int(np.ceil(np.log10(len(dataset))))}d}/{len(dataset)}], Loss: {loss.item():.8f}, Error: {error.item() * 100:.8f}%', end="")
                start += now - start
        print(f'\r\033[KEpoch [{epoch:0{int(np.ceil(np.log10(num_epochs)))}d}/{num_epochs}], Loss: {loss_sum / len(dataset):.8f}, Error: {error_sum / len(dataset) * 100:.8f}%')

model = MecanumDriveModel(4 + 3, 64, 6)
criterion = nn.MSELoss()
optimizer = optim.RAdam(model.parameters(), lr=0.00001, eps=torch.finfo(torch.float).eps, decoupled_weight_decay=True)

# loader = data.DataLoader(SyntheticDataset(20000), 1000, shuffle=False, pin_memory=True)
loader = SampleGenerator(20000, 20000)

train(model, criterion, optimizer, loader)

def print_model_weights(model: nn.Module):
    for param_tensor in model.state_dict():
        print(f"{param_tensor}: {model.state_dict()[param_tensor]}")

print_model_weights(model)

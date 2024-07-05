from math import prod
import multiprocessing.managers
import numpy as np
import scipy.integrate
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils
import torch.utils.data as data
from collections.abc import Iterator
import time
from sklearn import preprocessing
import torch.utils.data
import torch.multiprocessing as mp
import adabelief_pytorch

torch.set_num_threads(1)

# global_seed = 123456789

rng = np.random.Generator(np.random.SFC64()) # np.random.SeedSequence(global_seed)))
# torch.manual_seed(global_seed)

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

local_vel_to_wheel_vel = np.array([[ 1,  1,  1,  1],
                                   [-1,  1,  1, -1],
                                   [-a,  a, -a,  a]]) / r

wheel_vel_to_local_vel = np.array([[1, -1, -1 / a],
                                   [1,  1,  1 / a],
                                   [1,  1, -1 / a],
                                   [1, -1,  1 / a]]) * (r / 4)

control_duty_to_motor_duty = np.array([[ 1,  1,  1,  1],
                                       [-1,  1,  1, -1],
                                       [-1,  1, -1,  1]])


wheel_torque_to_local_accel = local_vel_to_wheel_vel.T / np.array([inertia, inertia, moment_of_inertia])[np.newaxis, :]

# Calculate the derivative of the state at time t.
# Given: t, x, y, theta, x', y', theta'
# Compute: x', y', theta', x'', y'', theta''
def equations_of_motion(t: float, state: np.ndarray[np.floating], control: np.ndarray[np.floating], dtype=None):
    state = state.astype(dtype)
    control = control.astype(dtype)

    x, y, theta, vx, vy, omega = state

    absolute_vel = np.array([vx, vy, omega])

    absolute_to_local = np.array([[ np.cos(-theta), np.sin(-theta), 0],
                                  [-np.sin(-theta), np.cos(-theta), 0],
                                  [              0,              0, 1]])
    
    local_vel = absolute_vel @ absolute_to_local

    wheel_vel = local_vel @ local_vel_to_wheel_vel

    motor_power = control @ control_duty_to_motor_duty
    is_same_direction = ((np.sign(motor_power * wheel_vel) + 1) / 2)
    wheel_torque = T_stall * (1 - np.abs(wheel_vel) * is_same_direction / w_free) * motor_power

    local_accel = wheel_torque @ wheel_torque_to_local_accel

    local_to_absolute = np.array([[ np.cos(theta), np.sin(theta), 0],
                                  [-np.sin(theta), np.cos(theta), 0],
                                  [             0,             0, 1]])

    absolute_accel = local_accel @ local_to_absolute

    return np.concatenate([absolute_vel, absolute_accel], dtype=dtype) # might not need dtype, but just to be safe...

def f(params):
    return scipy.integrate.solve_ivp(lambda t, y: equations_of_motion(t, y, params[1], dtype=params[2]), (0, delta_t), params[0], t_eval = np.linspace(0, delta_t, 10, dtype=params[2])).y[:, -1]

def create_data_tensors(num_samples: int, noisy = False,
                        device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    idtype = np.float64
    position = np.empty((num_samples, 3), dtype=idtype)
    position[:, 0:2] = 0
    position[:, 2] = rng.uniform(-np.pi, np.pi, num_samples).astype(idtype)

    r = rng.uniform(-1, 1, (num_samples, 3)).astype(idtype)
    wheel_velocity = r @ control_duty_to_motor_duty
    wheel_velocity = wheel_velocity / np.maximum(np.max(wheel_velocity, axis=-1), 1)[:, np.newaxis] * w_free
    velocity = wheel_velocity @ wheel_vel_to_local_vel

    state = np.concatenate([position, velocity], axis = -1, dtype=idtype)
    control = rng.uniform(-1, 1, (num_samples, 3)).astype(idtype) / 3

    predicted = np.stack(mp.Pool().map(f, zip(state, control, (idtype for _ in iter(int, 1))))) - state

    inputs = torch.from_numpy(np.concatenate([state[:, 2:], control], axis = -1, dtype=idtype)).to(**factory_kwargs)
    targets = torch.from_numpy(predicted).to(**factory_kwargs)

    return (inputs, targets)

def print_model_weights(model: nn.Module):
    for param_tensor in model.state_dict():
        print(f"{param_tensor}: {model.state_dict()[param_tensor]}")

class MecanumDriveModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MecanumDriveModel, self).__init__()

        eps = torch.finfo(dtype).eps

        self.fc1 = nn.Linear(input_dim,       hidden_dim, False, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_dim,      hidden_dim // 2, False, **factory_kwargs)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4, False, **factory_kwargs)
        self.fc4 = nn.Linear(hidden_dim // 4, output_dim, False, **factory_kwargs)

        self.bn1 = nn.BatchNorm1d(hidden_dim, eps, **factory_kwargs)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2, eps, **factory_kwargs)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4, eps, **factory_kwargs)

        self.fcx = nn.Linear(input_dim,       hidden_dim, False, **factory_kwargs)
        self.fca = nn.Linear(hidden_dim,      hidden_dim // 2, False, **factory_kwargs)
        self.fcb = nn.Linear(hidden_dim // 2, hidden_dim // 4, False, **factory_kwargs)

        self.do = nn.Dropout()
        self.relu = nn.LeakyReLU(negative_slope = 0.01)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        a = self.relu(self.bn1(self.fc1(x))) # + self.fcx(x)
        b = self.relu(self.bn2(self.fc2(a))) # + self.fca(a)
        c = self.relu(self.bn3(self.fc3(b))) # + self.fcb(b)
        d = self.fc4(c)
        return d

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, a = 0.01, nonlinearity='leaky_relu')

def train(model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler, data,
          device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}

    model.train()

    start = time.time_ns()
    for i, (inputs, targets) in enumerate(data):
        optimizer.zero_grad()

        outputs: torch.Tensor = model(inputs)
        loss: torch.Tensor = F.mse_loss(outputs, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        now = time.time_ns()
        if now - start > 1E9 * 0.1 and i < len(data) - 1:
            print(f"\r\033[KBatch [{i+1}/{len(data)}], Loss: {loss.item():.8f}, LR: {', '.join(str(i['lr']) for i in optimizer.param_groups)}", end="")
            start = now

def test(model: nn.Module, loader: data.DataLoader,
         device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for inputs, targets in loader:
            outputs: torch.Tensor = model(inputs)
            test_loss += F.mse_loss(outputs, targets, reduction='sum').item()

    test_loss /= len(loader.dataset) * targets.shape[1]
    return test_loss

def standardize(tensor: torch.Tensor):
    return (tensor - torch.mean(tensor, dim=0)) / torch.std(tensor, dim=0) # iqr

def main(device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    num_epochs = 2000
    num_test = 1000
    num_train = 20000
    batch_size = 256 # 128
    inputs, targets = create_data_tensors(num_train + num_test, True, **factory_kwargs)
    inputs = standardize(inputs)
    targets = standardize(targets)
    train_set, test_set = data.random_split(data.TensorDataset(inputs, targets), [num_train, num_test])
    train_loader = data.DataLoader(train_set, batch_size, True)
    test_loader = data.DataLoader(test_set, batch_size, False)

    model = MecanumDriveModel(4 + 3, 6, 128, **factory_kwargs); torch.compile(model, options={"triton.cudagraphs" : True})
    model.apply(init_weights)

    optimizer = adabelief_pytorch.AdaBelief(model.parameters(), lr=0.01, eps=torch.finfo(dtype).eps, weight_decay=0.001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 0.1, steps_per_epoch=len(train_loader), epochs=num_epochs)

    start = time.time_ns()
    for epoch in range(num_epochs):
        train(model, optimizer, scheduler, train_loader, **factory_kwargs)

        now = time.time_ns()
        if now - start > 1E9 * 0.1:
            loss = test(model, test_loader, **factory_kwargs)
            print(f"\r\033[KEpoch [{epoch+1}/{num_epochs}], Loss: {loss:.8f}, LR: {', '.join(str(i['lr']) for i in optimizer.param_groups)}")
            start = now

    print_model_weights(model)

if __name__=="__main__":
    main(torch.device("cpu"), torch.float32)

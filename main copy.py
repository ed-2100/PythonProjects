from math import prod
import multiprocessing.managers
import numpy as np
import scipy.integrate
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import torch.utils
import torch.utils.data as data
from collections.abc import Iterator
import time
from sklearn import preprocessing
import torch.utils.data
import torch.multiprocessing as mp
import adabelief_pytorch

torch.set_num_threads(1)

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

def generate_data(num_samples,
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
    predicted = np.stack(mp.Pool().map(f, zip(state, control, (idtype for _ in iter(int, 1)))))

    predicted -= state

    return torch.from_numpy(np.concatenate([state[:, 2:], control], axis = -1, dtype=idtype)).to(**factory_kwargs), torch.from_numpy(predicted).to(**factory_kwargs)

class SampleGenerator(Iterator):
    def __init__(self, num_samples, batch_size = 1,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SampleGenerator, self).__init__()
        self._batch = 0
        self.batch_size = batch_size
        self._num_samples = num_samples

        self._inputs, self._targets = generate_data(num_samples, **factory_kwargs)

        self._inputs_mean = torch.mean(self._inputs, dim=0)
        self._inputs_std = torch.sqrt(torch.sum(torch.square(self._inputs - self._inputs_mean), dim=0) / (self._inputs.shape[0] - 1))

        # print(torch.sort(torch.flatten(self._inputs[torch.abs(self._inputs - self._inputs_mean) > self._inputs_std]), dim=0))
        self._inputs = (self._inputs - self._inputs_mean) / self._inputs_std

        self._targets_mean = torch.mean(self._targets, dim=0)
        self._targets_std = torch.sqrt(torch.sum(torch.square(self._targets - self._targets_mean), dim=0) / (self._targets.shape[0] - 1))
        self._targets = (self._targets - self._targets_mean) / self._targets_std
        
    def __len__(self):
        return -(self._num_samples // -self.batch_size)

    def __next__(self):
        batch_index = self._batch * self.batch_size

        if batch_index >= self._num_samples:
            self._batch = 0
            raise StopIteration
        
        if self._batch == 0:
            rand = torch.randperm(self._num_samples)
            self._rand_inputs = self._inputs[rand]
            self._rand_targets = self._targets[rand]

        if self.batch_size == 1:
            temp = (self._rand_inputs[self._batch],
                    self._rand_targets[self._batch])
        elif self.batch_size >= self._num_samples:
            temp = (self._rand_inputs,
                    self._rand_targets)
        else:
            temp = (self._rand_inputs[batch_index:batch_index+self.batch_size],
                    self._rand_targets[batch_index:batch_index+self.batch_size])
        
        self._batch += 1

        return temp

class MecanumDriveModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MecanumDriveModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, **factory_kwargs)
        self.bn1 = nn.BatchNorm1d(hidden_dim, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)
        self.bn2 = nn.BatchNorm1d(hidden_dim, **factory_kwargs)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2, **factory_kwargs)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2, **factory_kwargs)
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim, **factory_kwargs)
        self.relu = nn.ReLU()
    
        # self.fc1 = nn.Linear(input_dim, hidden_dim, **factory_kwargs, bias=False)
        # self.hidden_layers = [{"bn0" : nn.BatchNorm1d(hidden_dim, **factory_kwargs),
        #                        "fc0" : nn.Linear(hidden_dim, hidden_dim, **factory_kwargs, bias=False),
        #                        "fc1" : nn.Linear(hidden_dim, hidden_dim, **factory_kwargs, bias=False)} for _ in range(num_hidden)]
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs, bias=False)
        # self.bn2 = nn.BatchNorm1d(hidden_dim, **factory_kwargs)

        # self.fc4 = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs, bias=False)
        # self.fc5 = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs, bias=False)
        # self.bn4 = nn.BatchNorm1d(hidden_dim, **factory_kwargs)
        
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)
        # self.fc6 = nn.Linear(hidden_dim, output_dim, **factory_kwargs, bias=False)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        # a = self.fc1(x)
        # for i in self.hidden_layers:
        #     a = i["fc1"](i["fc0"](functional.sigmoid(i["bn0"](a))) + a)
        # # a = functional.dropout(a)
        # a = self.fc5(self.fc4(functional.gelu(self.bn4(a))) + a)
        # # a = functional.dropout(a)
        # a = self.fc6(self.fc3(functional.sigmoid(self.bn2(a))) + a)
        return x

def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, data,
          device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}

    model.train()

    start = time.time_ns()
    for i, (inputs, targets) in enumerate(data):
        optimizer.zero_grad()

        outputs: torch.Tensor = model(inputs)
        loss: torch.Tensor = functional.mse_loss(outputs, targets)

        loss.backward()
        optimizer.step()

        now = time.time_ns()
        if now - start > 1E9 * 0.1:
            print(f"\r\033[KBatch [{i}/{len(data)}], Loss: {loss.item():.8f}", end="")
            start = now

def test(model, data,
         device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    model.eval()
    a = 0
    b = 0
    c = 0
    for inputs, targets in data:
        outputs: torch.Tensor = model(inputs)
        loss = functional.mse_loss(outputs, targets, reduction='sum')
        a += loss.item()
        b += prod(outputs.shape)
        c += functional.l1_loss(outputs, targets, reduction='sum')
    return a / b, c / b

def print_model_weights(model: nn.Module):
    for param_tensor in model.state_dict():
        print(f"{param_tensor}: {model.state_dict()[param_tensor]}")

def main(device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}

    model = MecanumDriveModel(4 + 3, 6, 128, 1, **factory_kwargs); torch.compile(model, options={"triton.cudagraphs" : True})
    optimizer = adabelief_pytorch.AdaBelief(model.parameters(), lr=0.1, eps=torch.finfo(dtype).eps, weight_decay=0)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15_000, gamma=1) # 0.99999)

    num_epochs = 20_000_000
    num_test = 1000
    num_train = 20000
    batch_size = 20000
    train_loader = SampleGenerator(num_train, batch_size, **factory_kwargs) # data.DataLoader(train_dataset, num_train, shuffle=True)
    test_loader = SampleGenerator(num_test, batch_size, **factory_kwargs) # data.DataLoader(test_dataset, num_test)

    start = time.time_ns()
    for epoch in range(1, num_epochs + 1):
        train(model, optimizer, train_loader, **factory_kwargs)
        
        scheduler.step()

        now = time.time_ns()
        if now - start > 1E9 * 0.1:
            loss, l1_loss = test(model, test_loader, **factory_kwargs)
            print(f"\r\033[KEpoch [{epoch}/{num_epochs}], Loss: {loss:.8f}, L1: {l1_loss}, LR: {', '.join(str(i['lr']) for i in optimizer.param_groups)}")
            start = now

    print_model_weights(model)

if __name__=="__main__":
    main(torch.device("cuda"), torch.float32)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchdiffeq
import time
import adabelief_pytorch

# global_seed = 123456789

# torch.manual_seed(global_seed)

delta_t = 1 / 30

T_stall = 5.4 * (1 / 100) # N * m
w_free = 1620 * (1 / 60) * 2 * torch.pi # rad / s
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

local_vel_to_wheel_vel = torch.Tensor([[1, -1, -a],
                                       [1,  1,  a],
                                       [1,  1, -a],
                                       [1, -1,  a]]) / r

wheel_vel_to_local_vel = torch.Tensor([[     1,     1,      1,     1],
                                       [    -1,     1,      1,    -1],
                                       [-1 / a, 1 / a, -1 / a, 1 / a]]) * (r / 4)

control_duty_to_motor_duty = torch.Tensor([[ 1, -1, -1],
                                           [ 1,  1,  1],
                                           [ 1,  1, -1],
                                           [ 1, -1,  1]])

wheel_torque_to_local_accel = local_vel_to_wheel_vel.T / torch.Tensor([inertia, inertia, moment_of_inertia])[:, None]

class MecanumSystem(nn.Module):
    def __init__(self, control_duty: torch.Tensor, device=None, dtype=None):
        super(MecanumSystem, self).__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.local_vel_to_wheel_vel = local_vel_to_wheel_vel.to(**self.factory_kwargs)
        self.control_duty_to_motor_duty = control_duty_to_motor_duty.to(**self.factory_kwargs)
        self.wheel_torque_to_local_accel = wheel_torque_to_local_accel.to(**self.factory_kwargs)

        self.control_duty = control_duty

    # TODO: Do less squeezing and unsqueezing.
    def forward(self, t: torch.Tensor, state: torch.Tensor):
        absolute_vel = state[..., 3:]
        theta = state[..., 2]

        c, s = torch.cos(-theta), torch.sin(-theta)
        absolute_to_local = torch.zeros(state.shape[:-1] + (3, 3), **self.factory_kwargs)
        absolute_to_local[..., 0, 0] = c
        absolute_to_local[..., 0, 1] = -s
        absolute_to_local[..., 1, 0] = s
        absolute_to_local[..., 1, 1] = c
        absolute_to_local[..., 2, 2] = 1

        local_vel = torch.matmul(absolute_to_local, absolute_vel.unsqueeze(-1)).squeeze(-1)

        wheel_vel = torch.matmul(self.local_vel_to_wheel_vel, local_vel.unsqueeze(-1)).squeeze(-1)

        motor_duty = torch.matmul(self.control_duty_to_motor_duty, self.control_duty.unsqueeze(-1)).squeeze(-1)

        is_same_direction = ((torch.sign(motor_duty * wheel_vel) + 1) / 2)

        wheel_torque = T_stall * (1 - torch.abs(wheel_vel) * is_same_direction / w_free) * motor_duty

        local_accel = torch.matmul(self.wheel_torque_to_local_accel, wheel_torque.unsqueeze(-1)).squeeze(-1)

        c, s = torch.cos(theta), torch.sin(theta)
        local_to_absolute = torch.zeros(state.shape[:-1] + (3, 3), **self.factory_kwargs)
        local_to_absolute[..., 0, 0] = c
        local_to_absolute[..., 0, 1] = -s
        local_to_absolute[..., 1, 0] = s
        local_to_absolute[..., 1, 1] = c
        local_to_absolute[..., 2, 2] = 1

        absolute_accel = torch.matmul(local_to_absolute, local_accel.unsqueeze(-1)).squeeze(-1)

        return torch.cat((absolute_vel, absolute_accel), dim=-1)

def create_data_tensors(num_samples: int,
                        device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    position = torch.zeros((num_samples, 3), **factory_kwargs)
    position[:, 2] = torch.distributions.Uniform(-torch.pi, torch.pi).sample((num_samples,)).to(position)

    local_duty = torch.distributions.Uniform(-1, 1).sample((num_samples, 3)).to(**factory_kwargs)
    wheel_duty = torch.matmul(control_duty_to_motor_duty.to(local_duty), local_duty.unsqueeze(-1)).squeeze(-1)
    highest_duty = torch.max(wheel_duty, dim=-1).values
    wheel_velocity = wheel_duty / torch.maximum(highest_duty, torch.full_like(highest_duty, 1))[:, None] * w_free
    velocity = torch.matmul(wheel_vel_to_local_vel.to(wheel_velocity), wheel_velocity.unsqueeze(-1)).squeeze(-1)

    state = torch.cat([position, velocity], dim = -1)
    control = torch.distributions.Uniform(-1, 1).sample((num_samples, 3)).to(**factory_kwargs) / 3

    model = MecanumSystem(control, **factory_kwargs)

    predicted = torchdiffeq.odeint_adjoint(model, state, torch.tensor((0, delta_t), **factory_kwargs))[-1] - state

    inputs = torch.cat([state[:, 2:], control], dim = -1)
    targets = predicted

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

        torch.nn.init.kaiming_uniform_(self.fc1.weight, a = 0.01, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight, a = 0.01, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc3.weight, a = 0.01, nonlinearity='leaky_relu')
        torch.nn.init.xavier_uniform_(self.fc4.weight)

        self.bn1 = nn.BatchNorm1d(hidden_dim, eps, **factory_kwargs)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2, eps, **factory_kwargs)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4, eps, **factory_kwargs)

        self.do = nn.Dropout()
        self.relu = nn.LeakyReLU(negative_slope = 0.01)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        a = self.relu(self.bn1(self.fc1(x)))
        b = self.relu(self.bn2(self.fc2(a)))
        c = self.relu(self.bn3(self.fc3(b)))
        d = self.fc4(c)
        return d

def train(model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler, dataset,
          device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}

    model.train()

    start = time.time_ns()
    for i, (inputs, targets) in enumerate(dataset):
        inputs, targets = inputs.to(**factory_kwargs), targets.to(**factory_kwargs)
        optimizer.zero_grad()

        outputs: torch.Tensor = model(inputs)
        loss: torch.Tensor = F.mse_loss(outputs, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        now = time.time_ns()
        if now - start > 1E9 * 0.1 and i < len(dataset) - 1:
            print(f"\r\033[KBatch [{i+1}/{len(dataset)}], Loss: {loss.item():.8f}, LR: {', '.join(str(i['lr']) for i in optimizer.param_groups)}", end="")
            start = now

def test(model: nn.Module, dataset,
         device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for inputs, targets in dataset:
            inputs, targets = inputs.to(**factory_kwargs), targets.to(**factory_kwargs)
            outputs: torch.Tensor = model(inputs)
            test_loss += F.mse_loss(outputs, targets, reduction='sum').item()

    test_loss /= dataset.dataset_len * targets.shape[-1]
    return test_loss

class BatchedInMemoryDatabase:
    def __init__(self, *tensors: torch.Tensor, batch_size: int = 1):
        self.dataset_len = tensors[0].shape[0]
        for i in tensors: assert i.shape[0] == self.dataset_len
        self.tensors = tensors
        self.batch_size = batch_size

    def __len__(self):
        return -((-self.dataset_len) // self.batch_size)

    # UNTESTED
    def __getitem__(self, idx: int):
        if idx > len(self) or idx < 0: raise IndexError
        if self.batch_size == 1: return tuple(i[idx] for i in self.tensors)
        if self.batch_size == self.dataset_len: return self.tensors
        return tuple(i[idx*self.batch_size:idx*self.batch_size+self.batch_size] for i in self.tensors)
    
    def __iter__(self):
        if self.batch_size == 1: return zip(*self.tensors)
        if self.batch_size == self.dataset_len: return iter((self.tensors,))
        randind = torch.randperm(self.dataset_len)
        randomtensors = tuple(i[randind] for i in self.tensors)
        return (tuple(i[idx*self.batch_size:idx*self.batch_size+self.batch_size] for i in randomtensors) for idx in range(len(self)))

def main(device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    num_epochs = 50000
    num_test = 1000
    num_train = 40000
    batch_size = 40000 # 128

    inputs, targets = create_data_tensors(num_train + num_test, **factory_kwargs)

    inputs_mean = torch.mean(inputs, dim=0)
    targets_mean = torch.mean(targets, dim=0)
    
    inputs_std = torch.std(inputs, dim=0)
    targets_std = torch.std(targets, dim=0)

    inputs = (inputs - inputs_mean) / inputs_std
    targets = (targets - targets_mean) / targets_std

    randind = torch.randperm(inputs.shape[0])
    randomized_inputs = inputs[randind]
    randomized_targets = targets[randind]

    train_inputs = randomized_inputs[:num_train]
    train_targets = randomized_targets[:num_train]
    test_inputs = randomized_inputs[num_train:]
    test_targets = randomized_targets[num_train:]

    # print(tuple((i.device, i.dtype) for i in (train_inputs, train_targets, test_inputs, test_targets)))

    train_inputs += torch.distributions.Normal(0, 0.1).sample(train_inputs.shape).to(train_inputs)
    train_targets += torch.distributions.Normal(0, 0.1).sample(train_targets.shape).to(train_targets)

    # train_inputs_mean = torch.mean(train_inputs, dim=0)
    # train_targets_mean = torch.mean(train_targets, dim=0)

    # train_inputs_std = torch.std(train_inputs, dim=0)
    # train_targets_std = torch.std(train_targets, dim=0)

    # train_inputs = (train_inputs - train_inputs_mean) / train_inputs_std
    # test_inputs = (test_inputs - train_inputs_mean) / train_inputs_std

    # train_targets = (train_targets - train_targets_mean) / train_targets_std
    # test_targets = (test_targets - train_targets_mean) / train_targets_std

    train_loader = BatchedInMemoryDatabase(train_inputs, train_targets, batch_size=batch_size)
    test_loader = BatchedInMemoryDatabase(test_inputs, test_targets, batch_size=batch_size)

    model = MecanumDriveModel(4 + 3, 6, 128, **factory_kwargs); torch.compile(model, options={"triton.cudagraphs" : True})
    optimizer = adabelief_pytorch.AdaBelief(model.parameters(), lr=0.01, eps=torch.finfo(dtype).eps, weight_decay=0.001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 0.3, pct_start=0.3, steps_per_epoch=len(train_loader), epochs=num_epochs)

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
    torch.set_num_threads(1)
    main(torch.device("cuda"), torch.float32)

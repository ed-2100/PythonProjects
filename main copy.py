import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchdiffeq
import time
import adabelief_pytorch

# global_seed = 123456789

# torch.manual_seed(global_seed)

import robotsystem

def create_data_tensors(num_samples: int,
                        device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    position = torch.zeros((num_samples, 3), **factory_kwargs)
    position[:, 2] = torch.distributions.Uniform(-torch.pi, torch.pi).sample((num_samples,)).to(position)

    local_duty = torch.distributions.Uniform(-1, 1).sample((num_samples, 3)).to(**factory_kwargs)
    wheel_duty = torch.matmul(robotsystem.control_duty_to_motor_duty.to(local_duty), local_duty.unsqueeze(-1)).squeeze(-1)
    highest_duty = torch.max(wheel_duty, dim=-1).values
    wheel_velocity = wheel_duty / torch.maximum(highest_duty, torch.full_like(highest_duty, 1))[:, None] * robotsystem.w_free
    velocity = torch.matmul(robotsystem.wheel_rot_to_local.to(wheel_velocity), wheel_velocity.unsqueeze(-1)).squeeze(-1)

    state = torch.cat([position, velocity], dim = -1)
    control = torch.distributions.Uniform(-1, 1).sample((num_samples, 3)).to(**factory_kwargs) / 3

    model = robotsystem.MecanumSystemModel(control, **factory_kwargs)

    predicted = torchdiffeq.odeint_adjoint(model, state, torch.tensor((0, 1 / 30), **factory_kwargs))[-1] - state

    return (torch.cat([state[:, 2:], control], dim = -1), predicted)

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

        outputs = model(inputs)
        loss = F.mse_loss(outputs, targets)

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
        if self.batch_size == self.dataset_len: return iter((self.tensors,))
        randind = torch.randperm(self.dataset_len)
        randomtensors = tuple(i[randind] for i in self.tensors)
        if self.batch_size == 1: return zip(*randomtensors)
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

    # train_inputs += torch.distributions.Normal(0, 0.1).sample(train_inputs.shape).to(train_inputs)
    # train_targets += torch.distributions.Normal(0, 0.1).sample(train_targets.shape).to(train_targets)

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
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 0.2, pct_start=0.3, steps_per_epoch=len(train_loader), epochs=num_epochs)

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

import torch
import torch.nn as nn
import torchdiffeq

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

factory_kwargs = {'device': 'cuda', 'dtype': torch.float32}

state = torch.randn(5, 6, **factory_kwargs)
control = torch.randn(5, 3, **factory_kwargs)

model = MecanumSystem(control, **factory_kwargs)

new_state = model(0, state)

print(state, control, new_state, sep="\n")
print()
print(torchdiffeq.odeint_adjoint(model, state, torch.linspace(0, delta_t, 2, **factory_kwargs)))

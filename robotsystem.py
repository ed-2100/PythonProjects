import torch

G = 13.7 # 5.2
w_free = 435 / 60 * 2 * torch.pi # 1150

r = 3 * 0.0254 # m

mass = 12 # kg
moment_of_inertia = mass * (12 * 0.0254) ** 2 / 6 # The moment of inertia of a 1 ft x 1 ft square.

inertia = torch.tensor((mass, mass, moment_of_inertia))

# ##MMMMMMMM##
# ##M      M## |
#   M      M   | l_y
#   M      M  
# ##M      M##
# ##MMMMMMMM##
#  -----
#    l_x

l_x = 0.129907
l_y = 0.095724

local_to_wheel = torch.tensor([[1, -1, -(l_x + l_y)],
                               [1,  1,  (l_x + l_y)],
                               [1,  1, -(l_x + l_y)],
                               [1, -1,  (l_x + l_y)]], dtype=torch.float64)
wheel_to_local = torch.pinverse(local_to_wheel)
local_to_wheel_rot = local_to_wheel / r
wheel_rot_to_local = wheel_to_local * r
wheel_torque_to_local_force = wheel_to_local / r
wheel_torque_to_local_accel = wheel_torque_to_local_force / inertia[:, None]

control_duty_to_motor_duty = torch.tensor([[ 1, -1, -1],
                                           [ 1,  1,  1],
                                           [ 1,  1, -1],
                                           [ 1, -1,  1]], dtype=torch.float64)
motor_duty_to_control_duty = torch.pinverse(control_duty_to_motor_duty)

def softsign(x, eps):
    return x / torch.sqrt(torch.square(x) + eps)

class MecanumSystemModel(torch.nn.Module):
    def __init__(self, device=None, dtype=None):
        super(MecanumSystemModel, self).__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.local_to_wheel_rot = local_to_wheel_rot.to(**self.factory_kwargs)                   # type: torch.Tensor
        self.control_duty_to_motor_duty = control_duty_to_motor_duty.to(**self.factory_kwargs)   # type: torch.Tensor
        self.wheel_torque_to_local_accel = wheel_torque_to_local_accel.to(**self.factory_kwargs) # type: torch.Tensor
    
    def set_control_duty(self, control_duty: torch.Tensor):
        self.control_duty = control_duty
    
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

        wheel_vel = self.local_to_wheel_rot @ absolute_to_local @ absolute_vel.unsqueeze(-1)

        motor_duty = self.control_duty_to_motor_duty @ self.control_duty.unsqueeze(-1)

        wheel_torque = (motor_duty * 0.193 - wheel_vel * G * 0.000304 - softsign(wheel_vel, 0.01) * 0.00317) * G

        local_to_absolute = torch.pinverse(absolute_to_local)

        absolute_accel = (local_to_absolute @ self.wheel_torque_to_local_accel @ wheel_torque).squeeze(-1)

        return torch.cat((absolute_vel, absolute_accel), dim=-1)

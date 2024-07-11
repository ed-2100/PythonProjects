import torch

T_stall = 5.4 * (1 / 100) # N * m
w_free = 1520 * (1 / 60) * 2 * torch.pi # rad / s
r = 4 * 0.0254 # m
mass = 6 # kg

# Percent of maximum motor torque.
Q = 0.65


# The moment of inertia of a 1 ft x 1 ft square.
moment_of_inertia = mass * (12 * 0.0254) ** 2 / 6

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
wheel_to_local = torch.linalg.pinv(local_to_wheel)

local_to_wheel_rot = local_to_wheel / r
wheel_rot_to_local = wheel_to_local * r

local_force_to_wheel_torque = local_to_wheel * r
wheel_torque_to_local_force = wheel_to_local / r

local_accel_to_wheel_torque = local_force_to_wheel_torque * inertia[None, :]
wheel_torque_to_local_accel = wheel_torque_to_local_force / inertia[:, None]

control_duty_to_motor_duty = torch.tensor([[ 1, -1, -1],
                                           [ 1,  1,  1],
                                           [ 1,  1, -1],
                                           [ 1, -1,  1]], dtype=torch.float64)
motor_duty_to_control_duty = torch.linalg.pinv(control_duty_to_motor_duty)

def softsign(x, eps):
    return x / torch.sqrt(torch.square(x) + eps)

class MecanumSystemModel(torch.nn.Module):
    def __init__(self, control_duty: torch.Tensor, device=None, dtype=None):
        super(MecanumSystemModel, self).__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.local_to_wheel_rot = local_to_wheel_rot.to(**self.factory_kwargs)
        self.wheel_to_local = wheel_to_local.to(**self.factory_kwargs)
        self.local_to_wheel = local_to_wheel.to(**self.factory_kwargs)
        self.inertia = inertia.to(**self.factory_kwargs)
        self.control_duty_to_motor_duty = control_duty_to_motor_duty.to(**self.factory_kwargs)
        self.wheel_torque_to_local_accel = wheel_torque_to_local_accel.to(**self.factory_kwargs)

        self.control_duty = control_duty
    
    def set_control_duty(self, control_duty: torch.Tensor):
        self.control_duty -= self.control_duty
        self.control_duty += control_duty

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

        epsilon = 0.001

        is_same_direction = ((torch.sgn(motor_duty * wheel_vel) + 1) / 2)

        wheel_torque = T_stall * (1 - torch.abs(wheel_vel) * is_same_direction / w_free) * motor_duty
        
        viscous_friction = 0.0003 * wheel_vel
        coulomb_friction = 0.01 * softsign(wheel_vel, epsilon)
        
        effective_wheel_torque = wheel_torque - viscous_friction - coulomb_friction

        local_to_absolute = torch.linalg.pinv(absolute_to_local)

        absolute_accel = (local_to_absolute @ self.wheel_torque_to_local_accel @ effective_wheel_torque).squeeze(-1)

        return torch.cat((absolute_vel, absolute_accel), dim=-1)

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

delta_t = 1 / 30

T_stall = 7.8 * (1 / 100) # N * m
w_free = 1150 * (1 / 60) * 2 * np.pi # rad / s
r = 4 * 0.0254 # m
mass = 6 # kg

l_x = 0.129907
l_y = 0.095724
a = l_x + l_y

# ##MMMMMMMM##
# ##M      M## |
#   M      M   + l_y
#   M      M   |
# ##M      M## |
# ##MMMMMMMM##
#  ---l_x----

Q = 0.65

inertia = mass

# The moment of inertia of a 1 ft x 1 ft square.
moment_of_inertia = mass * (12 * 0.0254) ** 2 / 6

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
def equations_of_motion(t: np.float32, state: np.ndarray, control: np.ndarray):
    x, y, theta, vx, vy, omega = state

    absolute_vel = np.array([vx, vy, omega])

    absolute_to_local = np.array([[np.cos(-theta), -np.sin(-theta),  0],
                                  [np.sin(-theta),  np.cos(-theta),  0],
                                  [             0,             0,  1]]).T
    
    local_vel = absolute_vel @ absolute_to_local

    wheel_vel = local_vel @ local_vel_to_wheel_vel

    motor_power = control @ control_duty_to_motor_duty
    is_same_direction = ((np.sign(motor_power * wheel_vel) + 1) / 2)
    wheel_torque = T_stall * (1 - np.abs(wheel_vel) * is_same_direction / w_free) * motor_power

    local_accel = wheel_torque @ wheel_torque_to_local_accel

    local_to_absolute = np.array([[np.cos(theta), -np.sin(theta),  0],
                                  [np.sin(theta),  np.cos(theta),  0],
                                  [            0,              0,  1]]).T

    absolute_accel = local_accel @ local_to_absolute

    return np.concatenate([absolute_vel, absolute_accel])

state = np.array([0, 0, 0, 0, 0, 0])

control = np.array([0.99, 0, 0.01])

# Integrate the state over the span of delta_t.

n_seconds = 20
t = np.linspace(0, n_seconds, int(n_seconds * 10 / delta_t))
sol = scipy.integrate.solve_ivp(lambda t, y: equations_of_motion(t, y, control),
                                (0, n_seconds),
                                state,
                                t_eval = t)

np.set_printoptions(formatter={'float': '{:0.2f}'.format})

fig = plt.figure()

from matplotlib.axes import Axes
ax = fig.subplots(2, 2)

state = sol.y.T

ax[0, 0].plot(state[:, 0], state[:, 1])
ax[0, 0].autoscale()
ax[0, 0].set_aspect('equal')

vec = ax[0, 0].arrow(state[0, 0], state[0, 1], state[0, 4], state[0, 5])
angle = ax[0, 0].arrow(state[0, 0] * np.cos(state[0, 2]) - state[0, 1] * np.sin(state[0, 2]),
                 state[0, 0] * np.sin(state[0, 2]) + state[0, 1] * np.cos(state[0, 2]), 0, 0)

def update(frame):
    x = state[frame * 10]
    vec.set_data(x = x[0], y = x[1], dx = x[3], dy = x[4])
    mag = np.sqrt(np.square(x[3]) + np.square(x[4]))
    angle.set_data(x = x[0], y = x[1], dx = mag * np.cos(x[2]), dy = mag * np.sin(x[2]))
    print(x)
    print(x[3:] @ local_vel_to_wheel_vel)
    return (vec, angle)

ani = animation.FuncAnimation(fig=fig, func=update, frames=state.shape[0] // 10, interval = delta_t * 1000)

ax[0, 1].plot(t, state[:, 3])
ax[1, 0].plot(t, np.linalg.norm(state[:, 3:5], axis = -1))
ax[1, 1].plot(t, state[:, 5])
ax[0, 1].autoscale()
ax[1, 0].autoscale()
ax[1, 1].autoscale()

plt.show()

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

# Calculate the derivative of the state at time t.
# Given: t, x, y, theta, x', y', theta'
# Compute: x', y', theta', x'', y'', theta''
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

state = np.array([0, 0, 0, 0, 0, 0])

control = np.array([0, 0.4, 0.6])

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
    print(xyt_to_tttt @ x[3:])
    return (vec, angle)

ani = animation.FuncAnimation(fig=fig, func=update, frames=state.shape[0] // 10, interval = delta_t * 1000)

ax[0, 1].plot(t, state[:, 3])
ax[1, 0].plot(t, np.linalg.norm(state[:, 3:5], axis = -1))
ax[1, 1].plot(t, state[:, 5])
ax[0, 1].autoscale()
ax[1, 0].autoscale()
ax[1, 1].autoscale()

plt.show()
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

delta_t = 1 / 30

T_stall = 5.4 * (1 / 100) # N * m
w_free = 1620 * (1 / 60) * 2 * np.pi # rad / s
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


# Mass of robot.
m_0 = np.float128(mass)

# Mass of wheel.
# m_1 = XXX

m_s = np.float128(m_0) # + 4 * m_1

# Moment of inertia of the entire system.
J_0 = np.float128(moment_of_inertia)
J_C = np.float128(J_0) # + 4 * (J_2 + m_1 * (p ** 2 + l ** 2))

# Radius of wheel.
R = np.float128(r)

# Distances from wheel to center.
p = np.float128(l_y)
l = np.float128(l_x)

# Magic numbers.
A = m_s * R ** 2 / 8 + J_C * R**2 / (16 * (p + l) ** 2)
B = J_C * R ** 2 / (16 * (p + l) ** 2)
C = m_s * R ** 2 / 8 - J_C * R**2 / (16 * (p + l) ** 2)

# More magic numbers.
A1 = (A * (A - C) - 2 * B ** 2) / ((A + C) * (A - 2 * B - C) * (A + 2 * B - C))
B1 = B / ((A - 2 * B - C) * (A - 2 * B - C))
C1 = (C * (A - C) + 2 * B ** 2) / ((A + C) * (A - 2 * B - C) * (A + 2 * B - C))

print(A, B, B, A1, B1, C1)
import sys
sys.exit()

# Calculate the derivative of the state at time t.
# Given: t, x, y, theta, x', y', theta'
# Compute: x', y', theta', x'', y'', theta''
def equations_of_motion(t: np.float32, state: np.ndarray, control: np.ndarray):
    x, y, theta, vx, vy, omega = state

    velocity = np.array([vx, vy, omega])

    dx_dt = velocity

    TTTT_to_axayat = xyt_to_tttt.T / np.array([mass, mass, moment_of_inertia])[..., np.newaxis]
    
    w = xyt_to_tttt @ velocity

    motor_powers = ctl_to_mot_pow @ control
    is_same_direction = ((np.sign(motor_powers * w) + 1) / 2)
    motor_torques = T_stall * (1 - np.abs(w) * is_same_direction / w_free) * motor_powers

    motor_accelerations = A1 * motor_torques + B1 * (motor_torques[..., [1, 0, 3, 2]] - motor_torques[..., [2, 3, 0, 1]]) - C1 * np.flip(motor_torques, -1)

    relative_acceleration = tttt_to_xyt @ motor_accelerations

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s,  0],
                  [s,  c,  0],
                  [0,  0,  1]])

    dv_dt = R @ relative_acceleration

    return np.concatenate([dx_dt, dv_dt])

state = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)

control = np.array([0, 0.1, 0.9], dtype=np.float64)

# Integrate the state over the span of delta_t.

n_seconds = 17
t = np.linspace(0, n_seconds, int(n_seconds * 10 / delta_t))
sol = scipy.integrate.solve_ivp(lambda t, y: equations_of_motion(t, y, control),
                                (0, n_seconds),
                                state,
                                t_eval = t)

np.set_printoptions(formatter={'float': '{:0.2f}'.format})

fig = plt.figure()
((ax1, ax2), (ax3, ax4)) = fig.subplots(2, 2)

state = sol.y.T

ax1.plot(state[:, 0], state[:, 1])
ax1.autoscale()
ax1.set_aspect('equal')

vec = ax1.arrow(state[0, 0], state[0, 1], state[0, 4], state[0, 5])
angle = ax1.arrow(state[0, 0] * np.cos(state[0, 2]) - state[0, 1] * np.sin(state[0, 2]),
                 state[0, 0] * np.sin(state[0, 2]) + state[0, 1] * np.cos(state[0, 2]), 0, 0)

def update(frame):
    x = state[frame * 10]
    vec.set_data(x = x[0], y = x[1], dx = x[3], dy = x[4])
    mag = np.sqrt(np.square(x[3]) + np.square(x[4]))
    angle.set_data(x = x[0], y = x[1], dx = mag * np.cos(x[2]), dy = mag * np.sin(x[2]))
    print(x)
    return (vec, angle)

ani = animation.FuncAnimation(fig=fig, func=update, frames=state.shape[0] // 10, interval = delta_t * 1000)

ax2.plot(t, state[:, 3])
ax3.plot(t, state[:, 4])
ax4.plot(t, state[:, 5])
ax2.autoscale()
ax3.autoscale()
ax4.autoscale()

plt.show()

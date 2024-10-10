import matplotlib.pyplot as plt
import numpy as np
import hummingbirdParam as P
from signalGenerator import SignalGenerator
from hummingbirdDynamics import HummingbirdDynamics
from hummingbirdAnimation import HummingbirdAnimation
from dataPlotter import DataPlotter
from _hummingbird_lab import ctrlEquilibrium

# Instantiate the dynamics with parameter variation (alpha=0 for no variation)
alpha = 0.0
hummingbird = HummingbirdDynamics(alpha)
reference = SignalGenerator(amplitude=0.5, frequency=0.1)

# Instantiate the simulation plots and animation
dataPlot = DataPlotter()
animation = HummingbirdAnimation()

phi_ref = SignalGenerator(amplitude=1.5, frequency=0.05)
theta_ref = SignalGenerator(amplitude=0.5, frequency=0.05)
psi_ref = SignalGenerator(amplitude=0.5, frequency=.05)

force = SignalGenerator(amplitude=0.5, frequency=.05)
torque = SignalGenerator(amplitude=0.1, frequency=0.1)

# Simulation parameters
t = P.t_start  # time starts at t_start
dt = P.Ts  # Time step

# Define input torques/forces for testing
pwm_left = 0.6  # Left motor PWM input (scaled 0-1)
pwm_right = 0.4  # Right motor PWM input (scaled 0-1)
u = np.array([[ctrlEquilibrium.pwm_left], [ctrlEquilibrium.pwm_right]])

# Run the simulation loop
t = P.t_start  # time starts at t_start
while t < P.t_end:  # Main simulation loop

    t_next_plot = t + P.t_plot
    while t < t_next_plot:  
        phi = phi_ref.sin(t)
        theta = theta_ref.sin(t)
        psi = psi_ref.sin(t)

        state = np.array([[phi], [theta], [psi], [0.0], [0.0], [0.0]])
        ref = np.array([[0], [0], [0]])
        f = (P.m1 + P.m2 + P.m3) * P.g + force.sin(t)
        tau = torque.sin(t)

        motor_thrust = P.mixing @ np.array([[f], [tau]])
        y = hummingbird.update(motor_thrust)
        t = t + P.Ts  # advance time by Ts

    # Update animation and data plotter
    animation.update(t, state)
    dataPlot.update(t, state, ref, pwm_left, pwm_right)
    

# Keeps the program from closing until the user presses a button
print('Press any key to close')
plt.waitforbuttonpress()
plt.close()
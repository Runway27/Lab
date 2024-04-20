import numpy as np
import matplotlib.pyplot as plt

Ts = 0.1
tau = 0.1
def matrix_a(tau):
    A = np.array([[1, Ts, 0, 0, 0, 0],
                  [0, 1, Ts, Ts*tau, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
    return A

# Define matrix C
C = np.array([[1, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 1, 0],
              [1, 0, 0, 0, 0, 1]])

Q = np.eye(6)
R = np.array([[2.77, 0, 0],
              [0, 4, 0],
              [0, 0, 69.44]])

def dynamics(x, tau, u):
    w = np.random.multivariate_normal(np.zeros(6), Q).reshape(-1, 1)
    B = np.array([[0], [Ts], [0], [0], [0], [0]])  # Control input matrix B
    x_next = matrix_a(tau) @ x + B * u + w
    return x_next

def output(x):
    v = np.random.multivariate_normal(np.zeros(3), R)
    y = (C @ x) + v[:, np.newaxis]
    return y

def measurement_update(x, sigma, y):
    S = C @ sigma @ C.T + R
    K = sigma @ C.T @ np.linalg.inv(S)
    innovation = y - C @ x
    x_meas = x + K @ innovation.reshape((-1, 1))
    sigma_meas = (np.eye(sigma.shape[0]) - K @ C) @ sigma
    return x_meas, sigma_meas

def time_update(x, sigma, tau, u):
    x_time = dynamics(x, tau, u)
    sigma_time = matrix_a(tau) @ sigma @ matrix_a(tau).T + Q
    return x_time, sigma_time

# Define the state and control weight matrices for DARE
Q_dare = np.eye(6)  # State weight matrix
R_dare = np.eye(1)  # Control weight matrix

# Extract the system matrices A and B
A = matrix_a(tau)
B = np.array([[0], [Ts], [0], [0], [0], [0]])

# Solve the Discrete Algebraic Riccati Equation (DARE)
P = np.eye(A.shape[0])
while True:
    P_next = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R_dare + B.T @ P @ B) @ B.T @ P @ A + Q_dare
    if np.allclose(P, P_next, rtol=1e-4):
        break
    P = P_next

# Calculate the optimal gain matrix K
K = np.linalg.inv(B.T @ P @ B + R_dare) @ (B.T @ P @ A)

class StateFeedbackController:
    def __init__(self, K):
        self.K = K

    def control_input(self, x):
        return -self.K @ x

# Initialize x and sigma
x = np.zeros((6, 1))
sigma = np.eye(6)
tau = 0.1

# Initialize the state feedback controller with the optimal K
controller = StateFeedbackController(K)

# Initialize lists to store state estimates for plotting
state_estimates = []

# Kalman Filter
for i in range(30):
    # Time update
    u = controller.control_input(x)  # Get the control input
    x, sigma = time_update(x, sigma, tau, u)  # Apply control input to time update

    # Measurement update
    y = output(x)  # Simulated measurement
    x, sigma = measurement_update(x, sigma, y)

    # Store the current state estimate for plotting
    state_estimates.append(x.flatten())

    # Print the current state estimate
    print(f"Step {i+1}: State estimate = {x.flatten()}")

# Convert the list of state estimates to a 2D array
state_estimates = np.array(state_estimates)

# Plot the state estimates over time
plt.figure(figsize=(10, 6))
for i in range(6):
    plt.plot(state_estimates[:, i], label=f"State {i+1}")

# Label the states
state_labels = ['Altitude (z)', 'Velocity (v)', 'alpha_0', 'alpha_1', 'ToF Bias (d_tof)', 'Barometer Bias (d_bar)']
plt.legend(state_labels)
plt.xlabel('Time step')
plt.ylabel('State estimate')
plt.grid(True)
plt.show()
#############################################################################
# Scenario (i): The quadcopter flies over a step, so the measurement of the ToF sensor increases suddenly
def scenario_i(x):
    x[0] += 10  # Increase altitude suddenly
    return x

# Scenario (ii): The GNSS sensor does not give any measurements for a few seconds
def scenario_ii(y):
    y[0] = None  # GNSS sensor fails to provide altitude measurement
    return y

# Scenario (iii): The battery drains gradually, so the value of α1 decreases slowly
def scenario_iii(x, i):
    i_old=0
    if i_old != i:
        x[3] -= 0.01  # Gradual decrease in α1 due to battery drain
        i_old =i
    return x

# Scenario (iv): There is a slow change in the bias of the barometer
def scenario_iv(x):
    x[5] += 0.01  # Slow increase in barometer bias
    return x
def time_update_2(x, sigma, tau):
  x_time = matrix_a(tau) @ x
  sigma_time = matrix_a(tau) @ sigma @ matrix_a(tau).T + Q
  return x_time, sigma_time
state_estimates_the_second = []
#Kalman filter without controller
for i in range(30):
    # Time update
    x, sigma = time_update_2(x, sigma, tau)  

    # Simulate scenarios
    if i == 10:  # At time step 10, simulate scenario (i)
        x = scenario_i(x)
    elif i == 15:  # At time step 15, simulate scenario (ii)
        y = scenario_ii(y)
    elif i >= 20:  # From time step 20 onwards, simulate scenarios (iii) and (iv)
        x = scenario_iii(x, i)
        x = scenario_iv(x)

    # Measurement update
    y = output(x)  # Simulated measurement
    x, sigma = measurement_update(x, sigma, y)

    # Store the current state estimate for plotting
    state_estimates_the_second.append(x.flatten())

    # Print the current state estimate
    print(f"Step {i+1}: State estimate = {x.flatten()}")

# Convert the list of state estimates to a 2D array
state_estimates_the_second = np.array(state_estimates_the_second)

# Plot the state estimates over time
plt.figure(figsize=(10, 6))
for i in range(6):
    plt.plot(state_estimates_the_second[:, i], label=f"State {i+1}")

# Label the states
state_labels = ['Altitude (z)', 'Velocity (v)', 'alpha_0', 'alpha_1', 'ToF Bias (d_tof)', 'Barometer Bias (d_bar)']
plt.legend(state_labels)
plt.xlabel('Time step')
plt.ylabel('State estimate')
plt.grid(True)
plt.show()
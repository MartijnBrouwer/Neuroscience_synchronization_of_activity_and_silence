import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)

# Parameters:
num_neurons = 100  # Number of neurons
time_steps = 2000 # Number of time steps
dt = 0.1          # Time step size

# Excitatory and inhibitory parameters:
tau_e = 5.0
tau_i = 3.0
w_ee = 16.0
w_ie = 24.0
w_ei = 10.0
w_ii = 6.0
I_e = -3.7
I_i = -6.7

c_e = 3     # Coupling strength for u
c_i = 3.5   # Coupling strength for v
noise = 0.4 # Noise strength

# Sigmoidal activation function
def F(x):
    return 1/(1 + np.exp(-x))

# Random initialization for the neurons
u = np.random.rand(num_neurons)
v = np.random.rand(num_neurons)

# Arrays in which neuronal activity is saved over time
activity_u = np.zeros((time_steps, num_neurons))
activity_v = np.zeros((time_steps, num_neurons))

# Looping over time
for t in range(time_steps):
    # Save activity at time = t in arrays:
    activity_u[t] = u
    activity_v[t] = v
    # Apply coupling and noise to neurons:
    u_next = np.zeros_like(u)
    v_next = np.zeros_like(v)
    for j in range(num_neurons):
        u_bar = (1 - c_e)*u[j] + (c_e/2)*(u[(j+1) % num_neurons] + u[(j-1) % num_neurons])
        v_bar = (1 - c_i)*v[j] + (c_i/2)*(v[(j+1) % num_neurons] + v[(j-1) % num_neurons])
        # Independent noise terms:
        eta_j = noise*np.random.randn()
        zeta_j = noise*np.random.randn()
        # Euler forward method:
        u_next[j] = u[j] + (-u[j] + F(w_ee*u_bar - w_ie*v_bar + I_e) + eta_j)*dt/tau_e
        v_next[j] = v[j] + (-v[j] + F(w_ei*u_bar - w_ii*v_bar + I_i) + zeta_j)*dt/tau_i

    # Update neurons
    u = u_next
    v = v_next

# Plotting
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(activity_u.T, cmap='hot', aspect='auto')
plt.title(f'Excitatory activity, $c_e$ = {c_e}, $c_i$ = {c_i}, noise = {noise}')
plt.clim([0,1])
plt.colorbar(label='Activity')
plt.ylabel('Neuron')
plt.xlabel('Time Step')
plt.subplot(1,2,2)
plt.imshow(activity_v.T, cmap='hot', aspect='auto')
plt.clim([0,1])
plt.colorbar(label='Activity')
plt.title(f'Inhibitory activity, $c_e$ = {c_e}, $c_i$ = {c_i}, noise = {noise}')
plt.ylabel('Neuron')
plt.xlabel('Time Step')
plt.show()

# Thresholds for active and silent states
active_threshold = 0.6
silent_threshold = 0.4

# We will consider activity_u
activity = activity_u

# Plotting of the bimodal distribution (scaled according to density)
plt.figure()
plt.hist(np.concatenate(activity), bins=50, density=True)
hist, bins = np.histogram(np.concatenate(activity), bins=50, density=True)
plt.axvline(active_threshold, color='green', linestyle='--', label='Active threshold')
plt.axvline(silent_threshold, color='red', linestyle='--', label='Silent threshold')
plt.xlabel('Activity')
plt.ylabel('Density')
plt.xlim([0,1])
plt.legend()
plt.show()
# Activity plot
plt.figure(figsize=(10, 6))
for i in range(num_neurons):
    plt.plot(activity[:, i], color='black',linewidth=0.5)
plt.axhline(active_threshold, color='green', linestyle='--', label='Active threshold')
plt.axhline(silent_threshold, color='red', linestyle='--', label='Silent threshold')
plt.title('Neural activity')
plt.xlabel('Time Step')
plt.ylabel('Activity')
plt.xlim([0,time_steps])
plt.ylim([0,1])
plt.show()

# State detection
active_regions = [[] for _ in range(num_neurons)]
silent_regions = [[] for _ in range(num_neurons)]

for i in range(num_neurons):
    state = 'neutral'
    start = 0
    for t in range(time_steps):
        if state == 'neutral':
            if activity[t, i] > active_threshold:
                state = 'active'
                start = t
            elif activity[t, i] < silent_threshold:
                state = 'silent'
                start = t
        elif state == 'active':
            if activity[t, i] < active_threshold:
                if t - start >= 10:
                    active_regions[i].append((start, t))
                state = 'neutral'
        elif state == 'silent':
            if activity[t, i] > silent_threshold:
                if t - start >= 10:
                    silent_regions[i].append((start, t))
                state = 'neutral'
    # Extend active and silent regions if the state crosses the threshold at the end
    if state == 'active':
        active_regions[i].append((start, time_steps))
    elif state == 'silent':
        silent_regions[i].append((start, time_steps))

# Extend regions for gaps shorter than the threshold
gap_threshold = 100
for i in range(num_neurons):
    for j in range(len(active_regions[i]) - 1, 0, -1):
        start_prev, end_prev = active_regions[i][j - 1]
        start_current, end_current = active_regions[i][j]
        if start_current - end_prev <= gap_threshold:
            active_regions[i][j - 1] = (start_prev, end_current)
            del active_regions[i][j]
    for j in range(len(silent_regions[i]) - 1, 0, -1):
        start_prev, end_prev = silent_regions[i][j - 1]
        start_current, end_current = silent_regions[i][j]
        if start_current - end_prev <= gap_threshold:
            silent_regions[i][j - 1] = (start_prev, end_current)
            del silent_regions[i][j]

# Plotting
plt.figure(figsize=(10, 6))
for i in range(num_neurons):
    for start, end in active_regions[i]:
        if end - start >= gap_threshold:
            plt.axvspan(start, end, color='green', alpha=0.2)
    for start, end in silent_regions[i]:
        if end - start >= gap_threshold:
            plt.axvspan(start, end, color='red', alpha=0.2)
    plt.plot(activity[:, i], color='black',linewidth=0.5)

# Plotting threshold lines
plt.axhline(y=active_threshold, color='green', linestyle='--', linewidth=1.5)
plt.axhline(y=silent_threshold, color='red', linestyle='--', linewidth=1.5)

plt.title('Neuronal activity with state regions')
plt.xlim([0, time_steps])
plt.ylim([0,1])
plt.xlabel('Time Step')
plt.ylabel('Activity')
plt.show()

# Calculate onset delay standard deviation for neighboring neurons
active_delays = []
silent_delays = []
for i in range(num_neurons):
    for j in range(len(active_regions[i])): #active regions
        start, _ = active_regions[i][j]
        if i > 0 and j < len(active_regions[i - 1]):
            prev_start, _ = active_regions[i - 1][j]
            active_delays.append(start - prev_start)
        if i < num_neurons - 1 and j < len(active_regions[i + 1]): #up to num_neurons-1 for pairs
            next_start, _ = active_regions[i + 1][j]
            active_delays.append(next_start - start)
    for j in range(len(silent_regions[i])): #silent regions
        start, _ = silent_regions[i][j]
        if i > 0 and j < len(silent_regions[i - 1]):
            prev_start, _ = silent_regions[i - 1][j]
            silent_delays.append(start - prev_start)
        if i < num_neurons - 1 and j < len(silent_regions[i + 1]):
            next_start, _ = silent_regions[i + 1][j]
            silent_delays.append(next_start - start)

active_sd = np.std(active_delays)
silent_sd = np.std(silent_delays)

print(f'Active SD for neighboring neurons: {active_sd}')
print(f'Silent SD for neighboring neurons: {silent_sd}')
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("weights_data.txt")

plt.figure(figsize=(12, 8))
for state_id in range(data.shape[1]):
    plt.plot(data[:, state_id], label=f"State {state_id}")

plt.xlabel("Episode")
plt.ylabel("Weight Value")
plt.title("Evolution of Weights in Baird's Counterexample")
plt.legend()
plt.show()

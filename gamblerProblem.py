import numpy as np
import matplotlib.pyplot as plt


def gambler_value_iteration(ph, max_capital=100, theta=1e-9):
    # Initialize value function
    V = np.zeros(max_capital + 1)
    V[max_capital] = 1
    policy = np.zeros(max_capital + 1)

    while True:
        delta = 0
        for capital in range(1, max_capital):
            old_value = V[capital]
            action_values = []
            for bet in range(1, min(capital, max_capital - capital) + 1):
                win = ph * V[capital + bet]
                loss = (1 - ph) * V[capital - bet]
                action_values.append(win + loss)
            V[capital] = max(action_values)
            delta = max(delta, abs(old_value - V[capital]))

        if delta < theta:
            break

    # Extract policy
    for capital in range(1, max_capital):
        action_values = []
        for bet in range(1, min(capital, max_capital - capital) + 1):
            win = ph * V[capital + bet]
            loss = (1 - ph) * V[capital - bet]
            action_values.append(win + loss)
        best_action = np.argmax(action_values) + 1
        policy[capital] = best_action

    return V, policy


def plot_result(ph):
    V, policy = gambler_value_iteration(ph)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(V)
    plt.title(f"Value Funciton for ph = {ph}")
    plt.xlabel("Capital")
    plt.ylabel("Value")

    plt.subplot(1, 2, 2)
    plt.plot(policy)
    plt.title(f"Policy for ph = {ph}")
    plt.xlabel("Capital")
    plt.ylabel("Bet Amount")

    plt.tight_layout()
    plt.show()


# Solve and plot for ph = 0.25, and ph = 0.55
plot_result(0.25)
plot_result(0.55)

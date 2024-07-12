#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "matplotlibcpp.h"  // C++ library for plotting

using namespace std;
namespace plt = matplotlibcpp;

// Constants for the grid world
const int GRID_SIZE = 5;
const int NUM_EPISODES = 1000;
const double GAMMA = 0.9; // Discount factor

// Function to perform TD(0) update
void TD0Update(vector<vector<double>>& values, int state, int nextState, double reward, double alpha) {
    double tdError = reward + GAMMA * values[nextState][nextState] - values[state][nextState];
    values[state][nextState] += alpha * tdError;
}

// Function to perform n-step return update
void NStepReturnUpdate(vector<vector<double>>& values, vector<int>& states, vector<double>& rewards, int currentState, int n, double alpha) {
    int T = states.size();
    int tau = currentState - n + 1;
    if (tau < 0) tau = 0;

    double G = rewards[currentState];
    for (int k = tau; k <= currentState - 1; ++k) {
        G += pow(GAMMA, currentState - k) * rewards[k];
    }
    if (currentState >= n) {
        G += pow(GAMMA, n) * values[states[tau + n]][states[tau + n]];
    }
    int state = states[tau];
    double tdError = G - values[state][currentState];
    values[state][currentState] += alpha * tdError;
}

// Function to simulate the environment and perform RL learning
void runRLSimulation(vector<double>& td0Errors, vector<double>& nStepErrors) {
    srand(time(0));

    // Initialize value function
    vector<vector<double>> values(GRID_SIZE, vector<double>(GRID_SIZE, 0.0));

    // Simulate episodes
    for (int episode = 0; episode < NUM_EPISODES; ++episode) {
        // Initialize the agent's starting position
        int currentState = rand() % GRID_SIZE;

        // Initialize storage for states and rewards
        vector<int> states;
        vector<double> rewards;

        // Simulate steps within the episode
        while (true) {
            // Perform actions based on a policy (random for simplicity)
            int action = rand() % 2; // Assume 2 possible actions for simplicity

            // Simulate transitions and rewards (simplified for illustration)
            int nextState = (action == 0) ? currentState + 1 : currentState - 1;
            if (nextState < 0) nextState = 0;
            if (nextState >= GRID_SIZE) nextState = GRID_SIZE - 1;
            double reward = (nextState == GRID_SIZE - 1) ? 1.0 : 0.0;

            // Store state and reward
            states.push_back(currentState);
            rewards.push_back(reward);

            // Update using TD(0)
            double alpha = 0.1; // Learning rate
            TD0Update(values, currentState, nextState, reward, alpha);

            // Calculate TD(0) errors for efficiency comparison
            double td0Error = abs(reward + GAMMA * values[nextState][nextState] - values[currentState][nextState]);
            td0Errors.push_back(td0Error);

            // Update using n-step return
            NStepReturnUpdate(values, states, rewards, currentState, 3, alpha);

            // Calculate n-step return errors for efficiency comparison
            double nStepError = abs(rewards[currentState] + GAMMA * values[states[currentState]][states[currentState]] - values[states[currentState]][states[currentState]]);
            nStepErrors.push_back(nStepError);

            // Move to the next state
            currentState = nextState;

            // Termination condition (end of episode)
            if (currentState == 0 || currentState == GRID_SIZE - 1)
                break;
        }
    }
}

int main() {
    vector<double> td0Errors;
    vector<double> nStepErrors;

    // Run RL simulation to collect errors
    runRLSimulation(td0Errors, nStepErrors);

    // Plot the results
    plt::figure_size(1200, 780);
    plt::plot(td0Errors, {{"label", "TD(0) Errors"}});
    plt::plot(nStepErrors, {{"label", "n-step Return Errors"}});
    plt::xlabel("Episode");
    plt::ylabel("TD Error");
    plt::title("TD(0) vs n-step Return Efficiency");
    plt::legend();
    plt::save("rl_errors_plot.png");
    plt::show();

    cout << "RL simulation completed and results plotted to rl_errors_plot.png." << endl;

    return 0;
}

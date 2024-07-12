#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

const double ALPHA = 0.1;
const double GAMMA = 0.99;
const int N = 3; // n-step TD
const int NUM_STATES = 5;
const int NUM_ACTIONS = 2;

double getReward(int state, int action) {
    if (action == 0) return 1.0;
    if (action == 1) return 0.5;
    return 0.0;
}

int getNextState(int state, int action) {
    if (action == 0) return (state + 1) % NUM_STATES;
    if (action == 1) return (state + 2) % NUM_STATES;
    return state;
}

int selectAction(int state) {
    return rand() % NUM_ACTIONS;
}

void nStepTD(std::vector<double>& V, const std::vector<int>& states, const std::vector<int>& actions, const std::vector<double>& reward) {
    int T = states.size();
    for (int t = 0; t < T - N; ++t) {
        double G = 0.0;
        for (int k = 0; k < N; ++k) {
            G += std::pow(GAMMA, k) * reward[t + k];
        }
        G += std::pow(GAMMA, N) * V[states[t + N]];
        double delta = G - V[states[t]];
        V[states[t]] += ALPHA * delta;
    }
}

int main() {
    std::srand(std::time(0));

    std::vector<int> states;
    std::vector<int> actions;
    std::vector<double> rewards;

    int currentState = 0;
    for (int t = 0; t < 10; ++t) {
        int action = selectAction(currentState);
        double reward = getReward(currentState, action);
        int nextState = getNextState(currentState, action);

        states.push_back(currentState);
        actions.push_back(action);
        rewards.push_back(reward);

        currentState = nextState;
    }

    states.push_back(currentState);

    std::vector<double> V(NUM_STATES, 0.0);
    nStepTD(V, states, actions, rewards);

    for (int i = 0; i < V.size(); ++i) {
        std::cout << "V[" << i << "] = " << V[i] << std::endl;
    }

    return 0;
}
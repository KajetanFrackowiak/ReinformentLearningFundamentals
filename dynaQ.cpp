#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <ctime>

const int NUM_STATES = 10000;
const int NUM_ACTIONS = 4;
const int BRANCHING_FACTOR = 3;
const double ALPHA = 0.1;
const double GAMMA = 0.95;
const double EPSILON = 0.1;
const int PLANNING_STEPS = 50;
const int EPISODES = 100;
const int STEPS_PER_EPISODE = 100;

// Helper function to generate random numbers
double rand_double() {
    return static_cast<double>(rand()) / RAND_MAX;
}

// Epsilon-greedy policy
int epsilon_greedy(int state, const std::vector<std::vector<double>>& Q) {
    if (rand_double() < EPSILON) {
        return rand() % NUM_ACTIONS;
    } else {
        return std::distance(Q[state].begin(), std::max_element(Q[state].begin(), Q[state].end()));
    }
}

// Step function simulating the environment
std::pair<int, double> step(int state) {
    std::vector<int> next_states(BRANCHING_FACTOR);
    for (int i = 0; i < BRANCHING_FACTOR; ++i) {
        next_states[i] = rand() % NUM_STATES;
    }
    int next_state = next_states[rand() % BRANCHING_FACTOR];
    double reward = ((rand() % 2000) - 1000) / 1000.0;  // Reward from Gaussian distribution with mean 0 and variance 1
    return {next_state, reward};
}

int main() {
    srand(time(0));

    // Initialize Q-values and model
    std::vector<std::vector<double>> Q(NUM_STATES, std::vector<double>(NUM_ACTIONS, 0.0));
    std::unordered_map<int, std::vector<std::pair<int, double>>> model;

    // Track the value of the start state under a greedy policy
    std::vector<double> values;

    for (int episode = 0; episode < EPISODES; ++episode) {
        int state = rand() % NUM_STATES;
        for (int t = 0; t < STEPS_PER_EPISODE; ++t) {
            int action = epsilon_greedy(state, Q);
            auto [next_state, reward] = step(state);
            Q[state][action] += ALPHA * (reward + GAMMA * *std::max_element(Q[next_state].begin(), Q[next_state].end()) - Q[state][action]);
            model[state * NUM_ACTIONS + action].emplace_back(next_state, reward);

            // Planning steps
            for (int p = 0; p < PLANNING_STEPS; ++p) {
                int rand_index = rand() % model.size();
                auto it = std::next(model.begin(), rand_index);
                int sim_state = it->first / NUM_ACTIONS;
                int sim_action = it->first % NUM_ACTIONS;
                auto [sim_next_state, sim_reward] = it->second[rand() % it->second.size()];
                Q[sim_state][sim_action] += ALPHA * (sim_reward + GAMMA * *std::max_element(Q[sim_next_state].begin(), Q[sim_next_state].end()) - Q[sim_state][sim_action]);
            }

            state = next_state;
        }
        values.push_back(*std::max_element(Q[0].begin(), Q[0].end()));
    }

    // Output results
    for (double value : values) {
        std::cout << value << std::endl;
    }

    return 0;
}

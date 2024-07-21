#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

const int NUM_STATES = 7;
const int NUM_ACTIONS = 2;
const double GAMMA = 0.99;
const double ALPHA = 0.01;
const double EPSILON = 0.1;

struct State {
    int id;
};

struct Action {
    int id;
};

std::mt19937 gen(0);
std::uniform_real_distribution<> dis(0, 1);

State next_state(const State& state, const Action& action) {
    State next;
    if (state.id == 6) {
        next.id = 0;
    } else {
        next.id = state.id + 1;
    }
    return next;
}

double reward(const State& state, const Action& action) {
    if (state.id == 6 && action.id == 0) {
        return 1.0;
    }
    return 0.0;
}

class QLearning {
   public:
    QLearning() {
        Q = std::vector<std::vector<double>>(
            NUM_STATES, std::vector<double>(NUM_ACTIONS, 0.1));
        weights = std::vector<double>(NUM_STATES, 0.1);
    }

    void update(const State& state, const Action& action, State& next_state,
                double reward) {
        double q_current = Q[state.id][action.id];
        double q_next =
            *std::max_element(Q[next_state.id].begin(), Q[next_state.id].end());
        double target = reward + GAMMA * q_next;

        // std::cout << "State: " << state.id << ", Action: " << action.id
        //           << ", Q_current: " << q_current << ", Target: " << target
        //           << ", Diff: " << target - q_current << std::endl;

        Q[state.id][action.id] += ALPHA * (target - q_current);

        weights[state.id] = Q[state.id][action.id];
    }

    Action select_action(const State& state) {
        if (dis(gen) < EPSILON) {
            return {std::uniform_int_distribution<>(0, NUM_ACTIONS - 1)(gen)};
        } else {
            return {static_cast<int>(std::distance(
                Q[state.id].begin(),
                std::max_element(Q[state.id].begin(), Q[state.id].end())))};
        }
    }

    void run(int num_episodes) {
        std::ofstream outfile("weights_data.txt");

        for (int episode = 0; episode < num_episodes; ++episode) {
            State state = {0};
            for (int step = 0; step < 100; ++step) {
                Action action = select_action(state);
                State next = next_state(state, action);
                double r = reward(state, action);

                update(state, action, next, r);

                state = next;
            }

            for (double w : weights) {
                outfile << w << " ";
            }
            outfile << std::endl;
        }

        outfile.close();
    }

    void print_weights() {
        for (double w : weights) {
            std::cout << w << " ";
        }
        std::cout << std::endl;
    }

   private:
    std::vector<std::vector<double>> Q;
    std::vector<double> weights;
};

int main() {
    QLearning qlearning;
    qlearning.run(1000);
    qlearning.print_weights();

    return 0;
}
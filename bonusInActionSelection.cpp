#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

const int GRID_SIZE = 5;
const int NUM_EPISODES = 500;
const double ALPHA = 0.1;
const double GAMMA = 0.9;
const double K = 0.01; // Exploration bonus

struct State {
    int x, y;
    bool operator==(const State& other) const { return x == other.x && y == other.y; }
};

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

int random_action() { return dis(gen) < 0.5 ? 0 : 1; }

std::pair<State, double> step(const State& state, int action) {
    State next_state = state;
    double reward = 0.0;

    if (action == 0 && state.x < GRID_SIZE - 1) next_state.x++;
    if (action == 1 && state.y < GRID_SIZE - 1) next_state.y++;

    if (next_state.x == GRID_SIZE - 1 && next_state.y == GRID_SIZE - 1) reward = 1.0;

    return std::make_pair(next_state, reward);
}

template<typename T>
int argmax(const std::vector<T>& vec) {
    return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

void update_Q(std::vector<std::vector<std::vector<double>>>& Q, const State& s, int a, double reward, const State& next_s, int next_a) {
    double td_target = reward + GAMMA * Q[next_s.x][next_s.y][next_a];
    double td_error = td_target - Q[s.x][s.y][a];
    Q[s.x][s.y][a] += ALPHA * td_error;
}

void dyna_q_plus(std::vector<std::vector<std::vector<double>>>& Q, std::vector<std::vector<int>>& N) {
    State start = {0, 0};
    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        State state = start;
        while (!(state.x == GRID_SIZE - 1 && state.y == GRID_SIZE - 1)) {
            int a1 = argmax(Q[state.x][state.y]);
            State next_state;
            double reward;
            std::tie(next_state, reward) = step(state, a1);

            N[state.x][state.y]++;

            int a2 = argmax(Q[next_state.x][next_state.y]);
            update_Q(Q, state, a1, reward, next_state, a2);

            state = next_state;
        }
    }
}

void experiment() {
    std::vector<std::vector<std::vector<double>>> Q(GRID_SIZE, std::vector<std::vector<double>>(GRID_SIZE, std::vector<double>(2, 0.0)));
    std::vector<std::vector<int>> N(GRID_SIZE, std::vector<int>(GRID_SIZE, 0));

    dyna_q_plus(Q, N);

    std::vector<double> values;
    for (int i = 0; i < GRID_SIZE; ++i) {
        for (int j = 0; j < GRID_SIZE; ++j) {
            values.push_back(*std::max_element(Q[i][j].begin(), Q[i][j].end()));
        }
    }
    plt::plot(values);
    plt::title("Q-values after Dyna-Q+ with exploration bonus");
    plt::show();
}

int main() {
    experiment();
    return 0;
}

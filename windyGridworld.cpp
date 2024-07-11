#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <tuple>

const int WIDTH = 10;
const int HEIGHT = 7;
const int START_X = 0;
const int START_Y = 3;
const int GOAL_X = 7;
const int GOAL_Y = 3;

const std::vector<int> WIND = {0, 0, 0, 1, 1, 1, 2, 2, 1, 0};

// Actions: {N, S, E, W, NE, NW, SE, SW, Stay}
const std::vector<std::pair<int, int>> ACTIONS = {
    {-1, 0}, {1, 0}, {0, 1}, {0, -1}, 
    {-1, 1}, {-1, -1}, {1, 1}, {1, -1}, 
    {0, 0}
};

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);
std::uniform_int_distribution<> wind_distrib(0, 2);

std::tuple<int, int> step(int x, int y, int action) {
    int new_x = x + ACTIONS[action].first;
    int new_y = y + ACTIONS[action].second;
    
    new_y = std::max(0, std::min(HEIGHT - 1, new_y));
    new_x = std::max(0, std::min(WIDTH - 1, new_x));
    
    int wind_effect = WIND[new_x];
    int stochastic_effect = wind_distrib(gen) - 1; // -1, 0, or 1
    new_y = std::max(0, std::min(HEIGHT - 1, new_y + wind_effect + stochastic_effect));
    
    return {new_x, new_y};
}

void q_learning() {
    const double alpha = 0.5;
    const double gamma = 0.9;
    const double epsilon = 0.1;
    const int episodes = 500;

    std::vector<std::vector<std::vector<double>>> Q(HEIGHT, std::vector<std::vector<double>>(WIDTH, std::vector<double>(ACTIONS.size(), 0.0)));

    for (int ep = 0; ep < episodes; ++ep) {
        int x = START_X;
        int y = START_Y;

        while (x != GOAL_X || y != GOAL_Y) {
            int action;
            if (dis(gen) < epsilon) {
                action = rand() % ACTIONS.size();
            } else {
                action = std::distance(Q[y][x].begin(), std::max_element(Q[y][x].begin(), Q[y][x].end()));
            }

            auto [new_x, new_y] = step(x, y, action);
            double reward = (new_x == GOAL_X && new_y == GOAL_Y) ? 0 : -1;

            int best_next_action = std::distance(Q[new_y][new_x].begin(), std::max_element(Q[new_y][new_x].begin(), Q[new_y][new_x].end()));
            double td_target = reward + gamma * Q[new_y][new_x][best_next_action];
            double td_error = td_target - Q[y][x][action];

            Q[y][x][action] += alpha * td_error;

            x = new_x;
            y = new_y;
        }
    }

    // Output policy
    std::vector<std::vector<char>> policy(HEIGHT, std::vector<char>(WIDTH, ' '));
    std::vector<std::string> action_chars = {"N", "S", "E", "W", "NE", "NW", "SE", "SW", " "};
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            if (x == GOAL_X && y == GOAL_Y) {
                policy[y][x] = 'G';
            } else {
                int best_action = std::distance(Q[y][x].begin(), std::max_element(Q[y][x].begin(), Q[y][x].end()));
                policy[y][x] = action_chars[best_action][0];
            }
        }
    }

    for (const auto& row : policy) {
        for (char cell : row) {
            std::cout << cell << ' ';
        }
        std::cout << std::endl;
    }
}

int main() {
    q_learning();
    return 0;
}

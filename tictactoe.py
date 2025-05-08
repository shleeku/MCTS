import torch
import torch.nn as nn
import numpy as np
import sys
import os


class TicTacToeEnv():
    def __init__(self,):
        self.state = [0]*9
        self.done = False

    def reset(self):
        self.state = [0]*9
        self.done = False
        return self.state

    def check_winner(self, player: int):
        # check rows
        for i in range(3):
            if self.state[i*3] == player and self.state[i*3+1] == player and self.state[i*3+2] == player:
                return True
        # check columns
        for i in range(3):
            if self.state[i] == player and self.state[i+3] == player and self.state[i+6] == player:
                return True
        # check diagonals
        if (self.state[0] == player and self.state[4] == player and self.state[8] == player) or \
           (self.state[2] == player and self.state[4] == player and self.state[6] == player):
            return True
        return False

    def step(self, action: int, player: int):
        if self.state[action] == 0:
            self.state[action] = player
        else:
            raise ValueError("Invalid action")
        if self.check_winner(1):
            reward = 1
            self.done = True
        elif self.check_winner(-1):
            reward = -1
            self.done = True
        elif 0 not in self.state:
            reward = 0
            self.done = True
        else:
            reward = 0
        return self.state, reward, self.done

    def render(self):
        print("Current state:")
        for i in range(3):
            row = self.state[i*3:i*3+3]
            print(" | ".join(str(x) if x != 0 else " " for x in row))
            if i < 2:
                print("-" * 9)
        print()
    def get_valid_actions(self):
        return [i for i in range(9) if self.state[i] == 0]
    # def get_state(self):
    #     return self.state
    # def get_done(self):
    #     return self.done
    # def get_reward(self):
    #     if self.check_winner(1):
    #         return 1
    #     elif self.check_winner(-1):
    #         return -1
    #     else:
    #         return 0
    # def get_current_player(self):
    #     return 1 if self.state.count(1) <= self.state.count(-1) else -1
    # def get_opponent(self):
    #     return -1 if self.get_current_player() == 1 else 1

class TicTacToeAgent():
    def __init__(self, env: TicTacToeEnv):
        self.env = env
        # self.state = env.reset()
        # self.done = False
        # self.player = 1
        # self.opponent = -1
        self.action_space = [i for i in range(9)]

        in_dim = 9
        out_dim = 9
        self.device = "cuda"

        self.mlp = nn.Sequential(nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim)).to(self.device)

    def act(self, state):
        valid_actions = self.env.get_valid_actions()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_probs = self.mlp(state_tensor)
        action_probs = torch.softmax(action_probs, dim=1)
        action_probs = action_probs.squeeze(0).cpu().detach().numpy()
        action_probs = [action_probs[i] if i in valid_actions else 0 for i in range(9)]
        action_probs = action_probs / sum(action_probs)
        action = np.random.choice(self.action_space, p=action_probs)
        return action







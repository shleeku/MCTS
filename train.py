import torch
import torch.nn as nn
import numpy as np
import sys
import os
import random
import time
import wandb
from tqdm import tqdm
from tictactoe import TicTacToeEnv, TicTacToeAgent

def play_loop(env, agent, opponent, num_episodes=10, max_steps=50):
    agent_wins = 0
    opponent_wins = 0
    draws = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        for step in range(max_steps):

            action = agent.act(state)
            next_state, reward, done = env.step(action, 1)
            env.render()
            state = next_state
            if done:
                if reward == 1:
                    agent_wins += 1
                elif reward == -1:
                    opponent_wins += 1
                else:
                    draws += 1
                print("Game Over! Winner: ", "Agent" if reward == 1 else "Opponent" if reward == -1 else "Draw")
                break
            action = opponent.act(state)
            next_state, reward, done = env.step(action, -1)
            env.render()
            state = next_state
            if done:
                if reward == 1:
                    agent_wins += 1
                elif reward == -1:
                    opponent_wins += 1
                else:
                    draws += 1
                print("Game Over! Winner: ", "Agent" if reward == 1 else "Opponent" if reward == -1 else "Draw")
                break
        print("agent wins: ", agent_wins, "opponent wins: ", opponent_wins, "draws: ", draws)
        print(f"Episode {episode+1}/{num_episodes} finished after {step+1} steps.")

class TreeNode:
    def __init__(self, id):
        self.id = id
        self.total = 0
        self.visits = 0
        self.prob = 0
        self.children = []
        self.parent = None

    def add_child(self, node):
        self.children.append(node)

    def __repr__(self, level=0):
        ret = "  " * level + repr(self.value) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

class MCTS():
    def __init__(self, env, agent, opponent, num_simulations=10):
        self.env = env
        self.agent = agent
        self.opponent = opponent
        self.tree = TreeNode("root")
        self.num_simulations = num_simulations

    def UCB1(self, node):
        if node.visits == 0:
            return float('inf')
        else:
            return node.total / node.visits + np.sqrt(2 * np.log(node.parent.visits) / node.visits)
    def select(self, current_node):
        if len(current_node.children) == 0:
            self.expand(current_node)
        else:
            best_child = max(current_node.children, key=self.UCB1)
            if best_child.visits == 0:
                return best_child
            else:
                return self.select(best_child)
    def expand(self, current_node):
        possible_actions = self.env.get_valid_actions()
        if len(possible_actions) == 0:
            return current_node
        else:
            for action in possible_actions:
                child_node = TreeNode(action)
                child_node.parent = current_node
                current_node.add_child(child_node)
            return current_node.children[0]
    def search(self, state):
        for _ in range(self.num_simulations):
            self.simulate(state)

    def simulate(self, current_node):
        # Simulate a random game from the given state
        while not self.env.done:
            valid_actions = self.env.get_valid_actions()
            action = random.choice(valid_actions)
            state, reward, done = self.env.step(action, 1)
            if done:
                if reward == 1:
                    current_node.total += 1

                elif reward == -1:
                    current_node.total -= 1
                break

if __name__ == "__main__":
    game = TicTacToeEnv()
    agent = TicTacToeAgent(game)
    opponent = TicTacToeAgent(game)
    play_loop(game, agent, opponent)

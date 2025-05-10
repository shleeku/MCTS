import torch
import torch.nn as nn
import numpy as np
import sys
import os
import copy
import random
import time
import wandb
from tqdm import tqdm
from tictactoe import TicTacToeEnv, TicTacToeAgent


class TreeNode:
    def __init__(self, id):
        self.id = id
        self.total = 0
        self.visits = 0
        self.prob = 0
        self.children = []
        self.parent = None
        self.depth = 0

    def add_child(self, node):
        self.children.append(node)

    def __repr__(self, level=0):
        ret = "  " * level + f"ID: {self.id}, Total: {self.total}, Visits: {self.visits}, Prob: {self.prob}, Depth: {self.depth}\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

class MCTS():
    def __init__(self, env, num_simulations=100):
        self.original_state = copy.deepcopy(env.state)
        self.env = TicTacToeEnv()
        self.env.state = copy.deepcopy(env.state)
        self.env.done = False
        self.root = TreeNode("root")
        num_moves = len([i for i in self.env.state if i != 0])
        self.root.depth = num_moves
        self.num_simulations = num_simulations
        self.current_root = self.root

    def reset(self):
        # print("Original state: ", self.original_state)
        self.env.state = copy.deepcopy(self.original_state)
        self.env.done = False
    def UCB1(self, node):
        if node.visits == 0:
            return float('inf')
        else:
            return node.total / node.visits + np.sqrt(2 * np.log(node.parent.visits) / node.visits)
    def take_action(self, current_node):
        num_moves = len([i for i in self.env.state if i != 0])
        next_player = (num_moves + 1) % 2
        action = current_node.id
        print("Action taken: ", action)
        state, reward, done = self.env.step(action, (-1) ** (next_player + 1))
        self.env.render()
        if done:
            self.backpropagate(current_node, reward)
    def select(self, current_node):
        if len(current_node.children) == 0:
            if current_node.visits == 0:
                return current_node
            else:
                return self.expand(current_node)
        else:
            best_child = max(current_node.children, key=self.UCB1)
            for child in current_node.children:
                print("Child ID: ", child.id, "Total: ", child.total, "Visits: ", child.visits, "Depth: ", child.depth, "UCB1: ", self.UCB1(child))
            print("taking action best child")
            self.take_action(best_child)
            return self.select(best_child)
    def expand(self, current_node):
        possible_actions = self.env.get_valid_actions()
        if len(possible_actions) == 0:
            return current_node
        else:
            for action in possible_actions:
                child_node = TreeNode(action)
                child_node.parent = current_node
                child_node.depth = current_node.depth + 1
                current_node.add_child(child_node)
            print("taking action expand")
            self.take_action(current_node.children[0])
            return current_node.children[0]
    def search(self):
        for _ in range(self.num_simulations):
            print("Simulation number: ", _)
            self.reset()
            # self.env.render()
            current_node = self.select(self.current_root)
            self.simulate(current_node)

    def simulate(self, current_node):
        # current_player = current_node.depth % 2
        # print("original current node depth: ", current_node.depth)
        # print("current player: ", current_player)
        num_moves = len([i for i in self.env.state if i != 0])
        next_player = num_moves % 2
        # print("START SIMULATION")
        while not self.env.done:
            next_player += 1
            next_player = next_player % 2
            # print("next player: ", next_player, ":", (-1)**(next_player+1))
            valid_actions = self.env.get_valid_actions()
            action = random.choice(valid_actions)
            # print("Action chosen: ", action)
            state, reward, done = self.env.step(action, (-1)**(next_player+1))
            print("Simulation action: ", action)
            self.env.render()
            # print("State: ", self.env.state)
            # print("Done: ", done)
            if done:
                self.backpropagate(current_node, reward)
                break

    def backpropagate(self, current_node, reward):
        print("reward: ", reward)
        if reward == 1:
            if current_node.depth % 2 == 1:
                current_node.total += 1
            else:
                current_node.total -= 1
            current_node.visits += 1
            print("Current node: ", current_node)
            while current_node.parent is not None:
                current_node = current_node.parent
                if current_node.depth % 2 == 1:
                    current_node.total += 1
                else:
                    current_node.total -= 1
                current_node.visits += 1
                print("Current node: ", current_node)
        elif reward == -1:
            if current_node.depth % 2 == 1:
                current_node.total -= 1
            else:
                current_node.total += 1
            current_node.visits += 1
            print("Current node: ", current_node)
            while current_node.parent is not None:
                current_node = current_node.parent
                if current_node.depth % 2 == 1:
                    current_node.total -= 1
                else:
                    current_node.total += 1
                current_node.visits += 1
                print("Current node: ", current_node)
        else:
            current_node.total += 0
            current_node.visits += 1
            while current_node.parent is not None:
                current_node = current_node.parent
                current_node.total += 0
                current_node.visits += 1

    def best_action(self):
        print("Current root: ", self.current_root)
        best_child = max(self.current_root.children, key=lambda x: x.visits)
        print("Best child ID: ", best_child.id)
        return best_child.id

def agent_vs_opponent(env, agent, opponent, num_episodes=20, max_steps=50):
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

def agent_vs_singleuseMCTS(env, agent, num_episodes=20, max_steps=50):
    agent_wins = 0
    opponent_wins = 0
    draws = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        for step in range(max_steps):
            action = agent.act(state)
            print("Agent action: ", action)
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
            singleuseMCTS = MCTS(env)
            singleuseMCTS.search()
            action = singleuseMCTS.best_action()
            print("Action chosen by MCTS: ", action)
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

if __name__ == "__main__":
    game = TicTacToeEnv()
    agent = TicTacToeAgent(game)
    opponent = TicTacToeAgent(game)
    agent_vs_singleuseMCTS(game, agent)

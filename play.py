import torch
from tictactoe import TicTacToeEnv, TicTacToeAgent
from selfplay import MCTS


game = TicTacToeEnv()
agent = TicTacToeAgent(game)
state = game.reset()
done = False

def MCTSmove(env):
    singleuseMCTS = MCTS(env)
    singleuseMCTS.search()
    action = singleuseMCTS.best_action()
    print("Action chosen by MCTS: ", action)
    return action

while not done:
    # action = agent.act(state)
    action = MCTSmove(game)
    state, reward, done = game.step(action, 1)
    game.render()
    if done:
        print("Game Over! Winner: ", "AI" if reward == 1 else "User" if reward == -1 else "Draw")
        break
    user_action = input("Enter your action (0-8): ")
    state, reward, done = game.step(int(user_action), -1)
    game.render()
    if done:
        print("Game Over! Winner: ", "AI" if reward == 1 else "User" if reward == -1 else "Draw")
        break




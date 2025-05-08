import torch
from tictactoe import TicTacToeEnv, TicTacToeAgent

game = TicTacToeEnv()
agent = TicTacToeAgent(game)
state = game.reset()
done = False
while not done:
    action = agent.act(state)
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




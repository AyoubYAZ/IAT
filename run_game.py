import os, glob
from time import sleep, time
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.qagent import QAgent

from epsilon_profile import EpsilonProfile
import logging


def main():
    game = SpaceInvaders(display=True)



    gamma = 0.9
    alpha = 1
    eps_profile = EpsilonProfile(1, 0.001)
    max_steps = 500
    n_episodes = 10000

    def Train():
        fileName = f"""Q_E{n_episodes}_S{max_steps}_G{gamma}_I{eps_profile.initial}_F{eps_profile.final}"""
        controller = QAgent(game, eps_profile, gamma, alpha, fileName)
        startTime = time()
        print("START Training")
        controller.learn(game, n_episodes, max_steps)
        endTime = time()
        controller.saveQToFile(os.path.join("Training", fileName))
        print("\n############################################################################")
        print("FINISHED Training")
        print(f"\ttime learning: , {endTime - startTime}")
        print("\n############################################################################")

    controller = QAgent(game, eps_profile, gamma, alpha)

    try:
        print("DIIIRECTLY")
        controller.loadQFromFile(os.path.join(os.path.abspath("Training"),f"""Q_E{n_episodes}_S{max_steps}_G{gamma}_I{eps_profile.initial}_F{eps_profile.final}.npy"""))
    except Exception:
        Train()
        controller.loadQFromFile(
        os.path.join(os.path.abspath("Training"),f"""Q_E{n_episodes}_S{max_steps}_G{gamma}_I{eps_profile.initial}_F{eps_profile.final}.npy"""))






    state = game.reset()
    while True:
        xsc = None
        action = controller.select_greedy_action(state)
        state, reward, is_done = game.step(action)
        print(f"\r#> Score: {game.score_val}  ", end=" ")
        sleep(0.0001)





if __name__ == '__main__':
    main()

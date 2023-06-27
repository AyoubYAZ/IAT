import os
import numpy as np
from epsilon_profile import EpsilonProfile
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

from game import SpaceInvaders

X_MIN = 0
X_MAX = 770 
Y_MIN = 0
Y_MAX = 500



class QAgent():
    def __init__(self,
                 spaceInvaders: SpaceInvaders,
                 eps_profile: EpsilonProfile,
                 gamma: float,
                 alpha: float,
                 fileLog="logQ"):
        
        self.spaceInvaders = spaceInvaders
        self.na=spaceInvaders.na 
        self.Q = np.zeros([X_MAX + 1, Y_MAX + 1, 1 + 1, self.na])
        self.gamma = gamma
        self.alpha = alpha
        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial
        self.qvalues = pd.DataFrame(data={'episode': [], 'score': [], 'Q_sum': []})
        self.fileLog = fileLog

    def getQ(self, state, action):
        return self.Q[state[0]][state[1]][state[2]][action]

    def setQ(self, state, action, value):
        self.Q[state[0]][state[1]][state[2]][action] = value

    def saveQToFile(self,
                    file=os.path.join(os.path.dirname(__file__),
                                      '../Training/LearnedQ.npy')):
        np.save(file, self.Q)

    def loadQFromFile(self,
                      file=os.path.join(os.path.dirname(__file__),
                                        '../Training/LearnedQ.npy')):
        self.Q = np.load(file)

    def learn(self, env: SpaceInvaders, n_episodes, max_steps):
        n_steps = np.zeros(n_episodes) + max_steps
        sum_rewards = np.zeros(n_episodes)
        for episode in range(n_episodes):
            state = env.reset()
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, terminal = env.step(action)
                self.updateQ(state, action, reward, next_state)
                sum_rewards[episode] += reward

                if terminal:
                    n_steps[episode] = step + 1
                    break
                state = next_state
            epsilon = max(
                self.epsilon - self.eps_profile.dec_episode /
                (n_episodes - 1.), self.eps_profile.final)
            if n_episodes >= 0:
                self.epsilon = max(epsilon - self.eps_profile.dec_episode / (n_episodes - 1.), self.eps_profile.final)
                print(
                    "\r#> Ep.: {}/{}    Sum(Q): {}    Curr. Score: {}  ".
                    format(episode, n_episodes - 1,  sum_rewards,
                           self.spaceInvaders.score_val),
                    end=" ")
                self.save_log(env, episode)
                state = env.reset()

        self.qvalues.to_csv(
            os.path.join(os.path.dirname(__file__), '../Remarques',
                         self.fileLog + '.csv'))

    def updateQ(self, state, action, reward, next_state):
        """À COMPLÉTER!
        Cette méthode utilise une transition pour mettre à jour la fonction de valeur Q de l'agent. 
        Une transition est définie comme un tuple (état, action récompense, état suivant).
        :param state: L'état origine
        :param action: L'action
        :param reward: La récompense perçue
        :param next_state: L'état suivant
        """

        # If invader reached boarder its y-position is set to a too small value. This one has to be
        # increased to make it at least Y_MAX
        if next_state[1] < Y_MIN:
            next_state[1] = Y_MIN

        # print(self.Q[next_state],next_state)
        val = (1. - self.alpha) * self.getQ(state, action) + self.alpha * (
            reward + self.gamma * np.max(self.Q[next_state]))

        # self.Q[state][action] = val
        # self.Q[state[0]][state[1]][state[2]][state[3]][action] = val
        self.setQ(state, action, val)

    def select_action(self, state: int):
        """À COMPLÉTER!
        Cette méthode retourne une action échantilloner selon le processus d'exploration (ici epsilon-greedy).
        :param state: L'état courant
        :return: L'action 
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.na)  # random action
        else:
            return self.select_greedy_action(state)

        

    def select_greedy_action(self, state: 'Tuple[int, int]'):
        """
        Cette méthode retourne l'action gourmande.

        :param state: L'état courant
        :return: L'action gourmande
        """

        # If invader reached boarder its y-position is set to a too small value. This one has to be
        # increased to make it at least Y_MAX
        if state[1] < Y_MIN:
            state[1] = Y_MIN

        mx = np.max(self.Q[state])
        # greedy action with random tie break
        return np.random.choice(np.where(self.Q[state] == mx)[0])

    def save_log(self, env, episode):
        """Sauvegarde les données d'apprentissage.
        :warning: Vous n'avez pas besoin de comprendre cette méthode
        """

        self.qvalues = self.qvalues.append(
            {
                'episode': episode,
                'score': self.spaceInvaders.score_val,
                'Q_sum': np.sum(self.Q)
            },
            ignore_index=True)

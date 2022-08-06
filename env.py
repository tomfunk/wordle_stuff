import gym
from gym import spaces
import numpy as np

class WordleEnv1(gym.Env):

    def __init__(self, valid_words, word_bank, max_attempts=6):
        self.valid_words = valid_words
        self.word_bank = word_bank
        self.max_attempts = max_attempts
        self.valid_words_matrix = np.array([list(word) for word in valid_words])
        self.attempt = 0
        self.prior_actions = []
        self.guesses = [['', '', '', '', '']] * self.max_attempts
        self.known_letters = np.array(['', '', '', '', ''])
        self.present_letters = []
        self.absent_letters = []
        self.observation_space = spaces.Box(low=-2, high=2, shape=[len(valid_words), 5], dtype=int)

        # each possible guess is an action
        self.action_space = spaces.Discrete(len(valid_words))
    

    def _get_reward(self, action):
        guess_word = self.valid_words[action]
        if guess_word == self.word:
            return 10
        elif action in self.prior_actions:
            return -1
        return 0

    def _check_done(self, action):
        guess_word = self.valid_words[action]
        if guess_word == self.word or self.attempt == self.max_attempts:
            return True
        return False

    def _update_state(self, action):
        if action not in self.prior_actions:
            self.prior_actions.append(action)
            guess_word = self.valid_words[action]
            guess_word_list = list(guess_word)
            self.guesses[self.attempt] = guess_word_list
            self.attempt += 1
            # update known letters
            g_corr_letter_msk = np.array(list(self.word)) == np.array(guess_word_list)
            self.known_letters[g_corr_letter_msk] = np.array(guess_word_list)[g_corr_letter_msk]
            # update present letters
            self.present_letters = list(set(self.present_letters).union(set([l for l in guess_word_list if l in self.word])))
            # update absent letters
            self.absent_letters = list(set(self.absent_letters).union(set([l for l in guess_word_list if l not in self.word])))

    def _get_obs(self, action=None):
        if action is not None:
            self._update_state(action)
   
        present_letter_msk = np.isin(self.valid_words_matrix, self.present_letters)
        absent_letter_msk = np.isin(self.valid_words_matrix, self.absent_letters)
        known_letter_msk = self.valid_words_matrix[:, :] == self.known_letters
        prior_guess_msk = self.valid_words_matrix[:, :] == self.known_letters

        obs = present_letter_msk * 1
        obs[known_letter_msk] = 2
        obs[absent_letter_msk] = -1
        obs[self.prior_actions] = -2

        return obs

    
    def _get_info(self):
        return {
            'attempt': self.attempt,
            'guesses': self.guesses,
            'known_letters': self.known_letters.tolist(),
            'present_letters': self.present_letters,
            'absent_letters': self.absent_letters,
        }
    
    def reset(self, return_info=False):
        self.word = self.word_bank[np.random.randint(len(self.word_bank))]
        self.attempt = 0
        self.prior_actions = []
        self.guesses = [['', '', '', '', '']] * self.max_attempts
        self.known_letters = np.array(['', '', '', '', ''])
        self.present_letters = []
        self.absent_letters = []
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation
    
    def render(self):
        print(self._get_info())
    
    def step(self, action):
        observation = self._get_obs(action)
        reward = self._get_reward(action)
        done = self._check_done(action)
        info = self._get_info()

        return observation, reward, done, info
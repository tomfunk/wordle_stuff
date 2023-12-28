import gymnasium as gym
from gymnasium import spaces
import numpy as np
import string
from termcolor import cprint


class WordleEnv2(gym.Env):
    def __init__(
        self,
        valid_words,
        word_bank,
        max_attempts=6,
        max_sub_attempts=100,
        render=False,
        flatten=True,
    ):
        self.valid_words = set(valid_words)
        self.word_bank = word_bank
        self.max_attempts = max_attempts
        self.max_sub_attempts = max_sub_attempts  # sub_attempts include attempts that aren't valid words or are duplicates
        self.letter_map = {i: l for i, l in enumerate(string.ascii_lowercase)}
        self.index_map = {l: i for i, l in enumerate(string.ascii_lowercase)}
        self.attempt = 0
        self.sub_attempt = 0
        self.prior_guess_words = []
        self.observation = np.array(
            [[[26, 26, 26, 26, 26], [1, 1, 1, 1, 1]]] * self.max_attempts
        )
        self.color_map = {
            0: "on_red",
            1: "on_white",
            2: "on_yellow",
            3: "on_green",
        }
        self.render = render
        self.flatten = flatten

        # a (2, 5) matrix for each guess. [0, :] is the guess letters (e.g. 0 = a, .. 25 = z, 26 = '' or no guess yet)
        #  and [1, :] is the score per letter:
        # 0 whole word is invalid (already guessed or not in valid_words)
        # 1 not in word,
        # 2 in word but different location,
        # 3 correct letter in correct location)
        if self.flatten:
            self.observation_space = spaces.MultiDiscrete(
                [27, 27, 27, 27, 27, 4, 4, 4, 4, 4] * self.max_attempts
            )
        else:
            self.observation_space = spaces.MultiDiscrete(
                [[[27, 27, 27, 27, 27], [4, 4, 4, 4, 4]]] * self.max_attempts
            )

        # letters in guess (e.g. 0 = a, .. 25 = z, 26)
        # NOTE can't guess ''
        self.action_space = spaces.MultiDiscrete([26, 26, 26, 26, 26])

    def _translate_word(self, word):
        return np.array(list(map(self.index_map.get, list(word))))

    def _translate_action(self, action):
        return "".join(map(self.letter_map.get, action))

    def _get_reward(self, action):
        guess_word = self._translate_action(action)
        if guess_word == self.word:
            return 10
        # elif guess_word in self.prior_guess_words:
        #     reward = -5
        # elif guess_word not in self.valid_words:
        #     reward = -1
        else:
            # bonus reward for new correct letters
            # msk = (
            #     (self.observation[: self.attempt - 1, 1] == 2).sum(axis=0).astype(bool)
            # )
            # reward = (self.observation[self.attempt - 1, 1] == 2)[~msk].sum()
            # # reward for any correct letters
            # reward += (self.observation[self.attempt - 1, 1] == 2).sum()
            reward = 0
        self.prior_guess_words.append(guess_word)
        return reward

    def _check_done(self, action):
        guess_word = self._translate_action(action)
        if (
            guess_word == self.word
            or self.attempt == self.max_attempts
            or self.sub_attempt == self.max_sub_attempts
        ):
            return True
        return False

    def _update_state(self, action):
        guess_word = self._translate_action(action)
        self.sub_attempt += 1

        g_corr_letter_msk = self.word_array == action
        g_present_letter_msk = np.isin(action, self.word_array)

        self.observation[self.attempt, 0] = action
        if guess_word in self.prior_guess_words or guess_word not in self.valid_words:
            self.observation[self.attempt, 1] = 0

        else:
            self.observation[self.attempt, 1][g_present_letter_msk] = 2
            self.observation[self.attempt, 1][g_corr_letter_msk] = 3
            self.attempt += 1

    def _get_obs(self, action=None):
        if action is not None:
            self._update_state(action)
        if self.flatten:
            return self.observation.flatten()
        return self.observation

    def _get_info(self):
        return {
            "attempt": self.attempt,
            "sub_attempt": self.sub_attempt,
        }

    def reset(self, return_info=True, seed=None):
        self.word = self.word_bank[np.random.randint(len(self.word_bank))]
        self.word_array = self._translate_word(self.word)
        self.attempt = 0
        self.sub_attempt = 0
        self.prior_guess_words = []
        self.observation = np.array(
            [[[26, 26, 26, 26, 26], [1, 1, 1, 1, 1]]] * self.max_attempts
        )
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def print_render(self):
        print(f"\nattempt {self.attempt}")
        for row in self.observation:
            for letter, color in zip(row[0], row[1]):
                cprint(
                    self.letter_map.get(letter, " "),
                    "grey",
                    self.color_map[color],
                    end="",
                )
            print()

    def step(self, action):
        observation = self._get_obs(action)
        reward = self._get_reward(action)
        done = self._check_done(action)
        info = self._get_info()
        if self.render:
            self.print_render()
        return observation, reward, done, False, info

    @classmethod
    def create_default(cls, **kwargs):
        with open("valid-words.csv") as f:
            valid_words = f.readlines()
        valid_words = [w.strip() for w in valid_words]

        with open("word-bank.csv") as f:
            word_bank = f.readlines()
        word_bank = [w.strip() for w in word_bank]
        return cls(valid_words, word_bank, **kwargs)

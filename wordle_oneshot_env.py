import gymnasium as gym
from gymnasium import spaces
import numpy as np
import string
from masked_word_env import build_bigram_table, build_position_freq, _load_words


class OneShotWordleEnv(gym.Env):
    """
    Wordle with single-shot guesses. Same obs space as DiffusionWordEnv
    for perfect weight transfer.

    Each step = one Wordle guess (no diffusion refinement).
    Agent proposes 5 letters, they're immediately submitted.
    Greens from prior guesses freeze positions.
    Episode = up to 6 steps (guesses).

    Obs: [current_letters(5), mask(5)]
      - Green positions: letter is frozen (mask=0)
      - Free positions: last guess letters or 26/blank (mask=1)
    """

    def __init__(self, valid_words, word_bank):
        super().__init__()
        self.valid_words_list = list(valid_words)
        self.valid_words = set(valid_words)
        self.word_bank = word_bank
        self.letter_map = {i: l for i, l in enumerate(string.ascii_lowercase)}
        self.index_map = {l: i for i, l in enumerate(string.ascii_lowercase)}

        self.bigram_table = build_bigram_table(self.valid_words_list)
        self.position_freq = build_position_freq(self.valid_words_list)

        self.observation_space = spaces.MultiDiscrete(
            [27, 27, 27, 27, 27, 2, 2, 2, 2, 2]
        )
        self.action_space = spaces.MultiDiscrete([26, 26, 26, 26, 26])

    def _translate_action(self, action):
        return "".join(self.letter_map[a] for a in action)

    def _score_guess(self, guess_arr, target_arr):
        scores = np.ones(5, dtype=np.int64)
        target_remaining = target_arr.copy()
        for i in range(5):
            if guess_arr[i] == target_arr[i]:
                scores[i] = 3
                target_remaining[i] = -1
        for i in range(5):
            if scores[i] == 3:
                continue
            match = np.where(target_remaining == guess_arr[i])[0]
            if len(match) > 0:
                scores[i] = 2
                target_remaining[match[0]] = -1
        return scores

    def _shaped_reward(self, letter_indices):
        bigram = 0.0
        for pos in range(4):
            bigram += self.bigram_table[pos, letter_indices[pos], letter_indices[pos + 1]]
        position = 0.0
        for pos in range(5):
            position += self.position_freq[pos, letter_indices[pos]]
        return 0.3 * bigram / 4.0 + 0.1 * position / 5.0

    def _build_obs(self):
        return np.concatenate([self.current.copy(), self.mask.copy()])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target_word = self.word_bank[self.np_random.integers(len(self.word_bank))]
        self.target_array = np.array([self.index_map[c] for c in self.target_word])
        self.guess_count = 0
        self.greens = {}
        self.mask = np.ones(5, dtype=np.int64)
        self.current = np.full(5, 26, dtype=np.int64)
        return self._build_obs(), {"target": self.target_word}

    def step(self, action):
        guess = action.copy()
        for pos, letter in self.greens.items():
            guess[pos] = letter

        guess_word = self._translate_action(guess)
        is_valid = guess_word in self.valid_words
        is_correct = guess_word == self.target_word
        self.guess_count += 1

        if is_correct:
            bonus = 10.0 * (7 - self.guess_count)
            self.current = guess
            return self._build_obs(), bonus, True, False, {
                "won": True, "guesses": self.guess_count,
                "produced_word": guess_word, "target_word": self.target_word,
            }

        if is_valid:
            scores = self._score_guess(guess, self.target_array)
            for i in range(5):
                if scores[i] == 3:
                    self.greens[i] = guess[i]
                    self.mask[i] = 0
            greens = float(np.sum(scores == 3))
            yellows = float(np.sum(scores == 2))
            reward = 1.0 * greens + 0.5 * yellows
        else:
            reward = float(self._shaped_reward(guess)) * 0.5 - 1.0

        self.current = guess
        for pos, letter in self.greens.items():
            self.current[pos] = letter

        done = self.guess_count >= 6
        return self._build_obs(), reward, done, False, {
            "won": False if done else None,
            "guesses": self.guess_count,
            "produced_word": guess_word, "is_valid": is_valid,
            "target_word": self.target_word if done else None,
        }

    @classmethod
    def create_default(cls, **kwargs):
        valid_words, word_bank = _load_words()
        return cls(valid_words, word_bank, **kwargs)

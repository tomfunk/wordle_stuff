import gymnasium as gym
from gymnasium import spaces
import numpy as np
import string


def build_bigram_table(valid_words):
    counts = np.zeros((5, 26, 26), dtype=np.float32)
    for word in valid_words:
        indices = [ord(c) - ord("a") for c in word]
        for pos in range(4):
            counts[pos, indices[pos], indices[pos + 1]] += 1
    totals = counts.sum(axis=(1, 2), keepdims=True)
    totals[totals == 0] = 1
    return counts / totals


def build_position_freq(valid_words):
    counts = np.zeros((5, 26), dtype=np.float32)
    for word in valid_words:
        for pos, c in enumerate(word):
            counts[pos, ord(c) - ord("a")] += 1
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    return counts / totals


def _load_words():
    with open("valid-words.csv") as f:
        valid_words = [w.strip() for w in f.readlines()]
    with open("word-bank.csv") as f:
        word_bank = [w.strip() for w in f.readlines()]
    return valid_words, word_bank


class MaskedWordEnv(gym.Env):
    """Single-step masked fill-in. Good for k=1-3."""

    def __init__(self, valid_words, word_bank, num_masked=1):
        super().__init__()
        self.valid_words_list = list(valid_words)
        self.valid_words = set(valid_words)
        self.word_bank = word_bank
        self.num_masked = num_masked
        self.letter_map = {i: l for i, l in enumerate(string.ascii_lowercase)}
        self.index_map = {l: i for i, l in enumerate(string.ascii_lowercase)}

        self.bigram_table = build_bigram_table(self.valid_words_list)
        self.position_freq = build_position_freq(self.valid_words_list)

        self.observation_space = spaces.MultiDiscrete(
            [27, 27, 27, 27, 27, 2, 2, 2, 2, 2]
        )
        self.action_space = spaces.MultiDiscrete([26, 26, 26, 26, 26])

    def _translate_word(self, word):
        return np.array([self.index_map[c] for c in word])

    def _translate_action(self, action):
        return "".join(self.letter_map[a] for a in action)

    def _bigram_score(self, letter_indices):
        score = 0.0
        for pos in range(4):
            score += self.bigram_table[pos, letter_indices[pos], letter_indices[pos + 1]]
        return score / 4.0

    def _position_score(self, letter_indices):
        score = 0.0
        for pos in range(5):
            score += self.position_freq[pos, letter_indices[pos]]
        return score / 5.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target_word = self.word_bank[self.np_random.integers(len(self.word_bank))]
        self.target_array = self._translate_word(self.target_word)

        positions = self.np_random.choice(5, size=self.num_masked, replace=False)
        self.mask = np.zeros(5, dtype=np.int64)
        self.mask[positions] = 1

        letters = self.target_array.copy()
        letters[self.mask == 1] = 26

        self.obs = np.concatenate([letters, self.mask])
        return self.obs, {"target": self.target_word, "num_masked": self.num_masked}

    def step(self, action):
        result = self.obs[:5].copy()
        result[self.mask == 1] = action[self.mask == 1]
        produced_word = self._translate_action(result)

        is_valid = produced_word in self.valid_words
        exact = produced_word == self.target_word

        correct_masked = float(np.sum((result == self.target_array) & (self.mask == 1)))
        target_bonus = 0.05 * correct_masked

        if is_valid:
            reward = 1.0 + target_bonus + (0.5 if exact else 0.0)
        else:
            shaping = 0.3 * self._bigram_score(result) + 0.1 * self._position_score(result)
            reward = float(shaping) + target_bonus

        info = {
            "produced_word": produced_word,
            "target_word": self.target_word,
            "is_valid": is_valid,
            "exact_match": exact,
        }
        return self.obs, reward, True, False, info

    @classmethod
    def create_default(cls, num_masked=1, **kwargs):
        valid_words, word_bank = _load_words()
        return cls(valid_words, word_bank, num_masked=num_masked, **kwargs)


class DiffusionWordEnv(gym.Env):
    """
    Multi-step iterative refinement env for learning to produce words.

    Start from random letters in masked positions, refine over multiple steps.
    Each step: agent proposes letters for masked positions, gets shaped reward.
    Episode ends when valid word produced or max_steps reached.
    """

    def __init__(self, valid_words, word_bank, num_masked=5, max_steps=5):
        super().__init__()
        self.valid_words_list = list(valid_words)
        self.valid_words = set(valid_words)
        self.word_bank = word_bank
        self.num_masked = num_masked
        self.max_steps = max_steps
        self.letter_map = {i: l for i, l in enumerate(string.ascii_lowercase)}
        self.index_map = {l: i for i, l in enumerate(string.ascii_lowercase)}

        self.bigram_table = build_bigram_table(self.valid_words_list)
        self.position_freq = build_position_freq(self.valid_words_list)

        self.observation_space = spaces.MultiDiscrete(
            [27, 27, 27, 27, 27, 2, 2, 2, 2, 2]
        )
        self.action_space = spaces.MultiDiscrete([26, 26, 26, 26, 26])

    def _translate_word(self, word):
        return np.array([self.index_map[c] for c in word])

    def _translate_action(self, action):
        return "".join(self.letter_map[a] for a in action)

    def _bigram_score(self, letter_indices):
        score = 0.0
        for pos in range(4):
            score += self.bigram_table[pos, letter_indices[pos], letter_indices[pos + 1]]
        return score / 4.0

    def _position_score(self, letter_indices):
        score = 0.0
        for pos in range(5):
            score += self.position_freq[pos, letter_indices[pos]]
        return score / 5.0

    def _shaped_reward(self, letter_indices):
        return 0.3 * self._bigram_score(letter_indices) + 0.1 * self._position_score(letter_indices)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target_word = self.word_bank[self.np_random.integers(len(self.word_bank))]
        self.target_array = self._translate_word(self.target_word)

        positions = self.np_random.choice(5, size=self.num_masked, replace=False)
        self.mask = np.zeros(5, dtype=np.int64)
        self.mask[positions] = 1

        self.current = self.target_array.copy()
        self.current[self.mask == 1] = self.np_random.integers(0, 26, size=self.num_masked)


        self.step_count = 0
        self.prev_shaped = float(self._shaped_reward(self.current))

        obs = np.concatenate([self.current.copy(), self.mask])
        return obs, {"target": self.target_word, "num_masked": self.num_masked}

    def step(self, action):
        self.current[self.mask == 1] = action[self.mask == 1]
        self.step_count += 1

        produced_word = self._translate_action(self.current)
        is_valid = produced_word in self.valid_words
        exact = produced_word == self.target_word

        correct_masked = float(np.sum((self.current == self.target_array) & (self.mask == 1)))
        target_bonus = 0.05 * correct_masked

        shaped = float(self._shaped_reward(self.current))
        improvement = shaped - self.prev_shaped
        self.prev_shaped = shaped

        if is_valid:
            reward = 5.0 + target_bonus + (1.0 if exact else 0.0)
            done = True
        elif self.step_count >= self.max_steps:
            reward = shaped + target_bonus
            done = True
        else:
            reward = 0.1 * max(improvement, 0.0) + target_bonus
            done = False

        obs = np.concatenate([self.current.copy(), self.mask])
        info = {
            "produced_word": produced_word,
            "target_word": self.target_word,
            "is_valid": is_valid,
            "exact_match": exact,
            "step": self.step_count,
        }
        return obs, reward, done, False, info

    @classmethod
    def create_default(cls, num_masked=5, max_steps=5, **kwargs):
        valid_words, word_bank = _load_words()
        return cls(valid_words, word_bank, num_masked=num_masked, max_steps=max_steps, **kwargs)

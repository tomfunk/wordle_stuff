import numpy as np
import pandas as pd
import string


class WordleAgent1:
    def __init__(self, valid_words):
        self.words = self.get_words(valid_words)
        self.letter_map = {i: l for i, l in enumerate(string.ascii_lowercase)}

    @staticmethod
    def get_score(row, freq):
        return sum([freq[i].get(row[i], 0) for i in range(5)])

    def get_words(self, valid_words):
        words = pd.DataFrame([list(w) for w in valid_words])
        assert words.shape[1] == 5

        words["whole"] = valid_words
        words["nunique"] = words["whole"].apply(lambda x: len(set(x)))

        # letter freq in position
        pos_freq = {
            i: words[0].value_counts(normalize=True).to_dict() for i in range(5)
        }
        nunique_freq = words["nunique"].value_counts(normalize=True).to_dict()

        # freq of number of unique letters

        words["score"] = words.apply(self.get_score, freq=pos_freq, axis=1)
        words["score"] = words["score"] * words["nunique"].map(nunique_freq)
        return words

    def act(self, obs):
        if obs.shape == (60,):
            obs = obs.reshape((6, 2, 5))

        assert obs.shape[1:] == (2, 5)
        letters = pd.DataFrame(obs[:, 0, :]).applymap(self.letter_map.get)
        scores = pd.DataFrame(obs[:, 1, :])

        # words already guessed
        prior_guesses = letters.dropna().apply("".join, axis=1).tolist()
        # letters known to not be in the word
        known_absent_letters = (
            letters[scores == 1].melt()["value"].dropna().unique().tolist()
        )
        # letters known to be in the word
        known_present_letters = (
            letters[scores > 1].melt()["value"].dropna().unique().tolist()
        )
        # letters known to be in the word but not in specific positions
        known_present_letters_misses = (
            letters[scores == 2]
            .melt()
            .dropna()
            .groupby("variable")["value"]
            .apply(list)
            .to_dict()
        )
        # letters known to be in specific positions
        known_exact_letters = (
            letters[scores == 3]
            .melt()
            .dropna()
            .set_index("variable")["value"]
            .to_dict()
        )

        # msk remaining valid self.words
        msk = ~self.words["whole"].isin(prior_guesses)
        if known_absent_letters:
            msk = msk & (
                ~self.words.whole.str.contains(f'[{"".join(known_absent_letters)}]')
            )

        if known_present_letters:
            msk = msk & (
                self.words.whole.str.contains(f'[{"".join(known_present_letters)}]')
            )

        if known_present_letters_misses:
            for k, v in known_present_letters_misses.items():
                msk = msk & (~self.words[k].isin(v))

        if known_present_letters:
            for k, v in known_exact_letters.items():
                msk = msk & (self.words[k] == v)

        action = self.words[msk].sort_values(by="score", ascending=False).iloc[0].whole
        return action

    @classmethod
    def create_default(cls, **kwargs):
        with open("valid-words.csv") as f:
            valid_words = f.readlines()
        valid_words = [w.strip() for w in valid_words]
        return cls(valid_words, **kwargs)


class WordleAgentRandom:
    def __init__(self, valid_words):
        self.valid_words = valid_words

    def act(self, obs):
        return np.random.choice(self.valid_words)

    @classmethod
    def create_default(cls, **kwargs):
        with open("valid-words.csv") as f:
            valid_words = f.readlines()
        valid_words = [w.strip() for w in valid_words]
        return cls(valid_words, **kwargs)

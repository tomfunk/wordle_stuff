"""Have Jev (TypeSafe System One) play Wordle in our OneShotWordleEnv.

Each round Jev chooses a letter per open position from the letters still
available given the feedback so far — one Jev "choice" question per position,
one HTTP call per guess, up to 6 guesses.

Calls the Jev REST API directly with urllib (same pattern as tag_n_bag_poc.py),
so no SDK install is needed. Reads TYPESAFE_API_KEY from ./.env.

    python play_wordle_jev.py [seed]
    python play_wordle_jev.py selftest   # offline logic check, no API calls
"""
import json
import os
import string
import sys
import urllib.request

import numpy as np

from wordle_oneshot_env import OneShotWordleEnv

ALPHABET = list(string.ascii_lowercase)
COLORS = {3: "🟩", 2: "🟨", 1: "⬜"}
JEV_URL = "https://api.typesafe.ai/v1/systemone"


def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.rstrip("\n").split("=", 1)
                os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def jev(state, questions):
    """POST one System One request; return the answers dict. Mirrors tag_n_bag_poc._jev."""
    body = json.dumps({"state": state[:16000], "model": "jev-latest",
                       "questions": questions}).encode()
    req = urllib.request.Request(JEV_URL, data=body, headers={
        "Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
        "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)["answers"]


def render_board(rows):
    if not rows:
        return "(no guesses yet)"
    return "\n".join(
        f"  {w.upper()}  " + "".join(COLORS[s] for s in scores) for w, scores in rows
    )


def available_letters(pos, greens, absent, present, banned_at):
    if pos in greens:
        return [greens[pos]]
    # ponytail: duplicate-letter grays handled by 'absent - present', not the full
    # "gray means <=N copies" rule. Upgrade to per-count if it matters.
    banned = (absent - present) | banned_at[pos]
    letters = [c for c in ALPHABET if c not in banned]
    return letters or ALPHABET  # never leave a position with zero options


def choose_word(board_rows, greens, absent, present, banned_at):
    """One Jev call: a choice question per open position -> a 5-letter guess."""
    state = (
        "You are playing Wordle. The hidden answer is a common 5-letter word.\n"
        f"Board so far ({len(board_rows)}/6 guesses):\n{render_board(board_rows)}\n"
        f"Letters known IN the word: {sorted(present) or 'none'}.\n"
        f"Letters known NOT in the word: {sorted(absent) or 'none'}.\n"
        "Pick the most likely letter for each open position to form a real word."
    )
    guess = [None] * 5
    questions = {}
    for pos in range(5):
        opts = available_letters(pos, greens, absent, present, banned_at)
        if len(opts) == 1:
            guess[pos] = opts[0]  # forced (green or only option) — no need to ask
            continue
        questions[f"pos{pos}"] = {
            "type": "choice",
            "instructions": f"Best letter for position {pos + 1} of the 5-letter word",
            "criteria": {c: f"the letter '{c.upper()}'" for c in opts},
        }

    if questions:
        answers = jev(state, questions)
        for key, ans in answers.items():
            guess[int(key[3:])] = ans["choice"]
            print(f"    {key}: {ans['choice'].upper()} (conf={ans.get('confidence', 0):.2f})")

    return "".join(guess)


def update_constraints(word, scores, greens, absent, present, banned_at):
    for i, (c, s) in enumerate(zip(word, scores)):
        if s == 3:
            greens[i] = c
            present.add(c)
        elif s == 2:
            present.add(c)
            banned_at[i].add(c)
        else:
            absent.add(c)


def play(seed=None):
    load_env()
    env = OneShotWordleEnv.create_default()
    env.reset(seed=seed)
    print(f"Target (hidden from Jev): {env.target_word.upper()}\n")

    greens, absent, present = {}, set(), set()
    banned_at = [set() for _ in range(5)]
    board_rows = []

    for turn in range(1, 7):
        print(f"Guess {turn}:")
        word = choose_word(board_rows, greens, absent, present, banned_at)
        if word not in env.valid_words:
            print(f"  {word.upper()} is not a valid word — counts as a miss.")

        guess_arr = np.array([env.index_map[c] for c in word])
        scores = env._score_guess(guess_arr, env.target_array)
        board_rows.append((word, scores))
        print(render_board([board_rows[-1]]))

        if word == env.target_word:
            print(f"\n🎉 Jev solved it in {turn} guesses: {word.upper()}")
            return
        update_constraints(word, scores, greens, absent, present, banned_at)

    print(f"\n❌ Out of guesses. The word was {env.target_word.upper()}.")


def selftest():
    """Offline check of the constraint logic — no API calls."""
    env = OneShotWordleEnv.create_default()
    env.reset(seed=0)
    # Force a known target to test scoring + constraint updates.
    env.target_word = "crane"
    env.target_array = np.array([env.index_map[c] for c in "crane"])

    greens, absent, present = {}, set(), set()
    banned_at = [set() for _ in range(5)]
    # "artsy" vs "crane": a=yellow(pos0), r=green(pos1), t/s/y=absent.
    scores = env._score_guess(np.array([env.index_map[c] for c in "artsy"]), env.target_array)
    assert list(scores) == [2, 3, 1, 1, 1], list(scores)
    update_constraints("artsy", scores, greens, absent, present, banned_at)
    assert greens == {1: "r"}, greens
    assert present == {"a", "r"} and {"t", "s", "y"} <= absent, (present, absent)
    # 'a' is present but yellow at pos0, so banned there, available elsewhere.
    assert "a" not in available_letters(0, greens, absent, present, banned_at)
    assert "a" in available_letters(2, greens, absent, present, banned_at)
    assert "s" not in available_letters(2, greens, absent, present, banned_at)
    assert available_letters(1, greens, absent, present, banned_at) == ["r"]
    print("selftest OK")


def interdep():
    """Fresh board, first guess two ways: (A) 5 independent choices in one call,
    (B) 5 choices made sequentially, each told the letters already chosen.
    If B forms real words and A doesn't, the choices only interrelate when we
    explicitly feed prior picks back in — i.e. they're independent within a call.
    """
    load_env()
    env = OneShotWordleEnv.create_default()
    base = "Playing Wordle. Empty board. Pick letters to spell a common 5-letter word."

    # (A) one call, five parallel questions
    qs = {f"pos{i}": {"type": "choice",
                      "instructions": f"Best letter for position {i + 1} of the word",
                      "criteria": {c: f"letter '{c.upper()}'" for c in ALPHABET}}
          for i in range(5)}
    a = jev(base, qs)
    word_a = "".join(a[f"pos{i}"]["choice"] for i in range(5))

    # (B) five calls, each conditioned on the letters chosen so far
    chosen = []
    for i in range(5):
        state = base + (f" Letters chosen so far: {''.join(chosen).upper()}." if chosen else "")
        r = jev(state, {"pos": {"type": "choice",
                                "instructions": f"Best letter for position {i + 1}, "
                                                "given the letters already chosen, to spell a real word",
                                "criteria": {c: f"letter '{c.upper()}'" for c in ALPHABET}}})
        chosen.append(r["pos"]["choice"])
    word_b = "".join(chosen)

    for label, w in (("A parallel (one call)", word_a), ("B sequential (conditioned)", word_b)):
        print(f"{label}: {w.upper()}  {'valid word' if w in env.valid_words else 'NOT a word'}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "selftest":
        selftest()
    elif arg == "interdep":
        interdep()
    else:
        play(int(arg) if arg else None)

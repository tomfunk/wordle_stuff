# Jev plays Wordle

`play_wordle_jev.py` — the Jev model (TypeSafe System One) plays Wordle in
`OneShotWordleEnv`. Each round it makes one `choice` question per open position,
picking a letter from those still allowed by the green/yellow/gray feedback.

Calls the REST API directly with stdlib `urllib` (same pattern as
`tag_n_bag_poc.py`), so no SDK install is needed. Reads `TYPESAFE_API_KEY`
from `.env` (gitignored — not in the repo).

```
python play_wordle_jev.py [seed]   # play a game
python play_wordle_jev.py selftest # offline constraint-logic check, no API
python play_wordle_jev.py interdep # the interdependence experiment below
```

## Learnings

**Choices in one `system_one` call are independent.** With 5 separate `choice`
questions in a single call, each is judged against the shared `state` alone —
not against the sibling answers. Proof: the parallel first guess is `SEEEE`
(the same letter in all five slots). Nothing reasoning jointly toward a word
would do that.

**Interdependence only appears across calls.** When we pick letters
sequentially and feed the already-chosen letters into the next question's
`state`, the picks become complementary: the first guess turns into `AERTS`
(five distinct high-frequency letters, no repeats). So Jev *can* condition on
other letters — but only on what's actually in the `state` it receives.

**Letter-level choice can't win Wordle**, interdependent or not. Even the
conditioned `AERTS` is the five best letters, not a real word. Per-letter
choice optimizes "good letter for this slot," never "these 5 letters ∈ the
valid-word set." That membership constraint is the missing piece.

**To make it win**, change the unit of choice from letters to whole words: one
`choice` over candidate valid words filtered by the feedback constraints. Then
every guess is legal by construction. (Not yet implemented.)

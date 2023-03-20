# PPO_1
## Setup
Train from scratch. 512 x 4 layers

## Rewards
- -5 for guessing the same word twice
- -1 for guessing an invalid word
- +1 for each existing letter
- +2 for each correct letter (correct position)

## Result
Learned how to guess lots of exiting letters in incorrect places

# PPO_2
## Setup
Train from PPO_1

## Rewards
- -5 for guessing the same word twice
- -1 for guessing an invalid word
- +10 correct word

## Result
Starts at low subattempts but no wins

# PPO_4
## Setup
Train from PPO_1

## Rewards
- -5 for guessing the same word twice
- -1 for guessing an invalid word
- +0.25 for each correct letter (correct position) no repeats
- +10 correct word

## Result
No wins

# PPO_5
## Setup
Train from scratch

## Rewards
- -5 for guessing the same word twice
- -1 for guessing an invalid word
- +0.25 for each correct letter (correct position) no repeats
- +10 correct word

## Result
No wins

# PPO_7
## Setup
Train from scratch. 1024 x 4 layers

## Rewards
- -5 for guessing the same word twice
- -1 for guessing an invalid word
- +1 for each correct letter (correct position)
- +100 correct word



## Result
After 8 million steps it generally finishes in 6 steps with at least 1 or 2 correct letters. Doesn't seem to know what to do with yellow letters.

# PPO_8
## Setup
Train from scratch. 512 x 4 layers
Initially trained with 100 instead of 6 attempts per game (and 200 instead of 100 sub attempts)

## Rewards
- -5 for guessing the same word twice
- -1 for guessing an invalid word
- +1 for each correct letter (correct position)
- +100 correct word

## Result
Started increasing reward much earlier (1M vs 3.5M for most other models). Trains much slower. Got stuck.


# PPO_9
## Setup
Train from scratch. 512 x 4 layers
Initially trained with 25 instead of 6 attempts per game
gamma = 0.9

## Rewards
- -5 for guessing the same word twice
- -1 for guessing an invalid word
- +1 for each correct letter (correct position)
- +1 bonus for each new correct letter (correct position)
- +1000 correct word

## Result


# PPO_10
## Setup
Train from scratch. 512 x 4 layers
Initially trained with 25 instead of 6 attempts per game
gamma = 0.95

## Rewards
- -5 for guessing the same word twice
- -1 for guessing an invalid word
- +1 for each correct letter (correct position)
- +1 bonus for each new correct letter (correct position)
- +1000 correct word

## Result

# PPO_11
## Setup
Train from scratch. 512 x 6 layers
Initially trained with 25 instead of 6 attempts per game
gamma = 0.8

## Rewards
- -5 for guessing the same word twice
- -1 for guessing an invalid word
- +1 for each correct letter (correct position)
- +1 bonus for each new correct letter (correct position)
- +1000 correct word

## Result


# PPO_11
## Setup
Train from scratch. 512 x 6 layers
Initially trained with 10 instead of 6 attempts per game
gamma = 0.95

## Rewards
- -5 for guessing the same word twice
- -1 for guessing an invalid word
- +1 for each correct letter (correct position)
- +1 bonus for each new correct letter (correct position)
- +1000 correct word

## Result
# Wordle Stuff

## Files
|file|desc|
|--|--|
|env.py |a reinforcement learning environment for playing wordle|
|agent.ipynb|a notebook with a simple agent, snippet for training stable_baselines agent and some manual analysis|
|valid-words.csv| words that wordle accepts|
|word-bank.csv| possible correct wordle words|


## Stats
### Letters
The following summary stats are for letters in word bank words.

Definitions:
- words_contain - number of words that contain a letter
- words_contain_percent - %age of words that contain a letter
- freq - average number of a letter contained in a word
- freq_if_present - average number of a letter contained in a word for words that contain the letter
- max_freq - max number of times a letter is in a word

| letter   |   words_contain |   words_contain_percent |      freq |   freq_if_present |   max_freq |
|:---------|----------------:|------------------------:|----------:|------------------:|-----------:|
| a        |             909 |               0.392657  | 0.422894  |           1.07701 |          2 |
| b        |             267 |               0.115335  | 0.121382  |           1.05243 |          3 |
| c        |             448 |               0.193521  | 0.206048  |           1.06473 |          2 |
| d        |             370 |               0.159827  | 0.169762  |           1.06216 |          3 |
| e        |            1056 |               0.456156  | 0.532613  |           1.16761 |          3 |
| f        |             207 |               0.0894168 | 0.0993521 |           1.11111 |          3 |
| g        |             300 |               0.12959   | 0.134341  |           1.03667 |          2 |
| h        |             379 |               0.163715  | 0.168035  |           1.02639 |          2 |
| i        |             647 |               0.279482  | 0.289849  |           1.03709 |          2 |
| j        |              27 |               0.0116631 | 0.0116631 |           1       |          1 |
| k        |             202 |               0.087257  | 0.0907127 |           1.0396  |          2 |
| l        |             648 |               0.279914  | 0.310583  |           1.10957 |          2 |
| m        |             298 |               0.128726  | 0.136501  |           1.0604  |          3 |
| n        |             550 |               0.237581  | 0.24838   |           1.04545 |          3 |
| o        |             673 |               0.290713  | 0.325702  |           1.12036 |          2 |
| p        |             346 |               0.14946   | 0.158531  |           1.06069 |          3 |
| q        |              29 |               0.012527  | 0.012527  |           1       |          1 |
| r        |             837 |               0.361555  | 0.388337  |           1.07407 |          3 |
| s        |             618 |               0.266955  | 0.288985  |           1.08252 |          3 |
| t        |             667 |               0.288121  | 0.314903  |           1.09295 |          3 |
| u        |             457 |               0.197408  | 0.201728  |           1.02188 |          2 |
| v        |             149 |               0.0643629 | 0.0660907 |           1.02685 |          2 |
| w        |             194 |               0.0838013 | 0.0842333 |           1.00515 |          2 |
| x        |              37 |               0.0159827 | 0.0159827 |           1       |          1 |
| y        |             417 |               0.18013   | 0.183585  |           1.01918 |          2 |
| z        |              35 |               0.0151188 | 0.0172786 |           1.14286 |          2 |

### Best First Pick
The following scores are the sum of unique letter `words_contain_freq` for word bank words:
| word   |   score | in_bank   |
|:-------|--------:|:----------|
| alter  | 1.7784  | True      |
| later  | 1.7784  | True      |
| alert  | 1.7784  | True      |
| irate  | 1.77797 | True      |
| arose  | 1.76803 | True      |
| stare  | 1.76544 | True      |
| raise  | 1.7568  | True      |
| arise  | 1.7568  | True      |
| learn  | 1.72786 | True      |
| renal  | 1.72786 | True      |
| saner  | 1.7149  | True      |
| snare  | 1.7149  | True      |
| cater  | 1.69201 | True      |
| trace  | 1.69201 | True      |
| react  | 1.69201 | True      |
| crate  | 1.69201 | True      |
| stale  | 1.6838  | True      |
| steal  | 1.6838  | True      |
| least  | 1.6838  | True      |
| slate  | 1.6838  | True      |

and for valid words:

| word   |   score | in_bank   |
|:-------|--------:|:----------|
| oater  | 1.7892  | False     |
| roate  | 1.7892  | False     |
| orate  | 1.7892  | False     |
| realo  | 1.78099 | False     |
| taler  | 1.7784  | False     |
| artel  | 1.7784  | False     |
| ratel  | 1.7784  | False     |
| alert  | 1.7784  | True      |
| alter  | 1.7784  | True      |
| later  | 1.7784  | True      |
| terai  | 1.77797 | False     |
| irate  | 1.77797 | True      |
| retia  | 1.77797 | False     |
| raile  | 1.76976 | False     |
| ariel  | 1.76976 | False     |
| arose  | 1.76803 | True      |
| aeros  | 1.76803 | False     |
| soare  | 1.76803 | False     |
| taser  | 1.76544 | False     |
| strae  | 1.76544 | False     |

### Best Second Pick
If you chose `alter` and have no qualms about guessing words that don't contain any known letters:
| word   |   score | in_bank   |
|:-------|--------:|:----------|
| sonic  | 1.26825 | True      |
| scion  | 1.26825 | True      |
| noisy  | 1.25486 | True      |
| disco  | 1.1905  | True      |
| bison  | 1.19006 | True      |
| sound  | 1.15248 | True      |
| synod  | 1.13521 | True      |
| shiny  | 1.12786 | True      |
| spiny  | 1.11361 | True      |
| suing  | 1.11102 | True      |
| using  | 1.11102 | True      |
| minus  | 1.11015 | True      |
| bonus  | 1.10799 | True      |
| doing  | 1.09719 | True      |
| dingo  | 1.09719 | True      |
| spicy  | 1.06955 | True      |
| music  | 1.06609 | True      |
| snowy  | 1.05918 | True      |
| bingo  | 1.0527  | True      |
| hound  | 1.04924 | True      |

and for `oater`:
| word   |   score | in_bank   |
|:-------|--------:|:----------|
| lysin  | 1.24406 | False     |
| linds  | 1.22376 | False     |
| sulci  | 1.21728 | False     |
| sling  | 1.19352 | True      |
| lings  | 1.19352 | False     |
| limns  | 1.19266 | False     |
| hilus  | 1.18747 | False     |
| blins  | 1.17927 | False     |
| incus  | 1.17495 | False     |
| pilus  | 1.17322 | False     |
| pulis  | 1.17322 | False     |
| shily  | 1.17019 | False     |
| clips  | 1.16933 | False     |
| idyls  | 1.16631 | False     |
| unlid  | 1.15421 | False     |
| linch  | 1.15421 | False     |
| gusli  | 1.15335 | False     |
| iglus  | 1.15335 | False     |
| muils  | 1.15248 | False     |
| simul  | 1.15248 | False     |
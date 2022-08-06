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

#### Position Frequency

This is the frequency of each letter in each position. For example the first (0th) letter is `s` in 15.8% of word bank words.

|    | 0   | freq | 1   | freq | 2 | freq | 3 | freq | 4 | freq |
|---:|:--------|-------------:|:--------|------------:|:--------|------------:|:--------|--------------:|:--------|--------------:|
|  0 | s       |   0.158099   | a       | 0.131317    | a       | 0.132613    | e       |   0.137365    | e       |   0.183153    |
|  1 | c       |   0.0855292  | o       | 0.120518    | i       | 0.114903    | n       |   0.0786177   | y       |   0.157235    |
|  2 | b       |   0.07473    | r       | 0.115335    | o       | 0.1054      | s       |   0.0738661   | t       |   0.109287    |
|  3 | t       |   0.0643629  | e       | 0.104536    | e       | 0.0764579   | a       |   0.0704104   | r       |   0.0915767   |
|  4 | p       |   0.0613391  | i       | 0.087257    | u       | 0.0712743   | l       |   0.0699784   | l       |   0.0673866   |
|  5 | a       |   0.0609071  | l       | 0.0868251   | r       | 0.0704104   | i       |   0.0682505   | h       |   0.0600432   |
|  6 | f       |   0.0587473  | u       | 0.0803456   | n       | 0.0600432   | c       |   0.0656587   | n       |   0.0561555   |
|  7 | g       |   0.049676   | h       | 0.062203    | l       | 0.0483801   | r       |   0.0656587   | d       |   0.0509719   |
|  8 | d       |   0.0479482  | n       | 0.037581    | t       | 0.0479482   | t       |   0.0600432   | k       |   0.0488121   |
|  9 | m       |   0.0462203  | t       | 0.0332613   | s       | 0.0345572   | o       |   0.0570194   | a       |   0.0276458   |
| 10 | r       |   0.0453564  | p       | 0.0263499   | d       | 0.0323974   | u       |   0.0354212   | o       |   0.025054    |
| 11 | l       |   0.038013   | w       | 0.0190065   | g       | 0.0289417   | g       |   0.0328294   | p       |   0.0241901   |
| 12 | w       |   0.0358531  | c       | 0.0172786   | m       | 0.0263499   | d       |   0.0298056   | m       |   0.0181425   |
| 13 | e       |   0.0311015  | m       | 0.0164147   | p       | 0.025054    | m       |   0.0293737   | g       |   0.0177106   |
| 14 | h       |   0.0298056  | y       | 0.00993521  | b       | 0.024622    | k       |   0.0237581   | s       |   0.0155508   |
| 15 | v       |   0.0185745  | d       | 0.00863931  | c       | 0.0241901   | p       |   0.0215983   | c       |   0.0133909   |
| 16 | o       |   0.0177106  | b       | 0.00691145  | v       | 0.0211663   | v       |   0.0198704   | f       |   0.0112311   |
| 17 | n       |   0.0159827  | s       | 0.00691145  | y       | 0.012527    | f       |   0.0151188   | w       |   0.00734341  |
| 18 | i       |   0.0146868  | v       | 0.00647948  | w       | 0.0112311   | h       |   0.012095    | i       |   0.00475162  |
| 19 | u       |   0.0142549  | x       | 0.00604752  | f       | 0.0107991   | w       |   0.0107991   | b       |   0.00475162  |
| 20 | q       |   0.00993521 | g       | 0.00518359  | k       | 0.00518359  | b       |   0.0103672   | x       |   0.00345572  |
| 21 | k       |   0.00863931 | k       | 0.00431965  | x       | 0.00518359  | z       |   0.00863931  | z       |   0.00172786  |
| 22 | j       |   0.00863931 | f       | 0.00345572  | z       | 0.00475162  | x       |   0.0012959   | u       |   0.000431965 |
| 23 | y       |   0.00259179 | q       | 0.00215983  | h       | 0.00388769  | y       |   0.0012959   | nan     | nan           |
| 24 | z       |   0.0012959  | z       | 0.000863931 | j       | 0.0012959   | j       |   0.000863931 | nan     | nan           |
| 25 | nan     | nan          | j       | 0.000863931 | q       | 0.000431965 | nan     | nan           | nan     | nan           |

#### Overall Frequency
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

#### Position Frequency
The following are the highest likelihood of correct letters in the correct place for word bank words (avg_score is the average number of correct letters in the correct place):
| word   |   avg_score | in_bank   |
|:-------|------------:|:----------|
| slate  |    0.620734 | True      |
| sauce  |    0.609503 | True      |
| slice  |    0.608639 | True      |
| shale  |    0.606048 | True      |
| saute  |    0.603888 | True      |
| share  |    0.601728 | True      |
| sooty  |    0.601296 | True      |
| shine  |    0.596976 | True      |
| suite  |    0.596544 | True      |
| crane  |    0.595248 | True      |
| saint  |    0.592225 | True      |
| soapy  |    0.590065 | True      |
| shone  |    0.587473 | True      |
| shire  |    0.584017 | True      |
| saucy  |    0.583585 | True      |
| slave  |    0.580562 | True      |
| sense  |    0.579698 | True      |
| cease  |    0.579698 | True      |
| saner  |    0.578402 | True      |
| stale  |    0.577106 | True      |

and for all valid words:

| word   |   avg_score | in_bank   |
|:-------|------------:|:----------|
| saree  |    0.680346 | False     |
| sooey  |    0.678618 | False     |
| soree  |    0.669546 | False     |
| saine  |    0.666091 | False     |
| soare  |    0.660043 | False     |
| saice  |    0.653132 | False     |
| sease  |    0.652268 | False     |
| seare  |    0.64406  | False     |
| seine  |    0.639309 | False     |
| slane  |    0.639309 | False     |
| soole  |    0.637149 | False     |
| siree  |    0.636285 | False     |
| seise  |    0.634557 | False     |
| cooee  |    0.631965 | False     |
| soote  |    0.627214 | False     |
| slate  |    0.620734 | True      |
| soily  |    0.620734 | False     |
| soave  |    0.614255 | False     |
| samey  |    0.610367 | False     |
| sauce  |    0.609503 | True      |

This is the same as above but avg_score also includes 0.5 for correct letters in the incorrect place. This is for word bank words:

| word   |   avg_score | in_bank   |
|:-------|------------:|:----------|
| stare  |     1.26026 | True      |
| arose  |     1.2486  | True      |
| slate  |     1.24536 | True      |
| raise  |     1.23564 | True      |
| arise  |     1.23542 | True      |
| saner  |     1.22981 | True      |
| snare  |     1.22916 | True      |
| irate  |     1.22721 | True      |
| stale  |     1.22354 | True      |
| crate  |     1.22073 | True      |
| trace  |     1.21296 | True      |
| later  |     1.20778 | True      |
| share  |     1.2013  | True      |
| store  |     1.19806 | True      |
| scare  |     1.19784 | True      |
| alter  |     1.19698 | True      |
| crane  |     1.19676 | True      |
| alert  |     1.18423 | True      |
| teary  |     1.18337 | True      |
| saute  |     1.18251 | True      |

and all valid words:

| word   |   avg_score | in_bank   |
|:-------|------------:|:----------|
| soare  |     1.30929 | False     |
| roate  |     1.26307 | False     |
| stare  |     1.26026 | True      |
| arose  |     1.2486  | True      |
| orate  |     1.24665 | False     |
| slate  |     1.24536 | True      |
| raile  |     1.24449 | False     |
| raise  |     1.23564 | True      |
| arise  |     1.23542 | True      |
| strae  |     1.23153 | False     |
| saner  |     1.22981 | True      |
| snare  |     1.22916 | True      |
| salet  |     1.22721 | False     |
| irate  |     1.22721 | True      |
| saine  |     1.22441 | False     |
| stale  |     1.22354 | True      |
| slane  |     1.22138 | False     |
| taler  |     1.22117 | False     |
| crate  |     1.22073 | True      |
| ariel  |     1.22009 | False     |
#### Overall Frequency

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

#### Best Second Pick
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
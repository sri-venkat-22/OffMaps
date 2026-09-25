**Validation drives, the shipped models: 1 km, the PS benchmark** (13 runs, 1000 m without GNSS at 64 km/h median)

| stage | exit error, median | p90 | drift, median | under 10 % | under 100 m | same stretches on their own road: drift, median |
|---|--:|--:|--:|--:|--:|--:|
| Gyro heading + entry speed (no AI) | 88 m | 172 m | 8.8 % | 54 % | 54 % | 8.6 % |
| + SpeedNet (AI speed) | 224 m | 322 m | 22.4 % | 15 % | 15 % | 19.1 % |
| + learned fusion head | 181 m | 266 m | 18.1 % | 23 % | 23 % | 18.8 % |
| + HMM road matching (shipped) | 184 m | 286 m | 18.4 % | 23 % | 23 % | 19.0 % |

**Validation drives, the shipped models: the underpass itself** (67 runs, 358 m without GNSS at 62 km/h median)

| stage | exit error, median | p90 | drift, median | under 10 % | under 100 m | same stretches on their own road: drift, median |
|---|--:|--:|--:|--:|--:|--:|
| Gyro heading + entry speed (no AI) | 32 m | 101 m | 9.0 % | 55 % | 90 % | 9.5 % |
| + SpeedNet (AI speed) | 73 m | 128 m | 20.0 % | 22 % | 76 % | 17.8 % |
| + learned fusion head | 60 m | 111 m | 17.0 % | 19 % | 84 % | 16.3 % |
| + HMM road matching (shipped) | 41 m | 108 m | 11.7 % | 40 % | 87 % | 12.9 % |

**Train drives, each run with models that never saw it: 1 km, the PS benchmark** (110 runs, 1000 m without GNSS at 61 km/h median)

| stage | exit error, median | p90 | drift, median | under 10 % | under 100 m | same stretches on their own road: drift, median |
|---|--:|--:|--:|--:|--:|--:|
| Gyro heading + entry speed (no AI) | 137 m | 359 m | 13.7 % | 35 % | 35 % | 13.8 % |
| + SpeedNet (AI speed) | 250 m | 431 m | 25.0 % | 21 % | 21 % | 25.2 % |
| + learned fusion head | 167 m | 283 m | 16.7 % | 17 % | 17 % | 13.3 % |
| + HMM road matching (shipped) | 163 m | 288 m | 16.3 % | 24 % | 24 % | 12.6 % |

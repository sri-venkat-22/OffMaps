**Train drives, leave-one-drive-out (~9.5 h; every drive run with models that never saw it)**

| stage | 10 s | 30 s | 60 s | 120 s | mean | < 10 % (30 / 60 s) |
|---|--:|--:|--:|--:|--:|--:|
| Gyro heading + entry speed (no AI) | 12.9 | 22.3 | 28.2 | 33.3 | **24.2** | 21 / 14 % |
| + SpeedNet (AI speed) | 12.9 | 19.6 | 19.4 | 18.9 | **17.7** | 19 / 21 % |
| + learned fusion head (shipped) | 11.7 | 12.5 | 14.1 | 15.9 | **13.6** | 33 / 35 % |
| + HMM road matching | 11.2 | 10.1 | 11.5 | 14.0 | **11.7** | 49 / 47 % |

Outages per duration: 392 / 365 / 338 / 294. Median drift % of distance.

Paired step, outage by outage (median change in points [90 % bootstrap CI], share of outages better):

- + SpeedNet (AI speed): 10 s +0.0 [+0.0, +0.0], 0 %; 30 s -0.8 [-3.2, +1.2], 51 %; 60 s -7.3 [-9.6, -4.2], 65 %; 120 s -11.2 [-13.5, -7.7], 70 %
- + learned fusion head (shipped): 10 s -1.4 [-2.1, -0.8], 59 %; 30 s -4.9 [-5.9, -3.5], 72 %; 60 s -2.5 [-3.4, -1.3], 62 %; 120 s -1.7 [-2.3, -1.0], 62 %
- + HMM road matching: 10 s +0.0 [-0.0, +0.1], 48 %; 30 s -2.3 [-2.5, -1.5], 72 %; 60 s -1.1 [-1.6, -0.8], 66 %; 120 s -0.5 [-0.8, -0.3], 64 %

**Validation drives (2 drives, ~1 h)**

| stage | 10 s | 30 s | 60 s | 120 s | mean | < 10 % (30 / 60 s) |
|---|--:|--:|--:|--:|--:|--:|
| Gyro heading + entry speed (no AI) | 9.7 | 19.0 | 23.2 | 19.4 | **17.8** | 36 / 28 % |
| + SpeedNet (AI speed) | 9.7 | 19.4 | 14.2 | 11.9 | **13.8** | 30 / 33 % |
| + learned fusion head (shipped) | 14.0 | 13.2 | 16.4 | 16.2 | **15.0** | 32 / 26 % |
| + HMM road matching | 14.0 | 13.0 | 13.5 | 16.4 | **14.2** | 30 / 28 % |

Outages per duration: 50 / 44 / 43 / 33. Median drift % of distance.

Paired step, outage by outage (median change in points [90 % bootstrap CI], share of outages better):

- + SpeedNet (AI speed): 10 s +0.0 [+0.0, +0.0], 0 %; 30 s +1.3 [-4.1, +7.2], 43 %; 60 s -3.2 [-14.1, +1.7], 58 %; 120 s -5.1 [-12.9, -0.6], 67 %
- + learned fusion head (shipped): 10 s -0.2 [-2.0, +3.3], 52 %; 30 s -3.6 [-12.2, +0.6], 61 %; 60 s -0.4 [-6.4, +2.8], 51 %; 120 s -0.1 [-4.1, +3.7], 52 %
- + HMM road matching: 10 s -0.1 [-0.8, +0.0], 58 %; 30 s -0.1 [-0.4, +0.2], 55 %; 60 s -0.0 [-0.5, +0.0], 53 %; 120 s -0.2 [-0.6, +0.0], 64 %

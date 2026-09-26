# Research behind each design choice

Each OffMaps component, the published idea it rests on, where it lives in the code, and
what it measured on real drives. Where we depart from the paper, the entry says so.
Numbers are median drift, % of distance driven without GNSS, from `py/ablation.py`
unless stated otherwise.

| Component | What OffMaps does | Built on | Code | Measured |
|---|---|---|---|---|
| Filter | Planar error-state Kalman filter, state (e, n, ψ, v, b_g); GNSS, speed, road and heading as scalar updates | Error-state filtering [1]; strapdown / aided-INS practice [2] | `core/eskf.cpp` | C++ equals its Python oracle to 1e-9 (`py/tests/`) |
| Updates that must not touch some states | A road fix must not change speed or gyro bias, so those gains are forced to 0 and P uses the Joseph form, valid for any gain | Joseph-form covariance update [3] | `core/eskf.cpp` (`update_skip`) | Before the keep mask, a cross-track innovation leaked into speed and hurt real outages (`py/tune_map.py`) |
| Non-holonomic constraint | Velocity lies along the heading by construction (planar model); an explicit NHC update in the 3D 16-state filter | Vehicle-model constraints for land vehicles [4] | `core/idr.h`, `core/eskf3d.cpp` | Heading is not the error: at outage end the error is mostly along-track (`py/phase9_map_eval.py`) |
| AI speed | SpeedNet reads speed and its σ from 2 s of accelerometer + gyro, replacing the missing odometer | Learned pseudo-measurements with learned noise for IMU dead reckoning [5] | `py/model/tcn.py`, `nav/SpeedNet.kt` | 24.2 → 17.7 % on train drives; −7 to −11 points per outage at 60–120 s |
| SpeedNet architecture | Dilated temporal convolutions (1–16) + a GRU + three heads | TCN [6]; GRU [7] | `py/model/tcn.py` | ~180 k parameters, a 620 KB ONNX model on the phone |
| Speed uncertainty | The net outputs a mean and a log-variance, trained by Gaussian NLL | Heteroscedastic regression [8, 9] | `py/model/loss.py` | σ is fused as the measurement noise |
| Calibrated σ and p(stopped) | σ: affine and variance scaling fitted once on held-out drives. p(stopped): Platt scaling of the motion head, fitted on held-out outputs | Calibration of neural networks [10]; calibrated regression [11]; Platt scaling [12] | `py/model/nn_model.py` (`apply_calib`), `py/model/pstop.py` | p(stopped) calibration error 0.020 → 0.012, cross-fitted by drive |
| Learned fusion head | A GRU ensemble turns SpeedNet's output, the entry speed and the recent GNSS context into the speed measurement *and* its σ: a learned, adaptive R | Adaptive Kalman filtering [13]; learned covariances [5]; β-NLL loss [14] | `py/model/fusion_head.py`, `nav/FusionHead.kt` | 17.7 → 13.6 % on train drives (with the selection caveat in the README) |
| Doppler self-calibration | While GNSS is healthy, fit v_true = k·v_nn + c by Deming regression, because both variables are noisy | Errors-in-variables regression [15]; regression dilution [16] | `core/speed_cal.cpp` | Ordinary least squares diluted k; Deming recovers it |
| Map matching | Online HMM on the directed OSM segment graph: distance and heading emissions, route-distance transitions by bounded Dijkstra; a road update only on a confident match, a tight filter and a passing χ² gate | HMM map matching [17]; Dijkstra [18]; innovation gating [19] | `py/road_hmm.py`, `nav/RoadHmm.kt` | 13.6 → 11.7 % on train drives; greedy snapping made it worse (`LESSONS.md` item 8) |
| GNSS handling | No mode switch: one trust score (C/N₀, satellites, DOP, χ² innovation, NavIC) scales R continuously | Innovation-based consistency checks [19] | `core/gnss_quality.cpp` | ≤ 1.7 m error 10 s after re-lock (`py/phase6_check.py`) |
| Spoof check | Strong signal + an INS that strongly disagrees (position χ²) means spoofing | INS monitors for GNSS spoofing [20] | `core/gnss_quality.cpp` | Flagged at onset, zero false positives in `py/phase4_gates.py` gate 3 |
| Re-mount detection | CUSUM on the angle between 5 s and 30 s gravity estimates | CUSUM change detection [21] | `core/align.cpp` | ~1 false alarm per hour; a 30° re-mount found in 3–8 s |
| Potholes and bumps | High-passed accelerometer at 200–400 Hz, an adaptive shock test, then a road-hazard filter: vertical jolt ≥ 1 g, 1 s refractory, 30 m merge; shocks are cut out of the vibration RMS | Pothole Patrol [22]; smartphone Z-threshold detectors [23] | `core/vib.cpp`, `nav/FusionEngine.kt` | Real Redmi drives: 27.3 → 1.7 marks per km (`py/tests/test_pothole_filter.py`) |
| Stops | Strict stop detector (low speed for 3 s and a still gyro) then ZUPT + ZARU; off by default | Zero-velocity updates [24] | `py/edge_engine.py`, `nav/FusionEngine.kt` | A naive ZUPT tripled drift (11.3 → 33.6 % on validation) |
| Heading on two-wheelers | Yaw rate about true vertical; the lean angle comes from φ = atan(v·ψ̇/g) and is compensated | Coordinated-turn kinematics; heading from inertial and magnetic sensors [2] | `py/heading_aids.py`, `nav/HeadingAids.kt` | 5.4° median heading error at 31° lean, vs 14.3° without compensation |
| Parked start | Tilt-compensated magnetometer heading seeds the filter (σ 20°); never fused continuously, because a car body distorts the field | Magnetic heading and its disturbances [2] | `py/heading_aids.py` | 17° median seed error vs 82° without it |
| Evaluation | Leave-one-drive-out with out-of-fold models; paired per-outage comparisons with bootstrap intervals; fixed outage windows; test looks counted | Selection bias in model evaluation [25]; bootstrap [26]; dataset [27] | `py/ablation.py`, `REALDATA.md` | See `README.md` §Results and `LESSONS.md` |

## References

1. J. Solà, "Quaternion kinematics for the error-state Kalman filter", arXiv:1711.02508, 2017.
2. P. D. Groves, *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, 2nd ed., Artech House, 2013.
3. R. S. Bucy and P. D. Joseph, *Filtering for Stochastic Processes with Applications to Guidance*, Interscience, 1968.
4. G. Dissanayake, S. Sukkarieh, E. Nebot and H. Durrant-Whyte, "The aiding of a low-cost strapdown inertial measurement unit using vehicle model constraints for land vehicle applications", *IEEE Transactions on Robotics and Automation*, 17(5), 2001.
5. M. Brossard, A. Barrau and S. Bonnabel, "AI-IMU Dead-Reckoning", *IEEE Transactions on Intelligent Vehicles*, 2020 (arXiv:1904.06064).
6. S. Bai, J. Z. Kolter and V. Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling", arXiv:1803.01271, 2018.
7. K. Cho et al., "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation", EMNLP 2014.
8. D. A. Nix and A. S. Weigend, "Estimating the mean and variance of the target probability distribution", IEEE ICNN 1994.
9. A. Kendall and Y. Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NeurIPS 2017.
10. C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, "On Calibration of Modern Neural Networks", ICML 2017.
11. V. Kuleshov, N. Fenner and S. Ermon, "Accurate Uncertainties for Deep Learning Using Calibrated Regression", ICML 2018.
12. J. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods", *Advances in Large Margin Classifiers*, MIT Press, 1999.
13. R. K. Mehra, "On the identification of variances and adaptive Kalman filtering", *IEEE Transactions on Automatic Control*, 15(2), 1970.
14. M. Seitzer, A. Tavakoli, D. Antic and G. Martius, "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks", ICLR 2022.
15. W. E. Deming, *Statistical Adjustment of Data*, Wiley, 1943.
16. C. Frost and S. G. Thompson, "Correcting for regression dilution bias: comparison of methods for a single predictor variable", *Journal of the Royal Statistical Society A*, 163(2), 2000.
17. P. Newson and J. Krumm, "Hidden Markov Map Matching Through Noise and Sparseness", ACM SIGSPATIAL GIS 2009.
18. E. W. Dijkstra, "A note on two problems in connexion with graphs", *Numerische Mathematik*, 1, 1959.
19. Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley, 2001.
20. Ç. Tanıl, S. Khanafseh, M. Joerger and B. Pervan, "An INS Monitor to Detect GNSS Spoofers Capable of Tracking Vehicle Position", *IEEE Transactions on Aerospace and Electronic Systems*, 2018.
21. E. S. Page, "Continuous Inspection Schemes", *Biometrika*, 41(1/2), 1954.
22. J. Eriksson, L. Girod, B. Hull, R. Newton, S. Madden and H. Balakrishnan, "The Pothole Patrol: Using a Mobile Sensor Network for Road Surface Monitoring", ACM MobiSys 2008.
23. A. Mednis, G. Strazdins, R. Zviedris, G. Kanonirs and L. Selavo, "Real time pothole detection using Android smartphones with accelerometers", IEEE DCOSS 2011.
24. E. Foxlin, "Pedestrian Tracking with Shoe-Mounted Inertial Sensors", *IEEE Computer Graphics and Applications*, 25(6), 2005.
25. G. C. Cawley and N. L. C. Talbot, "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation", *Journal of Machine Learning Research*, 11, 2010.
26. B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.
27. U. Onyekpe, V. Palade, S. Kanarachos and S.-R. G. Christopoulos, "IO-VNBD: Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning", *Data in Brief*, 2021.

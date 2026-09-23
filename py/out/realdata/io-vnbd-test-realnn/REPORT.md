# Real-data validation — S (Driver A)/S3a/S-S3a#0, Y (Driver D)/Y1/S-Y1#3

**REPORTABLE**  ·  2026-09-23 11:52:49

Dead-reckoning numbers on the **loaded drive**, distinct from the repo's synthetic pipeline-validation numbers. Drift % is against the ISRO metric (`--dist-source speed`); the 10 % line is the PS limit.

## Provenance

- source: `~/OffMaps-data/IO-VNBD-sync`  (dataset dir)
- manifest: `out/realdata/io-vnbd-test-synthnn/manifest.json`  (replayed (frozen))
- seed: 0  ·  windows/duration requested: 30  ·  min-dist: 20.0 m
- reproduce: `python -m validate_realdata --layout iovnbd-sync --data ~/OffMaps-data/IO-VNBD-sync --include /Y1/|/S3a/ --n-per-duration 30 --manifest out/realdata/io-vnbd-test-synthnn/manifest.json --nn-ckpt model/nn_real.pt --out out/realdata/io-vnbd-test-realnn`
- env: python 3.11.9 · numpy 2.4.1 · pandas 3.0.5 · macOS-27.0-arm64-arm-64bit

| input file | sha256 (first 16) | bytes |
|---|---|--:|
| `M (Driver B)/S-M.csv` | `26a1ee080da4db98` | 19,798,721 |
| `M (Driver B)/V-M.csv` | `ecf977cb798082d5` | 22,507,823 |
| `S (Driver A)/S1/S-S1.csv` | `e79a2eea18143b82` | 9,631,499 |
| `S (Driver A)/S1/V-S1.csv` | `29e92ed9bcb2d711` | 10,967,129 |
| `S (Driver A)/S2/S-S2.csv` | `8f48d68ef4ab225c` | 17,469,302 |
| `S (Driver A)/S2/V-S2.csv` | `0514c4dd38fe24dd` | 20,091,681 |
| `S (Driver A)/S3a/S-S3a.csv` | `b9e45dc8d856271f` | 4,580,130 |
| `S (Driver A)/S3a/V-S3a.csv` | `1cac1c8045c4f854` | 5,286,764 |
| `S (Driver A)/S3b/S-S3b.csv` | `19447009a96bd80b` | 1,253,565 |
| `S (Driver A)/S3b/V-S3b.csv` | `d44d29d46b7c46c5` | 1,464,196 |
| `S (Driver A)/S3c/S-S3c.csv` | `b87fa6978d7d5008` | 6,881,631 |
| `S (Driver A)/S3c/V-S3c.csv` | `d7a98f02899d3312` | 7,976,092 |
| `S (Driver A)/S4/S-S4.csv` | `a2bb1e1eb28d111f` | 17,574,514 |
| `S (Driver A)/S4/V-S4.csv` | `8a83103ce0f0b03f` | 20,164,189 |
| `Vf (Driver E)/V-Vfa01/S-Vfa01.csv` | `7f46f770cd49c851` | 2,134,667 |
| `Vf (Driver E)/V-Vfa01/V-Vfa01.csv` | `81ad22765340419e` | 2,398,099 |
| `Vf (Driver E)/V-Vfa02/S-Vfa02.csv` | `ec666af90866254a` | 12,603,657 |
| `Vf (Driver E)/V-Vfa02/V-Vfa02.csv` | `bce837348c306ba3` | 13,955,001 |
| `Vta (Driver E)/Vta01a/S-Vta1a.csv` | `19b9d26888e58c71` | 4,766,740 |
| `Vta (Driver E)/Vta01a/V-Vta1a.csv` | `9522a6f43094ef5d` | 5,444,513 |
| `Vta (Driver E)/Vta01b/S-Vta1b.csv` | `aca49109fc93729d` | 178,260 |
| `Vta (Driver E)/Vta01b/V-Vta1b.csv` | `35b4e74bd1597d9f` | 205,277 |
| `Vta (Driver E)/Vta02/S-Vta2.csv` | `97ef388d6e5cc3ce` | 2,038,713 |
| `Vta (Driver E)/Vta02/V-vta2.csv` | `e7d255100464c702` | 2,307,461 |
| `Vta (Driver E)/Vta03/S-Vta3.csv` | `7c5479fc893131a8` | 116,809 |
| `Vta (Driver E)/Vta03/V-vta3.csv` | `ef8bf81fdf5df786` | 137,891 |
| `Vta (Driver E)/Vta04/S-Vta4.csv` | `0d16cd6deaddf30a` | 331,302 |
| `Vta (Driver E)/Vta04/V-vta4.csv` | `71acb4e5c50919df` | 380,874 |
| `Vta (Driver E)/Vta05/S-Vta5.csv` | `b9030167541be560` | 57,485 |
| `Vta (Driver E)/Vta05/V-vta5.csv` | `6819d97f1291b377` | 66,401 |
| `Vta (Driver E)/Vta06/S-Vta6.csv` | `15ad54e093b7ff78` | 254,665 |
| `Vta (Driver E)/Vta06/V-vta6.csv` | `728629ca2a30457e` | 289,249 |
| `Vta (Driver E)/Vta07/S-Vta7.csv` | `eb6e66dbb3397aea` | 155,360 |
| `Vta (Driver E)/Vta07/V-vta7.csv` | `031765759d3336fa` | 179,784 |
| `Vta (Driver E)/Vta08/S-Vta8.csv` | `cc1b57b0b539f1e7` | 678,900 |
| `Vta (Driver E)/Vta08/V-vta8.csv` | `93cef7816fb58c25` | 753,939 |
| `Vta (Driver E)/Vta09/S-Vta9.csv` | `150b73335a4283e2` | 29,554 |
| `Vta (Driver E)/Vta09/V-vta9.csv` | `4f662af06a2ab067` | 33,110 |
| `Vta (Driver E)/Vta10/S-Vta10.csv` | `212ba000dbad28b3` | 279,232 |
| `Vta (Driver E)/Vta10/V-vta10.csv` | `6f9911ce3037bf56` | 317,921 |
| `Vta (Driver E)/Vta11/S-Vta11.csv` | `af147288035d510d` | 95,272 |
| `Vta (Driver E)/Vta11/V-vta11.csv` | `e9d6bf8959e32ea8` | 111,690 |
| `Vta (Driver E)/Vta12/S-Vta12.csv` | `9dc5cfe05e00e91f` | 114,418 |
| `Vta (Driver E)/Vta12/V-vta12.csv` | `21c4f30690769a05` | 128,904 |
| `Vta (Driver E)/Vta13/S-Vta13.csv` | `b4a8f30e4fa8c139` | 76,217 |
| `Vta (Driver E)/Vta13/V-vta13.csv` | `398df644de6920ef` | 86,963 |
| `Vta (Driver E)/Vta14/S-Vta14.csv` | `49535f49178bb6c5` | 540,523 |
| `Vta (Driver E)/Vta14/V-vta14.csv` | `ddd26e084b97fe66` | 612,352 |
| `Vta (Driver E)/Vta15/S-Vta15.csv` | `9404fbd7420b32f5` | 156,907 |
| `Vta (Driver E)/Vta15/V-vta15.csv` | `313565b6b7a44f7e` | 175,132 |
| `Vta (Driver E)/Vta16/S-Vta16.csv` | `efef0aa0fae9e4cc` | 2,111,083 |
| `Vta (Driver E)/Vta16/V-vta16.csv` | `07cc3f87b51c3e68` | 2,410,634 |
| `Vta (Driver E)/Vta17/S-Vta17.csv` | `d918e507f6cf2b0c` | 839,635 |
| `Vta (Driver E)/Vta17/V-vta17.csv` | `21dbfe17ff301ac6` | 972,437 |
| `Vta (Driver E)/Vta19/S-Vta19.csv` | `695cf51819c0f0f8` | 54,578 |
| `Vta (Driver E)/Vta19/V-vta19.csv` | `e5e7fb443b7a92a2` | 61,675 |
| `Vta (Driver E)/Vta20/S-Vta20.csv` | `82c7e730c3aefbc0` | 582,891 |
| `Vta (Driver E)/Vta20/V-vta20.csv` | `a5e1f1e6afba77c6` | 596,090 |
| `Vta (Driver E)/Vta21/S-Vta21.csv` | `f22951b5bc4fed96` | 388,495 |
| `Vta (Driver E)/Vta21/V-vta21.csv` | `82687cefc4799459` | 450,180 |
| `Vta (Driver E)/Vta22/S-Vta22.csv` | `9291970633a370e9` | 290,331 |
| `Vta (Driver E)/Vta22/V-vta22.csv` | `ddd4bb7ffd3aaf08` | 338,739 |
| `Vta (Driver E)/Vta23/S-Vta23.csv` | `c5f9f4161bd33ec8` | 206,507 |
| `Vta (Driver E)/Vta23/V-vta23.csv` | `8b3508e408d88064` | 242,444 |
| `Vta (Driver E)/Vta24/S-Vta24.csv` | `ae6059b51b52fd21` | 217,212 |
| `Vta (Driver E)/Vta24/V-vta24.csv` | `db3726bc5e11c73f` | 250,448 |
| `Vta (Driver E)/Vta25/S-Vta25.csv` | `9ff0c54126af8ccb` | 118,562 |
| `Vta (Driver E)/Vta25/V-vta25.csv` | `4241b50add001a19` | 135,915 |
| `Vta (Driver E)/Vta26/S-Vta26.csv` | `d2d0de4a8ff27938` | 358,553 |
| `Vta (Driver E)/Vta26/V-vta26.csv` | `e1067b6fd8561c4f` | 401,896 |
| `Vta (Driver E)/Vta27/S-Vta27.csv` | `9607f8bbfcb7e920` | 471,023 |
| `Vta (Driver E)/Vta27/V-vta27.csv` | `766e63d6975426dc` | 550,395 |
| `Vta (Driver E)/Vta28/S-Vta28.csv` | `30bece8d988e8989` | 781,902 |
| `Vta (Driver E)/Vta28/V-vta28.csv` | `4778fd71aeb0165a` | 901,238 |
| `Vta (Driver E)/Vta29/S-Vta29.csv` | `dc24cf9cd6b8cce2` | 4,387,027 |
| `Vta (Driver E)/Vta29/V-vta29.csv` | `6c6fbb29e6400817` | 5,117,097 |
| `Vta (Driver E)/Vta30/S-Vta30.csv` | `d6f65e16288c2fd1` | 3,175,640 |
| `Vta (Driver E)/Vta30/V-vta30.csv` | `d73b5ac746240670` | 3,628,928 |
| `Vtb (Driver E)/Vtb01/S-Vtb1.csv` | `7bdca3360c9411c5` | 6,052,896 |
| `Vtb (Driver E)/Vtb01/V-vtb1.csv` | `8ab039c67c83f62e` | 6,989,783 |
| `Vtb (Driver E)/Vtb02/S-Vtb2.csv` | `6552ab344fe50a01` | 1,053,029 |
| `Vtb (Driver E)/Vtb02/V-vtb2.csv` | `4a194e6885091663` | 1,200,565 |
| `Vtb (Driver E)/Vtb03/S-Vtb3.csv` | `7cadb0617ec69513` | 1,530,275 |
| `Vtb (Driver E)/Vtb03/V-vtb3.csv` | `17e9cc28a2a6b290` | 1,739,502 |
| `Vtb (Driver E)/Vtb04/S-Vtb4.csv` | `cdc2626adc124083` | 103,090 |
| `Vtb (Driver E)/Vtb04/V-vtb4.csv` | `b5aac0532f3eb638` | 118,374 |
| `Vtb (Driver E)/Vtb05/S-Vtb5.csv` | `4753710832962b5c` | 12,010,333 |
| `Vtb (Driver E)/Vtb05/V-vtb5.csv` | `a4aac15f70c4555e` | 13,567,208 |
| `Vtb (Driver E)/Vtb06/S-Vtb6.csv` | `8ab4e1f0373173eb` | 94,210 |
| `Vtb (Driver E)/Vtb06/V-vtb6.csv` | `3f55326e614a3777` | 105,659 |
| `Vtb (Driver E)/Vtb07/S-Vtb7.csv` | `b798d898dc371334` | 87,349 |
| `Vtb (Driver E)/Vtb07/V-vtb7.csv` | `d2a5d2206394b29e` | 101,086 |
| `Vtb (Driver E)/Vtb08/S-Vtb8.csv` | `cdd40e3a04fe2f39` | 126,023 |
| `Vtb (Driver E)/Vtb08/V-vtb8.csv` | `0f11eea4a2c183a7` | 138,162 |
| `Vtb (Driver E)/Vtb09/S-Vtb9.csv` | `0a9dcbe6816e28e6` | 85,640 |
| `Vtb (Driver E)/Vtb09/V-vtb9.csv` | `ce69ae88e13086a3` | 93,354 |
| `Vtb (Driver E)/Vtb10/S-Vtb10.csv` | `b81e128fd1594a07` | 37,176 |
| `Vtb (Driver E)/Vtb10/V-vtb10.csv` | `5b0fe6e9f967fe9e` | 42,725 |
| `Vtb (Driver E)/Vtb11/S-Vtb11.csv` | `cd3ec5fb7285089c` | 68,659 |
| `Vtb (Driver E)/Vtb11/V-vtb11.csv` | `e611eb46d62fb695` | 74,838 |
| `Vtb (Driver E)/Vtb12/S-Vtb12.csv` | `00e8dc12482919e0` | 83,892 |
| `Vtb (Driver E)/Vtb12/V-vtb12.csv` | `dad39fa7ff152dc1` | 94,747 |
| `Vw (Driver E)/Vw01/S-Vw1.csv` | `41fb71ae3223748b` | 3,524,952 |
| `Vw (Driver E)/Vw01/V-Vw1.csv` | `125a3ab348bca6b5` | 4,001,856 |
| `Vw (Driver E)/Vw02/S-Vw2.csv` | `c153268bf5f53e17` | 9,883,172 |
| `Vw (Driver E)/Vw02/V-Vw2.csv` | `6aeca8956af02a64` | 11,341,576 |
| `Vw (Driver E)/Vw03/S-Vw3.csv` | `6fa0f3d88a6c7c80` | 720,455 |
| `Vw (Driver E)/Vw03/V-Vw3.csv` | `b89ca7c566b668ab` | 839,545 |
| `Vw (Driver E)/Vw04/S-Vw4.csv` | `0d79ef3cd96f48b6` | 23,607,418 |
| `Vw (Driver E)/Vw04/V-Vw4.csv` | `db9ba7d930842b5f` | 27,124,526 |
| `Vw (Driver E)/Vw05/S-Vw5.csv` | `0750e1ddccb84440` | 188,883 |
| `Vw (Driver E)/Vw05/V-Vw5.csv` | `a9972e12e8ea995a` | 216,316 |
| `Vw (Driver E)/Vw06/S-Vw6.csv` | `9543a52492106a1e` | 238,315 |
| `Vw (Driver E)/Vw06/V-Vw6.csv` | `2399eafd439216c3` | 278,191 |
| `Vw (Driver E)/Vw07/S-Vw7.csv` | `953176680614bf88` | 300,843 |
| `Vw (Driver E)/Vw07/V-Vw7.csv` | `3ba9d21d1532b4e1` | 346,895 |
| `Vw (Driver E)/Vw08/S-Vw8.csv` | `f1321e689b66128e` | 287,123 |
| `Vw (Driver E)/Vw08/V-Vw8.csv` | `0b08f02c7568c47b` | 332,609 |
| `Vw (Driver E)/Vw09/S-Vw9.csv` | `40d96f99b100518c` | 103,572 |
| `Vw (Driver E)/Vw09/V-Vw9.csv` | `ec3f3b288fe0793b` | 121,383 |
| `Vw (Driver E)/Vw10/S-Vw10.csv` | `b2aa8e3d7bae40e9` | 123,092 |
| `Vw (Driver E)/Vw10/V-Vw10.csv` | `acec110be8eb7ef4` | 141,457 |
| `Vw (Driver E)/Vw11/S-Vw11.csv` | `b9dd70fc7282ba46` | 924,189 |
| `Vw (Driver E)/Vw11/V-Vw11.csv` | `5e50658cbdd33fc9` | 1,032,515 |
| `Vw (Driver E)/Vw12/S-Vw12.csv` | `e5598004be4250cb` | 170,958 |
| `Vw (Driver E)/Vw12/V-Vw12.csv` | `f474667235581e51` | 188,273 |
| `Vw (Driver E)/Vw13/S-Vw13.csv` | `8dbdb8d7c3c0790c` | 53,041 |
| `Vw (Driver E)/Vw13/V-Vw13.csv` | `2b5e7012d2fce268` | 59,207 |
| `Vw (Driver E)/Vw14a/S-Vw14a.csv` | `de5b5da05baa7bed` | 585,872 |
| `Vw (Driver E)/Vw14a/V-Vw14a.csv` | `442bef4ccc4464f0` | 651,054 |
| `Vw (Driver E)/Vw14b/S-Vw14b.csv` | `5f13f8ede6117bb2` | 3,668,924 |
| `Vw (Driver E)/Vw14b/V-Vw14b.csv` | `ee81d26af18170ab` | 4,120,915 |
| `Vw (Driver E)/Vw14c/S-Vw14c.csv` | `ce97c12ec0c9cff7` | 2,957,471 |
| `Vw (Driver E)/Vw14c/V-Vw14c.csv` | `1b3b6d1e2e4d7c89` | 3,363,425 |
| `Vw (Driver E)/Vw15/S-Vw15.csv` | `7e440cc28168a538` | 254,039 |
| `Vw (Driver E)/Vw15/V-Vw15.csv` | `bd65ae3932fc1b6e` | 263,885 |
| `Vw (Driver E)/Vw16a/S-Vw16a.csv` | `1b25f604e0754544` | 1,100,323 |
| `Vw (Driver E)/Vw16a/V-Vw16a.csv` | `0b92aa13c01d1436` | 1,241,037 |
| `Vw (Driver E)/Vw16b/S-Vw16b.csv` | `817fd5f786beebb8` | 211,892 |
| `Vw (Driver E)/Vw16b/V-Vw16b.csv` | `1e81b217eaf6c234` | 239,979 |
| `Vw (Driver E)/Vw17/S-Vw17.csv` | `cc55c4ce1a945a78` | 62,489 |
| `Vw (Driver E)/Vw17/V-Vw17.csv` | `62ab9ddd16621eb7` | 71,065 |
| `Y (Driver D)/Y1/S-Y1.csv` | `cf17b89f6eec6872` | 13,039,543 |
| `Y (Driver D)/Y1/V-Y1.csv` | `505749fb081d5a2b` | 14,986,075 |

## QC gate

| drive | rows | speed/track ratio | fast n | verdict |
|---|--:|--:|--:|---|
| S (Driver A)/S3a/S-S3a#0 | 24554 | 0.998 | 20116 | PASS |
| Y (Driver D)/Y1/S-Y1#3 | 65542 | 0.999 | 48481 | PASS |
- ⚠ S (Driver A)/S3a/S-S3a#0: sync: tz +1h, skew +0.2s, yaw axis gyro 'Pitch' sign +1, yaw corr 0.996
- ⚠ S (Driver A)/S3a/S-S3a#0: phone accel vs GNSS along-track accel corr 0.38 (weak: accel-driven models are handicapped on this drive)
- ⚠ Y (Driver D)/Y1/S-Y1#3: sync: tz +1h, skew -0.8s, yaw axis gyro 'Pitch' sign +1, yaw corr 0.961
- ⚠ Y (Driver D)/Y1/S-Y1#3: phone accel vs GNSS along-track accel corr 0.02 (weak: accel-driven models are handicapped on this drive)

<details><summary>72 input segment(s) rejected by the loader (not scored)</summary>

- S (Driver A)/S3b/S-S3b#0: phone-yaw vs ECU-yaw corr 0.77 < 0.8 (skew +0.3s) -- phone not rigidly mounted / unsyncable
- S (Driver A)/S4/S-S4#2: no overlapping vehicle rows
- Vf (Driver E)/V-Vfa01/S-Vfa01#0: phone-yaw vs ECU-yaw corr 0.61 < 0.8 (skew -1.1s) -- phone not rigidly mounted / unsyncable
- Vf (Driver E)/V-Vfa02/S-Vfa02#0: phone-yaw vs ECU-yaw corr 0.47 < 0.8 (skew -1.2s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta01a/S-Vta1a#0: phone-yaw vs ECU-yaw corr 0.68 < 0.8 (skew +0.2s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta01b/S-Vta1b: 1 phone segment(s) shorter than 120s
- Vta (Driver E)/Vta02/S-Vta2#0: phone-yaw vs ECU-yaw corr 0.77 < 0.8 (skew +0.2s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta03/S-Vta3: 1 phone segment(s) shorter than 120s
- Vta (Driver E)/Vta04/S-Vta4#0: phone-yaw vs ECU-yaw corr 0.43 < 0.8 (skew +0.4s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta05/S-Vta5: 1 phone segment(s) shorter than 120s
- Vta (Driver E)/Vta06/S-Vta6#0: 100s overlap after sync
- Vta (Driver E)/Vta07/S-Vta7: 1 phone segment(s) shorter than 120s
- Vta (Driver E)/Vta08/S-Vta8#0: phone-yaw vs ECU-yaw corr 0.48 < 0.8 (skew -0.0s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta09/S-Vta9: 1 phone segment(s) shorter than 120s
- Vta (Driver E)/Vta10/S-Vta10#0: phone-yaw vs ECU-yaw corr 0.45 < 0.8 (skew +0.3s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta11/S-Vta11: 1 phone segment(s) shorter than 120s
- Vta (Driver E)/Vta12/S-Vta12: 1 phone segment(s) shorter than 120s
- Vta (Driver E)/Vta13/S-Vta13: 1 phone segment(s) shorter than 120s
- Vta (Driver E)/Vta14/S-Vta14#0: phone-yaw vs ECU-yaw corr 0.21 < 0.8 (skew +0.1s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta15/S-Vta15: 1 phone segment(s) shorter than 120s
- Vta (Driver E)/Vta16/S-Vta16#0: phone-yaw vs ECU-yaw corr 0.62 < 0.8 (skew +0.1s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta17/S-Vta17#1: phone-yaw vs ECU-yaw corr 0.51 < 0.8 (skew -0.0s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta17/S-Vta17: 1 phone segment(s) shorter than 120s
- Vta (Driver E)/Vta19/S-Vta19: 1 phone segment(s) shorter than 120s
- Vta (Driver E)/Vta20/S-Vta20#0: phone-yaw vs ECU-yaw corr 0.65 < 0.8 (skew -1.1s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta21/S-Vta21#0: phone-yaw vs ECU-yaw corr 0.54 < 0.8 (skew -1.3s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta22/S-Vta22#0: phone-yaw vs ECU-yaw corr 0.59 < 0.8 (skew -1.0s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta23/S-Vta23: 1 phone segment(s) shorter than 120s
- Vta (Driver E)/Vta24/S-Vta24: 1 phone segment(s) shorter than 120s
- Vta (Driver E)/Vta25/S-Vta25: 1 phone segment(s) shorter than 120s
- Vta (Driver E)/Vta26/S-Vta26#0: phone-yaw vs ECU-yaw corr 0.27 < 0.8 (skew -1.6s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta27/S-Vta27#0: phone-yaw vs ECU-yaw corr 0.23 < 0.8 (skew -1.2s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta28/S-Vta28#0: phone-yaw vs ECU-yaw corr 0.52 < 0.8 (skew -1.0s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta29/S-Vta29#0: phone-yaw vs ECU-yaw corr 0.48 < 0.8 (skew -1.3s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta29/S-Vta29#4: phone-yaw vs ECU-yaw corr 0.37 < 0.8 (skew -1.4s) -- phone not rigidly mounted / unsyncable
- Vta (Driver E)/Vta29/S-Vta29: 3 phone segment(s) shorter than 120s
- Vta (Driver E)/Vta30/S-Vta30#0: phone-yaw vs ECU-yaw corr 0.52 < 0.8 (skew -1.3s) -- phone not rigidly mounted / unsyncable
- Vtb (Driver E)/Vtb01/S-Vtb1#5457: phone-yaw vs ECU-yaw corr 0.38 < 0.8 (skew -1.7s) -- phone not rigidly mounted / unsyncable
- Vtb (Driver E)/Vtb01/S-Vtb1: 5457 phone segment(s) shorter than 120s
- Vtb (Driver E)/Vtb02/S-Vtb2#0: phone-yaw vs ECU-yaw corr 0.44 < 0.8 (skew -1.8s) -- phone not rigidly mounted / unsyncable
- Vtb (Driver E)/Vtb03/S-Vtb3#0: phone-yaw vs ECU-yaw corr 0.52 < 0.8 (skew -1.7s) -- phone not rigidly mounted / unsyncable
- Vtb (Driver E)/Vtb04/S-Vtb4: 1 phone segment(s) shorter than 120s
- Vtb (Driver E)/Vtb05/S-Vtb5#0: phone-yaw vs ECU-yaw corr 0.29 < 0.8 (skew -1.8s) -- phone not rigidly mounted / unsyncable
- Vtb (Driver E)/Vtb06/S-Vtb6: 1 phone segment(s) shorter than 120s
- Vtb (Driver E)/Vtb07/S-Vtb7: 1 phone segment(s) shorter than 120s
- Vtb (Driver E)/Vtb08/S-Vtb8: 1 phone segment(s) shorter than 120s
- Vtb (Driver E)/Vtb09/S-Vtb9: 1 phone segment(s) shorter than 120s
- Vtb (Driver E)/Vtb10/S-Vtb10: 1 phone segment(s) shorter than 120s
- Vtb (Driver E)/Vtb11/S-Vtb11: 1 phone segment(s) shorter than 120s
- Vtb (Driver E)/Vtb12/S-Vtb12: 1 phone segment(s) shorter than 120s
- Vw (Driver E)/Vw01/S-Vw1#0: phone-yaw vs ECU-yaw corr 0.21 < 0.8 (skew +53.4s) -- phone not rigidly mounted / unsyncable
- Vw (Driver E)/Vw02/S-Vw2#0: phone-yaw vs ECU-yaw corr 0.47 < 0.8 (skew -0.3s) -- phone not rigidly mounted / unsyncable
- Vw (Driver E)/Vw03/S-Vw3#0: phone-yaw vs ECU-yaw corr 0.62 < 0.8 (skew -0.5s) -- phone not rigidly mounted / unsyncable
- Vw (Driver E)/Vw04/S-Vw4#0: phone-yaw vs ECU-yaw corr 0.48 < 0.8 (skew -0.3s) -- phone not rigidly mounted / unsyncable
- Vw (Driver E)/Vw05/S-Vw5: 1 phone segment(s) shorter than 120s
- Vw (Driver E)/Vw06/S-Vw6#0: phone-yaw vs ECU-yaw corr 0.69 < 0.8 (skew -0.3s) -- phone not rigidly mounted / unsyncable
- Vw (Driver E)/Vw07/S-Vw7#0: no overlapping vehicle rows
- Vw (Driver E)/Vw08/S-Vw8#0: no overlapping vehicle rows
- Vw (Driver E)/Vw09/S-Vw9: 1 phone segment(s) shorter than 120s
- Vw (Driver E)/Vw10/S-Vw10: 1 phone segment(s) shorter than 120s
- Vw (Driver E)/Vw11/S-Vw11#0: phone-yaw vs ECU-yaw corr 0.61 < 0.8 (skew -0.1s) -- phone not rigidly mounted / unsyncable
- Vw (Driver E)/Vw12/S-Vw12: 1 phone segment(s) shorter than 120s
- Vw (Driver E)/Vw13/S-Vw13: 1 phone segment(s) shorter than 120s
- Vw (Driver E)/Vw14a/S-Vw14a#0: phone-yaw vs ECU-yaw corr 0.15 < 0.8 (skew -1.7s) -- phone not rigidly mounted / unsyncable
- Vw (Driver E)/Vw14b/S-Vw14b#0: phone-yaw vs ECU-yaw corr 0.09 < 0.8 (skew -0.4s) -- phone not rigidly mounted / unsyncable
- Vw (Driver E)/Vw14c/S-Vw14c#0: phone-yaw vs ECU-yaw corr 0.43 < 0.8 (skew -0.2s) -- phone not rigidly mounted / unsyncable
- Vw (Driver E)/Vw15/S-Vw15#0: 79s overlap after sync
- Vw (Driver E)/Vw16a/S-Vw16a#0: phone-yaw vs ECU-yaw corr 0.24 < 0.8 (skew -0.2s) -- phone not rigidly mounted / unsyncable
- Vw (Driver E)/Vw16b/S-Vw16b: 1 phone segment(s) shorter than 120s
- Vw (Driver E)/Vw17/S-Vw17: 1 phone segment(s) shorter than 120s
- Y (Driver D)/Y1/S-Y1#2: phone-yaw vs ECU-yaw corr nan < 0.8 (skew +29.9s) -- phone not rigidly mounted / unsyncable
- Y (Driver D)/Y1/S-Y1: 2 phone segment(s) shorter than 120s

</details>

## Results

### `eskf_map`  (n=256 windows)

| dur (s) | n | drift % | CEP50 (m) | CEP95 (m) | along (m) | cross (m) | vMAE (m/s) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 60 | 25.7 | 24.6 | 121.9 | 24.5 | 0.3 | 3.33 |
| 30 | 60 | 34.8 | 85.4 | 312.5 | 80.7 | 4.3 | 3.30 |
| 60 | 58 | 32.2 | 155.2 | 567.9 | 90.0 | 44.6 | 3.46 |
| 120 | 42 | 32.0 | 289.1 | 1254.8 | 179.8 | 81.5 | 3.56 |
| 180 | 36 | 36.6 | 599.6 | 1591.1 | 361.8 | 283.6 | 3.94 |

overall **CEP50 113.3 m · CEP95 1018.0 m**

σ-calibration: z_var=33.03 z_mean=-0.56 (n=174256) → FAIL: z_var 33.03 outside [0.7,1.4]; |z_mean| 0.56 > 0.3 (biased speed)

### `eskf`  (n=256 windows)

| dur (s) | n | drift % | CEP50 (m) | CEP95 (m) | along (m) | cross (m) | vMAE (m/s) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 60 | 22.0 | 22.3 | 114.8 | 21.4 | 2.7 | 3.00 |
| 30 | 60 | 24.3 | 53.3 | 267.6 | 44.9 | 16.2 | 2.81 |
| 60 | 58 | 20.2 | 94.7 | 474.5 | 54.3 | 51.2 | 2.68 |
| 120 | 42 | 17.1 | 174.6 | 730.8 | 123.3 | 49.4 | 2.76 |
| 180 | 36 | 17.3 | 252.6 | 1141.5 | 193.7 | 100.7 | 2.80 |

overall **CEP50 71.7 m · CEP95 540.4 m**

σ-calibration: z_var=0.48 z_mean=-0.34 (n=174256) → FAIL: z_var 0.48 outside [0.7,1.4]; |z_mean| 0.34 > 0.3 (biased speed)

### `nn`  (n=256 windows)

| dur (s) | n | drift % | CEP50 (m) | CEP95 (m) | along (m) | cross (m) | vMAE (m/s) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 60 | 22.7 | 22.3 | 116.5 | 21.8 | 2.7 | 3.01 |
| 30 | 60 | 25.1 | 52.1 | 267.3 | 43.6 | 14.4 | 2.81 |
| 60 | 58 | 19.1 | 93.1 | 477.7 | 54.8 | 50.8 | 2.69 |
| 120 | 42 | 17.3 | 177.4 | 729.9 | 125.2 | 52.2 | 2.77 |
| 180 | 36 | 17.2 | 252.4 | 1140.0 | 194.3 | 102.6 | 2.80 |

overall **CEP50 73.0 m · CEP95 544.7 m**

σ-calibration: z_var=0.48 z_mean=-0.34 (n=174256) → FAIL: z_var 0.48 outside [0.7,1.4]; |z_mean| 0.34 > 0.3 (biased speed)

### `physics`  (n=256 windows)

| dur (s) | n | drift % | CEP50 (m) | CEP95 (m) | along (m) | cross (m) | vMAE (m/s) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 60 | 16.8 | 14.2 | 45.0 | 13.3 | 2.7 | 1.44 |
| 30 | 60 | 22.0 | 57.0 | 285.1 | 37.3 | 20.8 | 2.59 |
| 60 | 58 | 28.8 | 129.9 | 438.9 | 67.9 | 69.0 | 3.36 |
| 120 | 42 | 27.5 | 324.1 | 1033.9 | 196.9 | 162.6 | 3.98 |
| 180 | 36 | 25.2 | 383.4 | 1759.7 | 262.5 | 198.5 | 4.81 |

overall **CEP50 93.2 m · CEP95 929.7 m**

### `cv`  (n=256 windows)

| dur (s) | n | drift % | CEP50 (m) | CEP95 (m) | along (m) | cross (m) | vMAE (m/s) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 60 | 25.6 | 27.1 | 67.0 | 12.5 | 8.7 | 1.44 |
| 30 | 60 | 51.7 | 155.0 | 317.2 | 97.3 | 66.1 | 2.59 |
| 60 | 58 | 56.5 | 262.2 | 841.2 | 147.9 | 171.5 | 3.36 |
| 120 | 42 | 84.9 | 834.8 | 1627.3 | 508.1 | 344.2 | 3.98 |
| 180 | 36 | 91.2 | 1472.4 | 2726.3 | 900.1 | 761.1 | 4.81 |

overall **CEP50 206.7 m · CEP95 1631.8 m**

## Plots

- `drift_vs_duration.png` — median drift % vs outage length (10 % ISRO line)
- `examples_<model>.png` — dead-reckoned vs truth for the longest windows


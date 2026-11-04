# Implementation map

| Paper equations | Where to read |
| --- | --- |
| (1)–(4) | `diagnosis.bank` fits Ledoit–Wolf precision and the RBF bandwidth. `score_all` evaluates each cue per batch. `JointRouter` ranks the scores against clean batches and compares their maximum with a per-modality cutoff. |
| (5) | `GlobalChangeClock` accumulates 64 clips, projects standardized features and forms a mean/log-variance descriptor. `OnlineBoundary` compares 3 recent windows with 16 preceding windows and tracks three consecutive route indications. |
| (6) | `spectrum.correct` operates on RGB in pixel space, preserving the per-frame DC component. |
| (7)–(10) | `OnlineAdapters.step` uses unit mean alignment strength on the detected modality, a frozen stable residual on its original tokens, and an active plastic residual. The stable bottleneck has width 32. |
| (11)–(15) | Sorted clean APS scores set the candidate threshold. Alpha increases from 0.001 to 0.3 over 32 updates with gamma 0.2. Candidate weights use detached probabilities and the preceding modality frequency estimate. |
| (16)–(20) | Both unimodal KL calls use the frozen shared decoder. Entropy reaches both active adapter types. Reverse KL reaches clean stable parameters, CP-PLL reaches corrupted plastic parameters. |

The source backbone and data loader belong to the upstream DASP repository. The bundled calibration files use the audio/video order. The reference file contains token means, RGB spectral power and sorted APS scores. Each router file contains frozen reference features and clean score tables for its batch size. The change detector loads its projection and normalization from change_features.npz.

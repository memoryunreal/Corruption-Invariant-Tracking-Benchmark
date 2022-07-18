# Corruption-Invariant-Tracking-Benchmark (CITB)

This benchmark contains VOT2020ST-C, GOT-10k-C, VOT2020-LT-C, UAV20L-C, and DepthTrack-C datasets.

## Corruptions in Tracking

## Random Corruption Robustness

![image](https://github.com/memoryunreal/Corruption-Invariant-Tracking-Benchmark/blob/main/randomuav.png)

## Corrupted Data Corruption Visualization

![image](https://github.com/memoryunreal/Corruption-Invariant-Tracking-Benchmark/blob/main/visualization.png)

## Statics of our benchmark datasets for single and multiple modality object tracking.
| Dataset                      | Description         | \#Videos | Added Corruptions                               |
|------------------------------|---------------------|----------|-------------------------------------------------|
| VOT2020-ST \cite{vot2020}    | Short-term tracking | 60       | Noise; Blur; Digital                            |
| GOT-10k \cite{got10k}        | Short-term tracking | 180      | Noise; Blur; Digital                            |
| VOT2020-LT \cite{vot2020}    | Long-term tracking  | 50       | Noise; Blur; Digital; Transmission              |
| UAV20L \cite{uav123}         | UAV tracking        | 20       | Noise; Blur; Digital; Weather; Transmission     |
| DepthTrack \cite{depthtrack} | RGB-Depth Tracking  | 50       | Noise; Blur; Digital; Transmission; Multi-modal |

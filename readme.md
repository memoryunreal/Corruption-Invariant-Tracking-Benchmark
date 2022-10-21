# Corruption-Invariant-Tracking-Benchmark (CITB)

cvpr CITB is a rigorous benchmark established for tracking corruption robustness by re-constructing both single- and multi-modal tracking datasets with task-specific corruptions, resulting in VOT2020ST-C, GOT-10k-C, VOT2020-LT-C, UAV20L-C, and DepthTrack-C datasets.

# Execute environment
```
server: 172.18.36.46 
docker image: watchtowerss/cvpr2023:latest
container ID: 9e61af2c327d
project path: /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark
dataset: /home/dataset4/

dataset-generate:
env: pytracking
```

# Installation
For all the tracking methods are using 'for' loop instead of dataloader to load dataset in inference stage, to accelerate the tracking speed, we create the corruption dataset in advance.

- Environment install
```
cd Corruption-Invariant-Tracking-Benchmark/dist/
python setup.py install
```

- Generate corruption datset
```
# generate image corruption dataset
python random_corruption_datset.py 

# generate video corruption dataset
python video_corruption.py 

# random corruption fo UAV20L
python random_20l.py
```

- vot2020 and votlt2020
```
bash launchfile/vot.sh
```

- GOT10K


## Corruption Settings
## Benchmark Datasets Construction
| Dataset       | Description         | \#Videos | Added Corruptions                               |
|---------------|---------------------|----------|-------------------------------------------------|
| VOT2020-ST    | Short-term tracking | 60       | Noise; Blur; Digital                            |
| GOT-10k       | Short-term tracking | 180      | Noise; Blur; Digital                            |
| VOT2020-LT    | Long-term tracking  | 50       | Noise; Blur; Digital; Transmission              |
| UAV20L        | UAV tracking        | 20       | Noise; Blur; Digital; Weather; Transmission     |
| DepthTrack    | RGB-Depth Tracking  | 50       | Noise; Blur; Digital; Transmission; Multi-modal |

## pytracking trackers
```
# run the pytracking trackers dimp tomp keep_track eco kys lwl atom(failed)
/launchfile/pytracking/*.sh

$
```
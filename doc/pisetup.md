This document helps you to start an openpi server locally for our benchmark.
## 1. Install openpi
Download latest open-pi's repo and setup environment, recommend with uv to manage environment for pi,for more info, please check open-pi's documents.
```
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
# Or if you already cloned the repo:
git submodule update --init --recursive
```
In openpi's workspace:
```
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
## 2.Import our config for finetune model 
1. Copy [ur5_policy.py](../vla/openpi/ur5_policy.py) file to src/openpi/policies under pi's workspace
2. Copy and paste class [LeRobotUR5DataConfig](../vla/openpi/config.txt#L3) in src/openpi/training/config.py for Observation transform
3. Copy and paste TrainConfig [pi_fast_ur5](../vla/openpi/config.txt#L46) or [pi05_ur5](../vla/openpi/config.txt#L70) for our finetuned weight

## 3.Run openpi's server

```
uv run scripts/serve_policy.py policy:checkpoint --policy.config= --policy.dir=
```
`policy.onfig` for config's name in `TrainConfig` and `policy.dir` for path you store the model weight.
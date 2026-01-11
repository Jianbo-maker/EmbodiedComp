EmbodiedComp is a closed-loop benchmark frame for VLA(Vision-Language-Action) model.We use robosuite to build a environment with random texture and objects for an UR5 robotic arm with Pi0-FAST,Pi0.5 and OpenVLA-oft.

# Installing
Prepare environment
```bash
git clone ...
cd EmbodiedComp
conda env create
conda activate Ecomp
```
Install openpi-client:
```
cd packages/openpi-client
pip install -e .
cd ../..
```

# Benchmark with Pi0&Pi05
We use openpi-client to interact with pi0 and pi0.5, that means you have to start pi agent first with its own environment.You can see [pisetup](doc/pisetup.md) for our config for ur5 robot.And you can download our finetuned weight [here](https://huggingface.co/qruisjtu/pi_ur5_fintuned). After the pi client is start,you can change the `BENCHMARK PARAMETERS` in `pi.py` including text prompt, benchmarkname, agentname,etc.Then run,
```
python pi.py
```
the benchmark's result will be saved under folder `data/benchmark`

# Benchmark with openvla
For openvla, you can download our finetuned weight [here](https://huggingface.co/qruisjtu/openvla_ur5_finetuned).Then change the`pretrained_checkpoint= `in [openvla.py](openvla.py#L69) to your own weight path and run 

```
python openvla.py
```

the benchmark's result will be saved under folder `data/benchmark`

# Use your own compress codec
You can change code listed below to replace with your own compress codec:

- [pi](pi.py#L60) and [openvla](openvla.py#L137)'s function `build_obs_ur5` transfer image from camera to agent, you can modify our extract the image.

- [COMPRESSIMG](compress/compressimg.py#L367) collects many useful codec, you can add your own codec class in this file with function `compress`, then announce your class in `codecmap`

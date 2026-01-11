"""
Record trajectory data with the DataCollectionWrapper wrapper and play them back.

Example:
    $ python demo_collect_and_playback_data.py --environment Lift
"""

import argparse
import os
import time
from glob import glob

import numpy as np
import cv2
import robosuite as suite
from robosuite.wrappers import DataCollectionWrapper,Pi0UR5DataWrapper


def collect_random_trajectory(env, timesteps=500, max_fr=None):
    """Run a random policy to collect trajectories.

    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment instance to collect trajectories from
        timesteps(int): how many environment timesteps to run for a given trajectory
        max_fr (int): if specified, pause the simulation whenever simulation runs faster than max_fr
    """

    env.reset()
    dof = env.action_dim

    for t in range(timesteps):
        start = time.time()
        action = np.random.randn(dof)
        env.step(action)
        env.render()
        if t % 100 == 0:
            print(t)

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)


def playback_trajectory(env, ep_dir, max_fr=None):
    """Playback data from an episode.

    Args:
        env (MujocoEnv): environment instance to playback trajectory in
        ep_dir (str): The path to the directory containing data for an episode.
    """

    # first reload the model from the xml
    xml_path = os.path.join(ep_dir, "model.xml")
    with open(xml_path, "r") as f:
        env.reset_from_xml_string(f.read())

    state_paths = os.path.join(ep_dir, "state_*.npz")
    mydic=env.sim.get_state().flatten()
    print(mydic)
    # read states back, load them one by one, and render
    t = 0
    for state_file in sorted(glob(state_paths)):
        print(state_file)
        dic = np.load(state_file,allow_pickle=True)
        states = dic["states"]
        # print(states)
        actions=dic["action_infos"]
        # for action in actions:
        for state in states:
            print(state)
            start = time.time()
            env.sim.set_state_from_flattened(state)
            # env.sim.data.qpos[:7]=state[1:8]
            # env.step(action["actions"])
            env.sim.forward()
            env.viewer.update()
            env.render()
            t += 1
            if t % 100 == 0:
                print(t)
            # time.sleep(1)
            if max_fr is not None:
                elapsed = time.time() - start
                diff = 1 / max_fr - elapsed
                if diff > 0:
                    time.sleep(diff)
    env.close()
def showimgstored(ep_dir):
    os.makedirs("debug/test",exist_ok=True)
    state_paths = os.path.join(ep_dir, "state_*.npz")
    for state_file in sorted(glob(state_paths)):
        print(state_file)
        dic = np.load(state_file)
        baseimgs=dic["base_rgb"]
        for img in baseimgs:
            cv2.imwrite(f"debug/test/img_{time.time_ns()}.png",img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="UR5e", help="Which robot(s) to use in the env")
    parser.add_argument("--directory", type=str, default="/tmp/")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    args = parser.parse_args()

    # create original environment
    env = suite.make(
        args.environment,
        robots=args.robots,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=("frontview","robot0_eye_in_hand"),
        has_renderer=True,
        has_offscreen_renderer=True,
        control_freq=20,
    )
    data_directory = args.directory

    # wrap the environment with data collection wrapper
    env = Pi0UR5DataWrapper(env, data_directory,prompt="pick up the red cube")

    # testing to make sure multiple env.reset calls don't create multiple directories
    env.reset()
    env.reset()
    env.reset()

    # # collect some data
    # print("Collecting some random data...")
    # collect_random_trajectory(env, timesteps=args.timesteps, max_fr=args.max_fr)

    # # playback some data
    # _ = input("Press any key to begin the playback...")
    # print("Playing back the data...")
    data_directory = env.ep_directory
    playback_trajectory(env, "data/ur5e/ep_1753338478_740269", args.max_fr)
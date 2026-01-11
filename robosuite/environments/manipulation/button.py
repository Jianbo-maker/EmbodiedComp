from collections import OrderedDict
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import ButtonObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat


class Button(ManipulationEnv):
    def __init__(
        self,
        robots,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        horizon=1000,
        reward_shaping=False,
        reward_scale=1.0,
        has_renderer=False,
        has_offscreen_renderer=True,
        control_freq=20,
        **kwargs,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array([0, 0, 0.8])
        self.reward_shaping = reward_shaping
        self.reward_scale = reward_scale

        super().__init__(
            robots=robots,
            env_configuration="default",
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            control_freq=control_freq,
            horizon=horizon,
            **kwargs,
        )

    def reward(self, action=None):
        pressed_depth = -self.sim.data.get_joint_qpos(self.button_joint_name)
        return self.reward_scale if pressed_depth > 0.01 else 0.0

    def _load_model(self):
        super()._load_model()

        # 机器人基座
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # 创建桌子
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # 创建按钮对象
        self.button = ButtonObject(name="")

        # 创建 ManipulationTask
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.button],
        )

    def _setup_references(self):
        super()._setup_references()
        self.button_body_id = self.sim.model.body_name2id(self.button.root_body)
        joint0 = self.button.joints[0]
        self.button_joint_name = joint0 if isinstance(joint0, str) else joint0.name

    def _setup_observables(self):
        observables = super()._setup_observables()

        @sensor(modality="object")
        def button_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.button_body_id])

        @sensor(modality="object")
        def button_quat(obs_cache):
            return convert_quat(np.array(self.sim.data.body_xquat[self.button_body_id]), to="xyzw")

        @sensor(modality="object")
        def button_pressed(obs_cache):
            return -self.sim.data.get_joint_qpos(self.button_joint_name)

        observables["button_pos"] = Observable("button_pos", sensor=button_pos, sampling_rate=self.control_freq)
        observables["button_quat"] = Observable("button_quat", sensor=button_quat, sampling_rate=self.control_freq)
        observables["button_pressed"] = Observable("button_pressed", sensor=button_pressed, sampling_rate=self.control_freq)

        return observables

    def _reset_internal(self):
        super()._reset_internal()

        # 随机按钮位置
        x_min, x_max = -self.table_full_size[0] / 4 + 0.015, self.table_full_size[0] / 4 - 0.015
        y_min, y_max = -self.table_full_size[1] / 4 + 0.015, self.table_full_size[1] / 4 - 0.015

        rand_x = np.random.uniform(x_min, x_max)
        rand_y = np.random.uniform(y_min, y_max)
        button_z = self.table_offset[2] + 0.01

        self.sim.model.body_pos[self.button_body_id] = np.array([rand_x, rand_y, button_z])
        self.sim.data.set_joint_qpos(self.button_joint_name, 0.0)

        # 更新 sim
        self.sim.forward()

    def _check_success(self):
        pressed_depth = -self.sim.data.get_joint_qpos(self.button_joint_name)
        return pressed_depth > 0.005

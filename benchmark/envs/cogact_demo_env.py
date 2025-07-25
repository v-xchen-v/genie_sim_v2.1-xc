# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import time
import glob
import pickle
import numpy as np
import os
from .base_env import BaseEnv
from .demo_env import DemoEnv
from robot import Robot

from planner.manip_solver import (
    load_task_solution,
    generate_action_stages,
    split_grasp_stages,
)
from benchmark.tasks.demo_task import DemoTask
from tasks.dummy_task import DummyTask
from base_utils.transform_utils import calculate_rotation_matrix

DEBUGGING = True  # Set to True for debugging

from base_utils.logger import Logger

logger = Logger()  # Create singleton instance

from scipy.spatial.transform import Rotation as R


def xyzwxyz2mat(xyzwxyz):
    quat = xyzwxyz[3:7]
    rot = R.from_quat(quat, scalar_first=True).as_matrix()
    mat = np.eye(4)
    mat[0:3, 0:3] = rot
    mat[0:3, 3] = xyzwxyz[0:3]
    return mat


class CogActDemoEnv(DemoEnv):
    def __init__(self, robot: Robot, task_file: str, init_task_config, policy=None):
        super().__init__(robot, task_file, init_task_config)
        self.policy = policy

        self.init_task_config = init_task_config
        self.camera_list = self.init_task_config["recording_setting"]["camera_list"]
        self.specific_task_name = self.init_task_config["specific_task_name"]

        self.states_objects_by_name = {}
        # init task_file
        self.load(task_file)
        self.load_task_setup()

    def get_observation(self):
        """
        # Example
            data_keys = {
                'camera': {
                    'camera_prim_list': [
                        '/World/G1/head_link/D455_Solid/TestCameraDepth'
                    ],
                    'render_depth': True,
                    'render_semantic': True
                },
                'pose': [
                    '/World/G1/head_link/D455_Solid/TestCameraDepth'
                    '/World/G1/right_base_link',
                ],
                'joint_position': True,
                'gripper': True
            }
        """

        data_keys = {
            "camera": {
                "camera_prim_list": self.camera_list[:3],  # 1. head 2.3 wrist cameras
                "render_depth": False,
                "render_semantic": False,
            },
            "pose": [
                self.camera_list[0],
                # "/G1/right_base_link",
                # "/G1/left_base_link"
            ],
            "joint_position": True,
            "gripper": True,
        }

        # observation_raw = self.robot.get_observation(data_keys) # issue of get_gripper_status, will hang somehow
        observation_raw = self.robot.client.get_observation(data_keys)
        # right_ee_pose = self.robot.client.GetEEPose(is_right=True)
        # left_ee_pose = self.robot.client.GetEEPose(is_right=False)
        left_ee_pose = self.robot.get_ee_pose(ee_type="gripper", id="left")
        right_ee_pose = self.robot.get_ee_pose(ee_type="gripper", id="right")
        # left_ee_pose = self._get_ee_pose_in_world("/G1/left_base_link", observation_raw)
        # right_ee_pose = self._get_ee_pose_in_world("/G1/right_base_link", observation_raw)
        left_ee_pose2 = self._get_ee_pose_in_world_from_prim("/G1/arm_l_end_link")
        right_ee_pose2 = self._get_ee_pose_in_world_from_prim("/G1/arm_r_end_link")

        # # T_world2base = self._get_ee_pose_in_world_from_prim("/G1/base_link")
        # T_world2pitchhead = self._get_ee_pose_in_world_from_prim("/G1/head_link2")
        # T_world2cam = self._get_ee_pose_in_world_from_prim("/G1/head_link2/Head_Camera")
        # # T_pitchhead2base = np.linalg.inv(T_world2pitchhead) @ T_world2base
        # import pickle

        # log_data = {
        #     "head_cam": T_world2cam,
        #     "link_pitch_head": T_world2pitchhead,
        # }

        # # Use mode 'wb' (write binary) when writing with pickle
        # with open('head_cam_and_link_pitch_head.pkl', 'wb') as f:
        #     pickle.dump(log_data, f)

        observation_raw["right_ee_pose"] = right_ee_pose2
        observation_raw["left_ee_pose"] = left_ee_pose2
        return observation_raw

    def _get_ee_pose_in_world_from_prim(self, ee_prim_path):
        """
        Get the end-effector pose in world coordinates from the prim path.
        """
        # Get the end-effector pose in world coordinates
        pose = self.robot.client.get_object_pose(ee_prim_path)

        xyz = [
            pose.object_pose.position.x,
            pose.object_pose.position.y,
            pose.object_pose.position.z,
        ]
        wxyz = [
            pose.object_pose.rpy.rw,
            pose.object_pose.rpy.rx,
            pose.object_pose.rpy.ry,
            pose.object_pose.rpy.rz,
        ]

        ee_pose = xyzwxyz2mat(np.concatenate([xyz, wxyz]))  # Convert to 4x4 matrix

        if DEBUGGING:
            logger.debug(f"End-effector({ee_prim_path}) pose in world: {ee_pose}")

        return ee_pose

    def _get_ee_pose_in_world(self, ee_prim_path, observation_raw):
        """
        Get the end-effector pose in world coordinates.
        """
        position_xyz = observation_raw["pose"][ee_prim_path]["position"]
        rotation_wxyz = observation_raw["pose"][ee_prim_path]["rotation"]

        if DEBUGGING:
            logger.debug(
                f"End-effector({ee_prim_path}) pose in world: {position_xyz}, {rotation_wxyz}"
            )

        ee_pose = xyzwxyz2mat(
            np.concatenate([position_xyz, rotation_wxyz])
        )  # Covert to 4x4 matrix

        return ee_pose

    def _robot_move(self, actions):
        move_type = (
            "Normal"  # "Normal", "AvoidObs" # Normal is much more fast than AvoidObs
        )
        # move_type = "AvoidObs"  # TODO: change to AvoidObs for now

        # move arm
        logger.logger.info(
            f'ROBOT_RIGHT_POS_IN_HEAD_CAM: {actions["ROBOT_RIGHT_POSE_IN_HEAD_CAM"][:3, 3]}'
        )
        logger.logger.info(
            f'ROBOT_RIGHT_POS_IN_WORLD: {actions["ROBOT_RIGHT_POSE_IN_WORLD"][:3, 3]}'
        )
        target_right_ee_pose = actions["ROBOT_RIGHT_POSE_IN_WORLD"]
        self.robot.move_pose(
            target_right_ee_pose, type=move_type, arm="right", block=True
        )

        # TODO: move left arm and figure out do we need parallel execution
        target_left_ee_pose = actions["ROBOT_LEFT_POSE_IN_WORLD"]
        self.robot.move_pose(
            target_left_ee_pose, type=move_type, arm="left", block=True
        )

        # move gripper
        right_gripper_action = actions["ROBOT_RIGHT_GRIPPER"][0]
        if right_gripper_action > 0.5:
            right_gripper_action = "open"
            self.robot.set_gripper_action(right_gripper_action, arm="right")
        else:
            right_gripper_action = "close"
            self.robot.set_gripper_action(right_gripper_action, arm="right")

        left_gripper_action = actions["ROBOT_LEFT_GRIPPER"][0]
        if left_gripper_action > 0.5:
            left_gripper_action = "open"
            self.robot.set_gripper_action(left_gripper_action, arm="left")
        else:
            left_gripper_action = "close"
            self.robot.set_gripper_action(left_gripper_action, arm="left")

    def _get_per_step_actions(self, actions, step_id):
        """
        Get the action for the current step based on the actions dictionary and step_id.
        """
        action = {
            "ROBOT_RIGHT_POSE_IN_HEAD_CAM": actions["ROBOT_RIGHT_POSE_IN_HEAD_CAM"][
                step_id
            ],
            "ROBOT_LEFT_POSE_IN_HEAD_CAM": actions["ROBOT_LEFT_POSE_IN_HEAD_CAM"][
                step_id
            ],
            "ROBOT_RIGHT_GRIPPER": actions["ROBOT_RIGHT_GRIPPER"][step_id],
            "ROBOT_LEFT_GRIPPER": actions["ROBOT_LEFT_GRIPPER"][step_id],
            "ROBOT_RIGHT_POSE_IN_WORLD": actions["ROBOT_RIGHT_POSE_IN_WORLD"][step_id],
            "ROBOT_LEFT_POSE_IN_WORLD": actions["ROBOT_LEFT_POSE_IN_WORLD"][step_id],
        }
        return action

    def step(self, actions):
        self.current_step += 1
        action_len = len(actions["ROBOT_RIGHT_POSE_IN_HEAD_CAM"])

        execute_K = min(2, action_len)
        execute_step_N = 4
        for execute_step_id in range(execute_K):
            is_first_or_last = execute_step_id == 0 or execute_step_id == execute_K - 1
            is_last = execute_step_id == execute_K - 1
            is_selected_step = (execute_step_id + 1) % execute_step_N == 0
            if not is_last and not is_selected_step:
                continue

            self._robot_move(self._get_per_step_actions(actions, execute_step_id))

        observaion = None
        info = {}
        done = False
        # if self.current_step != 1 and self.current_step % 30 == 0:
        observaion = self.get_observation()
        # done, info = self.task.get_termination(self, info)
        self.task.step(self)
        need_update = True
        self.action_update()

        if done:
            info["final_step"] = self.current_step
            self.reset()

        # return observaion, None, done, info
        return observaion, self.has_done, need_update, self.task.task_progress

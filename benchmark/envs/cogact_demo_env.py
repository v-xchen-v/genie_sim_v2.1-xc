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

        self.right_gripper_close_enabled = True  # Enable gripper operation by default
        self.right_gripper_closed_count = 0  # Count how many times the gripper has been operated
        self.right_gripper_closed_threshold = 2 # After how many executions the gripper will be enable to operate again
        self.left_gripper_close_enabled = True  # Enable gripper operation by default
        self.left_gripper_closed_count = 0  # Count how many times the gripper has been operated
        self.left_gripper_closed_threshold = 2 # After how many executions the gripper will be enable to operate again
        
        # hardcode the minimal gripper close position per task
        self.min_gripper_close_pos = {
            "iros_make_a_sandwich": 0.0,
            "iros_clear_table_in_the_restaurant": 0.0,
            "iros_restock_supermarket_items": 0.30,
            "iros_stamp_the_seal": 0.01,
            "iros_pack_moving_objects_from_conveyor": 0.0,
            "iros_clear_the_countertop_waste": 0.0,
        }
        
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
        """
        list(observation_raw['joint'].keys())
        [
            'idx01_body_joint1', 
            'idx02_body_joint2', 
            'idx11_head_joint1', 
            'idx12_head_joint2', 
            'idx21_arm_l_joint1', 
            'idx61_arm_r_joint1', 
            'idx22_arm_l_joint2', 
            'idx62_arm_r_joint2', 
            'idx23_arm_l_joint3', 
            'idx63_arm_r_joint3',
            'idx24_arm_l_joint4', 
            'idx64_arm_r_joint4',
            'idx25_arm_l_joint5', 
            'idx65_arm_r_joint5', 
            'idx26_arm_l_joint6', 
            'idx66_arm_r_joint6', 
            'idx27_arm_l_joint7', 
            'idx67_arm_r_joint7', 
            'idx31_gripper_l_inner_joint1', 
            'idx41_gripper_l_outer_joint1', 
            'idx71_gripper_r_inner_joint1', 
            'idx81_gripper_r_outer_joint1', 
            'idx32_gripper_l_inner_joint3', 
            'idx42_gripper_l_outer_joint3', 
            'idx72_gripper_r_inner_joint3', 
            'idx82_gripper_r_outer_joint3', 
            'idx33_gripper_l_inner_joint4', 
            'idx43_gripper_l_outer_joint4', 
            'idx73_gripper_r_inner_joint4', 
            'idx83_gripper_r_outer_joint4', 
            'idx54_gripper_l_inner_joint0', 
            'idx53_gripper_l_outer_joint0', 
            'idx94_gripper_r_inner_joint0', 
            'idx93_gripper_r_outer_joint0']
            
        list(observation_raw['joint'].keys())[19]: 'idx41_gripper_l_outer_joint1'
        list(observation_raw['joint'].keys())[21]: 'idx81_gripper_r_outer_joint1'
        """
        """
        {
            'camera': {}, 
            'joint': 
            {
                'idx01_body_joint1': 0.2700676918029785, 
                'idx02_body_joint2': 0.5237442255020142,  ...}, 
            'pose': {
                '/G1/head_link2/Head_Camera': {...}}, 
                'gripper': {'left': {...}, 'right': {...}}, 
                'right_ee_pose': array([[ 0.88513639, -0.45113999,  0.11404508, -4.51743698],
                    [-0.36640036, -0.52461625,  0.76845857, 10.86233616],
                    [-0.28685249, -0.7219768 , -0.62965479,  0.97332233],
                    [ 0.        ,  0.        ,  0.        ,  1.        ]]), 
                'left_ee_pose': array([[-0.88493367, -0.45251531,  0.11010131, -4.51798534],
                    [-0.36348256,  0.52329309, -0.770743  , 11.69530678],
                    [ 0.29115776, -0.72207633, -0.6275611 ,  0.97288483],
                    [ 0.        ,  0.        ,  0.        ,  1.        ]])}"""
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
        # handle gripper close/open enable flag
        if not self.right_gripper_close_enabled and self.right_gripper_closed_count >= self.right_gripper_closed_threshold:
            self.right_gripper_close_enabled = True
            logger.logger.info(
                f"Right gripper operation reset enabled."
            )
        self.right_gripper_closed_count += 1

        if not self.left_gripper_close_enabled and self.left_gripper_closed_count >= self.left_gripper_closed_threshold:
            self.left_gripper_close_enabled = True
            logger.logger.info(
                f"Left gripper operation reset enabled."
            )
        self.left_gripper_closed_count += 1

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
        # right_gripper_action = 0
        if right_gripper_action > 0.5:
            # right_gripper_action = "open"
            # self.robot.set_gripper_action(right_gripper_action, arm="right")
            # self.robot.open_gripper("right")
            self._operate_gripper("right", right_gripper_action)
        else:
            if not self.right_gripper_close_enabled:
                logger.logger.warning("[RIGHT] Gripper operation is currently disabled.")
                return
            else:
                # right_gripper_action = "close"
                # self.robot.set_gripper_action(right_gripper_action, arm="right")
                # self.robot.close_gripper("right")
                logger.logger.info("Close right gripper")
                self._operate_gripper("right", right_gripper_action)
                self.right_gripper_close_enabled = False
                self.right_gripper_closed_count = 0  # Reset count after operation
                
        left_gripper_action = actions["ROBOT_LEFT_GRIPPER"][0]
        if left_gripper_action > 0.5:
            # left_gripper_action = "open"
            # self.robot.set_gripper_action(left_gripper_action, arm="left")
            # self.robot.open_gripper("left")
            self._operate_gripper("left", right_gripper_action)
        else:
            if not self.left_gripper_close_enabled:
                logger.logger.warning("[LEFT] Gripper operation is currently disabled.")
                return
            else:
                # left_gripper_action = "close"
                # self.robot.set_gripper_action(left_gripper_action, arm="left")
                # self.robot.close_gripper("left")
                logger.logger.info("Close left gripper")
                self._operate_gripper("left", right_gripper_action)
                self.left_gripper_close_enabled = False
                self.left_gripper_closed_count = 0  # Reset count after operation

    def _operate_gripper(self, gripper_type: str, gripper_value):
        """def set_init_pose(self, init_pose):
        target_joint_position = []
        target_joint_indices = []
        for idx, val in enumerate(init_pose):
            if val is None:
                continue
            if not np.isfinite(val):
                continue
            target_joint_position.append(val)
            target_joint_indices.append(idx)
        self.client.set_joint_positions(
            target_joint_position,
            is_trajectory=False,
            joint_indices=target_joint_indices,
        )"""
        """        msg_remap.position.append(joint_name_state_dict["idx21_arm_l_joint1"])
        msg_remap.position.append(joint_name_state_dict["idx22_arm_l_joint2"])
        msg_remap.position.append(joint_name_state_dict["idx23_arm_l_joint3"])
        msg_remap.position.append(joint_name_state_dict["idx24_arm_l_joint4"])
        msg_remap.position.append(joint_name_state_dict["idx25_arm_l_joint5"])
        msg_remap.position.append(joint_name_state_dict["idx26_arm_l_joint6"])
        msg_remap.position.append(joint_name_state_dict["idx27_arm_l_joint7"])
        left_gripper_pos = min(1, max(0.0, (0.8 - (joint_name_state_dict["idx41_gripper_l_outer_joint1"]))))
        msg_remap.position.append(left_gripper_pos)

        msg_remap.position.append(joint_name_state_dict["idx61_arm_r_joint1"])
        msg_remap.position.append(joint_name_state_dict["idx62_arm_r_joint2"])
        msg_remap.position.append(joint_name_state_dict["idx63_arm_r_joint3"])
        msg_remap.position.append(joint_name_state_dict["idx64_arm_r_joint4"])
        msg_remap.position.append(joint_name_state_dict["idx65_arm_r_joint5"])
        msg_remap.position.append(joint_name_state_dict["idx66_arm_r_joint6"])
        msg_remap.position.append(joint_name_state_dict["idx67_arm_r_joint7"])
        right_gripper_pos = min(1, max(0.0, (0.8 - (joint_name_state_dict["idx81_gripper_r_outer_joint1"]))))
        msg_remap.position.append(right_gripper_pos)"""
        for idx, name in enumerate(self.robot.joint_names):
            if name == "idx41_gripper_l_outer_joint1":
                left_gripper_joint_idx = idx
            if name == "idx81_gripper_r_outer_joint1":
                right_gripper_joint_idx = idx

        # gripper_pos = min(0.8, max(0.0, gripper_value))
        if gripper_value > 0.5:
            gripper_pos = 0.8  # open
            print(f"{gripper_type} gripper: {gripper_value}")
            gripper_joint_idx = 19 if gripper_type == "left" else 21

            self.robot.client.set_joint_positions(
                target_joint_position=[gripper_pos],
                is_trajectory=False,
                joint_indices=[gripper_joint_idx],
            )
        else:
            gripper_pos = 0  # close
            self._gentle_close_gripper(gripper_type)

    def _gentle_close_gripper(self, gripper_type: str):
        """
        Gently close the gripper to avoid damaging objects.
        """
        if gripper_type == "left":
            gripper_joint_idx = 19
        else:
            gripper_joint_idx = 21

        # Set a gentle close position delta, e.g., 0.1
        # This is to avoid damaging objects by closing too hard
        gentle_close_position_delta = 0.005
        # Get the current position of the gripper
        cur_obs = self.get_observation()
        current_position = cur_obs["joint"][
            list(cur_obs["joint"].keys())[gripper_joint_idx]
        ] - 0.01 # seems the gripper position is not accurate, so we need to subtract a small value
        print(f"Current {gripper_type} gripper position: {current_position}")

        # Calculate the gentle close positions, each time close delta, end position should close to 0
        def generate_decreasing_range(current_value, min_value, delta):
            """
            Generate a NumPy array from current_value down to (and including) min_value,
            decreasing by delta each step.

            Args:
                current_value (float): Starting value.
                min_value (float): Minimum value to reach.
                delta (float): Step size (positive number).

            Returns:
                np.ndarray: Decreasing values from current_value to min_value.
            """
            if delta <= 0:
                raise ValueError("delta must be positive")

            # np.arange excludes the endpoint, so we subtract a small epsilon to include min_value
            return np.arange(current_value, min_value - 1e-8, -delta)

        # gentle_closed_postion = 0.1  # Set a gentle close position, e.g., 0.01
        gentle_closed_position = self.min_gripper_close_pos.get(
            self.specific_task_name, 0.01
        )  # Default to 0.01 if not specified for the task
        if current_position > gentle_closed_position:
            gentle_close_positions = generate_decreasing_range(
                current_value=current_position,
                min_value=gentle_closed_position,
                delta=gentle_close_position_delta,
            )
            print(
                f"Gently closing {gripper_type} gripper in {len(gentle_close_positions)} steps from {current_position} to {gentle_closed_position} with delta {gentle_close_position_delta}"
            )
            for gripper_pos in gentle_close_positions:
                # Set the gripper position
                self.robot.client.set_joint_positions(
                    target_joint_position=[gripper_pos],
                    is_trajectory=False,
                    joint_indices=[gripper_joint_idx],
                )
                time.sleep(0.1)
                print(f"{gripper_type} gripper gently closed to {gripper_pos}")

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

        execute_K = min(4, action_len)
        execute_step_N = 1
        for execute_step_id in range(execute_K):
            is_first_or_last = execute_step_id == 0 or execute_step_id == execute_K - 1
            is_last = execute_step_id == execute_K - 1
            is_selected_step = (execute_step_id + 1) % execute_step_N == 0
            if not is_first_or_last and not is_selected_step:
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

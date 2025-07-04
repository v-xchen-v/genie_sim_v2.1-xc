# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from .base import BasePolicy
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import requests
import time
import io
import json
import os


from base_utils.logger import Logger

logger = Logger()  # Create singleton instance
# Optional: create a directory to store the logs
log_dir = "action_logs"
os.makedirs(log_dir, exist_ok=True)
MOCK_DELTA_TRANS_IN_REALCAM_COORD = False
LOG_MODEL_OUTPUT = False
LOG_OBS = True

def translation_sum(trans):
    return np.array(trans).sum(axis=0)

def rotation_sum(rotation_euler):
    rotation_sum = np.eye(3)
    for euler in rotation_euler:
        rotation_sum = R.from_euler('xyz', euler).as_matrix() @ rotation_sum
    return R.from_matrix(rotation_sum).as_euler('xyz', degrees=False)

def serialize_action_raw(action_raw):
    return {k: v.tolist() for k, v in action_raw.items()}


def mat2xyzrpy(mat):
    rpy = R.from_matrix(mat[0:3, 0:3]).as_euler("xyz", degrees=False)
    xyz = mat[0:3, 3]
    xyzrpy = np.concatenate([xyz, rpy])
    return xyzrpy


def xyzrpy2mat(xyzrpy):
    rot = R.from_euler("xyz", xyzrpy[3:6]).as_matrix()
    mat = np.eye(4)
    mat[0:3, 0:3] = rot
    mat[0:3, 3] = xyzrpy[0:3]
    return mat


def xyzwxyz2mat(xyzwxyz):
    quat = xyzwxyz[3:7]
    rot = R.from_quat(quat, scalar_first=True).as_matrix()
    mat = np.eye(4)
    mat[0:3, 0:3] = rot
    mat[0:3, 3] = xyzwxyz[0:3]
    return mat


def mat2xyzwxyz(mat):
    quat = R.from_matrix(mat[0:3, 0:3]).as_quat()
    xyz = mat[0:3, 3]
    xyzwxyz = np.concatenate([xyz, quat])
    return xyzwxyz
import rclpy

class CogActPolicy(BasePolicy):
    def __init__(self, task_name=None) -> None:
        super().__init__(task_name)

        SIM_INIT_TIME = 10
        # while rclpy.ok():
        #     sim_time = self.get_sim_time(self.sim_ros_node)
        #     if sim_time > SIM_INIT_TIME:
        #         print("cur sim time", sim_time)
        #         break
        #     time.sleep(0.5)
        time.sleep(SIM_INIT_TIME)

    def get_sim_time(self, sim_ros_node):
        sim_time = sim_ros_node.get_clock().now().nanoseconds * 1e-9
        return sim_time

    # def _wait_for_img_head(self, timeout_sec=50.0):
    #     start_time = time.time()
    #     while self.sim_ros_node.get_img_head() is None:
    #         if time.time() - start_time > timeout_sec:
    #             return False
    #         rclpy.spin_once(self.sim_ros_node, timeout_sec=0.5)
    #         time.sleep(0.5)
    #     return True
    
    # def _wait_for_img_left_wrist(self, timeout_sec=50.0):
    #     start_time = time.time()
    #     while self.sim_ros_node.get_img_left_wrist() is None:
    #         if time.time() - start_time > timeout_sec:
    #             return False
    #         rclpy.spin_once(self.sim_ros_node, timeout_sec=0.5)
    #         time.sleep(0.5)
    #     return True
    
    # def _wait_for_img_right_wrist(self, timeout_sec=50.0):
    #     start_time = time.time()
    #     while self.sim_ros_node.get_img_right_wrist() is None:
    #         if time.time() - start_time > timeout_sec:
    #             return False
    #         rclpy.spin_once(self.sim_ros_node, timeout_sec=0.5)
    #         time.sleep(0.5)
    #     return True
    
    def _obs_head_cam_rgb_image(self, observations):
        # head_cam_rgb_flat = observations["camera"][
        #     "/G1/head_link2/Head_Camera"
        # ]["rgb_camera"]
        # head_cam_rgb_width = observations["camera"][
        #     "/G1/head_link2/Head_Camera"
        # ]["camera_info"]["width"]
        # head_cam_rgb_height = observations["camera"][
        #     "/G1/head_link2/Head_Camera"
        # ]["camera_info"]["height"]
        # # TODO: here is a only temp resolution modification, for submission to challenge host,
        # # we only allowed processed on original resolution image.
        # # Reshape to (480, 640, 4) → (height, width, channels)
        # head_cam_rgba_image = np.array(head_cam_rgb_flat).reshape(
        #     (head_cam_rgb_height, head_cam_rgb_width, 4)
        # )
        # # Optional: remove alpha channel for RGB display
        # head_cam_rgb_image = head_cam_rgba_image[:, :, :3]
        # return head_cam_rgb_image.astype(np.uint8)


        # img_raw = None
        # if self._wait_for_img_head():
        while True:
            img_raw = self.sim_ros_node.get_img_head()
            time.sleep(0.1)
            if img_raw is not None:
                break

        return img_raw.astype(np.uint8)
        
    def _obs_left_wrist_cam_rgb_image(self, observations):
        # left_wrist_cam_rgb_flat = observations["camera"][
        #     "/G1/gripper_l_base_link/Left_Camera"
        # ]["rgb_camera"]
        # left_wrist_cam_rgb_width = observations["camera"][
        #     "/G1/gripper_l_base_link/Left_Camera"
        # ]["camera_info"]["width"]
        # left_wrist_cam_rgb_height = observations["camera"][
        #     "/G1/gripper_l_base_link/Left_Camera"
        # ]["camera_info"]["height"]
        # # Reshape to (480, 640, 4) → (height, width, channels)
        # left_wrist_cam_rgba_image = np.array(left_wrist_cam_rgb_flat).reshape(
        #     (left_wrist_cam_rgb_height, left_wrist_cam_rgb_width, 4)
        # )
        # # Optional: remove alpha channel for RGB display
        # left_wrist_cam_rgb_image = left_wrist_cam_rgba_image[:, :, :3]
        # return left_wrist_cam_rgb_image.astype(np.uint8)

        # img_raw = None
        # if self._wait_for_img_left_wrist():
        while True:
            img_raw = self.sim_ros_node.get_img_left_wrist()
            time.sleep(0.1)
            if img_raw is not None:
                break


        return img_raw.astype(np.uint8)

    def _obs_right_wrist_cam_rgb_image(self, observations):
        # right_wrist_cam_rgb_flat = observations["camera"][
        #     "/G1/gripper_r_base_link/Right_Camera"
        # ]["rgb_camera"]
        # right_wrist_cam_rgb_width = observations["camera"][
        #     "/G1/gripper_r_base_link/Right_Camera"
        # ]["camera_info"]["width"]
        # right_wrist_cam_rgb_height = observations["camera"][
        #     "/G1/gripper_r_base_link/Right_Camera"
        # ]["camera_info"]["height"]
        # # Reshape to (480, 640, 4) → (height, width, channels)
        # right_wrist_cam_rgba_image = np.array(right_wrist_cam_rgb_flat).reshape(
        #     (right_wrist_cam_rgb_height, right_wrist_cam_rgb_width, 4)
        # )
        # # Optional: remove alpha channel for RGB display
        # right_wrist_cam_rgb_image = right_wrist_cam_rgba_image[:, :, :3]
        # return right_wrist_cam_rgb_image.astype(np.uint8)

        # img_raw = None
        # if self._wait_for_img_right_wrist():
        while True:
            img_raw = self.sim_ros_node.get_img_right_wrist()
            time.sleep(0.1)
            if img_raw is not None:
                break

        return img_raw.astype(np.uint8)
    
    def _obs_instruction(self):
        # TODO(Xi): Hard-code now, redesign later.
        # Hardcoded for now
        # instruction = "Pick up the red ball and place it on the table."
        # instruction = "Pick up the bottle in blue blanket and place it on the table."
        instruction = "Pick up the bottle in blue blanket."
        # instruction = (
        #     "Pick up the plate containing pasta on the table with the right arm."
        # )
        return instruction

    # def _obs_robot_ee_left_translation(self, observations):
    #     return observations["gripper"]["left"]["position"]

    # def _obs_robot_ee_right_translation(self, observations):
    #     return observations["gripper"]["right"]["position"]

    # def _obs_robot_ee_left_rotation_wxyz(self, observations):
    #     return observations["gripper"]["left"]["rotation"]

    # def _obs_robot_ee_right_rotation_wxyz(self, observations):
    #     return observations["gripper"]["right"]["rotation"]

    def realcam_to_simcam(
        self,
        # T_realcam_in_world: np.ndarray,
        T_obj_in_realcam: np.ndarray,
    ) -> np.ndarray:
        """
        Convert object pose from realcam frame to simcam frame.

        Inputs:
            T_realcam_in_world: 4x4 pose of realcam in world coordinates
            T_obj_in_realcam: 4x4 pose of object in realcam coordinates

        Output:
            T_obj_in_simcam: 4x4 pose of object in simcam coordinates
        """
        # the vla model input obs-robot state should be in real camera coordinate
        R_real2sim = np.array(
            [
                [1, 0, 0],  # realcam +X → simcam +X
                [0, -1, 0],  # realcam +Y → simcam -Y
                [0, 0, -1],  # realcam +Z → simcam -Z
            ]
        )

        T_real2sim = np.eye(4)
        T_real2sim[:3, :3] = R_real2sim

        # T_obj_in_simcam = T_real2sim @ T_obj_in_realcam
        T_sim2real = np.linalg.inv(T_real2sim)  # T_sim2real
        T_obj_in_simcam = T_sim2real @ T_obj_in_realcam
        return T_obj_in_simcam

    def T_obj_in_simcam_to_T_obj_in_realcam(
        self,
        # T_simcam_in_world: np.ndarray,
        T_obj_in_simcam: np.ndarray,
    ) -> np.ndarray:
        """
        Convert object pose from simcam frame to realcam frame.

        Inputs:
            T_simcam_in_world: 4x4 pose of simcam in world coordinates
            T_obj_in_simcam: 4x4 pose of object in simcam coordinates

        Output:
            T_obj_in_realcam: 4x4 pose of object in realcam coordinates
        """
        # The vla model output action is in sim camera coordinate, but we should executa in sim camera then sim world cam
        R_real2sim = np.array(
            [
                [1, 0, 0],  # realcam +X → simcam +X
                [0, -1, 0],  # realcam +Y → simcam -Y
                [0, 0, -1],  # realcam +Z → simcam -Z
            ]
        )
        R_sim2real = R_real2sim.T
        # np.array(
        #     [
        #         [1, 0,  0],
        #         [0, 0,  1],
        #         [0, -1, 0],
        #     ]
        # )

        T_sim2real = np.eye(4)
        T_sim2real[:3, :3] = R_sim2real

        T_real2sim = np.linalg.inv(T_sim2real)  # T_real2sim
        T_obj_in_realcam = T_real2sim @ T_obj_in_simcam
        return T_obj_in_realcam

    def _obs_robot_ee_left_in_world_T(self, observations):
        return observations["left_ee_pose"]

    def _obs_robot_ee_right_in_world_T(self, observations):
        print("obs_robot_ee_right in world", observations["right_ee_pose"][:3, 3])
        return observations["right_ee_pose"]

    def _obs_sim_head_cam_pose_in_world_T(self, observations):
        head_cam_pose = observations["pose"]["/G1/head_link2/Head_Camera"]
        xyz = head_cam_pose["position"]
        wxyz = head_cam_pose["rotation"]  # quaternion (w, x, y, z)
        return xyzwxyz2mat(
            np.concatenate([xyz, wxyz])
        )  # Convert to transformation matrix

    def _obs_real_head_cam_pose_in_world_T(self, observations):
        T_simcam_in_world = self._obs_sim_head_cam_pose_in_world_T(observations)
        # """
        # realcam +X(right)   → simcam +Y
        # realcam +Y(down)    → simcam -Z
        # realcam +Z(forward) → simcam -X
        # """
        # R_real2sim = np.array(
        #     [
        #         [0, 0, -1],  # realcam +Z → simcam -X
        #         [1, 0, 0],  # realcam +X → simcam +Y
        #         [0, -1, 0],  # realcam +Y → simcam -Z
        #     ]
        # )
        """
        realcam +X(right)   → simcam +X
        realcam +Y(down)    → simcam -Y
        realcam +Z(forward) → simcam -Z
        """
        R_real2sim = np.array(
            [
                [1, 0, 0],  # realcam +X → simcam +X
                [0, -1, 0],  # realcam +Y → simcam -Y
                [0, 0, -1],  # realcam +Z → simcam -Z
            ]
        ) # seems align to the "/World" coordinate system in the simulation
        T_simcam2realcam = np.eye(4)
        T_simcam2realcam[:3, :3] = R_real2sim
        T_realcam_in_simcam = T_simcam2realcam
        # T_simcam_to_world = np.linalg.inv(T_simcam_in_world)  # T_world2simcam
        T_realcam_in_world = T_simcam_in_world @ T_realcam_in_simcam # FIX: changed from right multiply to left multiply, remember pose in right
        return T_realcam_in_world

    def _get_object_pose_in_camera(self, T_world2cam, T_object_in_world):
        """
        Get the pose of an object in the camera frame.
        :param T_world2cam: Transformation matrix from world to camera frame.
        :param T_world2obj: Transformation matrix from world to object frame.
        :return: Transformation matrix of the object in the camera frame.
        """
        T_cam2world = np.linalg.inv(T_world2cam)
        T_obj_in_cam = T_cam2world @ T_object_in_world
        return T_obj_in_cam

    def _get_object_pose_in_world(self, T_world2cam, T_obj_in_cam):
        """
        Get the pose of an object in the world frame.
        :param T_world2cam: Transformation matrix from world to camera frame.
        :param T_cam2obj: Transformation matrix from camera to object frame.
        :return: Transformation matrix of the object in the world frame.
        """
        # T_cam2world = np.linalg.inv(T_world2cam)
        T_obj_in_world = T_world2cam @ T_obj_in_cam
        return T_obj_in_world

    def _obs_robot_ee_left_translation_in_sim_head_cam(self, observations):
        T_simcam_in_world = self._obs_sim_head_cam_pose_in_world_T(observations)
        T_eeleft_in_world = self._obs_robot_ee_left_in_world_T(observations)

        T_eeleft_in_simcam = self._get_object_pose_in_camera(
            T_simcam_in_world, T_eeleft_in_world
        )
        return T_eeleft_in_simcam[:3, 3]

    def _obs_robot_ee_left_translation_in_real_head_cam(self, observations):
        T_realcam_in_world = self._obs_real_head_cam_pose_in_world_T(observations)
        T_eeleft_in_world = self._obs_robot_ee_left_in_world_T(observations)
        T_eeleft_in_realcam = self._get_object_pose_in_camera(
            T_realcam_in_world, T_eeleft_in_world
        )
        return T_eeleft_in_realcam[:3, 3]

    def _obs_robot_ee_right_translation_in_sim_head_cam(self, observations):
        T_simcam_in_world = self._obs_sim_head_cam_pose_in_world_T(observations)
        T_eeright_in_world = self._obs_robot_ee_right_in_world_T(observations)
        # xyzrpy2mat(
        #     np.concatenate(
        #         [
        #             self._obs_robot_ee_right_translation(observations),
        #             self._obs_robot_ee_right_rotation_wxyz(observations),
        #         ]
        #     )
        # )
        T_eeright_in_simcam = self._get_object_pose_in_camera(
            T_simcam_in_world, T_eeright_in_world
        )
        return T_eeright_in_simcam[:3, 3]

    def _obs_robot_ee_right_translation_in_real_head_cam(self, observations):
        T_realcam_in_world = self._obs_real_head_cam_pose_in_world_T(observations)
        T_eeright_in_world = self._obs_robot_ee_right_in_world_T(observations)
        T_eeright_in_realcam = self._get_object_pose_in_camera(
            T_realcam_in_world, T_eeright_in_world
        )
        return T_eeright_in_realcam[:3, 3]

    def _obs_robot_ee_left_rotation_euler_xyz_in_sim_head_cam(self, observations):
        T_simcam_in_world = self._obs_sim_head_cam_pose_in_world_T(observations)
        T_eeleft_in_world = self._obs_robot_ee_left_in_world_T(observations)

        T_eeleft_in_simcam = self._get_object_pose_in_camera(
            T_simcam_in_world, T_eeleft_in_world
        )
        return R.from_matrix(T_eeleft_in_simcam[:3, :3]).as_euler(
            "xyz", degrees=False
        )  # Convert rotation matrix to Euler angles (xyz)

    def _obs_robot_ee_left_rotation_euler_xyz_in_real_head_cam(self, observations):
        T_realcam_in_world = self._obs_real_head_cam_pose_in_world_T(observations)
        T_eeleft_in_world = self._obs_robot_ee_left_in_world_T(observations)
        T_eeleft_in_cam = self._get_object_pose_in_camera(
            T_realcam_in_world, T_eeleft_in_world
        )
        return R.from_matrix(T_eeleft_in_cam[:3, :3]).as_euler(
            "xyz", degrees=False
        )  # Convert rotation matrix to Euler angles (xyz)

    def _obs_robot_ee_right_rotation_euler_xyz_in_real_head_cam(self, observations):
        T_simcam_in_world = self._obs_sim_head_cam_pose_in_world_T(observations)
        T_eeright_in_world = self._obs_robot_ee_right_in_world_T(observations)

        T_eeright_in_simcam = self._get_object_pose_in_camera(T_simcam_in_world, T_eeright_in_world)
        return R.from_matrix(T_eeright_in_simcam[:3, :3]).as_euler("xyz", degrees=False)

    def _obs_robot_ee_right_rotation_euler_xyz_in_real_head_cam(self, observations):
        T_realcam_in_world = self._obs_real_head_cam_pose_in_world_T(observations)
        T_eeright_in_world = self._obs_robot_ee_right_in_world_T(observations)
        
        T_eeright_in_realcam = self._get_object_pose_in_camera(
            T_realcam_in_world, T_eeright_in_world
        )
        return R.from_matrix(T_eeright_in_realcam[:3, :3]).as_euler("xyz", degrees=False)

    def act(self, observations, **kwargs):
        # At First, got all required observations for CogAct, includes:
        # - robot state
        # - head camera rgb frame
        # - instruction

        # head camera rgb frame
        head_cam_rgb_image = self._obs_head_cam_rgb_image(observations)
        
        # wrist camera rgb frame
        left_wrist_cam_rgb_image = self._obs_left_wrist_cam_rgb_image(observations)
        right_wrist_cam_rgb_image = self._obs_right_wrist_cam_rgb_image(observations)

        # instruction
        instruction = self._obs_instruction()

        # robot current state in camera coordinate
        robot_ee_left_translation_in_head_cam = (
            self._obs_robot_ee_left_translation_in_real_head_cam(observations)
        )
        robot_ee_right_translation_in_head_cam = (
            self._obs_robot_ee_right_translation_in_real_head_cam(observations)
        )
        robot_ee_left_rotation_euler_xyz_in_head_cam = (
            self._obs_robot_ee_left_rotation_euler_xyz_in_real_head_cam(observations)
        )
        robot_ee_right_rotation_euler_xyz_in_head_cam = (
            self._obs_robot_ee_right_rotation_euler_xyz_in_real_head_cam(observations)
        )

        # Construct the observation dictionary
        obs_dict = {
            "task_description": instruction,
            "robot_state": {
                "ROBOT_LEFT_TRANS": robot_ee_left_translation_in_head_cam.tolist(),
                "ROBOT_LEFT_ROT_EULER": robot_ee_left_rotation_euler_xyz_in_head_cam.tolist(),
                "ROBOT_LEFT_GRIPPER": np.zeros((1,), dtype=np.float32).tolist(),
                "ROBOT_RIGHT_TRANS": robot_ee_right_translation_in_head_cam.tolist(),
                "ROBOT_RIGHT_ROT_EULER": robot_ee_right_rotation_euler_xyz_in_head_cam.tolist(),
                "ROBOT_RIGHT_GRIPPER": np.zeros((1,), dtype=np.float32).tolist(),
            },
            "images": {
                "cam_top": head_cam_rgb_image,
                "head_left": left_wrist_cam_rgb_image,
                "head_right": right_wrist_cam_rgb_image,
            },
        }

        action_raw = self._call_infer(
            image_list=[Image.fromarray(obs_dict["images"]["cam_top"])],
            task_description=obs_dict["task_description"],
            robot_status=obs_dict["robot_state"],
            url="http://10.190.172.212:5300/api/inference",  # Example URL, change as needed
        )

        # action_raw["ROBOT_LEFT_TRANS"] = translation_sum(action_raw["ROBOT_LEFT_TRANS"]).reshape(1, 3)
        # action_raw["ROBOT_RIGHT_TRANS"] = translation_sum(action_raw["ROBOT_RIGHT_TRANS"]).reshape(1, 3)
        # action_raw["ROBOT_LEFT_ROT_EULER"] = rotation_sum(action_raw["ROBOT_LEFT_ROT_EULER"]).reshape(1, 3)
        # action_raw["ROBOT_RIGHT_ROT_EULER"] = rotation_sum(action_raw["ROBOT_RIGHT_ROT_EULER"]).reshape(1, 3)
        # action_raw["ROBOT_LEFT_GRIPPER"] = action_raw["ROBOT_LEFT_GRIPPER"][:1]
        # action_raw["ROBOT_RIGHT_GRIPPER"] = action_raw["ROBOT_RIGHT_GRIPPER"][:1]

        # TODO: put the logging logic into a separate function
        # Quickly hard-code log the action_raw for debugging
        MANUAL_LOG = True
        if MANUAL_LOG:
            import time
            import pickle

            os.makedirs(log_dir, exist_ok=True)
            # Use timestamp or step_id as filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if LOG_MODEL_OUTPUT:
                filename = os.path.join(log_dir, f"action_raw_{timestamp}.pkl")
                with open(filename, "wb") as f:
                    pickle.dump(action_raw, f)

            if LOG_OBS:
                # also save the observations
                obs_filename = os.path.join(log_dir, f"observations_{timestamp}.pkl")
                with open(obs_filename, "wb") as f:
                    # pickle.dump(observations, f)
                    pickle.dump(obs_dict, f)

        # # dump action for debugging
        # import pickle

        # with open("cogact_action.pkl", "wb") as f:
        #     pickle.dump(action_raw, f)

        # print(f"CogAct Action: {action_raw}")

        action_ee_left_pose_in_sim_head_cam = self._action_ee_left_T_in_sim_head_cam(
            action_raw, observations
        )
        action_ee_right_pose_in_sim_head_cam = self._action_ee_right_T_in_sim_head_cam(
            action_raw, observations
        )
        action_ee_left_gripper = self._action_gripper_left(action_raw)
        action_ee_right_gripper = self._action_gripper_right(action_raw)
        action_ee_left_pose_in_world = self._action_ee_left_T(action_raw, observations)
        action_ee_right_pose_in_world = self._action_ee_right_T(
            action_raw, observations
        )
        action_dict = {
            "ROBOT_LEFT_POSE_IN_HEAD_CAM": action_ee_left_pose_in_sim_head_cam,
            "ROBOT_RIGHT_POSE_IN_HEAD_CAM": action_ee_right_pose_in_sim_head_cam,
            "ROBOT_LEFT_GRIPPER": action_ee_left_gripper,
            "ROBOT_RIGHT_GRIPPER": action_ee_right_gripper,
            "ROBOT_LEFT_POSE_IN_WORLD": action_ee_left_pose_in_world,
            "ROBOT_RIGHT_POSE_IN_WORLD": action_ee_right_pose_in_world,
        }

        return action_dict

    def _action_gripper_left(self, action_raw):
        return action_raw["ROBOT_LEFT_GRIPPER"]

    def _action_gripper_right(self, action_raw):
        return action_raw["ROBOT_RIGHT_GRIPPER"]

    def _action_ee_left_rot_eular_xyz_in_real_head_cam(
        self,
        action_raw,
        observations,
    ):
        # calculate s+1 in cam coordinate of ee
        ee_left_rot_euler_xyz_delta = action_raw[
            "ROBOT_LEFT_ROT_EULER"
        ]  # delta, shape [n_step, 3]
        ee_left_rot_eular_xyz_s = (
            self._obs_robot_ee_left_rotation_euler_xyz_in_real_head_cam(observations)
        )
        
        rotation_sum = R.from_euler(
            "xyz", ee_left_rot_eular_xyz_s, degrees=False
        )
        rotaion_list = []
        for delta in ee_left_rot_euler_xyz_delta:
            rotation_sum = R.from_euler(
                "xyz", delta, degrees=False
            ) * rotation_sum
            rotaion_list.append(rotation_sum.as_euler("xyz", degrees=False))
        return np.array(rotaion_list)

    def _action_ee_right_rot_eular_xyz_in_real_head_cam(self, action_raw, observations):
        ee_right_rot_euler_xyz_delta = action_raw["ROBOT_RIGHT_ROT_EULER"]  # delta
        if MOCK_DELTA_TRANS_IN_REALCAM_COORD:
            ee_right_rot_euler_xyz_delta = np.tile(
                np.array([0, 0, 0]), (16, 1)
            )
            
        ee_right_rot_eular_xyz_s = (
            self._obs_robot_ee_right_rotation_euler_xyz_in_real_head_cam(observations)
        )
        # iterate over first_k, plus delta to current state
        # [8, 3] + [3]
        # ee_right_rot_euler_xyz_splus1 = (
        #     ee_right_rot_eular_xyz_s + ee_right_rot_euler_xyz_delta
        # )
        rotation_sum = R.from_euler(
            "xyz", ee_right_rot_eular_xyz_s, degrees=False
        )
        rotaion_list = []
        for delta in ee_right_rot_euler_xyz_delta:
            rotation_sum = R.from_euler(
                "xyz", delta, degrees=False
            ) * rotation_sum
            rotaion_list.append(rotation_sum.as_euler("xyz", degrees=False))
        return np.array(rotaion_list)

    def _action_ee_left_translation_in_real_head_cam(
        self,
        action_raw,
        observations,
        is_mock=MOCK_DELTA_TRANS_IN_REALCAM_COORD,
    ):
        ee_left_translation = action_raw["ROBOT_LEFT_TRANS"].copy()  # delta, shape [n_step, 3]
        if is_mock:
            ee_left_translation = np.tile(np.array([0.01, 0, 0]), (16, 1))
        ee_left_translation_s = self._obs_robot_ee_left_translation_in_real_head_cam(
            observations
        )
        for i in range(len(ee_left_translation)):
            ee_left_translation_s += ee_left_translation[i]
            ee_left_translation[i] = ee_left_translation_s.copy()
        return ee_left_translation

    def _action_ee_right_translation_in_real_head_cam(
        self, action_raw, observations, is_mock=MOCK_DELTA_TRANS_IN_REALCAM_COORD
    ):
        ee_right_translation = action_raw["ROBOT_RIGHT_TRANS"].copy()  # delta
        if is_mock:
            ee_right_translation = np.tile(
                np.array([-0.01, 0, 0]), (16, 1)
            )  # expect go left in real cam coord
            # ee_right_translation = np.tile(
            #     np.array([0, 0.01, 0]), (16, 1)
            # )  # expect go down in real cam coord
            # ee_right_translation = np.tile(
            #     np.array([0, 0, -0.01]), (16, 1)
            # )  # expect go backward in real cam coord
        ee_right_translation_s = self._obs_robot_ee_right_translation_in_real_head_cam(
            observations
        )
        
        for i in range(len(ee_right_translation)):
            ee_right_translation_s += ee_right_translation[i]
            ee_right_translation[i] = ee_right_translation_s.copy()
        return ee_right_translation

    def _action_ee_left_T_in_sim_head_cam(self, action_raw, observations):
        ee_left_translation_splus1 = self._action_ee_left_translation_in_real_head_cam(
            action_raw, observations
        )
        ee_left_rot_euler_xyz_splus1 = (
            self._action_ee_left_rot_eular_xyz_in_real_head_cam(
                action_raw, observations
            )
        )
        ee_left_T_splus1 = [
            xyzrpy2mat(item)
            for item in np.concatenate(
                [ee_left_translation_splus1, ee_left_rot_euler_xyz_splus1], axis=-1
            )
        ]
        return ee_left_T_splus1

    def _action_ee_left_T_in_real_head_cam(self, action_raw, observations):
        ee_left_translation_splus1 = self._action_ee_left_translation_in_real_head_cam(
            action_raw, observations
        )
        ee_left_rot_euler_xyz_splus1 = (
            self._action_ee_left_rot_eular_xyz_in_real_head_cam(
                action_raw, observations
            )
        )
        ee_left_T_splus1 = [
            xyzrpy2mat(item)
            for item in np.concatenate(
                [ee_left_translation_splus1, ee_left_rot_euler_xyz_splus1], axis=-1
            )
        ]
        return ee_left_T_splus1

    def _action_ee_left_T_in_sim_head_cam(self, action_raw, observations):
        ee_left_T_in_real_head_cam = self._action_ee_left_T_in_real_head_cam(
            action_raw, observations
        )
        ee_left_T_in_sim_head_cam = [
            self.realcam_to_simcam(item) for item in ee_left_T_in_real_head_cam
        ]
        return ee_left_T_in_sim_head_cam

    def _action_ee_right_T_in_real_head_cam(self, action_raw, observations):
        ee_right_translation_splus1 = (
            self._action_ee_right_translation_in_real_head_cam(action_raw, observations)
        )
        ee_right_rot_euler_xyz_splus1 = (
            self._action_ee_right_rot_eular_xyz_in_real_head_cam(
                action_raw, observations
            )
        )
        ee_right_T_splus1 = [
            xyzrpy2mat(item)
            for item in np.concatenate(
                [ee_right_translation_splus1, ee_right_rot_euler_xyz_splus1], axis=-1
            )
        ]
        return ee_right_T_splus1

    def _action_ee_right_T_in_sim_head_cam(self, action_raw, observations):
        ee_right_T_in_real_head_cam = self._action_ee_right_T_in_real_head_cam(
            action_raw, observations
        )
        ee_right_T_in_sim_head_cam = [
            self.realcam_to_simcam(item) for item in ee_right_T_in_real_head_cam
        ]
        return ee_right_T_in_sim_head_cam

    def _action_ee_left_T(self, action_raw, observations):
        """
        Get the end-effector left transformation in head camera coordinate.
        :param action_raw: Raw action from CogAct.
        :param observations: Observations from the environment.
        :return: End-effector left transformation in world coordinate.
        """
        ee_left_T_splus1 = self._action_ee_left_T_in_sim_head_cam(
            action_raw, observations
        )
        T_world2cam = self._obs_sim_head_cam_pose_in_world_T(observations)
        # Convert from head camera coordinate to world coordinate
        ee_left_T_world = [
            self._get_object_pose_in_world(T_world2cam, item)
            for item in ee_left_T_splus1
        ]
        return ee_left_T_world

    def _action_ee_right_T(self, action_raw, observations):
        """
        Get the end-effector right transformation in head camera coordinate.
        :param action_raw: Raw action from CogAct.
        :param observations: Observations from the environment.
        :return: End-effector right transformation in world coordinate.
        """
        ee_right_T_splus1 = self._action_ee_right_T_in_sim_head_cam(
            action_raw, observations
        )
        T_world2cam = self._obs_sim_head_cam_pose_in_world_T(observations)
        # Convert from head camera coordinate to world coordinate
        ee_right_T_world = [
            self._get_object_pose_in_world(T_world2cam, item)
            for item in ee_right_T_splus1
        ]
        return ee_right_T_world

    def _call_infer(
        self,
        image_list: Image.Image,
        task_description: str,
        robot_status: dict,
        url: str,
        image_format: str = "JPEG",
        verbose: bool = False,
    ):
        # TODO(XI): Change and integrate calling CogAct Infer here.
        """
        Get action from the image and task description.
        :param image: PIL Image to be processed.
        :param task_description: Task description.
        :return: Action or None.
        """

        start_time = time.time()
        files = []
        for i, img in enumerate(image_list):
            if img is not None:
                img_bytes = io.BytesIO()
                if image_format == "JPEG":
                    img.save(img_bytes, format="JPEG")
                elif image_format == "PNG":
                    img.save(img_bytes, format="PNG")
                else:
                    raise ValueError("Unsupported image format. Use 'JPEG' or 'PNG'.")
                img_bytes.seek(0)
                files.append(
                    (
                        f"image_{i}",
                        (
                            (f"image_{i}.png", img_bytes, "image/png")
                            if image_format == "PNG"
                            else (f"image_{i}.jpg", img_bytes, "image/jpeg")
                        ),
                    )
                )

        json_bytes = io.BytesIO(
            json.dumps(
                {
                    "task_description": task_description,
                    "state": robot_status,
                }
            ).encode("utf-8")
        )
        files.append(("json", ("data.json", json_bytes, "application/json")))
        if verbose:
            print(f"Save time: {time.time() - start_time}")
        start_request_time = time.time()
        response = requests.post(url, files=files)
        end_request_time = time.time()

        if verbose:
            print("Request time: ", end_request_time - start_request_time)
            print("Total time: ", end_request_time - start_time)

        if response.status_code == 200:
            return response.json()
        else:
            print("Failed to get a response from the API")
            print(response.text)

    def reset(self):
        target_position = [
            0.34906611,
            0.34987221,
            0,
            0.436332313,
            -0.66857928,
            0.67156327,
            0.2008844,
            -0.20287371,
            0.27921745,
            -0.282218840,
            -1.28203404,
            1.28208637,
            0.84163094,
            -0.84068865,
            1.51518357,
            -1.51710308,
            -0.18715125,
            0.18636601,
            1,
            -1,
            1,
            -1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
        ]
        return target_position

import os
from glob import glob
import logging
import multiprocessing

import numpy as np
from transforms3d import quaternions as tq
import torch

from util.rot_util import torch_quaternion_to_matrix, torch_matrix_to_quaternion


def BODex(params):
    data_file, configs = params[0], params[1]

    raw_data = np.load(data_file, allow_pickle=True).item()
    robot_pose = raw_data["robot_pose"][0]
    new_data = {}
    new_data["obj_pose"] = raw_data["obj_pose"][0]
    new_data["obj_scale"] = raw_data["obj_scale"][0]
    new_data["obj_path"] = f"assets/DGNObj/{data_file.split('/')[-2]}"

    if configs.hand_name == "shadow":
        # Change qpos order of thumb
        robot_pose = np.concatenate(
            [robot_pose[:, :, :7], robot_pose[:, :, 12:], robot_pose[:, :, 7:12]],
            axis=-1,
        )
        # Add a translation bias of palm which is included in XML but ignored in URDF
        tmp_rot = torch_quaternion_to_matrix(torch.tensor(robot_pose[:, :, 3:7]))
        robot_pose[:, :, :3] -= (
            (tmp_rot @ torch.tensor([0, 0, 0.034]).view(1, 1, 3, 1)).squeeze(-1).numpy()
        )
    elif configs.hand_name == "allegro":
        # Add a rotation bias of palm which is included in XML but ignored in URDF
        tmp_rot = torch_quaternion_to_matrix(torch.tensor(robot_pose[:, :, 3:7]))
        delta_rot = torch_quaternion_to_matrix(torch.tensor([0, 1, 0, 1]).view(1, 1, 4))
        robot_pose[:, :, 3:7] = torch_matrix_to_quaternion(
            tmp_rot @ delta_rot.transpose(-1, -2)
        )
    elif configs.hand_name == "ur10e_shadow":
        robot_pose = np.concatenate(
            [robot_pose[:, :, :8], robot_pose[:, :, 13:], robot_pose[:, :, 8:13]],
            axis=-1,
        )
    else:
        raise NotImplementedError

    for i in range(len(robot_pose)):
        if configs.hand.mocap:
            new_data["init_qpos"] = new_data["pregrasp_qpos"] = robot_pose[i, 0]
            new_data["grasp_qpos"] = robot_pose[i, 1]
            new_data["squeeze_qpos"] = robot_pose[i, 2]
        else:
            new_data["init_qpos"] = robot_pose[i, 0]
            new_data["approach_qpos"] = robot_pose[i, :-4]
            new_data["pregrasp_qpos"] = robot_pose[i, -4]
            new_data["grasp_qpos"] = robot_pose[i, -3]
            new_data["squeeze_qpos"] = robot_pose[i, -2]
            new_data["lift_qpos"] = robot_pose[i, -1]
        save_path = (
            data_file.replace(configs.task.data_path, configs.grasp_dir)
            .replace("_grasp.npy", f"/{i}.npy")
            .replace("_mogen.npy", f"/{i}.npy")
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, new_data)
    return


def task_format(configs):
    if configs.task.data_name == "BODex":
        if configs.hand.mocap:
            raw_data_struct = ["**", "**_grasp.npy"]
        else:
            raw_data_struct = ["**", "**_mogen.npy"]
    else:
        raise NotImplementedError
    raw_data_path_lst = glob(os.path.join(configs.task.data_path, *raw_data_struct))
    raw_file_num = len(raw_data_path_lst)
    if configs.task.max_num > 0:
        raw_data_path_lst = np.random.permutation(sorted(raw_data_path_lst))[
            : configs.task.max_num
        ]
    logging.info(f"Find {raw_file_num} raw files, use {len(raw_data_path_lst)}")

    if len(raw_data_path_lst) == 0:
        return

    iterable_params = zip(raw_data_path_lst, [configs] * len(raw_data_path_lst))
    with multiprocessing.Pool(processes=configs.n_worker) as pool:
        result_iter = pool.imap_unordered(eval(configs.task.data_name), iterable_params)
        results = list(result_iter)

    grasp_lst = glob(os.path.join(configs.grasp_dir, *list(configs.data_struct)))
    logging.info(f"Get {len(grasp_lst)} grasp data in {configs.save_dir}")
    logging.info(f"Finish format conversion")
    return

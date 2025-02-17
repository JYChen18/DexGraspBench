import os
import multiprocessing
from glob import glob
import trimesh
import logging
import traceback

import numpy as np
from transforms3d import quaternions as tq

from util.hand_util import RobotKinematics
from util.usd_helper import UsdHelper, Material


def read_npy(params):
    npy_path, xml_path, configs = params[0], params[1], params[2]
    task_config = configs.task
    grasp_data = np.load(npy_path, allow_pickle=True).item()
    hand_fk = RobotKinematics(xml_path)

    obj_pose_lst = []
    hand_pose_lst = []
    for data_name in task_config.data_type:
        all_qpos = grasp_data[f"{data_name}_qpos"]
        all_qpos = all_qpos[None] if len(all_qpos.shape) == 1 else all_qpos

        for i in range(all_qpos.shape[0]):
            if configs.hand.mocap:
                hand_pose = all_qpos[i, :7]
                hand_qpos = all_qpos[i, 7:]
            else:
                hand_pose = np.array([0.0, 0, 0, 1, 0, 0, 0])
                hand_qpos = all_qpos[i]

            hand_fk.forward_kinematics(hand_qpos)
            hand_link_pose = hand_fk.get_poses(hand_pose)

            obj_pose_lst.append(
                np.concatenate(
                    [grasp_data["obj_pose"], [grasp_data["obj_scale"]]], axis=-1
                )
            )
            hand_pose_lst.append(hand_link_pose)

    obj_path = grasp_data["obj_path"]
    if not obj_path.endswith(".obj"):
        obj_path = os.path.join(grasp_data["obj_path"], "mesh/coacd.obj")
    if not os.path.exists(obj_path):
        raise NotImplementedError
    return {
        "obj_path": obj_path,
        "obj_pose_scale": np.stack(obj_pose_lst, axis=0),
        "hand_link_pose": np.stack(hand_pose_lst, axis=0),
    }


def read_npy_safe(params):
    try:
        return read_npy(params)
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.warning(f"{error_traceback}")
        return None


def task_vusd(configs):
    usd_helper = UsdHelper()
    hand_fk = RobotKinematics(configs.hand.xml_path)
    init_robot_name_lst, init_robot_mesh_lst = hand_fk.get_init_meshes()
    save_path = os.path.join(configs.vusd_dir, "grasp.usd")

    grasp_lst = glob(os.path.join(configs.grasp_dir, *list(configs.data_struct)))
    succ_lst = glob(os.path.join(configs.succ_dir, *list(configs.data_struct)))
    eval_lst = glob(os.path.join(configs.eval_dir, *list(configs.data_struct)))
    logging.info(
        f"Find {len(grasp_lst)} grasp data in {configs.grasp_dir}, {len(eval_lst)} evaluated, and {len(succ_lst)} succeeded in {configs.save_dir}"
    )

    if configs.task.vis_type == "succ":
        input_file_lst = succ_lst
    elif configs.task.vis_type == "fail":
        input_file_lst = list(
            set(eval_lst).difference(
                set([p.replace(configs.succ_dir, configs.eval_dir) for p in succ_lst])
            )
        )
    elif configs.task.vis_type == "raw":
        input_file_lst = grasp_lst
    else:
        raise NotImplementedError

    input_file_lst = sorted(input_file_lst)
    
    np.random.seed(0)
    if configs.task.max_num > 0 and len(input_file_lst) > configs.task.max_num:
        input_file_lst = np.random.permutation(input_file_lst)[: configs.task.max_num]
    # print(input_file_lst[86])
    logging.info(f"Visualize {len(input_file_lst)} grasp")

    param_lst = [(i, configs.hand.xml_path, configs) for i in input_file_lst]
    with multiprocessing.Pool(processes=configs.n_worker) as pool:
        result_iter = pool.map(read_npy_safe, param_lst)
        result_iter = [r for r in list(result_iter) if r is not None]

    obj_path_dict = {}
    for r in result_iter:
        op = r.pop("obj_path")
        if op not in obj_path_dict:
            obj_path_dict[op] = []
        obj_path_dict[op].append(r)

    data_length = result_iter[0]["hand_link_pose"].shape[0]

    hand_pose_scale_lst = np.ones(
        (
            len(result_iter) * data_length,
            result_iter[0]["hand_link_pose"].shape[-2],
            8,
        )
    )
    obj_pose_scale_lst = np.ones(
        (len(result_iter) * data_length, len(obj_path_dict.keys()), 8)
    )
    obj_vit_lst = []
    obj_name_lst = []
    obj_mesh_lst = []
    count = 0
    for i, (k, v_lst) in enumerate(obj_path_dict.items()):
        obj_name_lst.append(k.replace("/", "_"))
        obj_mesh_lst.append(trimesh.load(k, force="mesh"))
        obj_vit_lst.append([count, count + len(v_lst) * data_length])
        for v in v_lst:
            hand_pose_scale_lst[count : count + data_length, :, :-1] = v[
                "hand_link_pose"
            ]
            obj_pose_scale_lst[count : count + data_length, i] = v["obj_pose_scale"]
            count += data_length

    usd_helper.create_stage(
        save_path, timesteps=len(result_iter) * data_length, dt=0.01
    )

    # Add hands
    usd_helper.add_meshlst_to_stage(
        init_robot_mesh_lst,
        init_robot_name_lst,
        hand_pose_scale_lst,
        obstacles_frame="robot",
        material=Material(color=configs.hand.color, name="obj"),
    )

    # Add objects
    usd_helper.add_meshlst_to_stage(
        obj_mesh_lst,
        obj_name_lst,
        obj_pose_scale_lst,
        visible_time=obj_vit_lst,
        obstacles_frame="object",
        material=Material(color=[0.5, 0.5, 0.5, 1.0], name="obj"),
    )

    # Add table
    if configs.setting == "tabletop":
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, 3] = [0.0, 0.0, -0.01]
        usd_helper.add_mesh_to_stage(
            trimesh.primitives.Box([2.0, 2.0, 0.02], transformation_matrix),
            "table",
            base_frame="/world/table",
            material=Material(color=[0.5, 0.5, 0.5, 1.0], name="obj"),
        )

    usd_helper.write_stage_to_file(save_path)
    logging.info(f"Save to {os.path.abspath(save_path)}")
    logging.info(f"Finish visualization with USD")

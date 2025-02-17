import trimesh
import numpy as np
import os
from glob import glob
import logging
import multiprocessing

from util.hand_util import RobotKinematics


def _single_visd(params):

    data_path, data_folder, configs = (
        params[0],
        params[1],
        params[2],
    )
    task_config = configs.task

    out_path = (
        data_path.replace(data_folder, configs.vobj_dir)
        .replace(".npy", "")
        .replace(".yaml", "")
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    grasp_data = np.load(data_path, allow_pickle=True).item()
    hand_fk = RobotKinematics(configs.hand.xml_path)

    # Visualize hand
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
            visual_mesh = hand_fk.get_posed_meshes(hand_pose)
            visual_mesh.export(f"{out_path}_{data_name}_{i}.obj")

    # Visualize object
    obj_path = os.path.join(grasp_data["obj_path"], "mesh/coacd.obj")
    obj_tm = trimesh.load(obj_path, force="mesh")
    obj_tm.vertices *= grasp_data["obj_scale"]
    rotation_matrix = trimesh.transformations.quaternion_matrix(
        grasp_data["obj_pose"][3:]
    )
    rotation_matrix[:3, 3] = grasp_data["obj_pose"][:3]
    obj_tm.apply_transform(rotation_matrix)
    obj_tm.export(f"{out_path}_obj.obj")

    logging.info(f"Save to {os.path.dirname(out_path)}")

    return


def task_vobj(configs):
    grasp_lst = glob(os.path.join(configs.grasp_dir, *list(configs.data_struct)))
    succ_lst = glob(os.path.join(configs.succ_dir, *list(configs.data_struct)))
    eval_lst = glob(os.path.join(configs.eval_dir, *list(configs.data_struct)))
    logging.info(
        f"Find {len(grasp_lst)} grasp data in {configs.grasp_dir}, {len(eval_lst)} evaluated, and {len(succ_lst)} succeeded in {configs.save_dir}"
    )

    if configs.task.vis_type == "succ":
        data_folder = configs.succ_dir
        input_file_lst = succ_lst
    elif configs.task.vis_type == "fail":
        data_folder = configs.eval_dir
        input_file_lst = list(
            set(eval_lst).difference(
                set([p.replace(configs.succ_dir, configs.eval_dir) for p in succ_lst])
            )
        )
    elif configs.task.vis_type == "raw":
        data_folder = configs.grasp_dir
        input_file_lst = grasp_lst
    else:
        raise NotImplementedError

    input_file_lst = sorted(input_file_lst)
    if configs.task.max_num > 0 and len(input_file_lst) > configs.task.max_num:
        input_file_lst = np.random.permutation(input_file_lst)[: configs.task.max_num]

    logging.info(f"Visualize {len(input_file_lst)} grasp")

    iterable_params = [(inp, data_folder, configs) for inp in input_file_lst]

    with multiprocessing.Pool(processes=configs.n_worker) as pool:
        result_iter = pool.imap_unordered(_single_visd, iterable_params)
        results = list(result_iter)

    logging.info(f"Finish visualization with OBJ")
    
    return

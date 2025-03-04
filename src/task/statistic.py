import numpy as np
import os
from glob import glob
import multiprocessing
import logging

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc

from util.rot_util import torch_quaternion_to_matrix, torch_matrix_to_axis_angle


def compute_ROC_data(data_lst, metric_name):
    sim_results = np.array([d["succ_flag"] for d in data_lst])
    analytic_metrics = np.array([d[metric_name] for d in data_lst])

    sort_index = np.argsort(analytic_metrics)
    sim_results = sim_results[sort_index]
    analytic_metrics = analytic_metrics[sort_index]

    lens = len(sim_results)
    spilt_num = min(100, lens)
    spilt_lens = lens // spilt_num
    spilt_left = lens % spilt_num
    steps = [0]
    for i in range(spilt_left):
        steps.append(steps[-1] + spilt_lens + 1)
    for i in range(spilt_left, spilt_num):
        steps.append(steps[-1] + spilt_lens)
    assert steps[-1] == lens

    tp, fp, tn, fn = (
        np.zeros(spilt_num + 1),
        np.zeros(spilt_num + 1),
        np.zeros(spilt_num + 1),
        np.zeros(spilt_num + 1),
    )
    tp[0], fp[0] = 0, 0
    tn[0], fn[0] = np.sum(sim_results == 0), np.sum(sim_results == 1)
    for i in range(spilt_num):
        tp[i + 1] = tp[i] + np.sum(sim_results[steps[i] : steps[i + 1]] == 1)
        fp[i + 1] = fp[i] + np.sum(sim_results[steps[i] : steps[i + 1]] == 0)
        tn[i + 1] = tn[i] - np.sum(sim_results[steps[i] : steps[i + 1]] == 0)
        fn[i + 1] = fn[i] - np.sum(sim_results[steps[i] : steps[i + 1]] == 1)
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)  # shape=(spilt_num+1,)
    steps[-1] -= 1
    threshold = analytic_metrics[steps]
    return tpr, fpr, threshold


def draw_ROC_curve(data_lst, save_path):
    metric_name_lst = [
        "qp_metric",
        "qp_dfc_metric",
        "q1_metric",
        "tdg_metric",
        "dfc_metric",
    ]
    tpr_lst, fpr_lst, threshold_lst = [], [], []
    for metric_name in metric_name_lst:
        tpr, fpr, threshold = compute_ROC_data(data_lst, metric_name)
        tpr_lst.append(tpr)
        fpr_lst.append(fpr)
        threshold_lst.append(threshold)

    color_dict = {
        "qp_metric": "red",
        "qp_dfc_metric": "red",
        "dfc_metric": "blue",
        "tdg_metric": "green",
        "q1_metric": "cyan",
    }
    line_type_dict = {
        "qp_metric": "-",
        "qp_dfc_metric": "--",
        "dfc_metric": "-",
        "tdg_metric": "-",
        "q1_metric": "-",
    }
    name_dict = {
        "qp_metric": "QP_ours",
        "qp_dfc_metric": "QP_baseline",
        "dfc_metric": "DFC",
        "tdg_metric": "TDG",
        "q1_metric": "Q1",
    }
    plt.figure(figsize=(6, 6))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.rcParams["font.size"] = 15
    plt.rcParams["lines.linewidth"] = 2
    plt.plot([0, 1], [0, 1], color="black", label="Random", linestyle="--")

    for i in range(len(metric_name_lst)):
        tpr, fpr, threshold = tpr_lst[i], fpr_lst[i], threshold_lst[i]
        distance = tpr - fpr
        max_index = np.argmax(distance)
        auc_score = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=color_dict[metric_name_lst[i]],
            label=name_dict[metric_name_lst[i]],
            linestyle=line_type_dict[metric_name_lst[i]],
        )
        # print(
        #     "max distance is ",
        #     distance[max_index],
        #     "threshold is ",
        #     threshold[max_index],
        # )
        # print("tpr is ", tpr[max_index], "fpr is ", fpr[max_index])
        logging.info(f"AUC of {metric_name_lst[i]}: {auc_score}")
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    return


def draw_obj_scale_fig(data_lst, save_path):
    obj_scale_lst = [float(d["obj_scale"]) for d in data_lst]

    bins = np.linspace(0.05, 0.2, 11)

    # Create the histogram
    plt.hist(
        np.array(obj_scale_lst),
        bins=bins,
        color="skyblue",
        edgecolor="black",
        rwidth=0.8,
    )

    # Add labels and title
    plt.xlabel("Scale")
    plt.ylabel("Frequency")
    plt.title("Distribution of Object Scales")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return


def read_data(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    return data


def get_diversity(data_lst):
    from sklearn.decomposition import PCA

    hand_poses = torch.tensor(
        np.stack([d["grasp_qpos"][:7] for d in data_lst], axis=0)
    ).float()
    hand_qpos = torch.tensor(
        np.stack([d["grasp_qpos"][7:] for d in data_lst], axis=0)
    ).float()
    obj_poses = torch.tensor(
        np.stack([d["obj_pose"] for d in data_lst], axis=0)
    ).float()

    obj_rot = torch_quaternion_to_matrix(obj_poses[:, 3:])
    hand_rot = torch_quaternion_to_matrix(hand_poses[:, 3:])
    hand_real_trans = (
        obj_rot.transpose(-1, -2) @ (hand_poses[:, :3] - obj_poses[:, :3]).unsqueeze(-1)
    ).squeeze(-1)
    hand_real_rot = obj_rot.transpose(-1, -2) @ hand_rot
    hand_final_pose = torch.cat(
        [
            hand_real_trans,
            torch_matrix_to_axis_angle(hand_real_rot),
            hand_qpos,
        ],
        dim=-1,
    )

    pca = PCA()
    pca.fit(hand_final_pose.numpy())
    explained_variance = []
    for i in range(5):
        explained_variance.append(np.sum(pca.explained_variance_ratio_[: i + 1]))
    return explained_variance


def task_stat(configs):
    grasp_lst = glob(os.path.join(configs.grasp_dir, *list(configs.data_struct)))
    succ_lst = glob(os.path.join(configs.succ_dir, *list(configs.data_struct)))
    eval_lst = glob(os.path.join(configs.eval_dir, *list(configs.data_struct)))
    logging.info(
        f"Find {len(grasp_lst)} grasp data in {configs.grasp_dir}, {len(eval_lst)} evaluated, and {len(succ_lst)} succeeded in {configs.save_dir}"
    )

    # Grasp success rate
    logging.info(f"Grasp success rate: {len(succ_lst)/len(eval_lst)}")

    # Object success rate
    obj_eval_lst = glob(os.path.join(configs.eval_dir, *list(configs.data_struct[:-1])))
    obj_succ_lst = glob(os.path.join(configs.succ_dir, *list(configs.data_struct[:-1])))
    logging.info(f"Object success rate: {len(obj_succ_lst)/len(obj_eval_lst)}")

    if len(eval_lst) == 0:
        logging.error("No evaluated grasp!")

    with multiprocessing.Pool(processes=configs.n_worker) as pool:
        result_iter = pool.imap_unordered(read_data, eval_lst)
        data_lst = list(result_iter)

    if configs.task.scale_fig:
        save_path = os.path.join(configs.log_dir, "objscale_distribution.png")
        draw_obj_scale_fig(data_lst, save_path)

    if configs.task.roc_fig:
        if "qp_metric" not in data_lst[0]:
            logging.warning(
                "Please set 'eval_analytic' and 'eval_simulate' to True while evaluation"
            )
        else:
            save_path = os.path.join(configs.log_dir, "analytic_metric_ROC.png")
            draw_ROC_curve(data_lst, save_path)

    if configs.task.diversity:
        pca_eigenvalue = get_diversity(data_lst)
        logging.info(f"Diversity: {pca_eigenvalue}")

    if "ho_pene" not in data_lst[0]:
        logging.warning("Please set 'eval_dist' to True while evaluation")
    else:
        average_penetration_depth = np.mean([d["ho_pene"] for d in data_lst])
        logging.info(f"Penetration depth: {average_penetration_depth}")
        average_self_penetration_depth = np.mean([d["self_pene"] for d in data_lst])
        logging.info(f"Self-penetration depth: {average_self_penetration_depth}")
        average_contact_distance = np.mean([d["contact_dist"] for d in data_lst])
        logging.info(f"Contact distance: {average_contact_distance}")
        average_contact_number = np.mean([d["contact_num"] for d in data_lst])
        logging.info(f"Contact number: {average_contact_number}")
        average_contact_consis = np.mean([d["contact_consis"] for d in data_lst])
        logging.info(f"Contact consistency: {average_contact_consis}")
    
    logging.info(f"Finish statistics")
    
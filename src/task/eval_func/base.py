import os
import sys
from copy import deepcopy

import numpy as np
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from util.rot_util import (
    np_get_delta_qpos,
    np_normalize_vector,
)
from util.hand_util import MjHO
from util.file_util import load_json
from .fc_metric import *


class BaseEval:
    def __init__(self, input_npy_path, configs):
        self.input_npy_path = input_npy_path
        self.configs = configs
        self.grasp_data = np.load(input_npy_path, allow_pickle=True).item()

        # Fix object mass by setting density
        obj_info = load_json(
            os.path.join(self.grasp_data["obj_path"], "info/simplified.json")
        )
        obj_coef = obj_info["mass"] / (obj_info["density"] * (obj_info["scale"] ** 3))
        new_obj_density = configs.task.obj_mass / (
            obj_coef * (self.grasp_data["obj_scale"] ** 3)
        )

        # Build mj_spec
        self.mj_ho = MjHO(
            obj_path=self.grasp_data["obj_path"],
            obj_scale=self.grasp_data["obj_scale"],
            has_floor_z0=configs.setting == "tabletop",
            obj_density=new_obj_density,
            hand_xml_path=configs.hand.xml_path,
            hand_mocap=configs.hand.mocap,
            exclude_table_contact=configs.hand.exclude_table_contact,
            friction_coef=configs.task.miu_coef,
            debug_render=configs.task.debug_render,
            debug_viewer=configs.task.debug_viewer,
        )

        if self.configs.task.debug_viewer or self.configs.task.debug_render:
            with open("debug.xml", "w") as f:
                f.write(self.mj_ho.spec.to_xml())

        return

    def _simulate_under_extforce_details(self, pre_obj_qpos):
        raise NotImplementedError

    def _eval_pene_and_contact(self):
        eval_config = self.configs.task.pene_contact_metrics

        ho_contact, hh_contact = self.mj_ho.get_contact_info(
            self.grasp_data["grasp_qpos"],
            self.grasp_data["obj_pose"],
            obj_margin=eval_config.contact_margin,
        )

        contact_link_set = set()
        contact_dist_dict = {
            name: eval_config.contact_margin for name in self.configs.hand.finger_prefix
        }
        for c in ho_contact:
            hand_body_name = c["body1_name"]
            # Update the distance between the finger and the object
            for finger_prefix in contact_dist_dict:
                if hand_body_name.startswith(finger_prefix):
                    contact_dist_dict[finger_prefix] = min(
                        contact_dist_dict[finger_prefix], c["contact_dist"]
                    )
                    break
            # Update the name set of hand bodies in contact with the object
            if (
                np.abs(c["contact_dist"]) < eval_config.contact_threshold
                and hand_body_name in self.configs.hand.valid_body_name
            ):
                contact_link_set.add(hand_body_name)
        contact_dist_lst = list(contact_dist_dict.values())
        contact_distance = np.mean([max(i, 0.0) for i in contact_dist_lst])
        contact_consistency = np.max(contact_dist_lst) - np.min(contact_dist_lst)
        contact_number = len(contact_link_set)

        ho_pene = (
            -min([c["contact_dist"] for c in ho_contact]) if len(ho_contact) > 0 else 0
        )
        self_pene = (
            -min([c["contact_dist"] for c in hh_contact]) if len(hh_contact) > 0 else 0
        )

        return ho_pene, self_pene, contact_number, contact_distance, contact_consistency

    def _eval_simulate_under_extforce(self):
        eval_config = self.configs.task.simulation_metrics

        # Reset to init hand qpos and check contact
        init_qpos = (
            self.grasp_data["pregrasp_qpos"]
            if self.configs.hand.mocap
            else self.grasp_data["approach_qpos"][0]
        )
        ho_contact, hh_contact = self.mj_ho.get_contact_info(
            init_qpos, self.grasp_data["obj_pose"]
        )

        # Filter out bad initialization with severe penetration
        ho_dist = (
            min([c["contact_dist"] for c in ho_contact]) if len(ho_contact) > 0 else 0
        )
        hh_dist = (
            min([c["contact_dist"] for c in hh_contact]) if len(hh_contact) > 0 else 0
        )
        if ho_dist < -eval_config.max_pene or hh_dist < -eval_config.max_pene:
            if self.configs.task.debug_viewer or self.configs.task.debug_render:
                print(f"Severe penetration larger than {eval_config.max_pene}")
            return False, 100, 100

        # Record initial object pose
        pre_obj_qpos = deepcopy(self.mj_ho.get_obj_pose())
        if self.configs.setting == "tabletop":
            pre_obj_qpos[2] += 0.1

        # Detailed simulation methods for testing
        self._simulate_under_extforce_details(pre_obj_qpos)

        # Compare the resulted object pose
        latter_obj_qpos = self.mj_ho.get_obj_pose()
        delta_pos, delta_angle = np_get_delta_qpos(pre_obj_qpos, latter_obj_qpos)
        succ_flag = (delta_pos < eval_config.trans_thre) & (
            delta_angle < eval_config.angle_thre
        )

        if self.configs.task.debug_viewer or self.configs.task.debug_render:
            print(succ_flag, delta_pos, delta_angle)
            if self.configs.task.debug_render:
                debug_path = self.input_npy_path.replace(
                    self.configs.grasp_dir, self.configs.task.debug_dir
                ).replace(".npy", ".gif")
                os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                imageio.mimsave(debug_path, self.mj_ho.debug_images)
                print("Save GIF to ", debug_path)

        return succ_flag, delta_pos, delta_angle

    def _eval_analytic_fc_metric(self):
        eval_config = self.configs.task.analytic_fc_metrics

        ho_contact, _ = self.mj_ho.get_contact_info(
            self.grasp_data["grasp_qpos"],
            self.grasp_data["obj_pose"],
            obj_margin=eval_config.contact_threshold,
        )

        contact_point_dict = {}
        contact_normal_dict = {}
        for c in ho_contact:
            hand_body_name = c["body1_name"]
            # Check whether the hand contact body name is needed
            if (
                (hand_body_name not in self.configs.hand.valid_body_name)
                or (
                    eval_config.contact_tip_only
                    and hand_body_name not in self.configs.hand.tip_body_name
                )
                or (np.abs(c["contact_dist"]) > eval_config.contact_threshold)
            ):
                continue

            # Record valid contact point and normal
            if hand_body_name not in contact_point_dict:
                contact_point_dict[hand_body_name] = []
                contact_normal_dict[hand_body_name] = []
            contact_point_dict[hand_body_name].append(c["contact_pos"])
            contact_normal_dict[hand_body_name].append(c["contact_normal"])

        # If no contact, directly set a bad value as metric
        fc_metric_results = {}
        if len(contact_point_dict) == 0:
            print("111")
            for metric_name in eval_config.type:
                fc_metric_results[f"{metric_name}_metric"] = 2
            return fc_metric_results
        else:
            # Average all contacts on the same hand body
            contact_points = np.stack(
                [np.mean(np.array(v), axis=0) for v in contact_point_dict.values()]
            )
            contact_normals = np.stack(
                [
                    np_normalize_vector(np.mean(np.array(v), axis=0))
                    for v in contact_normal_dict.values()
                ]
            )
            # Use a smaller friction to leave some room to adjust
            miu_coef = 0.5 * np.array(self.configs.task.miu_coef)

            # Calculate analytic force closure metrics
            for metric_name in eval_config.type:
                fc_metric_results[f"{metric_name}_metric"] = eval(
                    f"calcu_{metric_name}_metric"
                )(contact_points, contact_normals, miu_coef)

        return fc_metric_results

    def run(self):
        eval_results = {}
        if self.configs.task.pene_contact_metrics is not None:
            (
                eval_results["ho_pene"],
                eval_results["self_pene"],
                eval_results["contact_num"],
                eval_results["contact_dist"],
                eval_results["contact_consis"],
            ) = self._eval_pene_and_contact()

        if self.configs.task.analytic_fc_metrics is not None:
            fc_metric_results = self._eval_analytic_fc_metric()
            for k, v in fc_metric_results.items():
                eval_results[k] = v

        if self.configs.task.simulation_metrics is not None:
            (
                eval_results["succ_flag"],
                eval_results["delta_pos"],
                eval_results["delta_angle"],
            ) = self._eval_simulate_under_extforce()
            # Save success data
            succ_npy_path = self.input_npy_path.replace(
                self.configs.grasp_dir, self.configs.succ_dir
            )
            if eval_results["succ_flag"] and not os.path.exists(succ_npy_path):
                os.makedirs(os.path.dirname(succ_npy_path), exist_ok=True)
                os.system(
                    f"ln -s {os.path.relpath(self.input_npy_path, os.path.dirname(succ_npy_path))} {succ_npy_path}"
                )

        # Save evaluation information
        eval_npy_path = self.input_npy_path.replace(
            self.configs.grasp_dir, self.configs.eval_dir
        )
        os.makedirs(os.path.dirname(eval_npy_path), exist_ok=True)
        for key in [
            "approach_qpos",
            "pregrasp_qpos",
            "grasp_qpos",
            "squeeze_qpos",
            "obj_scale",
            "obj_path",
            "obj_pose",
        ]:
            if key in self.grasp_data.keys():
                eval_results[key] = self.grasp_data[key]
        np.save(eval_npy_path, eval_results)
        return

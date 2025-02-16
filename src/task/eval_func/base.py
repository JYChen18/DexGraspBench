import os
import sys
import pdb
from copy import deepcopy
import logging

import numpy as np
import imageio
import mujoco
import mujoco.viewer


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from util.rot_util import (
    interplote_pose,
    interplote_qpos,
    np_get_delta_qpos,
    np_normalize_vector,
)
from util.hand_util import HOMjSpec
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
        self.hospec = HOMjSpec(
            obj_path=self.grasp_data["obj_path"],
            obj_pose=self.grasp_data["obj_pose"],
            obj_scale=self.grasp_data["obj_scale"],
            has_floor_z0="TableTop" in configs.setting,
            obj_density=new_obj_density,
            hand_xml_path=configs.hand.xml_path,
            hand_tendon_cfg=configs.hand.tendon,
            friction_coef=configs.task.miu_coef,
            grasp_pose=self.grasp_data["grasp_pose"],
            grasp_qpos=self.grasp_data["grasp_qpos"],
            pregrasp_pose=self.grasp_data["pregrasp_pose"],
            pregrasp_qpos=self.grasp_data["pregrasp_qpos"],
        )

        # Get ready for simulation
        self.mj_model = self.hospec.spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)

        if self.configs.debug:
            with open("debug.xml", "w") as f:
                f.write(self.hospec.spec.to_xml())

        return

    ###################################
    ####    Helping Functions      ####
    ###################################
    def _check_large_penetration(self, max_pene):
        for jj, contact in enumerate(self.mj_data.contact):
            body1_id = self.mj_model.geom(contact.geom1).bodyid
            body2_id = self.mj_model.geom(contact.geom2).bodyid
            total_body_num = self.mj_model.nbody
            object_id = total_body_num - 1
            hand_id = total_body_num - 2
            if contact.dist < -max_pene and (
                (body1_id < hand_id and body2_id == object_id)
                or (body2_id < hand_id and body1_id == object_id)
            ):
                return False
        return True

    def _control_hand_with_interp(
        self, pose1, pose2, ctrl1, ctrl2, step_outer=10, step_inner=10
    ):
        pose_interp = interplote_pose(pose1, pose2, step_outer)
        qpos_interp = interplote_qpos(ctrl1, ctrl2, step_outer)
        for j in range(step_outer):
            self.mj_data.mocap_pos[0] = pose_interp[j, :3]
            self.mj_data.mocap_quat[0] = pose_interp[j, 3:7]
            self.mj_data.ctrl[:] = qpos_interp[j]
            mujoco.mj_forward(self.mj_model, self.mj_data)
            for _ in range(step_inner):
                mujoco.mj_step(self.mj_model, self.mj_data)

            if self.configs.debug_render:
                self.debug_renderer.update_scene(
                    self.mj_data, "closeup", self.debug_options
                )
                pixels = self.debug_renderer.render()
                self.debug_images.append(pixels)
        return

    def _simulate_under_extforce_details(self, pre_obj_qpos):
        raise NotImplementedError

    ###################################
    #### Main Evaluation Functions ####
    ###################################
    def _eval_pene_and_contact(self):
        eval_config = self.configs.task.pene_contact_metrics
        # Change contact margin
        for i in range(self.mj_model.ngeom):
            if "object_collision" in self.mj_model.geom(i).name:
                self.mj_model.geom_margin[i] = self.mj_model.geom_gap[i] = (
                    eval_config.contact_margin
                )

        # Set to grasp pose
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.nkey - 1)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self_pene = 0
        ho_pene = 0
        contact_dist_dict = {
            name: eval_config.contact_margin for name in self.configs.hand.finger_prefix
        }
        contact_link_set = set()
        for jj, contact in enumerate(self.mj_data.contact):
            body1_id = self.mj_model.geom(contact.geom1).bodyid
            body2_id = self.mj_model.geom(contact.geom2).bodyid
            body1_name = self.mj_model.body(
                self.mj_model.geom(contact.geom1).bodyid
            ).name
            body2_name = self.mj_model.body(
                self.mj_model.geom(contact.geom2).bodyid
            ).name
            total_body_num = self.mj_model.nbody
            object_id = total_body_num - 1
            hand_id = total_body_num - 2
            # Check whether contact is between hand and object
            if (body1_id < hand_id and body2_id == object_id) or (
                body2_id < hand_id and body1_id == object_id
            ):
                # Update max penetration depth
                ho_pene = min(ho_pene, contact.dist)
                # Check whether the hand contact body is on fingers
                body_name = (
                    body1_name.removeprefix(self.hospec.hand_prefix)
                    if body2_id == object_id
                    else body2_name.removeprefix(self.hospec.hand_prefix)
                )
                for finger_prefix in contact_dist_dict:
                    if body_name.startswith(finger_prefix):
                        # Update the distance between the finger and the object
                        contact_dist_dict[finger_prefix] = min(
                            contact_dist_dict[finger_prefix], contact.dist
                        )
                        break
                # Update the name set of hand bodies in contact with the object
                if (
                    np.abs(contact.dist) < eval_config.contact_threshold
                    and body_name in self.configs.hand.valid_body_name
                ):
                    contact_link_set.add(body_name)
            elif body1_id < hand_id and body2_id < hand_id:
                # Update max self-penetration depth
                self_pene = min(self_pene, contact.dist)

        # Change contact margin back
        self.mj_model.geom_margin = 0
        self.mj_model.geom_gap = 0

        self_pene, ho_pene = -self_pene, -ho_pene
        contact_dist_lst = list(contact_dist_dict.values())
        contact_distance = np.mean([max(i, 0.0) for i in contact_dist_lst])
        contact_consistency = np.max(contact_dist_lst) - np.min(contact_dist_lst)
        contact_number = len(contact_link_set)
        return ho_pene, self_pene, contact_number, contact_distance, contact_consistency

    def _eval_simulate_under_extforce(self):
        eval_config = self.configs.task.simulation_metrics
        if self.configs.debug_viewer:
            viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            viewer.sync()
            pdb.set_trace()

        if self.configs.debug_render:
            self.debug_renderer = mujoco.Renderer(self.mj_model, 480, 640)
            self.debug_options = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.debug_options)
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            self.debug_images = []

        # Reset to pre-grasp pose
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.nkey - 2)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Filter out bad initialization with severe penetration
        if not self._check_large_penetration(eval_config.max_pene):
            if (
                self.configs.debug
                or self.configs.debug_viewer
                or self.configs.debug_render
            ):
                print(f"Severe penetration larger than {eval_config.max_pene}")
            return False, 100, 100

        # Record initial object pose
        _, _, pre_obj_pose = self.hospec.split_qpos_pose(self.mj_data.qpos)
        pre_obj_qpos = deepcopy(pre_obj_pose)
        if "TableTop" in self.configs.setting:
            pre_obj_qpos[2] += 0.1

        # Calculate ctrl signals from hand qpos
        dense_actu_moment = np.zeros((self.mj_model.nu, self.mj_model.nv))
        mujoco.mju_sparse2dense(
            dense_actu_moment,
            self.mj_data.actuator_moment,
            self.mj_data.moment_rownnz,
            self.mj_data.moment_rowadr,
            self.mj_data.moment_colind,
        )
        self.grasp_data["pregrasp_ctrl"] = (
            dense_actu_moment[:, 6:-6] @ self.grasp_data["pregrasp_qpos"]
        )
        self.grasp_data["grasp_ctrl"] = (
            dense_actu_moment[:, 6:-6] @ self.grasp_data["grasp_qpos"]
        )
        self.grasp_data["squeeze_ctrl"] = (
            dense_actu_moment[:, 6:-6] @ self.grasp_data["squeeze_qpos"]
        )

        # Detailed simulation methods for testing
        self._simulate_under_extforce_details(pre_obj_qpos)

        # Compare the resulted object pose
        _, _, latter_obj_qpos = self.hospec.split_qpos_pose(self.mj_data.qpos)
        delta_pos, delta_angle = np_get_delta_qpos(pre_obj_qpos, latter_obj_qpos)
        succ_flag = (delta_pos < eval_config.trans_thre) & (
            delta_angle < eval_config.angle_thre
        )

        if self.configs.debug or self.configs.debug_viewer or self.configs.debug_render:
            print(succ_flag, delta_pos, delta_angle)
            if self.configs.debug_render:
                debug_path = self.input_npy_path.replace(
                    self.configs.grasp_dir, self.configs.debug_dir
                ).replace(".npy", ".gif")
                os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                imageio.mimsave(debug_path, self.debug_images)
                print("save to ", debug_path)

        return succ_flag, delta_pos, delta_angle

    def _eval_analytic_fc_metric(self):
        eval_config = self.configs.task.analytic_fc_metrics
        # Change contact margin
        for i in range(self.mj_model.ngeom):
            if "object_collision" in self.mj_model.geom(i).name:
                self.mj_model.geom_margin[i] = self.mj_model.geom_gap[i] = (
                    eval_config.contact_threshold
                )

        # Set to grasp pose
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.nkey - 1)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        contact_point_dict = {}
        contact_normal_dict = {}
        for jj, contact in enumerate(self.mj_data.contact):
            body1_id = self.mj_model.geom(contact.geom1).bodyid
            body2_id = self.mj_model.geom(contact.geom2).bodyid
            body1_name = self.mj_model.body(
                self.mj_model.geom(contact.geom1).bodyid
            ).name
            body2_name = self.mj_model.body(
                self.mj_model.geom(contact.geom2).bodyid
            ).name
            total_body_num = self.mj_model.nbody
            object_id = total_body_num - 1
            hand_id = total_body_num - 2
            # Check whether contact is between hand and object
            if (
                (body1_id < hand_id and body2_id == object_id)
                or (body2_id < hand_id and body1_id == object_id)
            ) and (np.abs(contact.dist) < eval_config.contact_threshold):
                if body2_id == object_id:
                    body_name = body1_name.removeprefix(self.hospec.hand_prefix)
                    contact_normal = -contact.frame[0:3]
                else:
                    body_name = body2_name.removeprefix(self.hospec.hand_prefix)
                    contact_normal = contact.frame[0:3]

                # Check whether the hand contact body name is needed
                if body_name not in self.configs.hand.valid_body_name or (
                    eval_config.contact_region is not None
                    and eval_config.contact_region not in body_name
                ):
                    continue

                # Record valid contact point and normal
                if body_name not in contact_point_dict:
                    contact_point_dict[body_name] = []
                    contact_normal_dict[body_name] = []
                contact_point_dict[body_name].append(contact.pos)
                contact_normal_dict[body_name].append(contact_normal)

        # Change contact margin back
        self.mj_model.geom_margin = 0
        self.mj_model.geom_gap = 0

        # If no contact, directly set a bad value as metric
        fc_metric_results = {}
        if len(contact_point_dict) == 0:
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
            # NOTE use a smaller friction to leave some room to adjust
            miu_coef = 0.5 * np.array(self.configs.task.miu_coef)
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
        if not os.path.exists(eval_npy_path):
            os.makedirs(os.path.dirname(eval_npy_path), exist_ok=True)
            for key in [
                "grasp_pose",
                "grasp_qpos",
                "pregrasp_pose",
                "pregrasp_qpos",
                "squeeze_pose",
                "squeeze_qpos",
                "obj_scale",
                "obj_path",
                "obj_pose",
            ]:
                if key in self.grasp_data.keys():
                    eval_results[key] = self.grasp_data[key]
            np.save(eval_npy_path, eval_results)
        return

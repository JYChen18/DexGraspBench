import os
import sys

import numpy as np

from .base import BaseEval


class tabletopArmEval(BaseEval):
    def _simulate_under_extforce_details(self, pre_obj_qpos):
        # 1. Set object gravity
        external_force_direction = np.array([0.0, 0, -1, 0, 0, 0])
        self.mj_ho.set_ext_force_on_obj(
            10 * external_force_direction * self.configs.task.obj_mass
        )

        # 2. Approaching
        approach_length = self.grasp_data["approach_qpos"].shape[0]
        for i in range(approach_length - 1):
            self.mj_ho.control_hand_with_interp(
                self.grasp_data["approach_qpos"][i],
                self.grasp_data["approach_qpos"][i + 1],
                step_outer=3 if (i % 5 == 4 or i == approach_length - 2) else 1,
            )

        # 3. Move hand to pre-grasp pose
        self.mj_ho.control_hand_with_interp(
            self.grasp_data["approach_qpos"][-1],
            self.grasp_data["pregrasp_qpos"],
        )

        # 4. Move hand to grasp pose
        self.mj_ho.control_hand_with_interp(
            self.grasp_data["pregrasp_qpos"],
            self.grasp_data["grasp_qpos"],
        )

        # 5. Move hand to squeeze pose.
        # NOTE step 2 and 3 are seperate because pre -> grasp -> squeeze are stage-wise linear.
        # If step 2 and 3 are merged to one linear interpolation, the performance will drop a lot.
        self.mj_ho.control_hand_with_interp(
            self.grasp_data["grasp_qpos"],
            self.grasp_data["squeeze_qpos"],
        )

        # 6. Lift the object
        self.mj_ho.control_hand_with_interp(
            self.grasp_data["squeeze_qpos"],
            self.grasp_data["lift_qpos"],
        )

        return

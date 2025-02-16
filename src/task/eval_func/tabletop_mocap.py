from copy import deepcopy

import numpy as np

from .base import BaseEval


class TableTopMocapEval(BaseEval):
    def _simulate_under_extforce_details(self, pre_obj_qpos):
        # 1. Set object gravity
        external_force_direction = np.array([0.0, 0, -1, 0, 0, 0])
        self.mj_ho.set_ext_force_on_obj(
            10 * external_force_direction * self.configs.task.obj_mass
        )

        # 2. Move hand to grasp pose
        self.mj_ho.control_hand_with_interp(
            self.grasp_data["pregrasp_pose"],
            self.grasp_data["grasp_pose"],
            self.grasp_data["pregrasp_qpos"],
            self.grasp_data["grasp_qpos"],
        )

        # 3. Move hand to squeeze pose.
        # NOTE step 2 and 3 are seperate because pre -> grasp -> squeeze are stage-wise linear.
        # If step 2 and 3 are merged to one linear interpolation, the performance will drop a lot.
        self.mj_ho.control_hand_with_interp(
            self.grasp_data["grasp_pose"],
            self.grasp_data["squeeze_pose"],
            self.grasp_data["grasp_qpos"],
            self.grasp_data["squeeze_qpos"],
        )

        # 5. Lift the object
        lift_pose = deepcopy(self.grasp_data["squeeze_pose"])
        lift_pose[2] += 0.1
        self.mj_ho.control_hand_with_interp(
            self.grasp_data["squeeze_pose"],
            lift_pose,
            self.grasp_data["squeeze_qpos"],
            self.grasp_data["squeeze_qpos"],
        )

        return

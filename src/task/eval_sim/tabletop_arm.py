import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from util.rot_util import interplote_pose, interplote_qpos, np_get_delta_qpos
from task.eval_sim.basic import BasicEval


class TableTopArmEval(BasicEval):
    def _eval_external_force_details(self):
        raise NotImplementedError

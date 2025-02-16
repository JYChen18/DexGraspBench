import os
import multiprocessing
import logging
from glob import glob
import traceback

import numpy as np

from .eval_func import *


def safe_eval_one(params):
    input_npy_path, configs = params[0], params[1]
    try:
        eval(f"{configs.setting}Eval")(input_npy_path, configs).run()
        return
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.warning(f"{error_traceback}")
        return


def task_eval(configs):
    assert (
        configs.task.simulation_metrics is not None
        or configs.task.analytic_fc_metrics is not None
        or configs.task.pene_contact_metrics is not None
    ), "You should at least evaluate one kind of metrics"
    input_path_lst = glob(os.path.join(configs.grasp_dir, *list(configs.data_struct)))
    init_num = len(input_path_lst)

    if configs.skip:
        eval_path_lst = glob(os.path.join(configs.eval_dir, *list(configs.data_struct)))
        eval_path_lst = [
            p.replace(configs.eval_dir, configs.grasp_dir) for p in eval_path_lst
        ]
        input_path_lst = list(set(input_path_lst).difference(set(eval_path_lst)))
    skip_num = init_num - len(input_path_lst)
    input_path_lst = sorted(input_path_lst)
    if configs.task.max_num > 0:
        input_path_lst = np.random.permutation(input_path_lst)[: configs.task.max_num]

    logging.info(
        f"Find {init_num} grasp data, skip {skip_num}, and use {len(input_path_lst)}."
    )

    if len(input_path_lst) == 0:
        return

    iterable_params = zip(input_path_lst, [configs] * len(input_path_lst))
    if configs.task.debug_viewer or configs.task.debug_render:
        for ip in iterable_params:
            safe_eval_one(ip)
    else:
        with multiprocessing.Pool(processes=configs.n_worker) as pool:
            result_iter = pool.imap_unordered(safe_eval_one, iterable_params)
            results = list(result_iter)

    grasp_lst = glob(os.path.join(configs.grasp_dir, *list(configs.data_struct)))
    succ_lst = glob(os.path.join(configs.succ_dir, *list(configs.data_struct)))
    eval_lst = glob(os.path.join(configs.eval_dir, *list(configs.data_struct)))
    logging.info(
        f"Get {len(grasp_lst)} grasp data, {len(eval_lst)} evaluated, and {len(succ_lst)} succeeded in {configs.save_dir}"
    )
    logging.info(f"Finish evaluation")

    return

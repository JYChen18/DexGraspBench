rm -r output/debug_leap
python src/main.py hand=leap task=format exp_name=debug task.max_num=100 task.data_path=../BODex/src/curobo/content/assets/output/sim_leap/fc/debug/graspdata
python src/main.py hand=leap task=eval exp_name=debug task.max_num=1000
python src/main.py hand=leap task=stat exp_name=debug
python src/main.py hand=leap task=vusd exp_name=debug task.max_num=10
python src/main.py hand=leap task=vobj exp_name=debug task.max_num=10

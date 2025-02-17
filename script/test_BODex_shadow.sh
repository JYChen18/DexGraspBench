rm -r output/debug_shadow
python src/main.py task=format exp_name=debug task.max_num=100 task.data_path=/mnt/disk1/jiayichen/data/cuDex/sim_shadow/fc/debug/graspdata
python src/main.py task=eval exp_name=debug task.max_num=1000
python src/main.py task=stat exp_name=debug
python src/main.py task=vusd exp_name=debug task.max_num=10
python src/main.py task=vobj exp_name=debug task.max_num=10

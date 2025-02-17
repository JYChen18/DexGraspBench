rm -r output/debug_allegro
python src/main.py hand=allegro task=format exp_name=debug task.max_num=100 task.data_path=/mnt/disk1/jiayichen/data/cuDex/sim_allegro/fc/debug/graspdata
python src/main.py hand=allegro task=eval exp_name=debug task.max_num=1000
python src/main.py hand=allegro task=stat exp_name=debug
python src/main.py hand=allegro task=vusd exp_name=debug task.max_num=10
python src/main.py hand=allegro task=vobj exp_name=debug task.max_num=10

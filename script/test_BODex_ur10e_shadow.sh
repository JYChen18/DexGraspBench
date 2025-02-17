rm -r output/debug_ur10e_shadow
python src/main.py setting=Tabletop hand=ur10e_shadow task=format exp_name=debug task.max_num=100 task.data_path=/mnt/disk1/jiayichen/data/cuDex/sim_shadow/tabletop/debug/graspdata
python src/main.py setting=Tabletop hand=ur10e_shadow task=eval exp_name=debug task.max_num=1000
python src/main.py setting=Tabletop hand=ur10e_shadow task=vusd exp_name=debug task.max_num=10

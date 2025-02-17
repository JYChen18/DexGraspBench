rm -r output/DGN_allegro
python src/main.py hand=allegro task.data_name=DGN task=format exp_name=DGN task.max_num=1000 task.data_path=/mnt/disk1/keyubin/Released/BODEX/DexGraspNet/data/experiments/exp_33/results
python src/main.py hand=allegro task=eval exp_name=DGN task.max_num=1000
python src/main.py hand=allegro task=stat exp_name=DGN
python src/main.py hand=allegro task=vusd exp_name=DGN task.max_num=10
python src/main.py hand=allegro task=vobj exp_name=DGN task.max_num=10
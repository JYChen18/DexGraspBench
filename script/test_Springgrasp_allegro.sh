rm -r output/Spg_allegro
python src/main.py hand=allegro task.data_name=Spg task=format exp_name=Spg task.max_num=1000 task.data_path=/mnt/disk1/keyubin/Released/BODEX/SpringGrasp/results
python src/main.py hand=allegro task=eval exp_name=Spg task.max_num=1000
python src/main.py hand=allegro task=stat exp_name=Spg
python src/main.py hand=allegro task=vusd exp_name=Spg task.max_num=10
python src/main.py hand=allegro task=vobj exp_name=Spg task.max_num=10
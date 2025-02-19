rm -r output/Frogger_allegro
python src/main.py hand=allegro task.data_name=Frogger task=format exp_name=Frogger task.max_num=1000 task.data_path=/mnt/disk1/keyubin/Released/BODEX/frogger/results
python src/main.py hand=allegro task=eval exp_name=Frogger task.max_num=1000
python src/main.py hand=allegro task=stat exp_name=Frogger
python src/main.py hand=allegro task=vusd exp_name=Frogger task.max_num=10
python src/main.py hand=allegro task=vobj exp_name=Frogger task.max_num=10
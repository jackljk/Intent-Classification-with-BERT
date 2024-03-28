mkdir assets
# python main.py --n-epochs 10 --do-train
# python main.py --n-epochs 10 --do-train --task custom --reinit_n_layers 4
python main.py --contrast_n_epochs 3 --n-epochs 40 --do-train --task supcon --batch-size 64 --learning-rate 1e-3 --contrast_learning_rate 1e-5 
#python main.py --contrast_n_epochs 8 --n-epochs 30 --do-train --task supcon --batch-size 64 --learning-rate 1e-3 --contrast_learning_rate 1e-4 --slimCLR True






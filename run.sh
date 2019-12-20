CUDA_VISIBLE_DEVICES=2,3 python run.py --job train --id momojie --nd-train 1 --epoch 5 --resume 0 --gpu 2,3 --model dpn3d26 --batch-size 4
python run.py --job test --id lwq --model dpn3d26 --split 16
clear
mkdir models
cp ../1_step1/models/checkpoint.pth.tar ./models/checkpoint_ba.pth.tar
mkdir log
python3 fine_tune.py --data=/home/aparen/data/imagenet --batch_size=220 --learning_rate=1e-3 --reg=1e-6 --epochs=401 --weight_decay=0 | tee -a log/training.txt

clear
mkdir models
cp ./models/model_400.pth.tar ./models/checkpoint.pth.tar
mkdir log
python3 fine_tune.py --data=/home/aparen/data/imagenet --batch_size=220 --learning_rate=1e-3 --reg=1 --epochs=200 --weight_decay=0 | tee -a log/training.txt

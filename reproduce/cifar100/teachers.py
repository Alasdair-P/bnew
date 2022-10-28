import os
import time
import yaml

def create_jobs():

    jobs = []

    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar100  --proxquant --reg-type none --opt sgd --lr_mode step --T 100 150 --weight_decay 1e-4  --epochs 200 --batch-size 128 --tag teacher_1 --lr 1.0 --model rn --train_acc --use_tensorboard "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar100  --proxquant --reg-type none --opt sgd --lr_mode step --T 100 150 --weight_decay 1e-4  --epochs 200 --batch-size 128 --tag teacher_2 --lr 1.0 --model rn --train_acc --use_tensorboard "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar100  --proxquant --reg-type none --opt sgd --lr_mode step --T 100 150 --weight_decay 1e-4  --epochs 200 --batch-size 128 --tag teacher_3 --lr 1.0 --model rn --train_acc --use_tensorboard "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar100  --proxquant --reg-type none --opt sgd --lr_mode step --T 100 150 --weight_decay 1e-4  --epochs 200 --batch-size 128 --tag teacher_4 --lr 1.0 --model rn --train_acc --use_tensorboard "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar100  --proxquant --reg-type none --opt sgd --lr_mode step --T 100 150 --weight_decay 1e-4  --epochs 200 --batch-size 128 --tag teacher_5 --lr 1.0 --model rn --train_acc --use_tensorboard "
    jobs.append(template)

    return jobs

def run_command(command, noprint=True):
    command = " ".join(command.split())
    print(command)
    os.system(command)

def launch(jobs, interval):
    for i, job in enumerate(jobs):
        print("\nJob {} out of {}".format(i + 1, len(jobs)))
        run_command(job)
        time.sleep(interval)

if __name__ == "__main__":
    jobs = create_jobs()
    for job in jobs:
        print(job)
    print("Total of {} jobs to launch".format(len(jobs)))
    launch(jobs, 5)



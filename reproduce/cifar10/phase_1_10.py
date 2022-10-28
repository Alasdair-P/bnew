import os
import time
import yaml

def create_jobs():

    jobs = []

    # PHASE 1

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar10 --proxquant --reg-type convex_hull --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag phase_1_1 --lr 0.01 --model brn --train_acc --use_tensorboard "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar10 --proxquant --reg-type convex_hull --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag phase_1_2 --lr 0.01 --model brn --train_acc --use_tensorboard "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar10 --proxquant --reg-type convex_hull --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag phase_1_3 --lr 0.01 --model brn --train_acc --use_tensorboard "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar10 --proxquant --reg-type convex_hull --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag phase_1_4 --lr 0.01 --model brn --train_acc --use_tensorboard "
    jobs.append(template)
    
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar10 --proxquant --reg-type convex_hull --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag phase_1_5 --lr 0.01 --model brn --train_acc --use_tensorboard "
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



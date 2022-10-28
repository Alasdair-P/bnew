import os
import time
import yaml

def create_jobs():

    jobs = []
    
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template += "python main.py --dataset imagenet --proxquant --reg-type convex_hull --opt adam_wd --lr_mode linear --epochs 128 --batch-size 256 --tag reactnet18_improved_aug --model reactnet18 --lr 2.5e-3 --train_acc --use_tensorboard --weight_decay 1e-5 --loss KL "
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



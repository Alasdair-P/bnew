import os
import time
import yaml

def create_jobs():

    jobs = []
   
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset imagenet --proxquant --opt adam --lr_mode linear --epochs 100 --batch-size 256 --tag reactnet18_improved_aug --model reactnet18 --lr 2.5e-3 --reg 5e-4 --train_acc --use_tensorboard --weight_decay 1e-5 --reg-type our_binary_w_s --resume imagenet/reactnet18imagenet-adam--lr-linear-0.0025--wd-1e-05--b-256-epoch-128-reg-0.0-type-convex_hull-reactnet18_improved_aug/best_model.pkl --freeze 90 91 92 93 94 95 96 97 98 99 100 --tag lr_linear --train_acc "
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



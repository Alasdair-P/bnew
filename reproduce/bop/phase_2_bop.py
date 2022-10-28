import os
import time
import yaml

def create_jobs():

    jobs = []
    #STE
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag phase_2_1 --lr 0.01 --model brn --train_acc --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-phase_1_1/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag phase_2_2 --lr 0.01 --model brn --train_acc --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-phase_1_2/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_2/best_model.pkl "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag phase_2_3 --lr 0.01 --model brn --train_acc --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-phase_1_3/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_3/best_model.pkl "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag phase_2_4 --lr 0.01 --model brn --train_acc --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-phase_1_4/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_4/best_model.pkl "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag phase_2_5 --lr 0.01 --model brn --train_acc --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-phase_1_5/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_5/best_model.pkl "
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



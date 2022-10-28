import os
import time
import yaml

def create_jobs():

    jobs = []


    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template += "python binary_main.py --dataset cifar100  --proxquant --reg-type mirror_binary --opt adam --lr_mode linear --epochs 1000 --batch-size 128 --tag phase_2_1_ --lr 0.01 --reg 1.01 --freeze 980 981 982 983 984 985 987 988 989 990 991 992 993 994 995 996 997 998 999 --model brn --train_acc --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-phase_1_1/best_model.pkl "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template += "python binary_main.py --dataset cifar100  --proxquant --reg-type mirror_binary --opt adam --lr_mode linear --epochs 1000 --batch-size 128 --tag phase_2_2 --lr 0.01 --reg 1.01 --freeze 980 981 982 983 984 985 987 988 989 990 991 992 993 994 995 996 997 998 999 --model brn --train_acc --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-phase_1_2/best_model.pkl "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template += "python binary_main.py --dataset cifar100  --proxquant --reg-type mirror_binary --opt adam --lr_mode linear --epochs 1000 --batch-size 128 --tag phase_2_3 --lr 0.01 --reg 1.01 --freeze 980 981 982 983 984 985 987 988 989 990 991 992 993 994 995 996 997 998 999 --model brn --train_acc --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-phase_1_3/best_model.pkl "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template += "python binary_main.py --dataset cifar100  --proxquant --reg-type mirror_binary --opt adam --lr_mode linear --epochs 1000 --batch-size 128 --tag phase_2_4 --lr 0.01 --reg 1.01 --freeze 980 981 982 983 984 985 987 988 989 990 991 992 993 994 995 996 997 998 999 --model brn --train_acc --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-phase_1_4/best_model.pkl "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template += "python binary_main.py --dataset cifar100  --proxquant --reg-type mirror_binary --opt adam --lr_mode linear --epochs 1000 --batch-size 128 --tag phase_2_5 --lr 0.01 --reg 1.01 --freeze 980 981 982 983 984 985 987 988 989 990 991 992 993 994 995 996 997 998 999 --model brn --train_acc --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-phase_1_5/best_model.pkl "
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



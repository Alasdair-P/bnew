import os
import time
import yaml

def create_jobs():

    jobs = []

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 1delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-1000-reg-0.0-type-bop_binary-phase_2_1/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 2delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-1000-reg-0.0-type-bop_binary-phase_2_2/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 3delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-1000-reg-0.0-type-bop_binary-phase_2_3/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 4delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-1000-reg-0.0-type-bop_binary-phase_2_4/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 5delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-1000-reg-0.0-type-bop_binary-phase_2_5/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)

    '''

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 1delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-1000-reg-0.0-type-bop_binary-phase_2_1_dist/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 2delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-1000-reg-0.0-type-bop_binary-phase_2_2_dist/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 3delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-1000-reg-0.0-type-bop_binary-phase_2_3_dist/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 4delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-1000-reg-0.0-type-bop_binary-phase_2_4_dist/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 5delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-1000-reg-0.0-type-bop_binary-phase_2_5_dist/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 1delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-bop_binary-phase_2_1/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 2delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-bop_binary-phase_2_2/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 3delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-bop_binary-phase_2_3/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 4delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-bop_binary-phase_2_4/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 5delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-bop_binary-phase_2_5/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 1delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-bop_binary-phase_2_1_dist/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 2delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-bop_binary-phase_2_2_dist/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 3delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-bop_binary-phase_2_3_dist/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 4delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-bop_binary-phase_2_4_dist/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17 "
    template += "python bop_main2.py --dataset cifar100  --proxquant --reg-type bop_binary --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag 5delete --lr 0.01 --model brn --eval --resume cifar100/brncifar100-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-bop_binary-phase_2_5_dist/best_model.pkl --teacher cifar100/rncifar100-sgd--lr-step-1.0--wd-0.0001--b-128-epoch-200-reg-0.0-type-none-teacher_1/best_model.pkl "
    jobs.append(template)
    '''

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



import os
import time
import yaml

def create_jobs():

    jobs = []

    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    #template += "python binary_main.py --dataset cifar10  --proxquant --reg-type our_binary --reg 0.0003 --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag ours_fixed --lr 0.01 --model blrn --train_acc --use_tensorboard --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/cifar10/blrncifar10-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-none-phase_1/best_model.pkl --freeze 190 191 192 193 194 195 196 197 198 199 200 "
    #template += "python binary_main.py --dataset cifar10  --proxquant --reg-type our_binary_w_s --reg 0.001 --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag phase_2_with_prescaler --lr 0.01 --model blrn --train_acc --use_tensorboard --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/cifar10/blrncifar10-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-phase_1_w_scale/best_model.pkl --freeze 190 191 192 193 194 195 196 197 198 199 200 "
    template += "python binary_main.py --dataset cifar10  --proxquant --reg-type our_binary_w_s --reg 0.001 --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag phase_2_with_scaler --lr 0.01 --model blrn --train_acc --use_tensorboard --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/cifar10/blrncifar10-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-phase_1_with_scale/best_model.pkl --freeze 190 191 192 193 194 195 196 197 198 199 200 "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template += "python binary_main.py --dataset cifar10  --proxquant --reg-type our_binary_w_s --reg 0.0003 --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag phase_2_with_scaler --lr 0.01 --model blrn --train_acc --use_tensorboard --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/cifar10/blrncifar10-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-phase_1_with_scale/best_model.pkl --freeze 190 191 192 193 194 195 196 197 198 199 200 "
    jobs.append(template)
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template += "python binary_main.py --dataset cifar10  --proxquant --reg-type our_binary_w_s --reg 0.0001 --opt adam --lr_mode linear --epochs 200 --batch-size 128 --tag phase_2_with_scaler --lr 0.01 --model blrn --train_acc --use_tensorboard --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/cifar10/blrncifar10-adam--lr-linear-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-phase_1_with_scale/best_model.pkl --freeze 190 191 192 193 194 195 196 197 198 199 200 "
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


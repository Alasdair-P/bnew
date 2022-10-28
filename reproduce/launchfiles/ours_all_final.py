import os
import time
import yaml

def create_jobs():

    jobs = []
    '''
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template += "python binary_main.py --dataset cifar10  --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/cifar10/rncifar10-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --use_tensorboard --tag ours_final --train_acc "
    list_jobs(template, jobs)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template += "python binary_main.py --dataset cifar100 --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/cifar100/rncifar100-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --use_tensorboard --tag ours_final --train_acc "
    list_jobs(template, jobs)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template += "python binary_main.py --dataset tiny_imagenet --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/tiny_imagenet/rntiny_imagenet-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --use_tensorboard --tag ours_final --train_acc "
    list_jobs(template, jobs)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template += "python binary_main.py --dataset tiny_imagenet --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/tiny_imagenet/wrntiny_imagenet-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --use_tensorboard --tag ours_final --train_acc "
    list_jobs(template, jobs)
        

    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template += "python binary_main.py --dataset cifar10 --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/cifar10/lrncifar10-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-small/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --use_tensorboard --tag ours_final --train_acc --model lrn --lr 0.01 --reg 0.001 "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    Ptemplate += "python binary_main.py --dataset cifar10 --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/cifar10/wrncifar10-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --use_tensorboard --tag ours_final --train_acc --model wrn --lr 0.01 --reg 0.001 "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar100 --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/cifar100/rncifar100-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --use_tensorboard --tag ours_final --train_acc --model rn --lr 0.01 --reg 0.001 "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar100 --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/cifar100/wrncifar100-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --use_tensorboard --tag ours_final --train_acc --model wrn --lr 0.01 --reg 0.001 "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset tiny_imagenet --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/tiny_imagenet/wrntiny_imagenet-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --use_tensorboard --tag ours_final_2 --train_acc --model wrn --lr 0.01 --reg 0.001 "
    jobs.append(template)


    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset tiny_imagenet --proxquant --reg-type binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/tiny_imagenet/lrntiny_imagenet-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-small/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --tag w_reg --train_acc --model lrn --lr 0.01 --reg 0.001 "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset tiny_imagenet --proxquant --reg-type binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/tiny_imagenet/dntiny_imagenet-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --tag w_reg --train_acc --model dn --lr 0.01 --reg 0.001 "
    jobs.append(template)
    
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset tiny_imagenet --proxquant --reg-type binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/tiny_imagenet/rntiny_imagenet-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --tag w_reg --train_acc --model rn --lr 0.01 --reg 0.001 "
    jobs.append(template)
    
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset tiny_imagenet --proxquant --reg-type binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/tiny_imagenet/wrntiny_imagenet-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --tag w_reg --train_acc --model wrn --lr 0.01 --reg 0.001 "
    jobs.append(template)
 
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset tiny_imagenet --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/tiny_imagenet/lrntiny_imagenet-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-small/best_model.pkl --relu ste --opt adam --lr_mode step --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --tag const_eta --train_acc --model lrn --lr 0.01 --reg 0.001 "
    jobs.append(template)

    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset tiny_imagenet --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/tiny_imagenet/dntiny_imagenet-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --tag const_eta --train_acc --model dn --lr 0.01 --reg 0.001 "
    jobs.append(template)
    
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset tiny_imagenet --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/tiny_imagenet/rntiny_imagenet-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --tag  const_eta --train_acc --model rn --lr 0.01 --reg 0.001 "
    jobs.append(template)
    '''

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar10 --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/cifar10/dncifar10-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --use_tensorboard --tag without_scalars --train_acc --model dn --lr 0.01 --reg 0.001 "
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



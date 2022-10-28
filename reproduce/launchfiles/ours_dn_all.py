import os
import time
import yaml

def create_jobs():

    jobs = []
    '''
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar10  --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/cifar10/dncifar10-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --use_tensorboard --tag ours_final --train_acc "
    list_jobs(template, jobs)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar100 --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/cifar100/dncifar100-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --use_tensorboard --tag ours_final --train_acc "
    list_jobs(template, jobs)

    '''
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset tiny_imagenet --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/tiny_imagenet/dntiny_imagenet-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.0-type-convex_hull-pretrain_final/best_model.pkl --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --freeze 190 191 192 193 194 195 196 197 198 199 200 --use_tensorboard --tag ours_final --train_acc "
    list_jobs(template, jobs)

    return jobs


def list_jobs(template, jobs):

    #wrn_opts = " --depth 40 --width 4 --epochs 200 "
    #rn_opts = " --width 1 "
    #dn_opts = " --depth 40 --growth 40 --epochs 300 "
    #mlp_opts = " "

    wrn_opts = " "
    rn_opts =  " "
    dn_opts =  " "
    mlp_opts = " "

    with open("reproduce/test_pretrain.yaml", "r") as f:
        hparams = yaml.safe_load(f)
    for hparam in hparams:
        command = template + " ".join("--{} {}".format(key, value) for key, value in hparam.items())
        if hparam['model'] == "wrn":
            command += wrn_opts
        elif hparam['model'] == "rn":
            command += rn_opts
        elif hparam['model'] == "lrn":
            command += rn_opts
        elif hparam['model'] == "mlp":
            command += mlp_opts
        elif hparam['model'] == "dn":
            command += dn_opts
        else:
            raise ValueError("Model {} not recognized".format(hparam["model"]))
        jobs.append(command)
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



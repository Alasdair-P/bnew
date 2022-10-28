import os
import time
import yaml

def create_jobs():

    jobs = []
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template += "python binary_main.py --dataset cifar10  --proxquant --reg-type none --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --tag floating_point_weight --lr 0.01 --model lrn --train_acc  "
    jobs.append(template)

    '''
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar10  --proxquant --reg-type none --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --tag floating_point_weight "
    list_jobs(template, jobs)


    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template += "python binary_main.py --dataset cifar100  --proxquant --reg-type none --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --tag floating_point_weight "
    list_jobs(template, jobs)

    
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template += "python binary_main.py --dataset tiny_imagenet --proxquant --reg-type none --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --tag floating_point_weight "
    list_jobs(template, jobs)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar10 --proxquant --reg-type convex_hull --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --tag pretrain_final "
    list_jobs(template, jobs)
    
    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset cifar100 --proxquant --reg-type convex_hull --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --tag pretrain_final "
    list_jobs(template, jobs)

    template = "CUDA_VISIBLE_DEVICES=0 taskset -c 0-17,32-53 "
    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset tiny_imagenet --proxquant --reg-type convex_hull --relu ste --opt adam --lr_mode step --T 100 150 --epochs 200 --batch-size 128 --tag pretrain_final "
    list_jobs(template, jobs)
    '''

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

    with open("reproduce/grid_pretrain_all.yaml", "r") as f:
    #with open("reproduce/pretrain_dn.yaml", "r") as f:
        hparams = yaml.safe_load(f)
    for hparam in hparams:
        command = template + " ".join("--{} {}".format(key, value) for key, value in hparam.items())
        if hparam['model'] == "wrn":
            command += wrn_opts
        elif hparam['model'] == "rn":
            command += rn_opts
        elif hparam['model'] == "lrn":
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



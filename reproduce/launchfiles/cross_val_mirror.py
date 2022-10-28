import os
import time
import yaml

def create_jobs():

    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 python binary_main.py --dataset cifar100 --proxquant --reg-type mirror_binary --relu ste --opt adam --lr_mode step --T 200 300 --epochs 400 --batch-size 256 --use_tensorboard --tag mirror --freeze 390 391 392 393 394 395 396 397 398 399 400 "

    jobs = []

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
    with open("reproduce/grid_wrn_mirror.yaml", "r") as f:
        hparams = yaml.safe_load(f)
    for hparam in hparams:
        command = template + " ".join("--{} {}".format(key, value) for key, value in hparam.items())
        if hparam['model'] == "wrn":
            command += wrn_opts
        elif hparam['model'] == "rn":
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



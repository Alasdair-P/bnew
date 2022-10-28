import os
import time
import yaml

def create_jobs():

    jobs = []

    template = "CUDA_VISIBLE_DEVICES=1 taskset -c 18-31,54-71 "
    template += "python binary_main.py --dataset tiny_imagenet --proxquant --reg-type our_binary --resume /home/aparen/quantized-scnn/projects/pytorch_fscnn/results/tiny_imagenet/lrntiny_imagenet-adam--lr-step-0.01--wd-0.0--b-128-epoch-200-reg-0.001-type-our_binary-sgd_pretrain/best_model.pkl --relu ste --eval --batch-size 128 --tag ours_test --model lrn "
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



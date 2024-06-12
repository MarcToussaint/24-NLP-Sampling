from multiprocessing import Pool
import subprocess

from defs import *

#jobs = [jobs[0]]
#for j in jobs:
#    print(j)

#https://stackoverflow.com/a/59162124
#https://stackoverflow.com/a/9874484/4279

N_JOBS = 64

def run(*args):
    job = (args[0], *args[1])
    cmd = ['./x.exe', '-ex/verbose', '0', '-sam/verbose', '0']
    for a,b in zip(job, cmdjob):
        cmd.append('-' + b)
        cmd.append(a)
    if job[0]=='push':
        cmd.append('-sam/downhillMaxSteps')
        cmd.append('100')
    if job[1]=='grad':
        cmd.append('-sam/slackStepAlpha')
        cmd.append('.1')
    # if job[-1]=='dist':
    #     cmd.append('-sam/seedCandidates')
    #     cmd.append('100')
    print('running: ', cmd)
    subprocess.run(cmd)
    return cmd

with Pool(N_JOBS) as p:
    p.starmap(run, jobs)


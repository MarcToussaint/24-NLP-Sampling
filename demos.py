import subprocess
import matplotlib.pyplot as plt
import numpy as np

def run(args):
    cmd = ['./x.exe', '-ex/outPath', 'ex_demo', '-ex/verbose', '0', '-sam/verbose', '0']
    for key, value in args.items():
        cmd.append('-'+key)
        cmd.append(value)
    # cmd.append('-slackStepAlpha')
    # cmd.append('.1')
    print('running: ', cmd)
    subprocess.run(cmd)
    return cmd

def plot_scatter(datafile, title, pdffile, x_limits=[-2,2], y_limits=[-2,2]):
    samples = np.loadtxt(datafile)
    # plt.axes('square')
    fig = plt.figure(figsize=(1.5*(x_limits[1]-x_limits[0]),1.5*(y_limits[1]-y_limits[0])))
    ax = fig.add_subplot()
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_aspect('equal')
    ax.scatter(samples[:,0], samples[:,1], s=1)
    ax.set_title(title)
    fig.tight_layout(pad=0)
    fig.savefig(pdffile, format='pdf', pad_inches=0)
    plt.show()

def plot_msts(datafile, msts_p, title, pdffile):
    run = np.loadtxt(datafile)
    fig, ax = plt.subplots()
    ax.set_xlabel('samples')
    ax.set_ylabel('MSTS')
    ax.plot(run[:,0], run[:,1+1], label='MSTS$_1$') #sample eval msts1 msts2
    ax.plot(run[:,0], run[:,1+2], label='MSTS$_2$') #sample eval msts1 msts2
    plt.legend(loc='best')
    fig.tight_layout(rect=[0.01, 0., .99, 0.95], pad=0)
    plt.title(title)
    fig.savefig(pdffile, format='pdf', pad_inches=0)
    plt.show()

def main():
    args = {
        'problemCosts': '0',
        'ex/problem': 'box.2',
        'ex/runs': '1',
        'ex/maxEvals': '1000000',
        'sam/slackMaxStep': '2.',
        'sam/interiorBurnInSteps': '0',
        'sam/interiorSampleSteps': '1',
        'sam/ineqOverstep': '1.',
        'sam/interiorMethod': 'HR' }

    # run(args)
    # plot_scatter('ex_demo/samples.0.dat', 'downhill only (E/S: 1.75)', 'demo_ineqOverstep_1.pdf')

    # args['sam/ineqOverstep'] = '1.2'
    # run(args)
    # plot_scatter('ex_demo/samples.0.dat', 'downhill, ineqOverstep=1.2 (E/S: 1.75)', 'demo_ineqOverstep_12.pdf')

    # args['sam/ineqOverstep'] = '1'
    # args['sam/interiorBurnInSteps'] = '1'
    # run(args)
    # plot_scatter('ex_demo/samples.0.dat', 'NHR, K_burn=1 (E/S: 2.61)', 'demo_Kburn_1.pdf')

    # args['sam/interiorBurnInSteps'] = '2'
    # run(args)
    # plot_scatter('ex_demo/samples.0.dat', 'NHR, K_burn=2 (E/S: 3.55)', 'demo_Kburn_2.pdf')

    # # f>0
    
    args['problemCosts'] = '1'
    args['sam/interiorBurnInSteps'] = '2'
    run(args)
    plot_scatter('ex_demo/samples.0.dat', 'f>0, NHR, K_burn=2 (E/S: 3.54, EMD: 0.81)', 'demo_f_HR_Kburn_2.pdf')

    args['sam/interiorBurnInSteps'] = '10'
    run(args)
    plot_scatter('ex_demo/samples.0.dat', 'f>0, NHR, K_burn=10 (E/S: 11.43, EMD: 0.25)', 'demo_f_HR_Kburn_10.pdf')

    args['sam/interiorBurnInSteps'] = '50'
    run(args)
    plot_scatter('ex_demo/samples.0.dat', 'f>0, NHR, K_burn=50 (E/S: 51.41, EMD: 0.04)', 'demo_f_HR_Kburn_50.pdf')

    args['sam/interiorMethod'] = 'MCMC'
    args['sam/interiorBurnInSteps'] = '2'
    run(args)
    plot_scatter('ex_demo/samples.0.dat', 'f>0, MCMC, K_burn=2 (E/S: 3.76, EMD: 1.03)', 'demo_f_MCMC_Kburn_2.pdf')

    args['sam/interiorBurnInSteps'] = '10'
    run(args)
    plot_scatter('ex_demo/samples.0.dat', 'f>0, MCMC, K_burn=10 (E/S: 11.73, EMD: 0.50)', 'demo_f_MCMC_Kburn_10.pdf')

    args['sam/interiorBurnInSteps'] = '50'
    run(args)
    plot_scatter('ex_demo/samples.0.dat', 'f>0, MCMC, K_burn=50 (E/S: 51.75, EMD: 0.03)', 'demo_f_MCMC_Kburn_50.pdf')

    args['sam/interiorMethod'] = 'Langevin'
    args['sam/interiorBurnInSteps'] = '2'
    run(args)
    plot_scatter('ex_demo/samples.0.dat', 'f>0, Langevin, K_burn=2 (E/S: 5.75, EMD: 0.99)', 'demo_f_Langevin_Kburn_2.pdf')

    args['sam/interiorBurnInSteps'] = '10'
    run(args)
    plot_scatter('ex_demo/samples.0.dat', 'f>0, Langevin, K_burn=10 (E/S: 21.77, EMD: 0.29)', 'demo_f_Langevin_Kburn_10.pdf')

    args['sam/interiorBurnInSteps'] = '50'
    run(args)
    plot_scatter('ex_demo/samples.0.dat', 'f>0, Langevin, K_burn=50 (E/S: 101.77, EMD: 0.03)', 'demo_f_Langevin_Kburn_50.pdf')

    # modes

    # args['ex/problem'] = 'modes.2'
    # args['problemCosts'] = '0'
    # args['sam/interiorMethod'] = 'HR'
    # args['sam/interiorBurnInSteps'] = '0'
    # run(args)
    # plot_scatter('ex_demo/samples.0.dat', 'modes.2, K_burn=0 (E/S: 3.26)', 'demo_modes2_HR.pdf', x_limits=[-1.2, 1.2], y_limits=[-1.2,1.2])
    # plot_msts('ex_demo/run.0.dat', 1, 'modes.2, K_burn=0', 'demo_modes2_HR_msts.pdf')

    # args['sam/interiorBurnInSteps'] = '5'
    # run(args)
    # plot_scatter('ex_demo/samples.0.dat', 'modes.2, NHR, K_burn=5 (E/S: 9.44)', 'demo_modes2_HR5.pdf', x_limits=[-1.2, 1.2], y_limits=[-1.2,1.2])
    # plot_msts('ex_demo/run.0.dat', 1, 'modes.2, NHR, K_burn=5', 'demo_modes2_HR5_msts.pdf')

    # args['ex/problem'] = 'modes.6'
    # args['sam/interiorBurnInSteps'] = '5'
    # run(args)
    # plot_scatter('ex_demo/samples.0.dat', 'modes.6, NHR5 (E/S: 10.27)', 'demo_modes6_HR5.pdf', x_limits=[-1.2, 1.2], y_limits=[-1.2,1.2])
    # plot_msts('ex_demo/run.0.dat', 1, 'modes.6, NHR5', 'demo_modes6_HR5_msts.pdf')

    # args['ex/problem'] = 'modes.6'
    # args['sam/interiorBurnInSteps'] = '5'
    # args['sam/seedMethod'] = 'dist'
    # args['sam/seedCandidates'] = '100'
    # run(args)
    # plot_scatter('ex_demo/samples.0.dat', 'modes.6, NHR5+dist (E/S: 10.43)', 'demo_modes6_HR5dist.pdf', x_limits=[-1.2, 1.2], y_limits=[-1.2,1.2])
    # plot_msts('ex_demo/run.0.dat', 1, 'modes.6, NHR5+dist', 'demo_modes6_HR5dist_msts.pdf')

    # args['ex/problem'] = 'modes.6'
    # args['sam/interiorBurnInSteps'] = '5'
    # args['sam/seedMethod'] = 'nov'
    # args['sam/seedCandidates'] = '10'
    # run(args)
    # plot_scatter('ex_demo/samples.0.dat', 'modes.6, NHR5+nov (E/S: 18.79)', 'demo_modes6_HR5nov.pdf', x_limits=[-1.2, 1.2], y_limits=[-1.2,1.2])
    # plot_msts('ex_demo/run.0.dat', 1, 'modes.6, NHR5+nov', 'demo_modes6_HR5nov_msts.pdf')

    # IK

    args['ex/problem'] = 'IK'
    args['problemCosts'] = '1'
    args['sam/interiorMethod'] = 'manifoldRRT'
    args['sam/interiorBurnInSteps'] = '0'
    run(args)
    plot_scatter('ex_demo/samples.0.dat', 'IK, K_burn=0 (E/S: 25.79)', 'demo_IK_mRRT.pdf', x_limits=[-3., 3.], y_limits=[-2.,2.])

    args['sam/interiorBurnInSteps'] = '2'
    run(args)
    plot_scatter('ex_demo/samples.0.dat', 'IK, mRRT, K_burn=2 (E/S: 33.78)', 'demo_IK_mRRT_Kburn_2.pdf', x_limits=[-3., 3.], y_limits=[-2.,2.])

    args['sam/interiorBurnInSteps'] = '10'
    run(args)
    plot_scatter('ex_demo/samples.0.dat', 'IK, mRRT, K_burn=10 (E/S: 55.08)', 'demo_IK_mRRT_Kburn_10.pdf', x_limits=[-3., 3.], y_limits=[-2.,2.])

    args['sam/interiorMethod'] = 'Langevin'
    args['sam/langevinTauPrime'] = '.01'
    args['sam/interiorBurnInSteps'] = '2'
    run(args)
    plot_scatter('ex_demo/samples.0.dat', 'IK, Langevin, K_burn=2 (E/S: 336.26)', 'demo_IK_Langevin_Kburn_2.pdf', x_limits=[-3., 3.], y_limits=[-2.,2.])

    args['sam/interiorBurnInSteps'] = '10'
    run(args)
    plot_scatter('ex_demo/samples.0.dat', 'IK, Langevin, K_burn=10 (E/S: 671.85)', 'demo_IK_Langevin_Kburn_10.pdf', x_limits=[-3., 3.], y_limits=[-2.,2.])

if __name__ == "__main__":
    main()

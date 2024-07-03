from defs import *
import numpy as np
import os.path
import matplotlib.pyplot as plt

#for j in jobs:
#    print(j)
#exit()

MSTS_p = 1
max_run_samples = 10000
use_avg_xlim = False

def eval(path):
    return 0

def get_mean_dev_final(traces, samples_idx=0, evals_idx=1, msts_idx=MSTS_p+1):
    n = len(traces)
    x_values = []
    y_values = []
    for trace in traces:
        x_values.append(trace[-1, samples_idx] / trace[-1, evals_idx] )
        y_values.append(trace[-1, msts_idx] / trace[-1, evals_idx] )

    return [np.mean(x_values), np.mean(y_values), np.std(x_values), np.std(y_values)]
    
def get_mean_dev(traces, xidx=1, yidx=MSTS_p+1):
    n = len(traces)
    min_x = 1e10
    max_x = 0
    for trace in traces:
        min_x = min(min_x, min(trace[:,xidx]))
        max_x = max(max_x, max(trace[:,xidx]))

    x_grid = np.linspace(min_x, max_x, 50)
    y_mean = np.zeros(x_grid.shape)
    y_sqr = np.zeros(x_grid.shape)
    y_values = []

    for i in range(0, n):
        X = traces[i][:,xidx]
        Y = traces[i][:,yidx]
        y_values = np.interp(x_grid, X, Y)
        y_mean += y_values
        y_sqr += np.square(y_values)

        # plt.plot(X, Y)
        # plt.plot(x_grid, y_values)

    y_mean /= n
    y_sdv = np.sqrt(y_sqr/n - np.square(y_mean))
    # print('minmax:', min_x, max_x, x_grid[0], x_grid[-1])
    return x_grid, y_mean, y_sdv

def plot(traces, tag):
    x_grid, y_mean, y_sdv = get_mean_dev(traces)

    plt.xlabel('evaluations')
    plt.ylabel(f'MSTS$_{MSTS_p}$')
    plt.plot(x_grid, y_mean, label=tag)
    plt.fill_between(x_grid, y_mean-y_sdv, y_mean+y_sdv, alpha=.2)
    plt.legend(loc='best')
    # plt.plot(a2[:,0], a2[:,1],c='black')
    # plt.plot(midx, midy, '--', c='black')
    plt.show()

def color_by_key(key):
    if '+0+1_' in key:
        return 'C1'
    elif 'HR' in key:
        return 'C2'
    elif 'manifoldRRT' in key:
        return 'C3'
    elif 'Langevin' in key:
        return 'C4'
    return 'C0'

def name(key):
    n = key
    n = n.replace('ex_', '')
    n = n.replace('+none', '')
    n = n.replace('_100+', '_50+')
    n = n.replace('_HR_50+0+1_', '_') #_none')
    n = n.replace('_HR_50', '_NHR')
    n = n.replace('_MCMC_50', '_MCMC')
    n = n.replace('_Langevin_50', '_Langevin')
    n = n.replace('_manifoldRRT_50', '_mRRT')
    n = n.replace('+0+', '+')
    n = n.replace('+', '.')
    n = n.replace('_uni10', '_uni')
    n = n.replace('_dist10', '_dist.10')
    n = n.replace('_nov10', '_align.10')
    return n

def plot_traces(traces, keys, legend=True, labelsize=7, title=None):
    plt.clf()
    plt.xlabel('evaluations')
    plt.ylabel(f'MSTS$_{MSTS_p}$')
    if use_avg_xlim:
        lim = 0
        for tr in traces:
            lim += tr[0][-1]
        lim = lim/len(traces)
        plt.xlim(0,lim)
    for tr, k in zip(traces, keys):
        X = tr[0]+1
        if legend:
            plt.plot(X, tr[1], label=k) #, color=color_by_key(k))
        else:
            plt.plot(X, tr[1]) #, color=color_by_key(k))
            plt.annotate(name(k), (X[-1], tr[1][-1]), fontsize=labelsize, ha='right')
        plt.fill_between(X, tr[1]-tr[2], tr[1]+tr[2], alpha=.2)
    # plt.legend(loc='best')
    # plt.plot(a2[:,0], a2[:,1],c='black')
    # plt.plot(midx, midy, '--', c='black')
    if title:
        plt.title(title)
        plt.savefig(f'{title}-trace.pdf', format='pdf')
    #plt.show()

def plot_perf(finals, keys, labelsize=7, title=None):
    plt.clf()
    F = np.array(finals)
    fig, ax = plt.subplots()
    ax.set_xlabel('samples/evals')
    ax.set_ylabel(f'MSTS$_{MSTS_p}$/evals')
    # print(F.shape, len(keys))
    for f,k in zip(F,keys):
        ax.errorbar(f[0], f[1], xerr=f[2], yerr=f[3], ls='none', capsize=2, marker='s', ms=2, lw=0.5, mew=0.5, color=color_by_key(k))
    for f,k in zip(finals, keys):
        plt.annotate(name(k), (f[0], f[1]), fontsize=labelsize)
    ax.set_xlim(left=0.)
    ax.set_ylim(bottom=0.)
    _,X = ax.get_xlim()
    _,Y = ax.get_ylim()
    Z = Y / X
    x, y = np.meshgrid(np.linspace(0.05*X, X, 100), np.linspace(0.05*Y, Y, 100))
    z = y / x
    CS = ax.contour(x, y, z, levels=np.linspace(Z/5, Z*3, 7))
    ax.clabel(CS, inline=True, fontsize=3, colors=('0.8'))
    for c in CS.collections:
        c.set_linewidth(0.1)
        c.set_color('0.8')
    if title:
        plt.title(title)
        plt.savefig(f'{title}-perf.pdf', format='pdf')
    # plt.show()

def main():
    for p in problem:
        traces = []
        finals = []
        keys = []
        for m in methods:
            downhillMaxSteps = '50'
            if p=='push':
                downhillMaxSteps = '100'
            seedCandidates = '10'
            #if m[6]=='dist':
            #    seedCandidates = '100'
            path = 'ex_'+p+'_'+m[0]+'+'+m[1]+'+'+m[2]+'_'+m[3]+'_'+downhillMaxSteps+'+'+m[4]+'+'+m[5]+'_'+m[6]+seedCandidates #box.2_GN+cov+MH_HR_50+20+20/
            print(p, m, path)
            runs = []
            for i in range(0, num_runs):
                filename = f'{path}/run.{i}.dat'
                if os.path.isfile(filename):
                    run = np.loadtxt(filename)
                    if run.ndim==2:
                        if run.shape[0]>max_run_samples:
                            run = run[:max_run_samples, :]
                        runs.append(run)
                    else:
                        print('  -- skip: run is empty ', p, m, i)
                else:
                    print('  -- skip: run does not exist ', p, m, i)
                    break
            if len(runs)>3:
                keys.append(path)
                
                # store the mean and sdv trace of all runs:
                traces.append(get_mean_dev(runs))

                # store the mean and sdv FINAL scores:
                finals.append(get_mean_dev_final(runs))
            else:
                print('too few #runs:', len(runs))

        plot_traces(traces, keys, legend=False, labelsize=1, title=p)
        plot_perf(finals, keys, labelsize=1, title=p)

if __name__ == "__main__":
    main()
    #print('#experiments', len(list(jobs)))

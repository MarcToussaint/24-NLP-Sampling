from defs import *
from eval_helpers import *
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

# Constants and configuration
MSTS_p = 1
max_run_samples = 10000
use_avg_xlim = False

# Ensure the pickle directory exists
if not os.path.exists(PICKLE_PATH):
    os.makedirs(PICKLE_PATH)

def get_mean_dev_final(traces, samples_idx=0, evals_idx=1, msts_idx=MSTS_p+1):
    n = len(traces)
    x_values = []
    y_values = []
    for trace in traces:
        x_values.append(trace[-1, samples_idx] / trace[-1, evals_idx] )
        y_values.append(trace[-1, msts_idx] / trace[-1, evals_idx] )

    return [np.mean(x_values), np.mean(y_values), np.std(x_values), np.std(y_values)]


# Calculate mean and standard deviation across traces
def get_mean_dev(traces, xidx=1, yidx=MSTS_p + 1):
    min_x, max_x = np.inf, 0
    for trace in traces:
        min_x, max_x = min(min_x, trace[:, xidx].min()), max(max_x, trace[:, xidx].max())
    x_grid = np.linspace(min_x, max_x, 50)
    y_mean, y_sqr = np.zeros(x_grid.shape), np.zeros(x_grid.shape)

    for trace in traces:
        y_values = np.interp(x_grid, trace[:, xidx], trace[:, yidx])
        y_mean += y_values
        y_sqr += y_values ** 2

    y_mean /= len(traces)
    y_sdv = np.sqrt(y_sqr / len(traces) - y_mean ** 2)
    return x_grid, y_mean, y_sdv


# Plot traces with mean and standard deviation
def plot_traces(traces, keys, legend=True, labelsize=7, title=None):
    plt.clf()
    plt.xlabel('evaluations')
    plt.ylabel(f'MSTS$_{MSTS_p}$')
    if use_avg_xlim:
        plt.xlim(0, sum(tr[0][-1] for tr in traces) / len(traces))
    for tr, k in zip(traces, keys):
        X = tr[0] + 1
        plt.plot(X, tr[1], label=k if legend else None, color=color_by_key(k))
        if not legend:
            plt.annotate(name(k), (X[-1], tr[1][-1]), fontsize=labelsize, ha='right')
        plt.fill_between(X, tr[1]-tr[2], tr[1]+tr[2], alpha=.2, color=color_by_key(k))
    if legend:
        plt.legend(loc='best')
    # plt.plot(a2[:,0], a2[:,1],c='black')
    # plt.plot(midx, midy, '--', c='black')
    if title:
        plt.title(title)
        plt.savefig(f'{title}-trace.pdf', format='pdf')
    #plt.show()


# Plot performance with error bars
def plot_perf(finals, keys, labelsize=7, title=None):
    plt.clf()
    #F = np.array(finals)
    fig, ax = plt.subplots()
    ax.set_xlabel('samples/evals')
    ax.set_ylabel(f'MSTS$_{MSTS_p}$/evals')
    #print(F.shape, len(keys))
    #print(list(zip(F,keys)))
    for f, k in zip(finals, keys):
        for runf in f:
            ax.errorbar(runf[0], runf[1], xerr=runf[2], yerr=runf[3], ls='none', capsize=2, alpha=.5, marker='s', ms=2, lw=0.5, mew=0.5, color=color_by_key(k), label=k)
        #plt.annotate(name(k), f[:, :2].mean(axis=0), fontsize=labelsize)
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
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=labelsize)
        plt.title(title)
        plt.savefig(f'{title}-perf.pdf', format='pdf')
    # plt.show()


# Save and load functions for pickled results
def save_results(filename, data):
    with open(os.path.join(PICKLE_PATH, filename), 'wb') as f:
        pickle.dump(data, f)


def load_results(filename):
    with open(os.path.join(PICKLE_PATH, filename), 'rb') as f:
        return pickle.load(f)


# Aggregate and plot results based on different conditions
def plot_aggregated_results():
    conditions = [
        ('downhill method', downhillMethod),
        ('downhill noise method', downhillNoiseMethod),
        ('downhill reject method', downhillRejectMethod),
        ('interior sampling method', interiorMethod),
        ('interior sampling burn in steps', interiorBurnInSteps),
        ('interior samples taken', interiorSampleSteps),
        ('restart seed method', seedMethod)
    ]

    for p in problem:
        for condition, values in conditions:  # Go over each of the 7 conditions
            traces, finals, keys = [], [], []
            for val in values:
                filename = f"{p}_{condition}_{val}.pkl"   # Storage path for aggregated results
                if os.path.exists(os.path.join(PICKLE_PATH, filename)) and LOAD_DATA:
                    aggregated_traces, aggregated_finals = load_results(filename)
                    print("Using loaded data")
                else:
                    # For each condition aggregate all results over all methods
                    aggregated_traces = []
                    aggregated_finals = []
                    aggregated_keys = []
                    for m in methods:
                        downhillMaxSteps = '50' if p != 'push' else '100'
                        seedCandidates = '10'
                        if val in m:
                            path = f"{DATA_PATH}/ex_{p}_{m[0]}+{m[1]}+{m[2]}_{m[3]}_{downhillMaxSteps}+{m[4]}+{m[5]}_{m[6]}{seedCandidates}"
                            runs = [np.loadtxt(f"{path}/run.{i}.dat") for i in range(num_runs) if os.path.isfile(f"{path}/run.{i}.dat")]
                            runs = [run if run.ndim == 2 and len(run) <= max_run_samples else run[:max_run_samples] for
                                    run in runs if run.ndim == 2]
                            if len(runs) > 3:
                                aggregated_traces.extend(runs)
                                aggregated_finals.append(get_mean_dev_final(runs))
                                aggregated_keys.append(path)
                            # else:
                            #     print('too few #runs:', len(runs))

                    # Store
                    save_results(filename, (aggregated_traces, aggregated_finals))

                # Now get the meta mean: we aggregate across all traces
                if aggregated_traces:
                    traces.append(get_mean_dev(aggregated_traces)) # For traces we can simply do the same as aggregating across runs, but now over all methods
                    keys.append(val)
                if aggregated_finals:
                    finals.append(aggregated_finals)  # For finals, we simply aggregate

            # Use data for plotting
            plot_traces(traces, keys, legend=True, labelsize=7, title=f"{p} - {condition}")
            plot_perf(finals, keys, labelsize=7, title=f"{p} - {condition}")


if __name__ == "__main__":
    plot_aggregated_results()

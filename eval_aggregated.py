from defs import *
from eval_helpers import *
import numpy as np
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Constants and configuration
MSTS_p = 1
max_run_samples = 10000
use_avg_xlim = False


def get_mean_dev_final(traces, samples_idx=0, evals_idx=1, msts_idx=MSTS_p + 1):
    x_values = [trace[-1, samples_idx] / trace[-1, evals_idx] for trace in traces]
    y_values = [trace[-1, msts_idx] / trace[-1, evals_idx] for trace in traces]
    z_values = [np.nan_to_num(trace[-1, msts_idx] / trace[-1, samples_idx], nan=0.) for trace in traces]
    return [np.mean(x_values), np.mean(y_values), np.std(x_values), np.std(y_values), np.mean(z_values), np.std(z_values)]


def get_mean_dev(traces, evals_idx=1, yidx=MSTS_p + 1):
    min_x, max_x = np.inf, 0
    for trace in traces:
        min_x, max_x = min(min_x, trace[:, evals_idx].min()), max(max_x, trace[:, evals_idx].max())

    x_grid = np.linspace(min_x, max_x, 50)
    y_mean, y_sqr = np.zeros(x_grid.shape), np.zeros(x_grid.shape)

    for trace in traces:
        y_values = np.interp(x_grid, trace[:, evals_idx], trace[:, yidx])
        y_mean += y_values
        y_sqr += y_values ** 2

    y_mean /= len(traces)
    y_sdv = np.sqrt(y_sqr / len(traces) - y_mean ** 2)
    return x_grid, y_mean, y_sdv


def plot_traces(traces, keys, legend=True, labelsize=7, title=None):
    plt.clf()
    plt.xlabel('evaluations')
    plt.ylabel(f'MSTS$_{MSTS_p}$')
    if use_avg_xlim:
        plt.xlim(0, sum(tr[0][-1] for tr in traces) / len(traces))
    for tr, k in zip(traces, keys):
        X = tr[0] + 1
        plt.plot(X, tr[1], label=k if legend else None, color=color_by_key(k))
        plt.fill_between(X, tr[1] - tr[2], tr[1] + tr[2], alpha=.2, color=color_by_key(k))
    if legend:
        plt.legend(loc='best')
    if title:
        plt.title(title)
        plt.savefig(f'{title}-trace.pdf', format='pdf')
    # plt.show()


def plot_perf(finals, keys, labelsize=7, title=None):
    plt.clf()
    fig, ax = plt.subplots()
    ax.set_xlabel('samples/evals')
    ax.set_ylabel(f'MSTS$_{MSTS_p}$/evals')
    legend_labels = set()  # Set to keep track of labels already added to the legend
    for f, k in zip(finals, keys):
        for runf in f:
            if k not in legend_labels:
                ax.errorbar(runf[0], runf[1], xerr=runf[2], yerr=runf[3], ls='none', capsize=2, alpha=.5, marker='s',
                            ms=2, lw=0.5, mew=0.5, color=color_by_key(k), label=name(k))
                legend_labels.add(k)
            else:
                ax.errorbar(runf[0], runf[1], xerr=runf[2], yerr=runf[3], ls='none', capsize=2, alpha=.5, marker='s',
                            ms=2, lw=0.5, mew=0.5, color=color_by_key(k))
    ax.set_xlim(left=0.)
    ax.set_ylim(bottom=0.)
    _, X = ax.get_xlim()
    _, Y = ax.get_ylim()
    Z = Y / X
    x, y = np.meshgrid(np.linspace(0.05 * X, X, 100), np.linspace(0.05 * Y, Y, 100))
    z = y / x
    CS = ax.contour(x, y, z, levels=np.linspace(Z / 5, Z * 3, 7))
    ax.clabel(CS, inline=True, fontsize=3, colors=('0.8'))
    for c in CS.collections:
        c.set_linewidth(0.1)
        c.set_color('0.8')
    if title:
        plt.legend(loc='best')
        plt.title(title)
        plt.savefig(f'{title}-perf.pdf', format='pdf')
    # plt.show()


def plot_cdf(traces, keys, metric_idx=MSTS_p + 1, title=None):
    plt.clf()
    plt.xlabel('CDF')
    plt.ylabel(f'MSTS$_{MSTS_p}$')
    for tr, k in zip(traces, keys):
        X = tr[0] + 1
        cdf = np.arange(X.shape[0]) / float(X.shape[0] - 1)
        plt.plot(cdf, tr[1], label=k, color=color_by_key(k))
        plt.fill_between(cdf, tr[1] - tr[2], tr[1] + tr[2], alpha=.2, color=color_by_key(k))
    if title:
        plt.legend()
        plt.title(title)
        plt.savefig(f'{title}-cdf.pdf', format='pdf')
    # plt.show()


def aggregate_results():
    n_runs = 0
    fn = "all_probs.pkl"
    if os.path.exists(os.path.join(PICKLE_PATH, fn)) and LOAD_DATA:
        aggregated_traces, aggregated_finals = load_results(fn)
        print(f"Using loaded data for {fn}")
    else:
        aggregated_traces = {}
        aggregated_finals = {}
        for p in problem:
            aggregated_traces[p] = {}
            aggregated_finals[p] = {}
            for m in methods:
                downhillMaxSteps = '50' if p != 'push' else '100'
                seedCandidates = '10'
                path = f"{DATA_PATH}/ex_{p}_{m[0]}+{m[1]}+{m[2]}_{m[3]}_{downhillMaxSteps}+{m[4]}+{m[5]}_{m[6]}{seedCandidates}"
                runs = [np.loadtxt(f"{path}/run.{i}.dat") for i in range(num_runs) if os.path.isfile(f"{path}/run.{i}.dat")]
                runs = [run if run.ndim == 2 and len(run) <= max_run_samples else run[:max_run_samples] for run in runs if run.ndim == 2]
                if len(runs) > 3:
                    n_runs += len(runs)
                    aggregated_traces[p][m] = runs  # Store runs in the dictionary
                    aggregated_finals[p][m] = get_mean_dev_final(runs)
                # else:
                #     print('too few #runs:', len(runs), p, m)

        save_results(fn, (aggregated_traces, aggregated_finals))

    print("Found runs ", n_runs)

    return aggregated_traces, aggregated_finals


def count_best_performers(full_finals, keys, condition_idx):
    best_counts_samples = {key: [] for key in keys}
    best_counts_msts = {key: [] for key in keys}
    best_counts_mps = {key: [] for key in keys}
    for p in problem:
        for method_combination in full_finals[p].keys():
            results = []
            for k in keys:
                relevant_condition = method_combination[:condition_idx] + tuple([k]) + method_combination[condition_idx+1:]
                relevant_data = full_finals[p].get(relevant_condition, None)
                results.append(relevant_data)
            if None not in results:
                results = np.array(results)
                best_results = results.max(axis=0)
                percentages = results / best_results

                for i, key in enumerate(best_counts_samples.keys()):
                    best_counts_samples[key].append(percentages[i, 0])
                    best_counts_msts[key].append(percentages[i, 1])
                    best_counts_mps[key].append(percentages[i, -2])
            # else:
            #     print("No comparison possible for ", relevant_condition)

    return best_counts_samples, best_counts_msts, best_counts_mps


def plot_best_performers(best_counts_samples, best_counts_msts, best_counts_mps, condition):
    labels = ['Samples/Eval', 'MSTS/Eval']
    x = np.arange(len(labels))
    width = 0.2  # Adjust width to fit all methods

    fig, ax = plt.subplots()

    # Number of methods
    n_methods = len(best_counts_samples)

    # Create bar positions for each method
    bar_positions = [x + (i - n_methods / 2) * width for i in range(n_methods)]

    # Plot bars for each method
    label_set = set()
    for i, (method, color) in enumerate(zip(best_counts_samples.keys(), [color_by_key(key) for key in best_counts_samples.keys()])):
        sample_count = best_counts_samples[method]
        msts_count = best_counts_msts[method]
        # mps_count = best_counts_mps[method]

        if method not in label_set:
            ax.bar(bar_positions[i][0], np.mean(sample_count), width, label=name(method), color=color_by_key(method))
            label_set.add(method)
        else:
            ax.bar(bar_positions[i][0], np.mean(sample_count), width, color=color_by_key(method))
        ax.bar(bar_positions[i][1], np.mean(msts_count), width, color=color_by_key(method))
        #ax.bar(bar_positions[i][2], mps_count, width, color=color_by_key(method))

        # Errorbars
        plt.errorbar(bar_positions[i][0], np.mean(sample_count), np.std(sample_count), color='black', fmt='o', ms=2, lw=0.5, mew=0.5, alpha=.8)
        plt.errorbar(bar_positions[i][1], np.mean(msts_count), np.std(msts_count), color='black', fmt='o', ms=2, lw=0.5, mew=0.5, alpha=.8)

    ax.set_ylabel('Avg. \% of best method')
    ax.set_title(f'Performance comparison for {condition} II')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    legend = ax.legend(loc=8, frameon=True)
    legend.get_frame().set_alpha(.8)
    # legend.get_frame().set_facecolor((0, 0, 1, .8))
    # legend.get_frame().set_edgecolor('black')

    plt.tight_layout()
    plt.savefig(f'{condition}_best_performers.pdf', format='pdf')
    plt.show()


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

    # First aggregate all results
    aggregated_traces, aggregated_finals = aggregate_results()

    # Loop over conditions
    for condition_idx, (condition, values) in enumerate(conditions):
        keys = values

        # Collect traces for plotting
        condition_traces = []
        mean_dev_traces = []
        all_finals = []
        for i, val in enumerate(values):
            condition_traces.append([])
            all_finals.append([])
            for p in aggregated_traces:
                for m in aggregated_traces[p]:
                    if m[condition_idx] == val:
                        condition_traces[i].extend(aggregated_traces[p][m])
                        all_finals[i].append(aggregated_finals[p][m])
            mean_dev_traces.append(get_mean_dev(condition_traces[i]))

        # Plot traces and final performances
        plot_perf(all_finals, keys, labelsize=7, title=f"{condition}")
        plot_traces(mean_dev_traces, keys, legend=True, labelsize=7, title=f"{condition}")
        #plot_cdf(mean_dev_traces, keys, metric_idx=MSTS_p + 1, title=f"{condition}")

        # Plot head2head
        performance_counts = count_best_performers(aggregated_finals, keys, condition_idx)
        plot_best_performers(*performance_counts, condition)


if __name__ == "__main__":
    plot_aggregated_results()

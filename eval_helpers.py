import pickle
import os

import matplotlib.pyplot as plt
import scienceplots

# Style the plots
plt.style.use(['science', 'ieee', 'no-latex'])
# plt.rc('axes', labelsize=14)   # Axis labels
# plt.rc('xtick', labelsize=12)  # X-axis tick labels
# plt.rc('ytick', labelsize=12)  # Y-axis tick labels

LOAD_DATA = False
DATA_PATH = ""  # TODO: set correct data path
PICKLE_PATH = "./pickled_results"

PICKLE_PATH = "./pickled_results"

# Ensure the pickle directory exists
os.makedirs(PICKLE_PATH, exist_ok=True)

def color_by_key(key):
    # color_map = {
    #     'GN': 'brown', 'grad': 'teal',
    #     'none': 'orange', 'iso': 'cyan', 'cov': 'magenta',
    #     'HR': 'green', 'MCMC': 'blue', 'Langevin': 'purple', 'manifoldRRT': 'red',
    #     '0': 'pink', '5': 'gray', '20': 'olive',
    #     '1': 'black', '100': 'yellow',
    #     'uni': 'lime', 'dist': 'navy', 'nov': 'coral'
    # }
    color_map = {
        'GN': 'green', 'grad': 'blue',
        'none': 'green', 'iso': 'blue', 'cov': 'purple',
        'MH': 'blue', 'Wolfe': 'purple', 
        'HR': 'green', 'MCMC': 'blue', 'Langevin': 'purple', 'manifoldRRT': 'red',
        '0': 'green', '5': 'blue', '20': 'purple',
        '1': 'green', '100': 'red',
        'uni': 'green', 'dist': 'blue', 'nov': 'purple'
    }
    return color_map.get(key, 'C0')


def name(key):
    # return (key.replace('ex_', '').replace('+none', '').replace('_100+', '_50+')
    #         .replace('_HR_50+0+1_', '_').replace('_HR_50', '_NHR')
    #         .replace('_MCMC_50', '_MCMC').replace('_Langevin_50', '_Langevin')
    #         .replace('_manifoldRRT_50', '_mRRT').replace('+0+', '+')
    #         .replace('+', '.').replace('_uni10', '_uni').replace('_dist10', '_dist.10')
    #         .replace('_nov10', '_align.10'))
    name_map = {
        'GN': 'Gauss-Newton', 'grad': 'Plain Gradient', 'none': 'None', 'iso': 'Isometric', 'cov': 'Covariant',
        'HR': 'NHR', 'uni': 'Uniform', 'dist': 'Distance', 'nov': 'Novelty', 'manifoldRRT': 'mRRT'
    }
    return name_map.get(key, key)

def save_results(filename, data):
    with open(os.path.join(PICKLE_PATH, filename), 'wb') as f:
        pickle.dump(data, f)

def load_results(filename):
    with open(os.path.join(PICKLE_PATH, filename), 'rb') as f:
        return pickle.load(f)

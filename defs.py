import itertools

problem = ['box.2', 'modes.2', 'box.6', 'modes.6', 'linear-program', 'IK', 'cylinder-obstacle', 'push']
#problem = ['push']

downhillMethod = ['GN', 'grad']
downhillNoiseMethod = ['none', 'iso', 'cov']
downhillRejectMethod = ['none', 'Wolfe', 'MH']

interiorMethod = ['HR', 'manifoldRRT', 'MCMC', 'Langevin']

interiorBurnInSteps = ['0', '5', '20']
interiorSampleSteps = ['1', '5', '20', '100']

seedMethod = ['uni', 'dist', 'nov']

methods = itertools.product(downhillMethod,
                            downhillNoiseMethod,
                            downhillRejectMethod,
                            interiorMethod,
                            interiorBurnInSteps,
                            interiorSampleSteps,
                            seedMethod)

methods = [m for m in methods if not (m[-3]=='0' and m[-2]=='1' and m[-4]!='HR')]

methods = [m for m in methods if not (m[1]=='none' and m[2]=='MH')]

jobs = itertools.product(problem, methods)

num_runs = 10

cmdjob = ['ex/problem', 'sam/downhillMethod', 'sam/downhillNoiseMethod', 'sam/downhillRejectMethod',
           'sam/interiorMethod', 'sam/interiorBurnInSteps', 'sam/interiorSampleSteps', 'sam/seedMethod']

print('#experiments:', len(list(jobs)))

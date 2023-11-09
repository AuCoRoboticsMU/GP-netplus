import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from src.experiments.clutter_removal import Data


def fitness_function(gsr, cr):
    weight_success = 1
    weight_clearance = 1
    return (weight_success * gsr + weight_clearance * cr) / (weight_success + weight_clearance)


experiment_name = 'test'
rootdir = 'data/experiments/{}'.format(experiment_name)
filename = 'gpnetplus_eps_8_threshold_0.29'

logdir = Path(rootdir)

all_thresholds = []
all_gsrs = []
all_crs = []
all_fitness = []
step = 0.01
all_nums = []

for threshold in np.arange(0.0, 1.0, step):
    data = Data(logdir, filename)
    data.grasps = data.grasps[data.grasps['score'] > threshold]

    if data.grasps["label"].count() == 0:
        success_rate = np.nan
        clearance_rate = np.nan
    else:
        success_rate = data.success_rate()
        clearance_rate, _ = data.percent_cleared()

    if data.grasps["label"].count() == 0:
        fitness = np.nan
    else:
        fitness = fitness_function(success_rate, clearance_rate)

    all_thresholds.append(threshold)
    all_gsrs.append(success_rate)
    all_crs.append(clearance_rate)
    all_fitness.append(fitness)
    all_nums.append(data.grasps["label"].count())

all_thresholds = np.array(all_thresholds)
all_fitness = np.array(all_fitness)
max_thresh = all_thresholds[np.nanargmax(all_fitness)]
max_fitness = all_fitness[all_thresholds == max_thresh][0]

print("Maximum fitness {:.2f} at threshold {}: GSR {:.2f} VCR {:.2f}".format(max_fitness,
                                                                            max_thresh,
                                                                            all_gsrs[np.nanargmax(all_fitness)],
                                                                            all_crs[np.nanargmax(all_fitness)]))


sns.set_theme(font_scale=1.2)
fig, ax = plt.subplots(1, 1)
plt.subplots_adjust(bottom=0.13, right=0.95)
plt.plot(all_thresholds, all_gsrs)
plt.plot(all_thresholds, all_crs)
plt.plot(all_thresholds, all_fitness)
plt.plot([max_thresh, max_thresh], [0.0, max_fitness], 'r--')
plt.plot([0.0, max_thresh], [max_fitness, max_fitness], 'r--')
plt.text(x=max_thresh + 0.01, y=45, s='\u03b3($f_{{max}}$) = {}'.format(max_thresh), fontdict={'size': 12})
plt.text(x=0.04, y=max_fitness + 1, s='$f_{{max}}$ = {:.1f}%'.format(max_fitness), fontdict={'size': 12})
plt.xlim((0.0, 1.0))
plt.ylim((0, 100.0))
plt.xlabel('Detection threshold \u03b3')
plt.ylabel('Performance [%]')
plt.title('Effect of detection threshold \u03b3 on GP-Net+ performance', pad=13)
plt.legend(loc='best', labels=['Grasp Success Rate (GSR)', 'Visible Clearance Rate (VCR)',
                               'Fitness f'])

plt.savefig('{}/{}_fitness.png'.format(rootdir, filename))
plt.close()


from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.graph_objects as go


from src.experiments.clutter_removal import Data
from src.utils import visualise_egad_heatmap
import numpy as np


experiment_name = 'test'

def visualise_egad(data):
    egad_counts = {}

    count_df = data.grasps[["round_id", "visible_objects"]]
    count_df['visible_objects'] = count_df.apply(lambda x: data.get_objects(x.visible_objects), axis=1)

    aggregations = {
        'visible_objects': data.union_of_lists
    }

    cleared_count = (
        count_df
        .groupby("round_id")
        .agg(aggregations)
        .reset_index()
    )

    all_bodies_texts = [x for x in os.listdir('{}/scenes/'.format(rootdir)) if 'bodies' in x and not 'graspable' in x]
    for body_text in all_bodies_texts:
        current_round_id = int(body_text.split('_')[1])
        try:
            visible_objects = cleared_count[cleared_count['round_id'] == current_round_id]['visible_objects'].values[0]
        except IndexError:
            continue
        with open('{}/scenes/{}'.format(rootdir, body_text), 'r') as f:
            lines = f.readlines()
            for line_cnt, line in enumerate(lines):
                if 'egad' in line and line_cnt in visible_objects:
                    obj = line.split('_')[1].lower()
                    if obj in egad_counts.keys():
                        egad_counts[obj][0] += 1
                    else:
                        egad_counts[obj] = [1]

    annot_size = 13
    egad_result = data.rates_egad(egad_counts)
    cp_clearance = sns.color_palette("mako", n_colors=49)
    visualise_egad_heatmap(egad_result, annot_size, cp_clearance, type='Visible clearance rate')
    plt.savefig('{}/clearance_heatmap_egad_{}.png'.format(rootdir, filename))
    plt.close()
    cnt_colourmap = sns.color_palette("light:#5A9", n_colors=50)
    visualise_egad_heatmap(egad_result, annot_size, cnt_colourmap, type='Grasp attempts', percent=False)
    plt.savefig('{}/count_heatmap_egad_{}.png'.format(rootdir, filename))
    plt.close()
    # Plot heatmap
    cp = sns.color_palette("rocket", n_colors=49)
    visualise_egad_heatmap(egad_result, annot_size, cp, type='Grasp success rate')
    plt.savefig('{}/success_heatmap_egad_{}.png'.format(rootdir, filename))
    plt.close()
    cnt_colourmap = sns.color_palette("dark:#5A9", n_colors=50)
    visualise_egad_heatmap(egad_result, annot_size, cnt_colourmap, type='Object occurrence', percent=False)
    plt.savefig('{}/object_occurrence_egad_{}.png'.format(rootdir, filename))
    plt.close()

def visualise_success_rate_per_object():
    # cp = sns.color_palette("dark:#5A9", n_colors=3)
    cp1 = sns.color_palette("tab10", n_colors=3)
    palette = {'# unsuccessful grasps': cp[2],
               '# object occurred': cp[0],
               '# successful grasps': cp[1]}
    g = sns.catplot(data=success_rate_per_object, x='grasped_object_name',
                    y='value', hue='variable', kind='bar', palette=palette, legend=False, aspect=2, height=5)
    plt.ylabel("")
    # plt.ylim((0, 100))
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
    plt.legend(loc='best')
    plt.xlabel('')
    plt.subplots_adjust(top=0.9, bottom=0.45, left=0.1, right=0.95)

    # plt.tight_layout()
    plt.title(filename, pad=15)  # "Grasp success rate per object")
    plt.savefig('{}/success_rate_per_object_{}.png'.format(rootdir, filename))
    plt.close()

def visualise_overview():
    column_widths = [3, 2.8, 2.6, 3.7]

    try:
        fig = go.Figure(
            data=[go.Table(header=dict(values=['', '<b>GSR [%]</b>',
                                               r'<b>CR* [%]</b>', '<b>Grasps attempted</b>'],
                                       align='center',
                                       font={'size': 12}),
                           columnwidth=column_widths,
                           cells=dict(values=[['Table', 'Shelf', 'All'],
                                              [grasp_success_rate_furniture_type['Table'],
                                               grasp_success_rate_furniture_type['Shelf'],
                                               grasp_success_rate],
                                              [clearance_rate_furniture_type['Table'],
                                               clearance_rate_furniture_type['Shelf'],
                                               clearance_rate],
                                              [num_grasps_furniture_type['Table'],
                                               num_grasps_furniture_type['Shelf'],
                                               sum(num_grasps)]],
                                      format=["", ".1f", ".1f", ""])
                           )
                  ])
    except KeyError:
        fig = go.Figure(
            data=[go.Table(header=dict(values=['', '<b>GSR [%]</b>',
                                               r'<b>CR* [%]</b>', '<b>Grasps attempted</b>'],
                                       align='center',
                                       font={'size': 12}),
                           columnwidth=column_widths,
                           cells=dict(values=[['All'],
                                              [grasp_success_rate],
                                              [clearance_rate],
                                              [sum(num_grasps)]],
                                      format=["", ".1f", ".1f", ""])
                           )
                  ])

    fig.update_layout(title='{}, thresh={}'.format(filename.split('_')[0], filename.split('_')[-1]),
                      title_x=0.5, title_y=0.8, width=350, height=210, margin={"l": 0, "r": 0, "b": 5, "t": 70})
    fig.write_image('{}/overview_{}.png'.format(rootdir, filename))

def visualise_all_experiments(success_rates, clearance_rates, num_executed_grasps,
                              architectures, network_name, thresholds, epoch, source_objects):
    column_widths = [6, 7, 5, 6, 5, 6]

    success_rates = np.array(success_rates)
    clearance_rates = np.array(clearance_rates)

    sum = (success_rates + clearance_rates) / 2
    max_gsr_plus_cr = np.max(sum)

    gsr = []
    cr = []
    for each_gsr, each_cr in zip(success_rates, clearance_rates):
        if (each_gsr + each_cr) / 2 > max_gsr_plus_cr - 1:
            gsr.append(f'<b>{each_gsr:.1f}</b>')
            cr.append(f'<b>{each_cr:.1f}</b>')
        else:
            gsr.append(f'{each_gsr:.1f}')
            cr.append(f'{each_cr:.1f}')

    fig = go.Figure(
        data=[go.Table(header=dict(values=['Architecture', 'Description', 'Thresh',
                                           'GSR [%]', 'CR [%]', '# Grasps'],
                                   align='center',
                                   font={'size': 12}),
                       columnwidth=column_widths,
                       cells=dict(values=[architectures, network_name, thresholds, gsr, cr, num_executed_grasps],
                                  format=["", "", "", "", "", ""])
                       )
              ])

    fig.update_layout(title='Tabletop: {}'.format(source_objects),
                      title_x=0.5, title_y=0.9, width=600, height=450)
    fig.write_image('{}/overview.png'.format(rootdir, filename))


def visualise_threshold_influence(thresh_score, cut_off):
    fig, axes = plt.subplots(1, 1, figsize=(5, 4))
    fig.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.86)

    thresh_score_added = thresh_score.copy()
    thresh_score_added['threshold_bin'] += 0.05
    ax2 = axes.twinx()
    sns.histplot(thresh_score_added, x='threshold_bin', bins=np.linspace(0.0, 1.0, 11),
                 stat='count', ax=axes, color=number_grasps_color)
    g = sns.lineplot(thresh_score_added, x='threshold_bin', y='label', errorbar=None, ax=ax2,
                     color=gsr_color, marker=".", markeredgecolor=gsr_color)

    ax2.grid(visible=False, which='both')
    axes.set_ylabel("# Executed grasps", labelpad=2)
    axes.set_xlabel("Predicted grasp confidence")
    ax2.tick_params(axis='y', colors=gsr_color)
    ax2.yaxis.label.set_color(gsr_color)
    ax2.set_ylim((0.0, 1.01))
    max_val = 801
    axes.set_ylim((0.0, max_val))
    axes.tick_params(axis='y', colors=number_grasps_color)
    axes.yaxis.label.set_color(number_grasps_color)
    axes.yaxis.set_ticks(np.arange(0, max_val, 160))

    plt.ylabel("Grasp success rate")
    fig.suptitle("Furniture: GP-Net+ {}".format(source_objects), y=0.97)
    plt.savefig('{}/{}_threshold_analysis.png'.format(rootdir, filename))
    plt.close()

def visualise_failure_causes():
    bar_colors = []
    palette = sns.color_palette("flare", n_colors=2)
    for obj_name in failure_causes['furniture_type']:
        idx = ['Table', 'Shelf'].index(obj_name)
        bar_colors.append(palette[idx])
    failure_causes.sort_index()

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    fig.subplots_adjust(top=0.9, bottom=0.25, left=0.14, right=0.98)
    g = sns.histplot(data=failure_causes,  x="cause_of_failure", shrink=0.7, hue="furniture_type",
                     multiple="stack")

    plt.xlabel("")
    leg = g.axes.get_legend()
    leg.set_title('Furniture type')
    plt.ylabel("Number of grasps")
    # plt.ylim((0, 1))
    plt.xticks(rotation=25)
    plt.title('GP-Net+ failure causes', pad=10)
    plt.savefig('{}/failure_causes_{}.png'.format(rootdir, filename))
    plt.close()




if 'simple_shapes' in experiment_name:
    source_objects = 'blocks'
elif 'egad' in experiment_name:
    source_objects = 'EGAD!'
else:
    source_objects = 'Test'

rootdir = 'data/experiments/{}'.format(experiment_name)
all_files = ['_'.join(x.split('_')[1:])[:-4] for x in os.listdir(rootdir) if 'grasps_' in x]
all_files.sort()
sns.set_theme(font_scale=1.2)

obj_cnt = 'object_count'
success_rates = []
clearance_rates = []
architectures = []
network_name = []
num_executed_grasps = []
thresholds = []
epochs = []
all_threshold_scores = []

for filename in all_files:
    print("File {}".format(filename))
    logdir = Path(rootdir)
    data = Data(logdir, filename)
    num_grasps = data.num_per_object()
    num_grasps_furniture_type = data.attempted_grasps_per_furniture_type()
    grasp_success_rate = data.success_rate()
    grasp_success_rate_furniture_type = data.success_rate_per_furniture_type()
    failure_causes = data.failure_causes()
    success_rate_per_object = data.success_rate_per_object()

    (clearance_rate, clearance_rate_furniture_type) = data.percent_cleared(obj_count=obj_cnt)

    left_color = sns.color_palette()[1]
    right_color = sns.color_palette()[0]
    single_colors = sns.dark_palette("#69d", reverse=True, n_colors=7) # sns.color_palette("magma", n_colors=10)
    gsr_color = single_colors[6]
    number_grasps_color = single_colors[1]
    attempted_grasps_color = sns.color_palette("mako", n_colors=10)[4]

    try:
        bar_colors = []
        cp = sns.color_palette("bright", n_colors=9)
        palette = {"Boxes": cp[0],
                   "Cups": cp[1],
                   "Misc": cp[2],
                   "Bottles": cp[3],
                   "Tools": cp[4],
                   "Toys": cp[5],
                   "Bowls": cp[6],
                   "Fruit": cp[7],
                   "EGAD!": cp[8],
                   "Furniture": '#fffea3',
                   "Blocks": '#feaff4'}
    except ValueError:
        bar_colors = None
        palette = sns.color_palette("flare", n_colors=49)

    visualise_failure_causes()

    arch = 'GP-net+'
    thresh = filename.split('_')[-1]
    data_name = filename.split('_')[0]
    network_description = 'Predicted quality threshold: {}'.format(thresh)
    epoch = filename.split('_')[-3]

    visualise_success_rate_per_object()

    if 'egad' in experiment_name and not "simple" in experiment_name:
        visualise_egad(data)

    success_rates.append(grasp_success_rate)
    clearance_rates.append(clearance_rate)
    num_executed_grasps.append(sum(num_grasps))
    architectures.append(arch)
    network_name.append(data_name)
    thresholds.append(thresh)
    epochs.append(epoch)
    if not "simple" in experiment_name:
        visualise_overview()

visualise_all_experiments(success_rates, clearance_rates, num_executed_grasps, architectures, network_name,
                          thresholds, epochs, source_objects)

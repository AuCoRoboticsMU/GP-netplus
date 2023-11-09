import os
import numpy as np
from utils import create_urdf_for_mesh


root = 'data/urdfs/egad_test_set/'
all_files = os.listdir(root)

obj_files = [f for f in all_files if 'collision.obj' in f]
vhacd_files = [f for f in all_files if 'vhacd.obj' in f]

masses = {}
with open('data/urdfs/all_objects/masses.csv', 'r') as f:
    for line in f.readlines():
        line = line.strip('\n').split(',')
        if 'Ycb' in line[0]:
            line[0] = 'y' + line[0][1:]
        masses[line[0]] = line[1]

for obj in obj_files:
    dset = obj.split('_')[0]
    if dset == 'bigbird':
        name = '_'.join(obj.split('_')[1:-1])
        mass = masses[name]
    elif dset == 'ycb' or dset == 'Ycb':
        name = 'y' + '_'.join(obj.split('_')[0:2])[1:]
        mass = masses[name]
    elif dset == 'ShapeNet':
        name = '_'.join(obj.split('_')[:2])
        factor = np.random.uniform(0.5, 1.5)
        mass = float(masses[name]) * factor
    elif dset == 'egad':
        mass = np.random.uniform(0.4, 0.6)
    else:
        raise Warning("We didn't have that one: {}".format(obj))
    create_urdf_for_mesh('{}{}'.format(root, obj), concave=False, mass=mass)


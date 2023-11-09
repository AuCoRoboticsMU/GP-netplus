import scipy.spatial
from src.utils import Workspaces
import os
import numpy as np
from src.settings import MESH_DATA_PATH

"""
Generate object workspaces from the regions in the .csv output from blender_find_workspaces.py
"""

def check_hull(points):
    hull = scipy.spatial.ConvexHull(points[:, :2])
    return hull, points[:, 2].mean()


ws_root = '{}/data/Shapenet_furniture/workspaces_postprocessed/'.format(MESH_DATA_PATH)
sampling_space_path = 'data/urdfs/Shapenet_furniture/workspaces_postprocessed_2'

if not os.path.exists(sampling_space_path):
    os.mkdir(sampling_space_path)

all_files = os.listdir(ws_root)
all_point_files = [f for f in all_files if 'workspace_points' in f]

for single_obj in all_point_files:
    obj_name = '_'.join(single_obj.split('_')[:-1])

    ws = Workspaces()
    current_points = []
    with open('{}/{}'.format(ws_root, single_obj), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Region' in line:
                if current_points:
                    current_points = np.array(current_points)
                    hull, height = check_hull(current_points)
                    ws.add_workspace(hull, height)
                    current_points = []
            elif line != '\n':
                all = line.split(',')
                if all[0] != '':
                    current_points.append((float(all[0]), float(all[1]), float(all[2])))
        current_points = np.array(current_points)
        hull, height = check_hull(current_points)
        ws.add_workspace(hull, height)
        ws.save('{}/{}.pkl'.format(sampling_space_path, obj_name))
        Workspaces.load('{}/{}.pkl'.format(sampling_space_path, obj_name))

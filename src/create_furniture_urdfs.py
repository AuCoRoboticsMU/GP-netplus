import os
from utils import create_urdf_for_furniture_mesh


furniture_dir = 'data/urdfs/furniture/val/'
all_files = os.listdir(furniture_dir)

obj_files = [f for f in all_files if '.obj' in f]

for obj in obj_files:
    create_urdf_for_furniture_mesh('{}{}'.format(furniture_dir, obj), concave=True, mass=0.0, furniture=True)


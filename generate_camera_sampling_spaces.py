import sys
from src.settings import MESH_DATA_PATH, EXPERIMENT_PATH
sys.path.append('src')

import numpy as np
import os
import scipy.spatial
import scipy.ndimage
from src.utils import CameraSampleSpace, ObjectWorkspaces
from skimage.morphology import (disk, dilation)
from alphashape import alphashape
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point

"""
Script to generate the camera workspace limits for a given mesh file.
"""


def get_concave_hull(filepath):
    """
    Generate a concave hull of the mesh footprint from the original mesh file.
    :param filepath: File to the original mesh (not including wall).
    :return: Polygon object of concave hull. Minimum and maximum height of object mesh.
    """
    mesh_points = []
    faces = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('v '):
                # vertex line
                vertex = [float(x) for x in line.split()[1:4]]
                mesh_points.append(vertex)
            elif line.startswith('f '):
                # face line
                face = [int(x.split('/')[0]) for x in line.split()[1:]]
                faces.append(face)

        # Calculate face centers
        face_centers = []
        for face in faces:
            coords = [mesh_points[i - 1] for i in face]  # Subtract one because the face pointers start with 1
            center = [sum(c) / len(c) for c in zip(*coords)]
            face_centers.append(center)
    mesh_points.extend(face_centers)
    mesh_points = np.array(mesh_points).astype(float)

    alpha = 1.5
    try:
        concave_hull = alphashape(mesh_points[:, :2], alpha)   # Generate a concave hull from the points
        cnt = 0
        while concave_hull.area <= 0.2 and cnt < 5:
            alpha -= 0.2
            concave_hull = alphashape(mesh_points[:, :2], alpha)
            cnt += 1
        if cnt == 5:
            hull = scipy.spatial.ConvexHull(mesh_points[:, :2])
            vertices = hull.points[hull.vertices]

            # Convert the vertices to a Polygon object
            poly = Polygon(vertices)
        else:
            try:
                poly = Polygon(concave_hull)  # Transform the concave hull into a polygon
            except:
                print("Something happend")
                poly = concave_hull.envelope
    except Warning:
        hull = scipy.spatial.ConvexHull(mesh_points[:, :2])
        vertices = hull.points[hull.vertices]

        # Convert the vertices to a Polygon object
        poly = Polygon(vertices)

    return poly, (mesh_points[:, 2].astype(float).min(), mesh_points[:, 2].astype(float).max())

def pixelate(concave_hull, grid_min, grid_max, step_size):
    """
    Convert a concave hull into a pixel grid.

    :param concave_hull: Concave hull to convert.
    :param grid_min: Minimum x/y value for pixel grid.
    :param grid_max: Maximum x/y value for pixel grid.
    :param step_size: Length of a single grid cell in the pixel grid.
    :return: Grid cell with 1's for where the concave hull is and 0's where it is not.
    """
    x, y = np.meshgrid(np.arange(grid_min, grid_max, step_size), np.arange(grid_min, grid_max, step_size))

    # Find check which pixel indices lie within our hull
    pixel_indices = list()
    for i, current_x in enumerate(x[0]):
        try:
            if current_x < hull.bounds[0] or current_x > hull.bounds[2]:
                continue
        except IndexError:
            print('whoopsie')
        for j, current_y in enumerate(y[:, 0]):
            if concave_hull.contains(Point(current_x, current_y)):
                pixel_indices.append([i, j])
    pixel_indices = np.array(pixel_indices)

    grid = np.zeros(x.shape)
    grid[pixel_indices[:, 0], pixel_indices[:, 1]] = 1
    return grid

def read_walls(wall_csv):
    """
    Get the position of walls in the mesh file.
    :param wall_csv: Path to csv file that lists all furniture units and if and where they have walls.
    :return: Dictionary of the furniture units and their walls.
    """
    walls = {}
    with open(wall_csv, 'r') as f:
        lines = f.readlines()
        for line in lines:
            x_wall = ''
            y_wall = ''
            splitted = line.split(',')
            if splitted[1] != '' and splitted[1] != '\n':
                y_wall = splitted[1]
            if splitted[2] != '\n' and splitted[2] != '':
                x_wall = splitted[2]
            walls[splitted[0]] = {'x_wall': x_wall, 'y_wall': y_wall}
    return walls


obj_root = '{}/Shapenet_furniture/processed_meshes/'.format(MESH_DATA_PATH)
workspace_root = '{}/data/Shapenet_furniture/workspaces_postprocessed/'.format(EXPERIMENT_PATH)
save_root = '{}/data/urdfs/Shapenet/new'.format(EXPERIMENT_PATH)
walls = read_walls('{}/data/urdfs/Shapenet/all_walls.csv'.format(EXPERIMENT_PATH))

forward = 0.1
away = 1.0
sideways = 0.3
above = 0.2
below = 0.05
mesh_sampling_dist = 0.05
camera_min_height = 1.0
camera_max_height = 1.55

# Grid details
grid_min, grid_max = -5, 5
step_size = 0.05

all_files = os.listdir(obj_root)
all_pieces = [f for f in all_files if '.obj' in f]

for piece in all_pieces:
    obj_name = '_'.join(piece.split('.')[:-1])
    hull, z_mesh_values = get_concave_hull('{}/{}'.format(obj_root, piece))

    mesh_grid = pixelate(hull, grid_min, grid_max, step_size)

    mesh_disk = disk(int(mesh_sampling_dist // step_size))
    mesh_sampling_space = dilation(mesh_grid, mesh_disk)

    collision_disk = disk(int(forward // step_size))
    collision_boundary = dilation(mesh_grid, collision_disk)

    camera_max_disk = disk(int(away // step_size))
    camera_max_boundary = dilation(mesh_grid, camera_max_disk)

    camera_sampling_space = camera_max_boundary.astype(int) ^ collision_boundary.astype(int)

    # How far we want to go away
    ws = ObjectWorkspaces.load('{}/{}_workspace.pkl'.format(workspace_root, obj_name))
    z_ws_min = min(ws.heights) - below
    z_ws_max = max(ws.heights) + above

    if z_ws_min < 0:
        z_ws_min = 0

    current_walls = walls[obj_name]
    # Adjust the outer limits based on the position of the walls
    if current_walls['y_wall'] != '':
        # Walls are always above the object mesh (top-view, so in direction of positive y)
        wall_y_max = np.digitize(float(current_walls['y_wall']) - sideways, np.arange(grid_min, grid_max, step_size))
        camera_sampling_space[:, wall_y_max:] = 0
    if current_walls['x_wall'] != '':
        if hull.bounds[2] <= float(current_walls['x_wall']) or \
                hull.bounds[0] < float(current_walls['x_wall']) < hull.bounds[2]:
            # X-wall is to the right of the object
            wall_x_max = np.digitize(float(current_walls['x_wall']) - sideways, np.arange(grid_min, grid_max, step_size))
            camera_sampling_space[wall_x_max:, :] = 0
        else:
            # X-wall is to the left of the object
            wall_x_min = np.digitize(float(current_walls['x_wall']) + sideways, np.arange(grid_min, grid_max, step_size))
            camera_sampling_space[:wall_x_min, :] = 0

    sampling = CameraSampleSpace(mesh_sampling_space, camera_sampling_space,
                                 (z_ws_min, z_ws_max), (camera_min_height, camera_max_height),
                                 grid_min=grid_min, grid_max=grid_max, step_size=step_size)

    sampling.save('{}/{}_cameraspace.pkl'.format(save_root, obj_name))






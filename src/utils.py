import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import copy
from matplotlib.path import Path
import pickle
import seaborn as sns
from skimage.morphology import (disk, dilation)

import sys
import pathlib
rel_path = pathlib.Path(__file__).parent.resolve()
sys.path.append('{}'.format(rel_path))


scale_factor = 100  # converting depth from m to cm
scale_factor = 100  # converting depth from m to cm
focal_length = 480.0  # focal length of the camera used
baseline_m = 0.075  # baseline in m
invalid_disp_ = 99999999.9
dot_pattern_ = cv2.imread("{}/../data/kinect-pattern_3x3.png".format(rel_path), 0)

sns.set_theme()

class Workspaces:
    def __init__(self):
        """
        ObjectWorkspaces of where to uniformly random sample object positions from.
        """
        self.hulls = []
        self.heights = []
        self.areas = []

    def sample_object_count(self):
        """
        Sample the number of objects to place on this particular furniture unit.
        :return: Number of objects.
        """
        lam = sum(self.areas) / 0.16  # 1 object per 0.4 * 0.4m
        object_count = np.random.poisson(lam)
        return min(max(1, object_count), 100)

    def add_workspace(self, ws_hull, ws_height):
        """
        Add a convex hull to the object workspaces for this furniture unit.
        :param ws_hull: Convex hull to add to the workspace.
        :param ws_height: Height of the convex hull in the world coordinate frame.
        """
        self.hulls.append(ws_hull)
        self.heights.append(ws_height)
        self.areas.append(ws_hull.volume)

    def choose_workspace(self):
        """
        Choose a workspace to place the objects on. Sampling is weighted based on the workspace area, so that it is
            more likely to place objects on large workspaces than small ones.
        :return: Index of the chosen workspace.
        """
        total_area = sum(self.areas)
        nbrs = len(self.areas)
        weights = np.array(self.areas) / total_area
        chosen_area = np.random.choice(nbrs, p=weights)
        return chosen_area

    def uniform_rejection_sampling(self, hull, hull_path):
        """
        Perform uniform rejection sampling to uniformly sample points within the hull.
        :param hull: Workspace hull.
        :param hull_path: Path of the hull border, used to check if a point lies within the hull.
        :return: x, y position of the sampled point.
        """
        # Recursively sample within the hull bounds until we find a point (for non-rectangular hulls)
        x = np.random.uniform(hull.min_bound[0], hull.max_bound[0])
        y = np.random.uniform(hull.min_bound[1], hull.max_bound[1])
        if hull_path.contains_point((x, y)):
            return x, y
        else:
            return self.uniform_rejection_sampling(hull, hull_path)

    def sample_among_workspace(self):
        """
        Sample an object position in the object workspaces.
        :return: x, y, z of object pose.
        """
        chosen_area = self.choose_workspace()
        chosen_hull = self.hulls[chosen_area]
        hull_path = Path(chosen_hull.points[chosen_hull.vertices])
        x, y = self.uniform_rejection_sampling(chosen_hull, hull_path)
        return x, y, self.heights[chosen_area]

    def save(self, filename):
        """
        Save an ObjectWorkspace object to a .pkl file.
        :param filename: Path to the .pkl file to save the ObjectWorkspace to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """
        Load an ObjectWorkspace object from a .pkl file.
        :param filename: Path to the .pkl file to load the ObjectWorkspace from.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)


class CameraSampleSpace:
    def __init__(self, mesh_region, camera_region, mesh_height, camera_height, step_size, grid_min, grid_max):
        """
        Camera sampling space to enable robot-centric camera pose sampling in a domestic scene. We're using a
        pixelised grid for the regions and sample from the grid to speed up the process.

        :param mesh_region: Mesh footprint in the pixelised grid.
        :param camera_region: Camera region in the pixelised grid.
        :param mesh_height: minimum and maximum height of workspaces in mesh.
        :param camera_height: minimum and maximum height of the camera.
        :param step_size: Side length of a single grid cell in the pixelised grid.
        :param grid_min: Minimum x/y position in grid.
        :param grid_max: Maximum x/y position in grid.
        """
        self.mesh_region = mesh_region
        self.camera_region = camera_region
        self.mesh_height = mesh_height
        self.camera_height = camera_height
        self.step_size = step_size
        self.grid_min = grid_min
        self.grid_max = grid_max

    def get_camera_samples(self):
        """
        Sample the number of images to render for this particular furniture unit.
        :return: Number of images.
        """
        mean_number_of_camera_samples = self.mesh_volume / 0.01  # One image per 0.1m^3
        n = np.random.poisson(mean_number_of_camera_samples)
        return max(10, n)

    @property
    def mesh_volume(self):
        """
        Calculate volume of the furniture mesh, used for sampling the number of images.
        :return: Mesh volume
        """
        area = np.sum(self.mesh_region) * self.step_size**2
        volume = area * (self.mesh_height[1] - self.mesh_height[0])
        return volume

    def sample_camera_points(self):
        """
        Sample a full pose of the camera with a camera origin and a look-to-point.
        :return: Camera origin, look-to-point
        """
        origin = self.sample_from_pixel_grid(self.camera_region, self.camera_height)
        look_to_point = self.sample_from_pixel_grid(self.mesh_region, self.mesh_height)
        return origin, look_to_point

    def sample_from_pixel_grid(self, region, height):
        """
        Uniformly samples a 3D pose from a pixelised 2D region and a height range.
        :param region: Pixelised grid to sample from, with 1's indicating region and 0's indicating no-region pixels.
        :param height: Minimum and maximum height to sample from
        :return: 3D coordinates (x, y, z)
        """
        indices = np.where(region == 1)
        index = np.random.choice(range(len(indices[0])))
        x_ind, y_ind = indices[0][index], indices[1][index]
        x_centre = self.grid_min + x_ind * self.step_size
        y_centre = self.grid_min + y_ind * self.step_size

        # Add a random offset to the x and y coordinates to vary the position within the cell.
        x = x_centre + np.random.uniform(-self.step_size / 2, self.step_size / 2)
        y = y_centre + np.random.uniform(-self.step_size / 2, self.step_size / 2)
        z = np.random.uniform(height[0], height[1])
        return x, y, z

    def save(self, filename):
        """
        Save an CameraSampleSpace object to a .pkl file.
        :param filename: Path to the .pkl file to save the CameraSampleSpace to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """
        Load an CameraSampleSpace object from a .pkl file.
        :param filename: Path to the .pkl file to load the CameraSampleSpace from.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

def is_object_in_air(segmask, body_uid):
    """
    From the segmask image from a parallel projection, check if the object is balancing over the edge of the workspace.
    :param segmask: Parallel projection segmask image.
    :param body_uid: Physics simulation UID of the body.
    :return: True if object is balancing over workspace, False otherwise.
    """
    segmask = np.array(segmask)
    body_segmask = np.zeros(segmask.shape)
    body_segmask[segmask == body_uid] = 1

    body_border = dilation(body_segmask, disk(1)) - body_segmask

    if np.any(segmask[body_border == 1] == -1):
        return True
    else:
        return False


def visualise_egad_heatmap(data_map, annot_size, cmap, type, percent=True):
    fig, axes = plt.subplots(2, 2, gridspec_kw={'width_ratios': [6, 1],
                                                'height_ratios': [6, 1]},
                             figsize=(4, 4.3))
    # fig.tight_layout()
    plt.subplots_adjust(bottom=0.02, left=0.15, right=0.98, top=0.8, wspace=0.05, hspace=0.05)

    if 'Grasp success rate' == type:
        data = data_map[type]
        difficulty = (data_map['Successful grasp attempts'].sum(axis=1) / data_map['Grasp attempts'].sum(axis=1)).values
        difficulty = np.expand_dims(difficulty * 100.0, axis=1)
        complexity = (data_map['Successful grasp attempts'].sum(axis=0) / data_map['Grasp attempts'].sum(axis=0)).values
        complexity = np.expand_dims(complexity * 100.0, axis=0)
        overall = (data_map['Successful grasp attempts'].sum().sum() / data_map['Grasp attempts'].sum().sum()) * 100.0
        overall = np.reshape(overall, (1, 1))
    elif 'Visible clearance rate' == type:
        data = data_map['Clearance rate']
        difficulty = data.mean(axis=1)
        difficulty = np.expand_dims(difficulty.values * 100.0, axis=1)
        complexity = data.mean()
        complexity = np.expand_dims(complexity.values * 100.0, axis=0)
        overall = np.reshape(np.nanmean(data.values), (1, 1)) * 100.0
    else:
        data = data_map[type]
        difficulty = data.mean(axis=1)
        difficulty = np.expand_dims(difficulty.values, axis=1)
        complexity = data.mean()
        complexity = np.expand_dims(complexity.values, axis=0)
        overall = np.reshape(np.nanmean(data.values), (1, 1))

    if percent:
        vmin = 0.0
        vmax = 100
    else:
        vmin = np.nanmin(data.values)
        vmax = np.nanmax(data.values)
    if 'Grasp success rate' == type:
        sns.heatmap(data * 100.0, ax=axes[0, 0], vmin=vmin, vmax=vmax, cbar=False, annot=True, fmt='.0f', cmap=cmap,
                    annot_kws={"size": annot_size}, mask=data_map['Grasp attempts'] == 0)
    elif 'Visible clearance rate' == type:
        sns.heatmap(data * 100.0, ax=axes[0, 0], vmin=vmin, vmax=vmax, cbar=False, annot=True, fmt='.0f', cmap=cmap,
                    annot_kws={"size": annot_size}, mask=data_map['Object occurrence'] == 0)
    else:
        sns.heatmap(data, ax=axes[0, 0], vmin=vmin, vmax=vmax, cbar=False, annot=True, fmt='.0f', cmap=cmap,
                    annot_kws={"size": annot_size})
    axes[0, 0].xaxis.tick_top()
    axes[0, 0].xaxis.set_label_position('top')
    axes[0, 0].yaxis.tick_left()
    axes[0, 0].yaxis.set_label_position('left')

    # mean difficulty
    sns.heatmap(difficulty, ax=axes[0, 1], vmin=vmin, vmax=vmax, cbar=False, annot=True, fmt='.0f', cmap=cmap,
                annot_kws={"size": annot_size})
    axes[0, 1].set(xlabel="Mean", ylabel="")
    axes[0, 1].xaxis.set_label_position('top')
    axes[0, 1].tick_params(bottom=False,  # ticks along the bottom edge are off
                           top=False,
                           labelbottom=False,
                           left=False,
                           labelleft=False)

    # mean complexity
    sns.heatmap(complexity, ax=axes[1, 0], vmin=vmin, vmax=vmax, cbar=False, annot=True, fmt='.0f', cmap=cmap,
                annot_kws={"size": annot_size})
    axes[1, 0].set(xlabel="", ylabel="Mean")
    axes[1, 0].tick_params(bottom=False,  # ticks along the bottom edge are off
                           top=False,
                           labelbottom=False,
                           left=False,
                           labelleft=False)

    # mean overall
    sns.heatmap(overall, ax=axes[1, 1], vmin=vmin, vmax=vmax, cbar=False, annot=True, fmt='.0f', cmap=cmap,
                annot_kws={"size": annot_size})
    axes[1, 1].set(xlabel="", ylabel="")
    axes[1, 1].tick_params(bottom=False,  # ticks along the bottom edge are off
                           top=False,
                           labelbottom=False,
                           left=False,
                           labelleft=False)
    plt.suptitle(type)

    return fig

def get_tensor_list(root_dir):
    """
    Gather all tensor identifiers in the dataset.
    :param root_dir: Path to the dataset.
    :return: Numpy array of all points in the dataset.
    """
    files_in_dir = os.listdir('{}/'.format(root_dir))
    pointers = [string.split('_')[-1].split('.')[0] for string in files_in_dir if 'depth_image' in string]
    pointers.sort()
    return np.arange(0, int(pointers[-1]) + 1)

def depth_encoding(image, vmin=0.2, vmax=2.0):
    """
    Enocde a depth image into a jet-colourscale RGB image
    :param image: depth image
    :param vmin: minimum visible depth in RGB image
    :param vmax: maximum visible depth in RGB image
    :return: RGB image
    """
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colored_image = plt.cm.jet(norm(image))[:, :, :-1]
    return np.array((colored_image[:, :, 0], colored_image[:, :, 1], colored_image[:, :, 2]), dtype=np.float32)

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]  # w
    q1 = Q[1]  # x
    q2 = Q[2]  # y
    q3 = Q[3]  # z

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix

def add_kinetic_noise(depth):
    h, w = depth.shape

    depth_interp = add_gaussian_shifts(depth)

    disp_ = focal_length * baseline_m / (depth_interp + 1e-10)
    depth_f = np.round(disp_ * 8.0) / 8.0

    out_disp = filterDisp(depth_f, dot_pattern_, invalid_disp_)

    depth = focal_length * baseline_m / out_disp
    depth[out_disp == invalid_disp_] = 0

    # The depth here needs to converted to cms so scale factor is introduced
    # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects
    with np.errstate(divide='ignore'):
        noisy_depth = (35130 / np.round((35130 / np.round(depth * scale_factor))
                                        + np.random.normal(size=(h, w)) * (1.0 / 6.0) + 0.5)) / scale_factor
    return noisy_depth


def add_gaussian_shifts(depth, std=1 / 2.0):
    rows, cols = depth.shape
    gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates
    xx = np.linspace(0, cols - 1, cols)
    yy = np.linspace(0, rows - 1, rows)

    # get xpixels and ypixels
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp

def filterDisp(disp, dot_pattern_, invalid_disp_):
    size_filt_ = 9

    xx = np.linspace(0, size_filt_ - 1, size_filt_)
    yy = np.linspace(0, size_filt_ - 1, size_filt_)

    xf, yf = np.meshgrid(xx, yy)

    xf = xf - int(size_filt_ / 2.0)
    yf = yf - int(size_filt_ / 2.0)

    sqr_radius = (xf ** 2 + yf ** 2)
    vals = sqr_radius * 1.2 ** 2

    vals[vals == 0] = 1
    weights_ = 1 / vals

    fill_weights = 1 / (1 + sqr_radius)
    fill_weights[sqr_radius > 9] = -1.0

    disp_rows, disp_cols = disp.shape
    dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape

    lim_rows = np.minimum(disp_rows - size_filt_, dot_pattern_rows - size_filt_)
    lim_cols = np.minimum(disp_cols - size_filt_, dot_pattern_cols - size_filt_)

    center = int(size_filt_ / 2.0)

    window_inlier_distance_ = 0.1

    out_disp = np.ones_like(disp) * invalid_disp_

    interpolation_map = np.zeros_like(disp)

    r = np.arange(lim_rows)
    c = np.arange(lim_cols)

    r_all = np.repeat(r, len(c))
    c_all = np.tile(c, len(r))

    mask = dot_pattern_[r_all + center, c_all + center] > 0

    c_all = c_all[mask]
    r_all = r_all[mask]

    windows = []
    dot_windows = []
    for i in range(size_filt_):
        for k in range(size_filt_):
            windows.append(disp[r_all + i, c_all + k])
            dot_windows.append(dot_pattern_[r_all + i, c_all + k])

    windows = np.array(windows).T.reshape(len(c_all), 9, 9)
    dot_windows = np.array(dot_windows).T.reshape(len(c_all), 9, 9)

    valid_dots = np.where(windows < invalid_disp_, dot_windows, 0)

    all_n_valids = np.sum(np.sum(valid_dots, axis=1), axis=1) / 255.0
    all_n_thresh = np.sum(np.sum(dot_windows, axis=1), axis=1) / 255.0

    mask = np.where(all_n_valids > all_n_thresh / 1.2)

    filtered_windows = windows[mask]
    filtered_dot_windows = dot_windows[mask]
    filtered_n_thresh = all_n_thresh[mask]
    r_all = r_all[mask]
    c_all = c_all[mask]

    all_mean = np.nanmean(np.where(filtered_windows < invalid_disp_, filtered_windows, np.nan), axis=(1, 2))

    all_diffs = np.abs(np.subtract(filtered_windows, np.repeat(all_mean, 81).reshape(len(all_mean), 9, 9)))
    all_diffs = np.multiply(all_diffs, weights_)

    all_cur_valid_dots = np.multiply(np.where(filtered_windows < invalid_disp_, filtered_dot_windows, 0),
                                     np.where(all_diffs < window_inlier_distance_, 1, 0))

    all_n_valids = np.sum(all_cur_valid_dots, axis=(1, 2)) / 255.0

    mask = np.where(all_n_valids > filtered_n_thresh / 1.2)

    filtered_windows = filtered_windows[mask]
    r_all = r_all[mask]
    c_all = c_all[mask]

    accu = filtered_windows[:, center, center]

    for i in range(len(c_all)):
        r = r_all[i]
        c = c_all[i]
        out_disp[r + center, c + center] = np.round(accu[i] * 8.0) / 8.0

        interpolation_window = interpolation_map[r:r + size_filt_, c:c + size_filt_]
        disp_data_window = out_disp[r:r + size_filt_, c:c + size_filt_]

        substitutes = np.where(interpolation_window < fill_weights, 1, 0)
        interpolation_window[substitutes == 1] = fill_weights[substitutes == 1]

        disp_data_window[substitutes == 1] = out_disp[r + center, c + center]

    return out_disp

def create_urdf_for_furniture_mesh(mesh_dir, concave=False, out_dir=None, mass=0.4, has_collision=True, furniture=False):
    assert '.obj' in mesh_dir, f'mesh_dir={mesh_dir}'

    lateral_friction = 1.0
    spinning_friction = 0.001
    rolling_friction = 0.001
    mass = float(mass)

    model_name = mesh_dir.split('/')[-1].split('.')[0]
    if furniture:
        model_name = 'furniture_' + model_name

    material_block = f"""  <material name="Grey">
    <color rgba="0.7 0.7 0.7 1.0"/>
  </material>"""
    if furniture:
        inertial_block = f"""<inertial>
      <origin rpy="0 0 0" xyz="90 0 0"/>
      <mass value="0.0" />
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>"""
    else:
        inertial_block = f"""<contact>
      <lateral_friction value="{lateral_friction}"/>
      <rolling_friction value="{rolling_friction}"/>
      <spinning_friction value="{spinning_friction}"/>
    </contact>
    <inertial>
          <mass value="{mass}" />
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>"""
    concave_str = 'no'
    collision_mesh_dir = copy.deepcopy(mesh_dir.split('/')[-1])
    if concave:
        concave_str = 'yes'
        if mass != 0:
            collision_mesh_dir = mesh_dir.replace('.obj', '_vhacd.obj')
    collision_block = ""
    if has_collision:
        collision_block = f"""<collision concave="{concave_str}">
      <geometry>
        <mesh filename="{collision_mesh_dir}"/>
      </geometry>
      <material name="Grey"/>
    </collision>"""
    link_str = f"""  <link concave="{concave_str}" name="base_link">
    {inertial_block}
    <visual>
      <geometry>
        <mesh filename="{collision_mesh_dir}"/>
      </geometry>
      <material name="Grey"/>
    </visual>
    {collision_block}
  </link>"""
    urdf_str = f"""<?xml version="1.0"?>
<robot name="{model_name}.urdf">
{material_block}
{link_str}
</robot>"""
    if out_dir is None:
        out_dir = mesh_dir.replace('.obj', '.urdf')
    with open(out_dir, 'w') as ff:
        ff.write(urdf_str)

    return out_dir

def create_urdf_for_mesh(mesh_dir, concave=False, out_dir=None, mass=0.4, has_collision=True):
    assert '.obj' in mesh_dir, f'mesh_dir={mesh_dir}'

    lateral_friction = 1.0
    spinning_friction = 0.001
    rolling_friction = 0.001
    contact_cfm = 0.0
    conctact_erp = 1.0

    model_name = mesh_dir.split('/')[-1].split('.')[0]

    visual_dir = mesh_dir.split('/')[-1] # .split('.')[-2].split('_collision')[:-1][0] + '.obj'

    material_block = f"""  <material name="Grey">
    <color rgba="0.7 0.7 0.7 1.0"/>
  </material>"""
    concave_str = 'no'
    collision_mesh_dir = copy.deepcopy(visual_dir)
    if concave:
        concave_str = 'yes'
    if mass != 0:
        collision_mesh_dir = visual_dir.replace('.obj', '_vhacd.obj')
    collision_block = ""
    if has_collision:
        collision_block = f"""<collision>
      <geometry>
        <mesh filename="{collision_mesh_dir}"/>
      </geometry>
      <material name="Grey"/>
    </collision>"""
    link_str = f"""  <link name="base_link">
    <contact>
      <lateral_friction value="{lateral_friction}"/>
      <rolling_friction value="{rolling_friction}"/>
      <spinning_friction value="{spinning_friction}"/>
    </contact>
    <inertial>
      <mass value="{mass}" />
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="{visual_dir}"/>
      </geometry>
      <material name="Grey"/>
    </visual>
    {collision_block}
  </link>"""
    urdf_str = f"""<?xml version="1.0"?>
<robot name="{model_name}.urdf">
{material_block}
{link_str}
</robot>"""
    if out_dir is None:
        out_dir = mesh_dir.replace('.obj', '.urdf')
    with open(out_dir, 'w') as ff:
        ff.write(urdf_str)

    return out_dir


def scale_data(data, scaling='max', sc_min=0.6, sc_max=0.8):
    """Scales a numpy array to [0, 255].

        Parameters
        ----------
        data: :obj:`numpy.ndarray`
            Data to be scaled.
        scaling: str
            Scaling method. Can be 'fixed' to scale between fixed values,
            or 'max' to scale between the minimum and maximum of the data.
            Defaults to 'max'.
        sc_min: float
            Lower bound for fixed scaling. Defaults to 0.6.
        sc_max: float
            Upper bound for fixed scaling. Defaults to 0.8

        Returns
        -------
        :obj:`numpy.ndarray`
            Scaled numpy array with the same shape as input array data.
    """
    data_fl = data.flatten()
    if scaling == 'fixed':
        scaled = np.interp(data_fl, (sc_min, sc_max), (0, 255), left=0, right=255)
    elif scaling == 'max':
        scaled = np.interp(data_fl, (min(data_fl), max(data_fl)), (0, 255), left=0, right=255)
    else:
        raise AttributeError
    integ = scaled.astype(np.uint8)
    integ.resize(data.shape)
    return integ


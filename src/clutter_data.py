import torch.utils.data
import numpy as np

from src.utils import depth_encoding, get_tensor_list


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, image_size=None):
        self.root = root
        self.image_size = image_size

        self.depth_min = 0.2
        self.depth_max = 1.5
        self.width_max = 0.08

        self.tensor_list = get_tensor_list(self.root)

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, i):
        x, y = self._get_ground_truth_data(i)

        return x, y

    def _select_grasps(self, grasp_data):
        """
        Selects the entities of the grasp_data tensor
        :param grasp_data: numpy array containing details about the grasp with g in
        {u, v, z, contact_u, contact_v, width, quaternion, hinge_rot, quality, collision_free}
        :return: pose input used to train the network.
        """
        try:
            u = grasp_data[:, 3]
            v = grasp_data[:, 4]
            width = grasp_data[:, 5] / self.width_max
            rot = grasp_data[:, 6:10]
            metric = grasp_data[:, 10]
            success = grasp_data[:, 11]
        except IndexError:
            # Only one grasp in array - index accordingly
            u = grasp_data[3]
            v = grasp_data[4]
            width = grasp_data[5] / self.width_max
            rot = grasp_data[6:10]
            metric = grasp_data[10]
            success = grasp_data[11]

        return u.astype(int), v.astype(int), width, rot, success

    def _get_ground_truth_data(self, image_label):
        """
        Chooses a grasp in the loaded tensor, loads the data for network training and updates the indices for
        future selections.
        :return: image_data: numpy array in shape (1, 300, 300) for the depth image
                 pose_data: numpy array in shape (1,) for the pose input (shape varies for alternative self.pose_input)
                 label_data: [0/1] for ground truth negative/positive grasps
        """

        image = np.load('{}/depth_image_{:07d}.npz'.format(self.root, image_label))['arr_0'].squeeze()
        grasps = np.load('{}/grasps_{:07d}.npz'.format(self.root, image_label))['arr_0']
        segmask = np.load('{}/segmask_image_{:07d}.npz'.format(self.root, image_label))['arr_0'].squeeze()

        if self.image_size is not None:
            sz = image.shape
            u0 = None
            if sz != self.image_size:
                v0 = int(sz[0] / 2 - self.image_size[0] / 2)
                v1 = int(sz[0] / 2 + self.image_size[0] / 2)
                u0 = int(sz[1] / 2 - self.image_size[1] / 2)
                u1 = int(sz[1] / 2 + self.image_size[1] / 2)
                image = image[v0:v1, u0:u1]
                segmask = segmask[v0:v1, u0:u1]

        too_small = np.where(image < self.depth_min)
        too_big = np.where(image > self.depth_max)

        jet_im = depth_encoding(image, vmin=self.depth_min, vmax=self.depth_max)
        y_true = np.zeros(shape=(image.shape[0], image.shape[1], 8))

        # We can get the non-object pixel from the segmask - all those are negative grasps
        non_object = np.where(segmask == 0)

        # Indicate we have ground truth at the table indices, but leaving the quality as zeros
        # Numpy coordinates, since they come from np.where()!!
        y_true[non_object[0], non_object[1], 0] = 1

        try:
            u_ind, v_ind, width, quaternion, labels = self._select_grasps(grasps)
        except IndexError:
            # Didn't have any grasps for this image - we just return the table grasps
            return jet_im, y_true

        if len(u_ind) == 0:
            # Didn't have any contact grasps for this image - we just return the table grasps
            return jet_im, y_true

        if self.image_size is not None and u0 is not None:
            mask = (u_ind > u0) & (u_ind < u1) & (v_ind > v0) & (v_ind < v1)
            u_ind = u_ind[mask] - u0
            v_ind = v_ind[mask] - v0
            labels = labels[mask]
            quaternion = quaternion[mask, :]
            width = width[mask]

        # Image coordinates (u, v) to numpy coordinates (row, column)
        y_true[v_ind, u_ind, 0] = 1
        y_true[too_small[0], too_small[1], 0] = 0  # Treating everything that can't be seen as unknown
        y_true[too_big[0], too_big[1], 0] = 0  # Treating everything that can't be seen as unknown
        y_true[v_ind, u_ind, 1] = labels
        y_true[v_ind, u_ind, 2] = quaternion[:, 0]
        y_true[v_ind, u_ind, 3] = quaternion[:, 1]
        y_true[v_ind, u_ind, 4] = quaternion[:, 2]
        y_true[v_ind, u_ind, 5] = quaternion[:, 3]
        y_true[v_ind, u_ind, 6] = width

        return jet_im, y_true

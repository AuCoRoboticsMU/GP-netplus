import time

import numpy as np
import torch
from PIL import Image
import os

from src.utils import depth_encoding, scale_data
from src.experiments.vis_utils import Grasp
from src.experiments.transform import Transform, Rotation
from src.model import load_network
from skimage.feature import peak_local_max


class GPnetplus(object):
    def __init__(self, model_path, detection_threshold=0.9):
        """
        Initiate a GP-Net+ model.

        :param model_path: path to the .pt file
        :param detection_threshold: Detection threshold used when applying Non-Maximum Suppression to the predicted
                                    quality output
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device)
        self.detection_threshold = detection_threshold
        self.depth_min = 0.2
        self.depth_max = 1.5
        self.width_max = 0.08
        self.debug_root = 'data/debug'
        if not os.path.exists(self.debug_root):
            os.mkdir(self.debug_root)

    def __call__(self, state):
        """
        Run inference with a GP-Net+ model and map the output to grasp proposals
        :param state: object including a depth image, segmask image and camera intrinsics
        :return: grasps, scores, indices, object_ids, quality_precision, time_to_propose_grasps
        """
        depth_im = state.depth_im
        segmask = state.segmask_im
        camera_intr = state.camera_intrinsics
        camera_extr = Transform.from_list(state.camera_extrinsics)

        tic = time.time()
        qual_pred, rot_pred, width_pred = self.predict(depth_im, self.net, self.device)

        image_depth = Image.fromarray(scale_data(depth_im, scaling='fixed',
                                                 sc_min=self.depth_min, sc_max=self.depth_max))
        image_depth.save('{}/depth.png'.format(self.debug_root))
        image_jet = Image.fromarray(np.moveaxis(scale_data(depth_encoding(depth_im,
                                                                          vmin=self.depth_min,
                                                                          vmax=self.depth_max)),
                                                [0, 1, 2], [2, 0, 1]))
        image_jet.save('{}/jet.png'.format(self.debug_root))
        image_segmask = Image.fromarray(scale_data(segmask))
        image_segmask.save('{}/segmask.png'.format(self.debug_root))
        image_quality = Image.fromarray(scale_data(qual_pred, scaling='fixed', sc_min=0.0, sc_max=1.0))
        image_quality.save('{}/quality.png'.format(self.debug_root))
        blended_image = Image.blend(image_depth, image_quality, alpha=0.6)
        blended_image.save('{}/overlay.png'.format(self.debug_root))

        grasps, scores, indices, object_ids = self.select_grasps(qual_pred, rot_pred, width_pred, depth_im,
                                                                 camera_intr, camera_extr, segmask)

        toc = time.time() - tic

        grasps, scores = np.asarray(grasps), np.asarray(scores)

        # Sort after grasp scores:
        if len(grasps) > 0:
            p = np.argsort(scores)[::-1]
            grasps = grasps[p]
            scores = scores[p]
            indices = indices[p]
            object_ids = object_ids[p]

        return grasps, scores, indices, object_ids, toc

    def predict(self, depth_image, net, device):
        """
        Encode a depth image in a jet-colourscale and run a forward pass with a GP-Net+ network.
         Set predicted quality of invisible pixels to 0 (as they cannot be mapped into a grasp proposal).
        :param depth_image: Depth image to run forward pass with.
        :param net: GP-Net model implementation.
        :param device: Device to map the tensors to (GPU or CPU)
        :return: Quality, Orientation and Width channels of the network output
        """
        x = depth_encoding(depth_image, vmin=self.depth_min, vmax=self.depth_max)
        x = torch.from_numpy(x).unsqueeze(0).to(device)

        # forward pass
        with torch.no_grad():
            qual, rot, width = net(x)

        # move output back to the CPU
        qual = qual.cpu().squeeze().numpy()
        rot = rot.cpu().squeeze().numpy()
        width = width.cpu().squeeze().numpy()
        qual[depth_image < self.depth_min] = 0.0
        qual[depth_image > self.depth_max] = 0.0
        return qual, rot, width

    def select_grasps(self, pred_qual, pred_quat, pred_width, depth_im, camera_intr,
                      camera_extr, segmask):
        """
        Select pixels with the highest predicted grasp quality by running NMS and
        reconstructing grasp proposals from them.

        :param pred_qual: Predicted quality tensor.
        :param pred_quat: Orientation tensor.
        :param pred_width: Width tensor.
        :param depth_im: Input depth image.
        :param camera_intr: Camera intrinsics.
        :param camera_extr: Camer extrinsics (pose of camera).
        :param segmask: Segmask image.
        :return: Selected grasp proposals, Quality of selected grasp proposals, Image coordinates of selected
                    grasp proposals and UID's of selected grasp proposals.
        """
        indices = peak_local_max(pred_qual, threshold_abs=self.detection_threshold)
        object_ids = segmask[indices[:, 0], indices[:, 1]]
        grasps = []
        qualities = []

        selected = np.zeros(pred_qual.shape)
        selected[indices[:, 0], indices[:, 1]] = 255
        image_quality = Image.fromarray(selected.astype(np.uint8))
        image_quality.save('{}/selected_im.png'.format(self.debug_root))

        for index in indices:
            quaternion = pred_quat[:, index[0], index[1]]
            quality = pred_qual[index[0], index[1]]

            contact = (index[1], index[0])
            width = pred_width[index[0], index[1]] * self.width_max
            grasp = self.reconstruct_grasp_from_variables(depth_im, contact, quaternion, width, camera_intr, camera_extr)
            grasps.append(grasp)
            qualities.append(quality)
        return grasps, qualities, indices, object_ids

    @staticmethod
    def reconstruct_grasp_from_variables(depth_im, contact, quaternion, width,
                                         camera_intr, T_camera_world):
        """
        Reconstruct a grasp proposal from the contact in the depth image, quaternion and its width.
        :param depth_im: Input depth image.
        :param contact: Image coordinates of the grasp contact.
        :param quaternion: Orientation of the grasp proposal in form of a quaternion.
        :param width: Width of the grasp proposal
        :param camera_intr: Intrinsics of the camera, used to de-project the grasp contact into 3D coordinates.
        :param T_camera_world: Transform between the camera and world coordinates to map the grasp proposals to the
                                world coordinate frame
        :return: Grasp object
        """
        # Deproject from depth image into image coordinates
        # Note that homogeneous coordinates have the image coordinate order (x, y), while accessing the depth image
        # works with numpy coordinates (row, column)
        homog = np.array((contact[0], contact[1], 1)).reshape((3, 1))
        point = depth_im[contact[1], contact[0]] * np.linalg.inv(camera_intr.intrinsic.K).dot(homog)
        point = point.squeeze()

        # Transform the quaternion into a rotation matrix
        rot = Rotation.from_quat([quaternion[1],
                                  quaternion[2],
                                  quaternion[3],
                                  quaternion[0]]).as_matrix()

        # Move from contact to grasp centre by traversing 0.5*grasp width in grasp axis direction
        centre_point = point + width / 2 * rot[:, 0]

        # Construct transform Camera --> gripper
        T_camera_grasp = Transform.from_matrix(np.r_[np.c_[rot, centre_point], [[0, 0, 0, 1]]])

        # Express grasp in world coordinates
        T_gripper_world = T_camera_world.inverse() * T_camera_grasp

        return Grasp(T_gripper_world, width)


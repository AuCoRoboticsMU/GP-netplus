import time

import numpy as np
import pybullet
from pybullet_utils import bullet_client

from src.experiments.transform import Rotation, Transform

assert pybullet.isNumpyEnabled(), "Pybullet needs to be built with NumPy"


class BtWorld(object):
    """Interface to a PyBullet physics server.

    Attributes:
        dt: Time step of the physics simulation.
        solver_iterations: Number of iterations for the solver
    """

    def __init__(self, gui=True):
        connection_mode = pybullet.GUI if gui else pybullet.DIRECT
        self.p = bullet_client.BulletClient(connection_mode)

        self.gui = gui
        self.dt = 1.0 / 240.0
        self.solver_iterations = 150
        if self.gui:
            self.p.configureDebugVisualizer(2, 1, (0, -5, 1.5))

        self.reset()

    def set_gravity(self, gravity):
        self.p.setGravity(*gravity)

    def load_urdf(self, urdf_path, pose, scale=1.0):
        body = Body.from_urdf(self.p, urdf_path, pose, scale)
        self.bodies[body.uid] = body
        return body

    def remove_body(self, body):
        self.p.removeBody(body.uid)
        del self.bodies[body.uid]

    def add_constraint(self, *argv, **kwargs):
        """See `Constraint` below."""
        constraint = Constraint(self.p, *argv, **kwargs)
        return constraint

    def add_camera(self, intrinsic, near, far):
        camera = Camera(self.p, intrinsic, near, far)
        return camera

    def generate_parallel_projection(self, object_top_coords, workspace_height, projection_scale):
        """
        Render an image using a parallel projection with the camera pointing in negative z direction (downwards).
        :param object_top_coords: 3D coordinates of where to look at.
        :param workspace_height: Height of plane to look at.
        :param projection_scale: Scale of projection plane that will be covered in the image.
        :return:
        """
        width = 500
        height = 500
        distance_of_camera = 1000
        camera_target_position = [object_top_coords[0], object_top_coords[1], workspace_height]
        camera_eye_position = [object_top_coords[0], object_top_coords[1], object_top_coords[2]+distance_of_camera]
        view_matrix = self.p.computeViewMatrix(
            cameraTargetPosition=camera_target_position,
            cameraEyePosition=camera_eye_position,
            cameraUpVector=[1, 0, 0]
        )

        cameraVisualizationLine = self.p.addUserDebugLine(camera_eye_position, camera_target_position,
                                                          lineColorRGB=[1, 0, 0], lineWidth=2)
        projection_matrix = self.p.computeProjectionMatrix(
            left=-projection_scale,
            right=projection_scale,
            bottom=-projection_scale,
            top=projection_scale,
            nearVal=distance_of_camera-0.01,
            farVal=distance_of_camera + object_top_coords[2] - workspace_height + 0.05
        )
        image_arr = self.p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )
        self.p.removeUserDebugItem(cameraVisualizationLine)

        segmask = image_arr[4]
        return segmask

    def get_contacts(self, bodyA):
        points = self.p.getContactPoints(bodyA.uid)
        contacts = []
        for point in points:
            contact = Contact(
                bodyA=self.bodies[point[1]],
                bodyB=self.bodies[point[2]],
                point=point[5],
                normal=point[7],
                depth=point[8],
                force=point[9],
            )
            contacts.append(contact)
        return contacts

    def reset(self):
        self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(
            fixedTimeStep=self.dt, numSolverIterations=self.solver_iterations
        )
        self.bodies = {}
        self.sim_time = 0.0

    def step(self):
        self.p.stepSimulation()
        self.sim_time += self.dt
        if self.gui:
            time.sleep(self.dt)

    def save_state(self):
        return self.p.saveState()

    def restore_state(self, state_uid):
        self.p.restoreState(stateId=state_uid)

    def save_world(self, file_dir):
        """
        Save a world configurations by capturing the urdf names of all bodies in the scene in a .txt file and
        saving the current pose of the bodies and physics parameters in a .bullet file.
        :param file_dir: Full path to the .bullet file. The .txt file will be based on the bullet filename.
        """
        with open('{}_bodies.txt'.format(file_dir.split('.')[0]), 'w') as f:
            for body in self.bodies:
                f.write('{},{}\n'.format(self.bodies[body].name, self.bodies[body].scale))
        self.p.saveBullet(file_dir)

    def load_world(self, file_dir, urdf_files_path, base_urdf_files_path):
        """
        Load a world configuration based on a .bullet file and a .txt file containing all urdf filenames.
        :param file_dir: Filename of the saved world configuration.
        :param urdf_files_path: Path where the object set is saved.
        :param base_urdf_files_path: Path where the urdfs are saved.
        :return: Number of background objects are in the scene, used to calculate the number of objects.
        """
        base_urdf_files_path = str(base_urdf_files_path)
        self.reset()
        self.set_gravity([0.0, 0.0, -9.81])
        gripper_uid = None
        cnt_background = 0
        with open('{}_bodies.txt'.format(file_dir), 'r') as f:
            all_bodies = f.readlines()
        for cnt, body in enumerate(all_bodies):
            try:
                body, scale = body.strip().split(',')
            except ValueError:
                body = body.strip()
                scale = 1.0
            if 'ground' in body:
                cnt_background += 1
                urdf_base = base_urdf_files_path + '/furniture/'
                body += '.urdf'
            elif 'furniture' in body:
                body = '_'.join(body.split('_')[1:])
                cnt_background += 1
                urdf_base = base_urdf_files_path + '/furniture/test/'
            elif 'plane.urdf' in body:
                body = 'plane.urdf'
                cnt_background += 1
                urdf_base = base_urdf_files_path + '/setup/'
            elif 'pal_gripper' in body:
                urdf_base = base_urdf_files_path + '/pal_gripper/'
                body = 'gripper_exact.urdf'
                cnt_background += 1
                gripper_uid = cnt
            else:
                urdf_base = urdf_files_path
            self.load_urdf('{}/{}'.format(urdf_base, body),
                           Transform(Rotation.identity(), [0.0, 0.0, 0.0]),
                           scale=float(scale))
        self.p.restoreState(fileName='{}.bullet'.format(file_dir))
        self.set_gravity([0.0, 0.0, -9.81])
        if gripper_uid is not None:
            cnt_background -= 1
            self.remove_body(self.bodies[gripper_uid])
        return cnt_background

    def close(self):
        self.p.disconnect()


class Body(object):
    """Interface to a multibody simulated in PyBullet.

    Attributes:
        uid: The unique id of the body within the physics server.
        name: The name of the body.
        joints: A dict mapping joint names to Joint objects.
        links: A dict mapping link names to Link objects.
    """

    def __init__(self, physics_client, body_uid, scale):
        self.p = physics_client
        self.uid = body_uid
        self.name = self.p.getBodyInfo(self.uid)[1].decode("utf-8")
        self.joints, self.links = {}, {}
        self.scale = scale
        for i in range(self.p.getNumJoints(self.uid)):
            joint_info = self.p.getJointInfo(self.uid, i)
            joint_name = joint_info[1].decode("utf8")
            self.joints[joint_name] = Joint(self.p, self.uid, i)
            link_name = joint_info[12].decode("utf8")
            self.links[link_name] = Link(self.p, self.uid, i)

    @classmethod
    def from_urdf(cls, physics_client, urdf_path, pose, scale):
        body_uid = physics_client.loadURDF(
            str(urdf_path),
            pose.translation,
            pose.rotation.as_quat(),
            globalScaling=scale,
        )
        return cls(physics_client, body_uid, scale)

    def get_pose(self):
        pos, ori = self.p.getBasePositionAndOrientation(self.uid)
        return Transform(Rotation.from_quat(ori), np.asarray(pos))

    def set_pose(self, pose):
        self.p.resetBasePositionAndOrientation(
            self.uid, pose.translation, pose.rotation.as_quat()
        )

    def get_velocity(self):
        linear, angular = self.p.getBaseVelocity(self.uid)
        return linear, angular

    def deactivate_collisions(self):
        self.p.setCollisionFilterGroupMask(self.uid, linkIndexA=-1,
                                           collisionFilterGroup=0, collisionFilterMask=0)

    def activate_collisions(self):
        self.p.setCollisionFilterGroupMask(self.uid, linkIndexA=-1,
                                           collisionFilterGroup=1, collisionFilterMask=1)


class Link(object):
    """Interface to a link simulated in Pybullet.

    Attributes:
        link_index: The index of the link.
        p: The pybullet physics client.
        body_uid: UID of the body the link belongs to.
    """

    def __init__(self, physics_client, body_uid, link_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.link_index = link_index

    def get_pose(self):
        link_state = self.p.getLinkState(self.body_uid, self.link_index)
        pos, ori = link_state[0], link_state[1]
        return Transform(Rotation.from_quat(ori), pos)


class Joint(object):
    """Interface to a joint simulated in PyBullet.

    Attributes:
        joint_index: The index of the joint.
        p: The pybullet physics client.
        body_uid: UID of the body the joint belongs to.
        lower_limit: Lower position limit of the joint.
        upper_limit: Upper position limit of the joint.
        effort: The maximum joint effort.
    """

    def __init__(self, physics_client, body_uid, joint_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.joint_index = joint_index

        joint_info = self.p.getJointInfo(body_uid, joint_index)
        self.lower_limit = joint_info[8]
        self.upper_limit = joint_info[9]
        self.effort = joint_info[10]

    def get_position(self):
        joint_state = self.p.getJointState(self.body_uid, self.joint_index)
        return joint_state[0]

    def set_position(self, position, kinematics=False):
        if kinematics:
            self.p.resetJointState(self.body_uid, self.joint_index, position)
        self.p.setJointMotorControl2(
            self.body_uid,
            self.joint_index,
            pybullet.POSITION_CONTROL,
            targetPosition=position,
            force=self.effort,
        )


class Constraint(object):
    """Interface to a constraint in PyBullet.

    Attributes:
        uid: The unique id of the constraint within the physics server.
    """

    def __init__(
        self,
        physics_client,
        parent,
        parent_link,
        child,
        child_link,
        joint_type,
        joint_axis,
        parent_frame,
        child_frame,
    ):
        """
        Create a new constraint between links of bodies.

        Args:
            parent:
            parent_link: None for the base.
            child: None for a fixed frame in world coordinates.

        """
        self.p = physics_client
        parent_body_uid = parent.uid
        parent_link_index = parent_link.link_index if parent_link else -1
        child_body_uid = child.uid if child else -1
        child_link_index = child_link.link_index if child_link else -1

        self.uid = self.p.createConstraint(
            parentBodyUniqueId=parent_body_uid,
            parentLinkIndex=parent_link_index,
            childBodyUniqueId=child_body_uid,
            childLinkIndex=child_link_index,
            jointType=joint_type,
            jointAxis=joint_axis,
            parentFramePosition=parent_frame.translation,
            parentFrameOrientation=parent_frame.rotation.as_quat(),
            childFramePosition=child_frame.translation,
            childFrameOrientation=child_frame.rotation.as_quat(),
        )

    def change(self, **kwargs):
        self.p.changeConstraint(self.uid, **kwargs)


class Contact(object):
    """Contact point between two multibodies.

    Attributes:
        point: Contact point.
        normal: Normal vector from ... to ...
        depth: Penetration depth
        force: Contact force acting on body ...
    """

    def __init__(self, bodyA, bodyB, point, normal, depth, force):
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.point = point
        self.normal = normal
        self.depth = depth
        self.force = force


class Camera(object):
    """Virtual RGB-D camera based on the PyBullet camera interface.

    Attributes:
        intrinsic: The camera intrinsic parameters.
    """

    def __init__(self, physics_client, intrinsic, near, far):
        self.intrinsic = intrinsic
        self.near = near
        self.far = far
        self.proj_matrix = _build_projection_matrix(intrinsic, near, far)
        self.p = physics_client

    def render(self, extrinsic):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.as_matrix()
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")
        gl_proj_matrix = self.proj_matrix.flatten(order="F")

        result = self.p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_proj_matrix,
            renderer=pybullet.ER_TINY_RENDERER,
        )

        rgb, z_buffer, segmask = result[2][:, :, :3], result[3], result[4]
        depth = (1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer))
        return rgb, depth, segmask


def _build_projection_matrix(intrinsic, near, far):
    perspective = np.array(
        [
            [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
            [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
            [0.0, 0.0, near + far, near * far],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
    return np.matmul(ortho, perspective)


def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
    )
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho

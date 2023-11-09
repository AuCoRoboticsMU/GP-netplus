from pathlib import Path
import pybullet

from src.experiments.perception import *
from src.experiments.vis_utils import workspace_lines, Grasp
from src.experiments.transform import Rotation, Transform
from src.experiments import btsim
from src.utils import Workspaces, CameraSampleSpace, is_object_in_air, add_kinetic_noise


class ClutterRemovalSim(object):
    def __init__(self, object_set, split, gui=True, seed=None, urdf_root=None):
        """
        Initiate a simulation environment for tabletop or domestic grasping scenes.
        :param object_set: Object set to use. This is a folder in the urdf_root or 'data/urdfs' directory including
                            URDFS of different objects to be used in the simulation.
        :param split: Object and furniture split to use. Can be 'train', 'val' and 'test'
        :param gui: Boolean value, defaults to True in order to visualise a GUI of the simulation environment
        :param seed: Seed using when drawing random samples
        :param urdf_root: root path for all URDF folders and files. When None, defaults to 'data/urdfs'.
        """
        if urdf_root is None:
            self.urdf_root = Path("data/urdfs")
        else:
            self.urdf_root = Path(urdf_root)
        self.object_set = object_set
        self.split = split
        self.discover_objects()
        self.discover_furniture()
        self.cnt_background = 0

        self.global_scaling = {"blocks": 1.67}.get(object_set, 1.0)
        self.gui = gui

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui)
        self.object_count = None
        self.table_name = ''
        intrinsic = CameraIntrinsic(640, 480, 620.3, 620.7, 331.3, 235.2)
        self.camera = self.world.add_camera(intrinsic, 0.2, 2.5)


    @property
    def num_objects(self):
        """
        Calculate the number of objects in the scene using the body count in simulation and substracting the number
        of background objects (e.g. ground, furniture unit, gripper, ...)
        """
        return max(0, self.world.p.getNumBodies() - self.cnt_background)  # remove table, ground and gripper from  count

    def discover_objects(self):
        """
        Load the available object URDFS to be used for sampling when generating tabletop or domestic scenes
        :return:
        """
        root = self.urdf_root / self.object_set / self.split
        self.object_urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]

    def discover_furniture(self):
        """
        Load the available furniture URDFS to be used for sampling when generating domestic scenes
        """
        root = self.urdf_root / 'furniture' / self.split
        self.furniture_urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]

    def save_state(self):
        """
        Save the state and remember the ID of the state in self._snapshot_id
        """
        self._snapshot_id = self.world.save_state()

    def save_world(self, filename):
        """
        Save a btsim world to a .bullet and a .txt file
        :param filename: Path to the .bullet and .txt file including the scene identifier
        """
        self.world.save_world(file_dir=filename)

    def load_world(self, filename, panda=False):
        """
        Load a btsim world from a .bullet and a .txt file.
        :param filename: Path to the .bullet and .txt file including the scene identifier
        :param panda: Boolean indicator if using a panda gripper. Defaults to false, using a PAL gripper instead.
        """
        urdf_files_path = self.urdf_root / self.object_set / self.split
        self.cnt_background = self.world.load_world(filename, urdf_files_path, self.urdf_root)

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=0.0,
                cameraPitch=-40,
                cameraTargetPosition=[0.0, 0.0, 0.2],
            )
        for body in self.world.bodies:
            self.world.p.setCollisionFilterGroupMask(body, linkIndexA=-1,
                                                     collisionFilterGroup=1, collisionFilterMask=1)
            if 'furniture' in self.world.bodies[body].name:
                table_name = Path('_'.join(self.world.bodies[body].name.split('_')[1:]).split('.')[0])
                self.table_name = self.urdf_root / 'furniture' / self.split / table_name
        self.wait_for_objects_to_rest()
        self.gripper = Gripper(self.world, panda=panda)
        self.cnt_background += 1
        self.size = 6 * self.gripper.finger_depth
        self.draw_workspace()

    def restore_state(self):
        """
        Restore a state previously saved using self._snapshot_id
        """
        self.world.restore_state(self._snapshot_id)

    def reset_domestic_scene(self, object_count):
        """
        Reset a domestic scene by placing a furniture unit, placing objects on the furniture unit
                and loading a gripper model
        :param object_count:
        :return:
        """
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.cnt_background = 0

        # Set ground
        self.world.load_urdf('data/urdfs/furniture/ground.urdf', Transform(Rotation.identity(), [0.0, 0.0, 0.0]))

        if self.gui:
            # Reset camera pose of the GUI
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=0.0,
                cameraPitch=-5,
                cameraTargetPosition=[0.0, 0.0, 1.3],
            )

        # Set scene
        self.place_furniture(object_count=object_count)
        self.place_objects_in_domestic_scene()

        # Load gripper
        self.gripper = Gripper(self.world)
        self.cnt_background += 1
        self.size = 6 * self.gripper.finger_depth

    def reset_simplistic(self, object_count):
        """
        Create a new tabletop scene, with objects dropped in a 30cm x 30cm workspace on a planar surface
        :param object_count: Number of objects that will be (attempted) to dropped into the workspace
        """
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.size = 0.3
        self.cnt_background = 0

        # Set ground
        self.place_planar_surface(0.005)
        self.cnt_background += 1

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=0.5,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.0, 0.0, 0.01],
            )

        self.pile_objects_in_tabletop_scene(object_count, 0.005)
        self.gripper = Gripper(self.world)
        self.cnt_background += 1
        self.draw_workspace()

    def draw_workspace(self):
        """
        Draw the workspace in the pybullet GUI as a 30cm x 30cm x 30cm box. Note that this is only visible if setting
        gui = True when initiating the ClutterRemovalSim
        """
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color)

    def place_furniture(self, object_count=None, urdf=None):
        """
        Place furniture (table or shelf) into the scene.
        :param object_count: When set to None, the object_count will be sampled based on a poisson distribution to get
                             an even density of objects on the furniture
        :param urdf: When set to None, a furniture unit will be sampled uniformly randomly from the set of all
                             available furniture units
        """
        if urdf is None:
            urdf = self.rng.choice(self.furniture_urdfs)
        pose = Transform(Rotation.identity(), [0.0, 0.0, 0.0])
        self.cnt_background += 1

        body = self.world.load_urdf(urdf, pose, scale=1.0)
        self.table_name = str(urdf).split('.')[0]
        self.workspaces = Workspaces.load(self.table_name + '_workspace.pkl')
        if object_count is None:
            self.object_count = self.workspaces.sample_object_count()
        else:
            self.object_count = object_count

    def place_planar_surface(self, height):
        """
        Place a planar surface (ground plane) into the scene.
        :param height: At which height to place the ground plane
        """
        urdf = self.urdf_root / "setup" / "plane.urdf"
        pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        self.world.load_urdf(urdf, pose, scale=0.6)
        self.cnt_background += 1

        # define valid volume for sampling grasps
        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height + 0.005, self.size
        self.lower = np.r_[lx, ly, lz]
        self.upper = np.r_[ux, uy, uz]

    def pile_objects_in_tabletop_scene(self, object_count, table_height):
        """
        Piles the object in the workspace using a temporary box for a simple tabletop scene.
        :param object_count: Number of objects that will be (attempted) to be dropped into the workspace
        :param table_height: Height of the planar surface
        """
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3)

        # drop objects
        urdfs = self.rng.choice(self.object_urdfs, size=object_count)
        for urdf in urdfs:
            rotation = Rotation.random(random_state=self.rng)
            # xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            xy = np.array((self.size / 2.0, self.size / 2.0))
            pose = Transform(rotation, np.r_[xy, table_height + 0.1])
            scale = 1.0  # self.rng.uniform(0.8, 1.0)
            self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            self.wait_for_objects_to_rest(timeout=1.0)

        # remove box
        self.world.remove_body(box)
        self.remove_and_wait_simple()

    def place_objects_in_domestic_scene(self):
        """
        Places objects on furniture units (tables, shelves) using area-weighted uniform rejection sampling
        on all workspaces of the furniture unit.
        """
        attempts = 0
        max_attempts = max(int(np.round(self.object_count * 2, 0)), 10)

        while self.num_objects < self.object_count and attempts < max_attempts:
            self.save_state()

            urdf = self.rng.choice(self.object_urdfs)
            x, y, z = self.workspaces.sample_among_workspace()
            obj_scale = self.rng.uniform(0.8, 1.2)
            if self.rng.choice([0, 1], p=[0.5, 0.5]):
                # Pose completely random
                rotation = Rotation.random(random_state=self.rng)
            else:
                # Pose upright and randomly rotated around the positive z axis.
                rotation = Rotation.from_rotvec(self.rng.uniform(0, 2 * np.pi) * np.array([0, 0, 1]))
            pose = Transform(rotation, np.r_[x, y, z])

            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * obj_scale)
            new_mass = body.p.getDynamicsInfo(body.uid, -1)[0] * self.global_scaling * obj_scale
            body.p.changeDynamics(body.uid, -1, mass=new_mass, angularDamping=0.0, linearDamping=0.0)
            # Readjust position so that object is 2cm above the workspace plane
            lower, upper = self.world.p.getAABB(body.uid)
            z_new = z + (z - lower[2]) + 0.02
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z_new]))
            self.world.step()

            if self.world.get_contacts(body):
                # Object is in collision with furniture unit or other object (could cause simulation to "implode")
                # Remove object and restore original state before object was dropped
                self.world.remove_body(body)
                self.restore_state()
            else:
                # Forward simulation to let object come to rest
                all_objects_in_rest = self.remove_and_wait(added_body_uid=body.uid)
                if not all_objects_in_rest:
                    # Latest object moved some of the other objects and they didn't settle
                    self.world.remove_body(body)
                    self.restore_state()
                # Filter out those that fell down to the ground.
                if self.world.bodies[max(list(self.world.bodies.keys()))].uid == body.uid and \
                        self.world.p.getAABB(body.uid)[0][2] < z - 0.1:  # 10cm buffer, as getAABB returns bounding box
                    # Object fell to the ground
                    self.world.remove_body(body)
                    self.restore_state()
                elif self.world.bodies[max(list(self.world.bodies.keys()))].uid == body.uid:
                    current_lower, current_upper = self.world.p.getAABB(body.uid)
                    current_pose = self.world.p.getBasePositionAndOrientation(body.uid)
                    scale = max(current_upper[0] - current_lower[0], current_upper[1] - current_lower[1])
                    segmask = self.world.generate_parallel_projection([current_pose[0][0], current_pose[0][1],
                                                                       current_upper[2]], z, scale)
                    in_air = is_object_in_air(segmask, body.uid)
                    if in_air:
                        # Object is balancing over the workspace border
                        self.world.remove_body(body)
                        self.restore_state()

            attempts += 1

    def render_images_domestic_scene(self, n=None, extrinsic_matr=None):
        """
        Render images from a camera pose uniformly sampled in a robot-based camera sampling space in a domestic
        grasping scene (including furniture units)
        :param n: Number of images to sample. If None, n is sampled from poisson distribution based on camera sampling volume.
        :param extrinsic_matr: Option to use specific camera extrinsics.
        :return: depth_images, segmask_images, extrinsics, point_clouds, tsdf, timing, rgb_images
        """
        height, width = self.camera.intrinsic.height, self.camera.intrinsic.width

        camera_sampling = CameraSampleSpace.load("{}_cameraspace.pkl".format(self.table_name))
        if n is None:
            n = camera_sampling.get_camera_samples()

        extrinsics = np.empty((n, 7), np.float32)
        depth_imgs = np.empty((n, height, width), np.float32)
        rgb_ims = np.empty((n, height, width, 3), np.uint8)
        segmask_imgs = np.empty((n, height, width), np.uint8)
        point_clouds = []

        tsdf = TSDFVolume(self.size, 40)

        if extrinsic_matr is None:
            max_attempts = 25
        else:
            max_attempts = 1

        i = 0
        k = 0
        while i < n:
            if extrinsic_matr is None:
                camera_origin, look_to_point = camera_sampling.sample_camera_points()

                extrinsic = Transform.look_at(np.r_[camera_origin[0], camera_origin[1], camera_origin[2]],
                                              np.r_[look_to_point[0], look_to_point[1], look_to_point[2]],
                                              np.array([0.0, 0.0, 1.0]))
            else:
                extrinsic = Transform.from_list(extrinsic_matr)

            rgb_img, depth_img, segmask_img = self.camera.render(extrinsic)

            if segmask_img.max() <= 1 and k < max_attempts:
                # We use k so that we don't loop in this one forever.
                k += 1
                continue

            # Combine background, ground & furniture to a single mask and set to 0
            segmask_img[segmask_img <= 1] = 1
            segmask_img -= 1

            extrinsics[i] = extrinsic.to_list()
            depth_imgs[i] = add_kinetic_noise(depth_img)
            segmask_imgs[i] = segmask_img
            rgb_ims[i] = rgb_img

            point_clouds.append(create_point_cloud(depth_imgs[i], segmask_imgs[i],
                                                   self.camera.intrinsic, extrinsic.inverse().as_matrix()))

            tsdf.integrate(depth_imgs[i], self.camera.intrinsic, extrinsic)

            i += 1

        return depth_imgs, segmask_imgs, extrinsics, point_clouds, tsdf, rgb_ims

    def render_images_tabletop(self, n=None, extrinsic_matr=None, perfect_tsdf=False, noise=True):
        """
        Render images for a 30cm x 30cm tabletop workspace scenario.
        :param n: Number of images to sample
        :param extrinsic_matr: Option to use specific camera extrinsics.
        :param noise: Boolean value, defaults to True. Set to False if you want to omit adding synthetic
                        depth camera noise to the depth image, TSDF and point cloud
        :return: depth_images, segmask_images, extrinsics, point_clouds, tsdf, timing, rgb_images
        """
        height, width = self.camera.intrinsic.height, self.camera.intrinsic.width
        origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, 0.0])

        extrinsics = np.empty((n, 7), np.float32)
        depth_imgs = np.empty((n, height, width), np.float32)
        rgb_ims = np.empty((n, height, width, 3), np.uint8)
        segmask_imgs = np.empty((n, height, width), np.uint8)
        point_clouds = []

        tsdf = TSDFVolume(self.size, 40)

        for i in range(n):
            if extrinsic_matr is None:
                r = 0.6
                phi = np.pi / 6.0
                if not perfect_tsdf:
                    theta = np.deg2rad(np.random.uniform(0, 180))
                else:
                    theta = 2.0 * np.pi * i / n

                extrinsic = camera_on_sphere(origin, r, theta, phi)
            else:
                extrinsic = Transform.from_list(extrinsic_matr)
            rgb_img, depth_img, segmask_img = self.camera.render(extrinsic)

            segmask_img[segmask_img == -1] = 0
            # segmask_img[segmask_img == 1] = 0

            extrinsics[i] = extrinsic.to_list()
            if noise:
                depth_imgs[i] = add_kinetic_noise(depth_img)
            else:
                depth_imgs[i] = depth_img
            segmask_imgs[i] = segmask_img
            rgb_ims[i] = rgb_img

            point_clouds.append(create_point_cloud(depth_imgs[i], segmask_imgs[i],
                                                   self.camera.intrinsic, extrinsic.inverse().as_matrix()))

            tsdf.integrate(depth_imgs[i], self.camera.intrinsic, extrinsic)
        return depth_imgs, segmask_imgs, extrinsics, point_clouds, tsdf, rgb_ims

    def grasp_collision_free(self, grasp):
        """
        Check if a grasp causes the gripper to be in collision with objects in the scene
        :param grasp: Grasp pose
        :return: True if collision-free, False if gripper is in collision
        """
        self.gripper.reset(grasp.pose)
        self.world.step()
        collision_free = True
        if self.gripper.detect_contact():
            collision_free = False
        self.gripper.deactivate_gripper()
        return collision_free

    def execute_grasp(self, grasp, remove=True, allow_contact=False, body_uid=None, furniture_uid=1):
        """
        Execute a grasp, report the success and cause of failure for the grasp.
        :param grasp: Grasp pose to test.
        :param remove: True if object should be removed upon successful grasping of the object.
        :param allow_contact: Allowing contact on the approach of the gripper
        :param body_uid: Physics simulation UID of the body that should be grasped
        :param furniture_uid: Physics simulation UID of furniture units, to check if trying to grasp the furniture
        :return: Success of object, width of the grasp, failure cause, physics simulation UID of the grasped object
        """
        T_world_grasp = grasp.pose

        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.07])
        T_world_retreat = T_grasp_pregrasp_world * T_world_grasp

        grasp_retreat = Grasp(T_world_retreat, width=self.gripper.max_opening_width, uid=body_uid)
        if not self.grasp_collision_free(Grasp(T_world_pregrasp, width=self.gripper.max_opening_width, uid=body_uid)):
            # Colliding with something in pregrasp pose
            result = 0, self.gripper.max_opening_width, 'approach collides', -1
            return result
        elif not self.grasp_collision_free(grasp_retreat):
            # We don't have space to move lift the object here..
            result = 0, self.gripper.max_opening_width, 'retreat collides', -1
            return result
        self.gripper.reset(T_world_pregrasp)

        if self.gripper.detect_contact():
            # Colliding with something in pregrasp pose
            result = 0, self.gripper.max_opening_width, 'approach collides', -1
        else:
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=False, vel=0.3)
            if self.gripper.detect_contact() and not allow_contact:
                result = 0, self.gripper.max_opening_width, 'approach collides', -1
            else:
                self.gripper.close_gripper(0.0)
                self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False, vel=0.3)
                if self.check_success(self.gripper):
                    grasped_body_uid = self.world.get_contacts(self.gripper.body)[0].bodyB.uid
                    result = 1, self.gripper.read_gripper_width(), 'success', grasped_body_uid
                    if body_uid is not None and grasped_body_uid != body_uid:
                        # Picking up wrong object
                        result = 0, self.gripper.max_opening_width, 'grasped wrong object', grasped_body_uid
                    if grasped_body_uid <= furniture_uid:
                        # In contact with furniture or ground
                        result = 0, self.gripper.max_opening_width, 'grasped furniture unit', grasped_body_uid
                    if remove:
                        contacts = self.world.get_contacts(self.gripper.body)
                        self.world.remove_body(contacts[0].bodyB)
                else:
                    result = 0, self.gripper.max_opening_width, 'failed to grasp', -1

        self.gripper.open_gripper()
        self.gripper.deactivate_gripper()

        if remove:
            self.remove_and_wait()

        return result

    def remove_and_wait(self, added_body_uid=None):
        """
        Wait for objects to come to a rest and check if the added body is still moving
        :param added_body_uid: Physics simulation UID of last object which was added to the scene
        :return: False if last body added is not at rest or, if no added_body_uid is given, any body is not at rest
        """
        # wait for objects to rest and remove those that haven't come to rest after 3 seconds
        all_objects_at_rest = self.wait_for_objects_to_rest()
        if not all_objects_at_rest:
            for body in list(self.world.bodies.values()):
                if np.linalg.norm(body.get_velocity()) > 0.1:
                    if (added_body_uid is not None and body.uid != added_body_uid) or added_body_uid is None:
                        return False
        return True

    def remove_and_wait_simple(self):
        """
        Wait for objects to rest while removing bodies that fell outside the workspace.
        """
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=3.0, tol=0.01):
        """
        Forward simulation time until all objects are resting or timeout has been reached.
        :param timeout: Timeout for waiting for objects to rest.
        :param tol: Tolerance for speed of objects.
        :return: True if objects are resting, false if objects are still in motion.
        """
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for body in list(self.world.bodies.values()):
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break
        return objects_resting

    def remove_objects_outside_workspace(self):
        """
        For simple tabletop case, remove objects outside of the workspace by  checking their absolute pose.
        :return: True if removed an objects, False if no objects were removed.
        """
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if body.name != 'simple_plane.urdf' and (np.any(xyz < 0.0) or np.any(xyz > self.size)):
                self.world.remove_body(body)
                removed_object = True

        return removed_object

    def check_success(self, gripper):
        """
        Check that the fingers are in contact with some object (not the plane) and not fully closed
        :param gripper: Physics simulation body of the gripper object
        :return: True if object in gripper and has been lifted above plane, False if grasp unsuccessful
        """
        contacts = self.world.get_contacts(gripper.body)
        if (len(contacts) > 0 and gripper.read_gripper_width() > 0.05 * gripper.max_opening_width and
                contacts[0].bodyB.name != 'plane'):
            contacts_of_object = self.world.get_contacts(contacts[0].bodyB)
            for obj_contact in contacts_of_object:
                if 'gripper' not in obj_contact.bodyB.name:
                    return False
            return True
        return False

    @staticmethod
    def project_deproject_contact(contact, intrinsic_matrix, depth_im):
        """
        Project and then deproject a 3D pose into image coordinates and then back into 3D coordinates.
        :param contact: 3D coordinates of contact in camera coordinate frame.
        :param intrinsic_matrix: Intrinsic matrix of the camera
        :param depth_im: Depth image rendered by the camera.
        :return: Image coordinates of contact (u,v) and deprojected contact point in 3D coordinates in camera frame.
        """
        contact_2d = intrinsic_matrix.dot(contact)
        contact_2d /= contact_2d[2]

        u = np.round(contact_2d[0]).astype(int)
        v = np.round(contact_2d[1]).astype(int)

        # Deproject contact points back to 3D
        try:
            deprojected = (depth_im[v, u] * np.linalg.inv(intrinsic_matrix).dot(np.array((u, v, 1))))
        except IndexError:
            u = -1
            v = -1
            deprojected = (None, None, None)

        return u, v, deprojected

    def ray_test(self, grasp_contact, camera_origin, uid):
        """
        Perform a ray test and check if physics simulation UID at a 3D contact matches the desired UID.
        :param grasp_contact: 3D contact coordinates in base frame.
        :param camera_origin: 3D camera origin coordinates in base frame.
        :param uid: physics simulation UID that we want to check if visible at the 3D point.
        :return: True if UID at grasp contact matches the desired UID, False otherwise.
        """
        ray = self.world.p.rayTest(camera_origin, grasp_contact)
        if ray[0][0] != uid:
            return False
        return True

    @staticmethod
    def get_quaternion_in_camera_frame(grasp, camera_transform):
        """
        Convert the grasp pose from base coordinates to camera coordinates and represent as a quaternion
        :param grasp: Grasp pose in base coordinates
        :param camera_transform: Camera transform in base coordinates
        :return: Quaternion of grasp pose in camera coordinates
        """
        grasp_in_camera = camera_transform * grasp.pose
        return grasp_in_camera.rotation.as_quat()

    def check_visibility(self, grasp, camera_transform, depth_im):
        """
        Check if a grasp is visible in a given depth image.
        :param grasp: Grasp pose in base coordinates
        :param camera_transform: Camera transform in base coordinates
        :param depth_im: Rendered depth image.
        :return: u_right_contact, v_right_contact, u_centre, v_centre,
                    distance TCP-camera, left contact 3D coordinates, right contact 3D coordinates
        """
        # Convert to camera coordinate frame
        T_left_contact_camera = camera_transform * Transform(Rotation.identity(), np.array(grasp.left_contact))
        T_right_contact_camera = camera_transform * Transform(Rotation.identity(), np.array(grasp.right_contact))
        T_centre_camera = camera_transform * Transform(Rotation.identity(), np.array(grasp.pose.translation))

        K = self.camera.intrinsic.K

        # Get image coordinates and de-projected contacts
        l_u, l_v, l_im = self.project_deproject_contact(T_left_contact_camera.translation, K, depth_im)
        r_u, r_v, r_im = self.project_deproject_contact(T_right_contact_camera.translation, K, depth_im)
        c_u, c_v, _ = self.project_deproject_contact(T_centre_camera.translation, K, depth_im)

        if l_u == -1 and r_u == -1 or c_u == -1:
            # No contacts of the grasps are visible (e.g. because the grasp is outside of the image boundaries)
            grasp.visible = False
            return -1, -1, -1, -1, -1, -1, -1

        l_3d = T_left_contact_camera.translation
        r_3d = T_right_contact_camera.translation

        # Calculate grasp distance between the two contact points
        dist = l_3d - r_3d
        grasp.width = np.sqrt(np.sum(dist**2))

        # Calculate Euclidean distance of original contact points to camera
        left_closer = True if np.sqrt(np.sum(l_3d**2)) < np.sqrt(np.sum(r_3d**2)) else False
        grasp.visible = False

        if left_closer and l_u != -1 and l_v != -1:
            # Left contact is up front - switch contacts
            l_diff = np.sqrt(np.sum((l_3d - l_im) ** 2))
            if l_diff < 0.01:
                # Grasp is judged visible if the depth at contact point is less than 1cm away from real contact point
                # and the hit test yields the correct object ID
                grasp.visible = self.ray_test(np.array(grasp.left_contact),
                                              camera_transform.inverse().translation,
                                              grasp.uid)
            # We switch the contacts because we want always describe the same gripper plate in contact with the visible
            # grasp contact to have a consistent orientation definition.
            grasp.switch_contacts()

            if l_u == -1:
                raise AttributeError("Somehow, this got messed up with the contact deprojection. Check l_u/l_v!")

            return l_u, l_v, c_u, c_v, T_centre_camera.translation[2], l_im, r_im
        elif not left_closer and r_u != -1 and r_v != -1:
            r_diff = np.sqrt(np.sum((r_3d - r_im) ** 2))
            if r_diff < 0.01:
                grasp.visible = self.ray_test(np.array(grasp.right_contact),
                                              camera_transform.inverse().translation,
                                              grasp.uid)
            if r_u == -1:
                raise AttributeError("Somehow, this got messed up with the contact deprojection. Check r_u/r_v!")
            return r_u, r_v, c_u, c_v, T_centre_camera.translation[2], l_im, r_im
        # The grasp is not visible
        grasp.visible = False
        return -1, -1, -1, -1, -1, -1, -1


class Gripper(object):
    """Simulated a gripper for the clutter simulation."""
    def __init__(self, world, panda=False):
        """
        Initiate a gripper object for our pybullet simulation
        :param world: BtSim world
        :param panda: Boolean value, defaults to False and using a PAL parallel jaw gripper.
                        Set to True if using a Franka Emika Panda gripper.
        """
        self.world = world
        self.panda = panda
        if self.panda:
            self.urdf_path = Path("data/urdfs/panda/hand.urdf")
            self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.022])
        else:
            self.urdf_path = Path("data/urdfs/pal_gripper/gripper_exact.urdf")
            self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.2])

        self.max_opening_width = 0.08
        self.finger_depth = 0.05

        self.T_tcp_body = self.T_body_tcp.inverse()
        # To prevent loading the gripper every time we test a grasp, we set it into a storage pose below the ground
        # and deactivate it when setting up the scene or rendering images.
        self.storage_pose = Transform(Rotation.identity(), [0.0, 0.0, -1.0])
        self.load_gripper()
        self.deactivate_gripper()

    def load_gripper(self):
        """
        Load a gripper URDF and set all of the links and constraints in order to use it in pybullet.
        """
        self.body = self.world.load_urdf(self.urdf_path, self.storage_pose)
        # constraint to keep fingers centered
        if self.panda:
            self.world.add_constraint(
                self.body,
                self.body.links["panda_leftfinger"],
                self.body,
                self.body.links["panda_rightfinger"],
                pybullet.JOINT_GEAR,
                [1.0, 0.0, 0.0],
                Transform.identity(),
                Transform.identity(),
            ).change(gearRatio=-1, erp=0.1, maxForce=500)
            self.joint1 = self.body.joints["panda_finger_joint1"]
            self.joint2 = self.body.joints["panda_finger_joint2"]
        else:
            self.world.add_constraint(
                self.body,
                self.body.links["gripper_left_finger_link"],
                self.body,
                self.body.links["gripper_right_finger_link"],
                pybullet.JOINT_GEAR,
                [1.0, 0.0, 0.0],
                Transform.identity(),
                Transform.identity(),
            ).change(gearRatio=-1, erp=0.1, maxForce=500)
            self.joint1 = self.body.joints["gripper_left_finger_joint"]
            self.joint2 = self.body.joints["gripper_right_finger_joint"]
        self.joint1.set_position(0.5 * self.max_opening_width, kinematics=True)
        self.joint2.set_position(0.5 * self.max_opening_width, kinematics=True)
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            self.storage_pose,
        )
        self.update_tcp_constraint(self.storage_pose)

    def deactivate_gripper(self):
        """
        Move the gripper to the storage pose
        """
        self.body.set_pose(self.storage_pose)

    def reset(self, T_world_tcp):
        """
        Move the gripper to a pose and update its constraints
        :param T_world_tcp: Pose to set the TCP of the gripper to
        """
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_world_body)  # sets the position of the COM, not URDF link
        self.update_tcp_constraint(T_world_tcp)

    def update_tcp_constraint(self, T_world_tcp):
        """
        Update the position and orientation of the TCP for a new pose.
        :param T_world_tcp: Pose to set the TCP of the gripper to.
        """
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=500,
        )

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        """
        Move the gripper TCP linearly to a new target pose.
        :param target: target pose of the gripper TCP.
        :param eef_step: Step size.
        :param vel: Velocity of gripper.
        :param abort_on_contact: Boolean value, true if approach should be aborted when contact is detected.
        """
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        if n_steps == 0:
            return
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self):
        """
        Check if the gripper body is in contact with any other object.
        :return: True if gripper is in contact with any other object
        """
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def close_gripper(self, width):
        """
        Close the gripper by moving both gripper plates towards each other.
        :param width: Distance between the gripper plates at the end of the motion.
        """
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def open_gripper(self):
        """
        Open gripper fully by moving both gripper plates away from each other.
        """
        self.joint1.set_position(0.5 * self.max_opening_width)
        self.joint2.set_position(0.5 * self.max_opening_width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def read_gripper_width(self):
        """
        Get the current width of the gripper.
        :return: Distance between the gripper plates.
        """
        width = self.joint1.get_position() + self.joint2.get_position()
        return width

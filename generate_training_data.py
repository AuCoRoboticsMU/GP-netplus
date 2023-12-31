import argparse
from pathlib import Path
import os
from tqdm import tqdm
import copy

from src.experiments.perception import *
from src.experiments.simulation import ClutterRemovalSim
from src.experiments.vis_utils import Grasp, Label
from src.utils import quaternion_rotation_matrix
from src.experiments.transform import Transform, Rotation

def main(args):
    """
    Generate dataset with single worker.
    """
    all_image_count = [0]
    simulate_scenes(args, args.num_images, -1, all_image_count)
    print("Finished")


def simulate_scenes(args, num_images, worker_id, all_image_count):
    """
    Simulate domestic grasping scenes and save them in a training/validation dataset
    :param args:
    :param num_images: Number of images to generate.
    :param worker_id: Worker ID, used when using multiprocessing for generating the scenes.
    :param all_image_count: List of images generated by workers. Has length of 1 when not using multiprocessing.
    """
    if worker_id == -1:
        save_dir = args.root / args.split
    else:
        save_dir = args.root / args.split / Path('worker_{}'.format(worker_id))
    if not os.path.exists(args.root):
        os.mkdir(args.root)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    progress_bar = tqdm(total=num_images, position=worker_id, desc=f"worker #{worker_id}", leave=False)
    sim = ClutterRemovalSim(args.object_set, args.split, gui=args.sim_gui)
    data_root = sim.urdf_root / sim.object_set / sim.split
    scene_cnt = 0

    while sum(all_image_count) < args.num_images:
        all_grasps = []
        all_labels = []

        # Start new simulated scene (place furniture piece, objects)
        sim.reset_domestic_scene(None)
        sim.save_state()

        # render synthetic depth images
        depth_imgs, segmask_imgs, extrinsics, _, _, _ = sim.render_images_domestic_scene(10)
        bodies = sim.world.bodies

        for body in bodies:
            if bodies[body].name.split('_')[0] == 'furniture' or \
                    bodies[body].name == 'ground' or\
                    'gripper' in bodies[body].name:
                continue
            object_name = '_'.join(bodies[body].name.split('.')[0].split('_')[:-1])
            grasps = np.load('{}/{}_grasps.npy'.format(data_root, object_name))

            indices = np.random.choice(np.arange(len(grasps)), min(len(grasps), 250))

            grasps = grasps[indices]
            metrics = np.load('{}/{}_metrics.npy'.format(data_root, object_name))[indices]
            contacts = np.load('{}/{}_contacts.npy'.format(data_root, object_name))[indices]

            trans, q_xyzw = sim.world.p.getBasePositionAndOrientation(body)
            ob_in_world = np.eye(4)
            ob_in_world[:3, 3] = trans
            q_wxyz = [q_xyzw[-1], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
            R = quaternion_rotation_matrix(q_wxyz)
            ob_in_world[:3, :3] = R.copy()

            for i, grasp in enumerate(grasps):
                # Scale up the grasp and contact coordinates with the scale of the body (local coordinates)
                grasp[:3, 3] *= sim.world.bodies[body].scale
                contact = np.ones((2, 4))
                contact[:2, :3] = (contacts[i] * sim.world.bodies[body].scale)
                # Convert to world base coordinate frame
                contact_in_world = ob_in_world @ contact.T
                grasp_in_world = ob_in_world @ grasp
                grasp_dist = np.sqrt(sum((contact_in_world[:3, 0] - contact_in_world[:3, 1]) ** 2))

                if grasp_dist > 0.08:
                    # Don't execute grasp if too wide
                    pos = grasp_in_world[:3, 3]
                    R = Rotation.from_matrix(grasp_in_world[:3, :3])
                    # Convert dex-net gripper representation to pybullet gripper representation
                    ori = R * Rotation.from_euler("z", np.pi / 2)

                    grasp = Grasp(Transform(ori, pos), width=grasp_dist, metric=metrics[i], uid=body)
                    label = 0
                else:
                    grasp, label, _ = evaluate_grasp(sim, grasp_in_world, body, metric=metrics[i],
                                                     num_rotations=12, grasp_width=grasp_dist)
                if isinstance(grasp, list):
                    # List of grasps, multiple collision-free areas with grasp success. Store each grasp in that case.
                    for each_grasp in grasp:
                        each_grasp.left_contact = contact_in_world[:3, 1]
                        each_grasp.right_contact = contact_in_world[:3, 0]
                        all_grasps.append(each_grasp)
                        all_labels.append(label)
                else:
                    grasp.left_contact = contact_in_world[:3, 1]
                    grasp.right_contact = contact_in_world[:3, 0]
                    all_grasps.append(grasp)
                    all_labels.append(label)

        # Scene is set up and grasps are tested, now project them into the image and check visibility
        for i, extrinsic in enumerate(extrinsics):
            # All images
            tmp = []
            all_contacts = []
            camera_transform = Transform.from_list(extrinsic)
            im_grasp_stats = []
            im_stats = np.array((scene_cnt, sim.num_objects))
            for grasp in all_grasps:
                # All grasps
                current_grasp = copy.deepcopy(grasp)
                store_contact = False
                if current_grasp.left_contact is None:
                    print("No contact??")
                else:
                    l_u, l_v, centre_u, centre_v, centre_depth, l_im, r_im = sim.check_visibility(current_grasp,
                                                                                                  camera_transform,
                                                                                                  depth_imgs[i])
                    rotation = sim.get_quaternion_in_camera_frame(current_grasp, camera_transform)
                    if current_grasp.visible:
                        if (l_u, l_v) in all_contacts:
                            # Another grasp is visible at the same position
                            ind = all_contacts.index((l_u, l_v))
                            new_in_z = Rotation.from_quat(rotation).as_matrix()[2, 2]
                            old_in_z = Rotation.from_quat([tmp[ind][1],
                                                           tmp[ind][2],
                                                           tmp[ind][3],
                                                           tmp[ind][4]]).as_matrix()[2, 2]
                            if current_grasp.success > tmp[ind][-1]:
                                # Replace if this one is better than the one originally saved
                                store_contact = True
                                tmp.pop(ind)
                                all_contacts.pop(ind)
                                im_grasp_stats.pop(ind)
                            elif current_grasp.success and new_in_z > old_in_z:
                                # Replace if this one is "more aligned" with the viewpoint, e.g. grasping the object
                                # from the front rather than from the back
                                store_contact = True
                                tmp.pop(ind)
                                all_contacts.pop(ind)
                                im_grasp_stats.pop(ind)
                        else:
                            store_contact = True
                    if store_contact:
                        im_grasp_stats.append([centre_depth, current_grasp.pose.translation[2]])
                        all_contacts.append((l_u, l_v))
                        grasp_array = np.r_[centre_u,
                                            centre_v,
                                            centre_depth,
                                            l_u,
                                            l_v,
                                            current_grasp.width,
                                            rotation[3],  # w
                                            rotation[0],  # x
                                            rotation[1],  # y
                                            rotation[2],  # z
                                            current_grasp.metric,
                                            current_grasp.success]
                        tmp.append(grasp_array)
            # Save images, grasps and statistics.
            np.savez_compressed('{}/depth_image_{:07d}.npz'.format(save_dir,
                                                                   all_image_count[worker_id]), depth_imgs[i])
            np.savez_compressed('{}/segmask_image_{:07d}.npz'.format(save_dir,
                                                                     all_image_count[worker_id]), segmask_imgs[i])
            np.savez_compressed('{}/grasps_{:07d}.npz'.format(save_dir, all_image_count[worker_id]), np.array(tmp))
            np.savez_compressed('{}/im_grasp_stats_{:07d}.npz'.format(save_dir, all_image_count[worker_id]),
                                np.array(im_grasp_stats))
            np.savez_compressed('{}/im_stats_{:07d}.npz'.format(save_dir, all_image_count[worker_id]), im_stats)
            all_image_count[worker_id] += 1
        progress_bar.n = all_image_count[worker_id]
        progress_bar.refresh()
        scene_cnt += 1

def evaluate_grasp(sim, grasp_pose, body_uid, metric, num_rotations=6, grasp_width=None):
    """
    Evaluate a grasp pose by hinge-rotating it around the contacts and testing the grasps in simulation.
    :param sim: Pybullet simulation environment.
    :param grasp_pose: Grasp pose to test.
    :param body_uid: Physics simulation UID of the body.
    :param metric: Robust force closure metric of the sampled grasp pose.
    :param num_rotations: Number of hinge-rotations to check for grasp success.
    :param grasp_width: Width of the grasp.
    :return: grasp pose(s), success, number of grasps tested
    """
    # define initial grasp frame on object surface
    pos = grasp_pose[:3, 3]
    R = Rotation.from_matrix(grasp_pose[:3, :3])
    # Convert dex-net gripper representation to pybullet gripper representation
    R = R * Rotation.from_euler("z", np.pi / 2)
    if grasp_width is None:
        grasp_width = sim.gripper.max_opening_width

    # try to grasp with different hinge-rotations
    yaws = np.linspace(0.0, 2 * np.pi, num_rotations)
    outcomes, candidates = [], []
    num_tested = 0
    for yaw in yaws:
        ori = R * Rotation.from_euler("x", yaw)
        sim.restore_state()
        candidate = Grasp(Transform(ori, pos), width=grasp_width, metric=metric, uid=body_uid)
        if sim.grasp_collision_free(candidate):
            outcome, width, _, _ = sim.execute_grasp(candidate, remove=False, body_uid=body_uid)
            num_tested += 1
        else:
            width = sim.gripper.max_opening_width
            outcome = 0
        outcomes.append(outcome)
        candidates.append(Grasp(Transform(ori, pos), width=width, success=outcome, metric=metric, uid=body_uid))

    # detect mid-point of widest peak of successful yaw angles
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    if np.sum(successes):
        if successes[0] and successes[-1] and num_rotations != 1 and np.sum(successes) != len(successes):
            # The first and last orientation were a success, so we'll align them to one block
            while successes[-1] == 1:
                successes = np.roll(successes, 1)
                candidates = np.roll(candidates, 1)
        successful = np.where(successes == 1)[0]
        regions = np.split(successful, np.where(np.diff(successful) != 1)[0] + 1)
        if len(regions) == 1:
            # We have one region where grasps are successful - take the median of those
            grasp = candidates[regions[0][len(regions[0]) // 2]]
        else:
            grasp = []
            for cnt, region in enumerate(regions):
                grasp.append(candidates[regions[cnt][len(region) // 2]])
    else:
        grasp = candidates[0]
    sim.restore_state()
    return grasp, int(np.max(outcomes)), num_tested


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--object-set", type=str, default="clutter_objects")
    parser.add_argument("--split", type=Path, choices=[Path('train'), Path('val'), Path('test')],
                        default=Path('train'))
    parser.add_argument("--num-images", type=int, default=10000)
    parser.add_argument("--sim-gui", action="store_true")
    args = parser.parse_args()
    main(args)

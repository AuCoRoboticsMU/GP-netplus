import collections
import argparse

import numpy as np
import tqdm
import sys
import os

from src.experiments.simulation import ClutterRemovalSim
from src.experiments.detection import GPnetplus
from src.experiments.clutter_removal import Logger
from pathlib import Path

State = collections.namedtuple("State", ["depth_im", "segmask_im", "camera_intrinsics",
                                         "camera_extrinsics", "points"])


def run(grasp_plan_fn, logdir, logname, description, object_set, sim_gui=False):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    sim = ClutterRemovalSim(object_set, gui=sim_gui, split='test')
    logger = Logger(logdir, logname, network_name=description)
    all_files = os.listdir(logger.scenes_dir)
    all_scenes = [x.split('.')[0] for x in all_files if '.bullet' in x]
    all_scenes.sort()

    if 'simple' in logname:
        uid_comp = 0
        simple_setup = True
        max_consecutive_failures = 2
    else:
        uid_comp = 1
        simple_setup = False
        max_consecutive_failures = 5

    for scene_name in tqdm.tqdm(all_scenes):
        sim.load_world('{}/{}'.format(logger.scenes_dir, scene_name))
        sim.save_state()

        round_id = scene_name.split('_')[1].split('.')[0]
        consecutive_failures = 0
        last_label = None
        im_counter = 0
        scene_extrinsics = np.load('{}/{}_extrinsics.npz'.format(logger.scenes_dir, scene_name))['arr_0']

        # extrinsic = None
        while sim.num_objects > 0 and consecutive_failures < max_consecutive_failures:
            # scan the scene
            try:
                if simple_setup:
                    (depth_ims, segmask_ims, extrinsics, pc,
                     tsdf_pc, rgb_ims) = sim.render_images_tabletop(n=1,
                                                                    extrinsic_matr=scene_extrinsics[im_counter])
                else:
                    (depth_ims, segmask_ims, extrinsics, pc,
                     tsdf_pc, rgb_ims) = sim.render_images_domestic_scene(n=1,
                                                                          extrinsic_matr=scene_extrinsics[im_counter])

            except IndexError:
                print("Ran out of scene extrinscis")
                consecutive_failures += 1
                if simple_setup:
                    (depth_ims, segmask_ims, extrinsics, pc,
                     tsdf_pc, rgb_ims) = sim.render_images_tabletop(n=1)
                else:
                    depth_ims, segmask_ims, extrinsics, pc, tsdf_pc, rgb_ims = sim.render_images_domestic_scene(n=1)

            depth_im, segmask_im, extrinsic, rgb_im = depth_ims[0], segmask_ims[0], extrinsics[0], rgb_ims[0]

            if not simple_setup:
                try:
                    if segmask_im[np.where((depth_im <= grasp_plan_fn.depth_max) &
                                           (depth_im > grasp_plan_fn.depth_min))].max() == 0:
                        im_counter += 1
                        continue
                except ValueError:
                    im_counter += 1
                    continue

            num_pixels = np.unique(segmask_im[np.where((depth_im <= grasp_plan_fn.depth_max) &
                                                       (depth_im > grasp_plan_fn.depth_min))].flatten(),
                                   return_counts=True)

            visible_objects = ''
            for uid, total_pixels in zip(num_pixels[0], num_pixels[1]):
                visible_objects += '{}_{};'.format(total_pixels, uid)

            im_counter += 1

            # plan grasps
            state = State(depth_im, segmask_im, sim.camera, extrinsic, tsdf_pc)
            grasps, scores, coords, object_ids, _ = grasp_plan_fn(state)

            if len(grasps) == 0:
                if last_label == 0:
                    consecutive_failures += 1
                else:
                    consecutive_failures = 1
                last_label = 0
                continue  # no detections found, count as consecutive failure
            # execute grasp
            proposed_objects = []
            num_proposals = np.unique(object_ids, return_counts=True)
            for uid in object_ids:
                proposed_objects.append(sim.world.bodies[uid + uid_comp].name)

            pixel_proposals = ''
            for uid, total_pixels in zip(num_pixels[0], num_pixels[1]):
                if uid == 0:
                    continue
                if uid in num_proposals[0]:
                    pixel_proposals += '{}_{};'.format(total_pixels,
                                                       num_proposals[1][np.where(num_proposals[0] == uid)[0]][0])
                else:

                    pixel_proposals += '{}_{};'.format(total_pixels, 0)

            proposed_objects_str = ';'.join(proposed_objects)
            idx = np.argmax(scores)
            grasp, score, planned_uid = grasps[idx], scores[idx], object_ids[idx] + uid_comp
            try:
                object_name = sim.world.bodies[planned_uid].name
            except KeyError:
                print("Proposed objects: ", proposed_objects_str)
                print("Object ids: ", object_ids)
                print("Index: {}; Planned body uid: {}".format(idx, planned_uid))
                break

            if not simple_setup and planned_uid <= uid_comp:
                # Planned grasp contact either touches the furniture or the ground (body uid's 1 and 0, respectively)
                label = 0
                fail_cause = "grasped furniture unit"
            else:
                if simple_setup:
                    label, _, fail_cause, body_uid = sim.execute_grasp(grasp, allow_contact=True, remove=False,
                                                                       furniture_uid=uid_comp)
                else:
                    label, _, fail_cause, body_uid = sim.execute_grasp(grasp, allow_contact=True, remove=False,
                                                                       body_uid=planned_uid, furniture_uid=uid_comp)
                    sim.restore_state()

                if label:
                    object_name = sim.world.bodies[body_uid].name
                    sim.world.remove_body(sim.world.bodies[body_uid])
                    sim.save_state()
                else:
                    sim.restore_state()

            camera_dist = depth_im[coords[0][0], coords[0][1]]

            logger.log_grasp(round_id, grasp, camera_dist, score, label, visible_objects, object_name,
                             fail_cause, proposed_objects_str, pixel_proposals)

            if last_label == 0 and label == 0:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            last_label = label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--logdir", type=Path, default="data/experiments")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--logname", type=str, default="test")
    parser.add_argument("--detection-threshold", type=float, default=0.29)
    parser.add_argument("--object-set", type=str, default="clutter_objects")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--sim-gui", action="store_true")
    args = parser.parse_args()

    # Load model
    if args.model is None:
        test_model = 'gpnetplus'
    else:
        test_model = args.model
    model_name = args.model_name
    if model_name is None:
        all_files = os.listdir('data/runs/{}'.format(test_model))
        highest_nbr = -1
        for each_file in all_files:
            if '.pt' in each_file and 'gpnet_plus' in each_file and 'ros' not in each_file:
                nbr = int(each_file.split('_')[-1].split('.')[0])
                if nbr > highest_nbr:
                    highest_nbr = nbr
        model_name = 'gpnet_plus_{}.pt'.format(highest_nbr)
    else:
        highest_nbr = model_name.split('.')[0].split('_')[-1]
    path_dir = 'data/runs/{}/{}'.format(test_model, model_name)

    grasp_planner = GPnetplus(Path(path_dir), detection_threshold=args.detection_threshold)

    if args.description == '':
        description = '{}_eps_{}_threshold_{}'.format(test_model, highest_nbr, args.detection_threshold)
    else:
        description = args.description

    run(grasp_plan_fn=grasp_planner,
        logdir=args.logdir,
        logname=args.logname,
        description=description,
        object_set=args.object_set,
        sim_gui=args.sim_gui,
        )

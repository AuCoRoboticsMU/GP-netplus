import argparse

import numpy as np
import tqdm
import sys
import gc
import pathlib
rel_path = pathlib.Path(__file__).parent.resolve()
sys.path.append('{}/../..'.format(rel_path))

from src.experiments.simulation import ClutterRemovalSim
from src.experiments.clutter_removal import Logger
from pathlib import Path


def generate_scenes(logdir, object_set, logname, num_scenes=1, sim_gui=False, domestic_scene=True):
    """
    Generate scenes to be used for evaluation with objects.

    :param logdir: Path of where to save the generated scenes.
    :param object_set: Object set to use in the generated scenes.
    :param logname: Name to use for the experiments.
    :param num_scenes: Number of scenes to generate.
    :param sim_gui: Boolean variable defining if a GUI is displayed. Defaults to False.
    :param domestic_scene: Boolean variable defining if domestic or tabletop scenes are created. Defaults to true,
                                where domestic scenes are generated
    """
    sim = ClutterRemovalSim(object_set, gui=sim_gui, split='test')
    logger = Logger(logdir, logname)

    if logger.last_round_id() != -1:
        num_scenes = num_scenes - logger.last_round_id()
    furniture_type = 'Table'

    for _ in tqdm.tqdm(range(num_scenes)):
        if domestic_scene:
            sim.reset_domestic_scene(None)
            _, _, extrinsics, _, _, _ = sim.render_images_domestic_scene(n=50)
            if 'Table' in sim.table_name:
                furniture_type = 'Table'
            elif 'Shelf' in sim.table_name:
                furniture_type = 'Shelf'
            else:
                raise UserWarning("Can't recognised furniture type.")
        else:
            sim.reset_simplistic(5)
            _, _, extrinsics, _, _, _ = sim.render_images_tabletop(n=15)
        sim.save_state()

        round_id = logger.last_round_id() + 1

        sim.save_world('{}/round_{}.bullet'.format(str(logger.scenes_dir), round_id))
        np.savez_compressed('{}/round_{}_extrinsics.npz'.format(str(logger.scenes_dir), round_id), extrinsics)

        logger.log_round(round_id, sim.num_objects, furniture_type)
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, default="data/experiments")
    parser.add_argument("--logname", type=str, default="test")
    parser.add_argument("--object-set", type=str, default="clutter_objects")
    parser.add_argument("--domestic-scene", action='store_true')
    parser.add_argument("--no-domestic-scene", dest='domestic-scene', action='store_false')
    parser.add_argument("--num-scenes", type=int, default=250)
    parser.add_argument("--sim-gui", action="store_true")
    parser.set_defaults(domestic_scene=True)
    args = parser.parse_args()

    generate_scenes(logdir=args.logdir, object_set=args.object_set, logname=args.logname, num_scenes=args.num_scenes,
                    sim_gui=args.sim_gui, domestic_scene=args.domestic_scene)

import uuid
import pandas as pd
import os

from src.experiments.vis_utils import append_csv, create_csv


class Logger(object):
    def __init__(self, root, description, network_name=None):

        self.logdir = root / description
        self.scenes_dir = self.logdir / "scenes"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path = self.logdir / "rounds.csv"
        if network_name is None:
            self.grasps_csv_path = None
        else:
            self.grasps_csv_path = self.logdir / "grasps_{}.csv".format(network_name)
        self._create_csv_files_if_needed()

    def _create_csv_files_if_needed(self):
        if not self.rounds_csv_path.exists():
            create_csv(self.rounds_csv_path, ["round_id", "object_count", "furniture_type"])

        if self.grasps_csv_path is not None and not self.grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "camera_distance",
                "width",
                "score",
                "label",
                "grasped_object_name",
                "cause_of_failure",
                "objects_with_grasp_proposals",
                "pixels_grasp_proposals",
                "visible_objects"
            ]
            create_csv(self.grasps_csv_path, columns)

    def last_round_id(self):
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count, furniture_type):
        append_csv(self.rounds_csv_path, round_id, object_count, furniture_type)

    def log_grasp(self, round_id, grasp, camera_dist, score, label, visible_objects,
                  grasped_object_name, cause_of_failure, proposed_objects,
                  pixel_proposals):
        scene_id = uuid.uuid4().hex

        # log grasp
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        label = int(label)
        append_csv(
            self.grasps_csv_path,
            round_id,
            scene_id,
            qx,
            qy,
            qz,
            qw,
            x,
            y,
            z,
            camera_dist,
            width,
            score,
            label,
            grasped_object_name,
            cause_of_failure,
            proposed_objects,
            pixel_proposals,
            visible_objects
        )


class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, logdir, network_name):
        self.logdir = logdir
        self.network_name = network_name
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps_{}.csv".format(network_name))

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def num_per_object(self):
        df = (
            self.grasps[["round_id"]]
            .groupby("round_id")
            .size()
        )
        values = list(df.values)
        for i in range(len(values), 100):
            values.append(0)
        return values

    def success_rate(self):
        return self.grasps["label"].sum() / self.grasps["label"].count() * 100

    def failure_causes(self):
        df = (
            self.grasps[["round_id", "cause_of_failure"]]
            .merge(self.rounds, on="round_id")
        )
        return df.sort_values(by=['cause_of_failure', 'furniture_type'], ascending=True)

    def success_rate_per_furniture_type(self):
        df = (
            self.grasps[["round_id", "label"]]
            .merge(self.rounds, on="round_id")
            .groupby('furniture_type')
        )
        return df["label"].sum() / df["label"].size() * 100

    def attempted_grasps_per_furniture_type(self):
        df = (
            self.grasps[["round_id", "label"]]
            .merge(self.rounds, on="round_id")
            .groupby('furniture_type')
        )
        return df["label"].size()

    def adjust_names_for_plotting(self, df):
        df['grasped_object_name'] = df['grasped_object_name'] \
            .apply(lambda x: x.split('.')[0])
        df['grasped_object_name'] = df['grasped_object_name'] \
            .apply(lambda x: x[:19] if 'Bowl' in x or 'Mug' in x else x)
        df['grasped_object_name'] = df['grasped_object_name'] \
            .apply(lambda x: x.lower())
        df['grasped_object_name'] = df['grasped_object_name'] \
            .apply(lambda x: x.replace("gluten_free_roasted_", ""))
        df['grasped_object_name'] = df['grasped_object_name'] \
            .apply(lambda x: x.replace("_almond_crunch", ""))
        df['grasped_object_name'] = df['grasped_object_name'] \
            .apply(lambda x: x.replace("nature_valley_", ""))
        df['grasped_object_name'] = df['grasped_object_name'] \
            .apply(lambda x: x.replace("colored_wood", "wood"))
        return df

    def combine_furniture(self, df):
        df['grasped_object_name'] = df['grasped_object_name'] \
            .apply(lambda x: '_'.join(x.split('_')[:-1]))
        df['grasped_object_name'] = df['grasped_object_name'] \
            .apply(lambda x: '_'.join(x.split('_')[1:]) if 'furniture' in x else x)
        return df

    def mean_successful_grasps(self):
        return self.grasps["label"].sum() / 100

    def mean_all_grasps(self):
        return self.grasps["label"].size / 100

    def get_objects(self, object_string):
        objects = [int(x.split('_')[1]) for x in object_string.split(';') if len(x) > 2]
        objects_without_background = [x + 1 for x in objects if x != 0]
        return objects_without_background

    @staticmethod
    def union_of_lists(series_of_lists):
        return list(set().union(*series_of_lists))

    def discover_object_occurrences(self):
        body_counts = {}
        all_bodies_texts = [x for x in os.listdir('{}/scenes/'.format(self.logdir)) if 'bodies' in x and not 'graspable' in x]
        for body_text in all_bodies_texts:
            with open('{}/scenes/{}'.format(self.logdir, body_text), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if not 'ground' in line and not 'pal_gripper' in line:
                        obj = line.rstrip()
                        if obj in body_counts.keys():
                            body_counts[obj][0] += 1
                        else:
                            body_counts[obj] = [1]
        return body_counts

    def success_rate_per_object(self):
        df = self.grasps[["label", "grasped_object_name", "visible_objects"]]
        df['visible_objects'] = df.apply(lambda x: self.get_objects(x.visible_objects), axis=1)
        object_occurrence = self.discover_object_occurrences()
        df_counts = pd.DataFrame().from_dict(object_occurrence).melt(value_name="object_count",
                                                                     var_name="grasped_object_name")
        df = self.combine_furniture(df)
        df_counts = self.combine_furniture(df_counts)
        df_counts = df_counts.groupby(df_counts['grasped_object_name']).aggregate({'object_count': 'sum'}).reset_index()
        df_merged = (df
                     .merge(df_counts, on="grasped_object_name")
                     )

        df_merged = self.adjust_names_for_plotting(df_merged)
        df_counts = self.adjust_names_for_plotting(df_counts)

        df_grouped = df_merged.groupby("grasped_object_name")

        s = df_grouped['label'].sum() / df_grouped['label'].count() * 100.0
        new_df = s.to_frame()
        new_df['# grasp attempts'] = df_grouped['label'].count()
        new_df['# unsuccessful grasps'] = df_grouped['label'].count() - df_grouped['label'].sum()
        new_df['# successful grasps'] = df_grouped['label'].sum()
        new_df['# object occurred'] = df_grouped['object_count'].first()

        new_df.loc['shapenet_shelf', '# object occurred'] = df_counts[
            df_counts.grasped_object_name == 'furniture_ShapeNet_Shelf'].sum().object_count
        new_df.loc['shapenet_table', '# object occurred'] = df_counts[
            df_counts.grasped_object_name == 'furniture_ShapeNet_Table'].sum().object_count
        new_df.loc['shapenet_shelf', '# successful grasps'] = 0
        new_df.loc['shapenet_table', '# successful grasps'] = 0
        if new_df.loc['shapenet_table', '# unsuccessful grasps'] == '':
            new_df.loc['shapenet_table', '# unsuccessful grasps'] = 0
        if new_df.loc['shapenet_shelf', '# unsuccessful grasps'] == '':
            new_df.loc['shapenet_shelf', '# unsuccessful grasps'] = 0

        for obj_name in df_counts.grasped_object_name:
            if obj_name not in new_df.index.values and not 'furniture' in obj_name and obj_name != 'simple':
                object_count = df_counts.loc[df_counts['grasped_object_name'] == obj_name]['object_count'].values[0]
                new_df.loc[obj_name] = [0, 0, 0, 0, object_count]

        new_df = new_df.rename(columns={'label': 'grasp_success_rate'})
        new_df = new_df.reset_index()
        new_df["# successful grasps"] = new_df["# successful grasps"].apply(pd.to_numeric)
        new_df["# unsuccessful grasps"] = new_df["# unsuccessful grasps"].apply(pd.to_numeric)

        melted_df = pd.melt(new_df.reset_index(), id_vars='grasped_object_name')
        melted_df.fillna(0)

        melted_df = melted_df[melted_df.variable != 'index']
        melted_df = melted_df[melted_df.variable != '# grasp attempts']
        melted_df = melted_df[melted_df.variable != 'grasp_success_rate']
        melted_df.to_csv('{}/success_per_object_{}.csv'.format(self.logdir, self.network_name), index=False)

        return melted_df

    def percent_cleared(self, obj_count='object_count'):
        try:
            df = self.grasps[["round_id", "label", "visible_objects"]]
            df['visible_objects'] = df.apply(lambda x: self.get_objects(x.visible_objects), axis=1)
        except KeyError:
            df = self.grasps[["round_id", "label"]]
            # df["visible_objects"] = [['1', '2', '3', '4' , '5'] for i in df.index]
            df["visible_objects"] = [['1'] for i in df.index]

        aggregations = {
            'visible_objects': self.union_of_lists,
            'label': 'sum'
        }

        cleared_count = (
            df
            .groupby("round_id")
            .agg(aggregations)
            .rename(columns={"label": "cleared_count"})
        )

        # df_graspable_objects = pd.DataFrame(graspable_object_dict)
        #
        # all_rounds = pd.merge(self.rounds, df_graspable_objects, on='round_id', how='outer')

        # Merge with self.rounds to include all round_id values
        df = self.rounds.merge(cleared_count, on="round_id", how="left")

        # Fill NaN values in cleared_count column with 0 to counter for rounds that have not tested a single grasp.
        df["cleared_count"] = df["cleared_count"].fillna(0)
        df['visible_objects'] = df['visible_objects'].apply(lambda d: d if isinstance(d, list) else [])

        df['num_visible'] = df.apply(lambda row: len(list(set(row['visible_objects']))), axis=1)

        grouped = df.groupby('furniture_type').agg({'cleared_count': 'sum',
                                                    'num_visible': 'sum',
                                                    'object_count': 'sum'})

        return df["cleared_count"].sum() / df['num_visible'].sum() * 100, \
               grouped['cleared_count'] / grouped['num_visible'] * 100,

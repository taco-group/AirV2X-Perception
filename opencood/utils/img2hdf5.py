import os
from multiprocessing import Process

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

mp = 24


def load_camera_data(camera_files, preload=False):
    """
    Args:
        camera_files: list,
            store camera path
        shape : tuple
            (width, height), resize the image, and overcoming the lazy loading.
    Returns:
        camera_data_list: list,
            list of Image, RGB order
    """
    camera_data_list = []
    for camera_file in camera_files:
        camera_data = Image.open(camera_file)
        if preload:
            camera_data = camera_data.copy()
        camera_data_list.append(camera_data)
    return camera_data_list


def load_camera_files(cav_path, timestamp, name):
    """
    Retrieve the paths to all camera files.

    Parameters
    ----------
    cav_path : str
        The full file path of current cav.

    timestamp : str
        Current timestamp

    Returns
    -------
    camera_files : list
        The list containing all camera png file paths.
    """
    camera0_file = os.path.join(cav_path, timestamp + f"_{name}0.png")
    camera1_file = os.path.join(cav_path, timestamp + f"_{name}1.png")
    camera2_file = os.path.join(cav_path, timestamp + f"_{name}2.png")
    camera3_file = os.path.join(cav_path, timestamp + f"_{name}3.png")
    return [camera0_file, camera1_file, camera2_file, camera3_file]


def parallel_transform(scenario_folders):
    print("subprocess...")
    for scenario_folder in tqdm(scenario_folders):
        cav_list = sorted(os.listdir(scenario_folder))

        assert len(cav_list) > 0

        # loop over all CAV data
        for j, cav_id in enumerate(cav_list):
            cav_path = os.path.join(scenario_folder, cav_id)

            yaml_files = sorted(
                [
                    os.path.join(cav_path, x)
                    for x in os.listdir(cav_path)
                    if x.endswith(".yaml")
                ]
            )
            timestamps = []

            # extract timestamp
            for file in yaml_files:
                res = file.split("/")[-1]
                timestamp = res.replace(".yaml", "")
                timestamps.append(timestamp)

            for timestamp in timestamps:
                if os.path.exists(os.path.join(cav_path, timestamp + "_imgs.hdf5")):
                    continue
                camera_files = load_camera_files(cav_path, timestamp, name="camera")
                depth_files = load_camera_files(cav_path, timestamp, name="depth")
                camera_data = load_camera_data(camera_files, True)
                depth_data = load_camera_data(depth_files, True)
                with h5py.File(
                    os.path.join(cav_path, timestamp + "_imgs.hdf5"), "w"
                ) as f:
                    for i in range(4):
                        f.create_dataset(f"camera{i}", data=camera_data[i])
                    for i in range(4):
                        f.create_dataset(f"depth{i}", data=depth_data[i])


if __name__ == "__main__":
    split_folders = [
        f"/GPFS/public/OPV2V_MoreAgents/{split}"
        for split in ["train", "validate", "test"]
    ]
    scenario_folders = []
    print(split_folders)

    for root_dir in split_folders:
        scenario_folders += sorted(
            [
                os.path.join(root_dir, x)
                for x in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, x))
            ]
        )

    mp_split = np.array_split(scenario_folders, mp)
    mp_split = [x.tolist() for x in mp_split]

    for i in range(mp):
        p = Process(target=parallel_transform, args=(mp_split[i],))
        p.start()

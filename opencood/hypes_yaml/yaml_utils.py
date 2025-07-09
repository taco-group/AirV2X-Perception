# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>
# Modifier: Yuheng Wu <yuhengwu@kaist.ac.kr>, Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: TDG-Attribution-NonCommercial-NoDistrib


import math
import os
import re

import numpy as np
import yaml


def load_yaml(file, opt=None):
    """
    Load yaml file and return a dictionary.

    Parameters
    ----------
    file : string
        yaml file path.

    opt : argparser
         Argparser.
    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """
    if opt and opt.model_dir:
        file = os.path.join(opt.model_dir, 
                            opt.config_file if hasattr(opt, "config_file") else "config.yaml")

    stream = open(file, "r")
    loader = yaml.Loader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    param = yaml.load(stream, Loader=loader)
    if "yaml_parser" in param:
        param = eval(param["yaml_parser"])(param)

    return param


def load_pickle(file):
    """
    Load pickle file and return a dictionary.

    Parameters
    ----------
    file : string
        Pickle file path.

    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """
    import pickle

    try:
        with open(file, "rb") as f:
            param = pickle.load(f)
    except Exception as e:
        import traceback

        traceback.print_exc()
        import pdb

        pdb.set_trace()
    return param


def load_voxel_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute `anchor_args[W][H][L]`
    """
    anchor_args = param["postprocess"]["anchor_args"]
    cav_lidar_range = anchor_args["cav_lidar_range"]
    voxel_size = param["preprocess"]["args"]["voxel_size"]

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args["vw"] = vw
    anchor_args["vh"] = vh
    anchor_args["vd"] = vd

    anchor_args["W"] = int((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args["H"] = int((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args["D"] = int((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param["postprocess"].update({"anchor_args": anchor_args})

    # sometimes we just want to visualize the data without implementing model
    if "model" in param:
        param["model"]["args"]["W"] = anchor_args["W"]
        param["model"]["args"]["H"] = anchor_args["H"]
        param["model"]["args"]["D"] = anchor_args["D"]

    return param


def load_point_pillar_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param["preprocess"]["cav_lidar_range"]
    voxel_size = param["preprocess"]["args"]["voxel_size"]

    grid_size = (
        np.array(cav_lidar_range[3:6]) - np.array(cav_lidar_range[0:3])
    ) / np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    param["model"]["args"]["point_pillar_scatter"]["grid_size"] = grid_size

    anchor_args = param["postprocess"]["anchor_args"]

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args["vw"] = vw
    anchor_args["vh"] = vh
    anchor_args["vd"] = vd

    anchor_args["W"] = math.ceil(
        (cav_lidar_range[3] - cav_lidar_range[0]) / vw
    )  # W is image width, but along with x axis in lidar coordinate
    anchor_args["H"] = math.ceil(
        (cav_lidar_range[4] - cav_lidar_range[1]) / vh
    )  # H is image height
    anchor_args["D"] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param["postprocess"].update({"anchor_args": anchor_args})

    return param


def load_cross_modal_point_pillar_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param["preprocess"]["cav_lidar_range"]
    voxel_size = param["preprocess"]["args"]["voxel_size"]

    grid_size = (
        np.array(cav_lidar_range[3:6]) - np.array(cav_lidar_range[0:3])
    ) / np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    print("grid_size: ", grid_size)
    param["model"]["args"]["pc_params"]["point_pillar_scatter"]["grid_size"] = grid_size

    anchor_args = param["postprocess"]["anchor_args"]

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args["vw"] = vw
    anchor_args["vh"] = vh
    anchor_args["vd"] = vd

    anchor_args["W"] = math.ceil(
        (cav_lidar_range[3] - cav_lidar_range[0]) / vw
    )  # W is image width, but along with x axis in lidar coordinate
    anchor_args["H"] = math.ceil(
        (cav_lidar_range[4] - cav_lidar_range[1]) / vh
    )  # H is image height
    anchor_args["D"] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param["postprocess"].update({"anchor_args": anchor_args})

    return param


def load_airv2x_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param["preprocess"]["cav_lidar_range"]
    voxel_size = param["preprocess"]["args"]["voxel_size"]
    max_cav = param["train_params"]["max_cav"]
    max_cav_num = (
        max_cav.get("vehicle", 0) + max_cav.get("rsu", 0) + max_cav.get("drone", 0)
    )
    print(f"max cav number: {max_cav_num}")
    param["train_params"]["max_cav_num"] = max_cav_num
    param["model"]["args"]["max_cav_num"] = max_cav_num


    veh_lidar_range = param["model"]["args"]["vehicle"]["lidar"]["lidar_range"]
    veh_voxel_size = param["model"]["args"]["vehicle"]["lidar"]["voxel_size"]
    veh_grid_size = (
        np.array(veh_lidar_range[3:6]) - np.array(veh_lidar_range[0:3])
    ) / np.array(veh_voxel_size)
    veh_grid_size = np.round(veh_grid_size).astype(np.int64)
    param["model"]["args"]["vehicle"]["lidar"]["point_pillar_scatter"]["grid_size"] = veh_grid_size

    print("vehicle grid_size: ", veh_grid_size)

    rsu_lidar_range = param["model"]["args"]["rsu"]["lidar"]["lidar_range"]
    rsu_voxel_size = param["model"]["args"]["rsu"]["lidar"]["voxel_size"]
    rsu_grid_size = (
        np.array(rsu_lidar_range[3:6] - np.array(rsu_lidar_range[0:3]))
    ) / np.array(rsu_voxel_size)
    rsu_grid_size = np.round(rsu_grid_size).astype(np.int64)
    param["model"]["args"]["rsu"]["lidar"]["point_pillar_scatter"]["grid_size"] = rsu_grid_size
    print("rsu grid_size: ", rsu_grid_size)
    anchor_args = param["postprocess"]["anchor_args"]
    
    
    drone_lidar_range = param["model"]["args"]["drone"]["lidar"]["lidar_range"]
    drone_voxel_size = param["model"]["args"]["drone"]["lidar"]["voxel_size"]
    drone_grid_size = (
        np.array(drone_lidar_range[3:6] - np.array(drone_lidar_range[0:3]))
    ) / np.array(drone_voxel_size)
    drone_grid_size = np.round(drone_grid_size).astype(np.int64)
    param["model"]["args"]["drone"]["lidar"]["point_pillar_scatter"]["grid_size"] = rsu_grid_size
    print("drone grid_size: ", drone_grid_size)
    anchor_args = param["postprocess"]["anchor_args"]

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args["vw"] = vw
    anchor_args["vh"] = vh
    anchor_args["vd"] = vd

    anchor_args["W"] = math.ceil(
        (cav_lidar_range[3] - cav_lidar_range[0]) / vw
    )  # W is image width, but along with x axis in lidar coordinate
    anchor_args["H"] = math.ceil(
        (cav_lidar_range[4] - cav_lidar_range[1]) / vh
    )  # H is image height
    anchor_args["D"] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param["postprocess"].update({"anchor_args": anchor_args})

    return param


def load_skylink_bm2cp_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param["preprocess"]["cav_lidar_range"]
    voxel_size = param["preprocess"]["args"]["voxel_size"]

    # TODO(YH): it allows different config of agents, but for now we still use same config for veh & rsu
    veh_grid_size = (
        np.array(cav_lidar_range[3:6]) - np.array(cav_lidar_range[0:3])
    ) / np.array(voxel_size)
    veh_grid_size = np.round(veh_grid_size).astype(np.int64)

    print("vehicle grid_size: ", veh_grid_size)

    rsu_grid_size = (
        np.array(rsu_lidar_range[3:6] - np.array(rsu_lidar_range[0:3]))
    ) / np.array(voxel_size)
    rsu_grid_size = np.round(rsu_grid_size).astype(np.int64)
    print("rsu grid_size: ", rsu_grid_size)
    param["model"]["args"]["vehicle"]["pc_params"]["point_pillar_scatter"][
        "grid_size"
    ] = veh_grid_size
    param["model"]["args"]["rsu"]["pc_params"]["point_pillar_scatter"]["grid_size"] = (
        rsu_grid_size
    )

    anchor_args = param["postprocess"]["anchor_args"]

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args["vw"] = vw
    anchor_args["vh"] = vh
    anchor_args["vd"] = vd

    anchor_args["W"] = math.ceil(
        (cav_lidar_range[3] - cav_lidar_range[0]) / vw
    )  # W is image width, but along with x axis in lidar coordinate
    anchor_args["H"] = math.ceil(
        (cav_lidar_range[4] - cav_lidar_range[1]) / vh
    )  # H is image height
    anchor_args["D"] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param["postprocess"].update({"anchor_args": anchor_args})

    return param


def load_second_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param["preprocess"]["cav_lidar_range"]
    voxel_size = param["preprocess"]["args"]["voxel_size"]

    grid_size = (
        np.array(cav_lidar_range[3:6]) - np.array(cav_lidar_range[0:3])
    ) / np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    param["model"]["args"]["grid_size"] = grid_size

    anchor_args = param["postprocess"]["anchor_args"]

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args["vw"] = vw
    anchor_args["vh"] = vh
    anchor_args["vd"] = vd

    anchor_args["W"] = int((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args["H"] = int((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args["D"] = int((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param["postprocess"].update({"anchor_args": anchor_args})

    return param


def load_bev_params(param):
    """
    Load bev related geometry parameters s.t. boundary, resolutions, input
    shape, target shape etc.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute `geometry_param`.

    """
    res = param["preprocess"]["args"]["res"]
    L1, W1, H1, L2, W2, H2 = param["preprocess"]["cav_lidar_range"]
    downsample_rate = param["preprocess"]["args"]["downsample_rate"]

    def f(low, high, r):
        return int((high - low) / r)

    input_shape = (
        int((f(L1, L2, res))),
        int((f(W1, W2, res))),
        int((f(H1, H2, res)) + 1),
    )
    label_shape = (
        int(input_shape[0] / downsample_rate),
        int(input_shape[1] / downsample_rate),
        7,
    )
    geometry_param = {
        "L1": L1,
        "L2": L2,
        "W1": W1,
        "W2": W2,
        "H1": H1,
        "H2": H2,
        "downsample_rate": downsample_rate,
        "input_shape": input_shape,
        "label_shape": label_shape,
        "res": res,
    }
    param["preprocess"]["geometry_param"] = geometry_param
    param["postprocess"]["geometry_param"] = geometry_param
    param["model"]["args"]["geometry_param"] = geometry_param
    return param


def save_yaml(data, save_name):
    """
    Save the dictionary into a yaml file.

    Parameters
    ----------
    data : dict
        The dictionary contains all data.

    save_name : string
        Full path of the output yaml file.
    """

    with open(save_name, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def load_point_pillar_params_stage1(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param["preprocess"]["cav_lidar_range"]
    voxel_size = param["preprocess"]["args"]["voxel_size"]

    grid_size = (
        np.array(cav_lidar_range[3:6]) - np.array(cav_lidar_range[0:3])
    ) / np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    param["box_align_pre_calc"]["stage1_model_config"]["point_pillar_scatter"][
        "grid_size"
    ] = grid_size

    anchor_args = param["box_align_pre_calc"]["stage1_postprocessor_config"][
        "anchor_args"
    ]

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args["vw"] = vw
    anchor_args["vh"] = vh
    anchor_args["vd"] = vd

    anchor_args["W"] = int(
        (cav_lidar_range[3] - cav_lidar_range[0]) / vw
    )  # W is image width, but along with x axis in lidar coordinate
    anchor_args["H"] = int(
        (cav_lidar_range[4] - cav_lidar_range[1]) / vh
    )  # H is image height
    anchor_args["D"] = int((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param["box_align_pre_calc"]["stage1_postprocessor_config"].update(
        {"anchor_args": anchor_args}
    )

    return param


def load_lift_splat_shoot_params(param):
    """
    Based on the detection range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param["preprocess"]["cav_lidar_range"]
    voxel_size = param["preprocess"]["args"]["voxel_size"]

    grid_size = (
        np.array(cav_lidar_range[3:6]) - np.array(cav_lidar_range[0:3])
    ) / np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)

    anchor_args = param["postprocess"]["anchor_args"]

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args["vw"] = vw
    anchor_args["vh"] = vh
    anchor_args["vd"] = vd

    anchor_args["W"] = math.ceil(
        (cav_lidar_range[3] - cav_lidar_range[0]) / vw
    )  # W is image width, but along with x axis in lidar coordinate
    anchor_args["H"] = math.ceil(
        (cav_lidar_range[4] - cav_lidar_range[1]) / vh
    )  # H is image height
    anchor_args["D"] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param["postprocess"].update({"anchor_args": anchor_args})

    return param


def load_general_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param["preprocess"]["cav_lidar_range"]
    voxel_size = param["preprocess"]["args"]["voxel_size"]
    anchor_args = param["postprocess"]["anchor_args"]

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args["vw"] = vw
    anchor_args["vh"] = vh
    anchor_args["vd"] = vd

    anchor_args["W"] = math.ceil(
        (cav_lidar_range[3] - cav_lidar_range[0]) / vw
    )  # W is image width, but along with x axis in lidar coordinate
    anchor_args["H"] = math.ceil(
        (cav_lidar_range[4] - cav_lidar_range[1]) / vh
    )  # H is image height
    anchor_args["D"] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param["postprocess"].update({"anchor_args": anchor_args})

    return param

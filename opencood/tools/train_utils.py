# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import glob
import importlib
import os
import re
import shutil
import sys
from datetime import datetime
from tkinter.messagebox import NO

import torch
import torch.optim as optim
import yaml


def backup_script(full_path, folders_to_save=["models", "data_utils", "utils", "loss"]):
    target_folder = os.path.join(full_path, "scripts")
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    current_path = os.path.dirname(
        __file__
    )  # __file__ refer to this file, then the dirname is "?/tools"

    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f"../{folder_name}")
        shutil.copytree(source_folder, ttarget_folder)


def load_saved_model(saved_path, model, epoch=None):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), "{} not found".format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, "*epoch*.pth"))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    # if os.path.exists(os.path.join(saved_path, 'net_latest.pth')):
    #     model.load_state_dict(torch.load(os.path.join(saved_path, 'net_latest.pth')))
    # file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))

    if False:
        pass
    # if file_list:
    #    assert len(file_list) == 1
    #    model.load_state_dict(torch.load(file_list[0], map_location='cpu'), strict=False)
    #    return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")), model

    # if os.path.exists(os.path.join(saved_path, 'net_epoch_bestval*.pth')):
    #    model.load_state_dict(torch.load(os.path.join(saved_path, 'net_epoch_bestval*.pth')))
    #    return 100, model
    else:
        if epoch is None:
            initial_epoch = findLastCheckpoint(saved_path)
        else:
            initial_epoch = int(epoch)

        if initial_epoch > 0:
            print("resuming by loading epoch %d" % initial_epoch)

        state_dict_ = torch.load(
            os.path.join(saved_path, "net_epoch%d.pth" % initial_epoch)
        )
        state_dict = {}
        # convert data_parallal to model
        for k in state_dict_:
            if k.startswith("module") and not k.startswith("module_list"):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]

        model_state_dict = model.state_dict()

        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(
                        "Skip loading parameter {}, required shape{}, "
                        "loaded shape{}.".format(
                            k, model_state_dict[k].shape, state_dict[k].shape
                        )
                    )
                    state_dict[k] = model_state_dict[k]
            else:
                print("Drop parameter {}.".format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print("No param {}.".format(k))
                state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        return initial_epoch, model


def load_model(saved_path, model, epoch=None, start_from_best=True):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), "{} not found".format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, "*epoch*.pth"))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                try:
                    _epoch = int(result[0])
                except Exception as e:
                    pass
                else:
                    epochs_exist.append(_epoch)
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    if epoch is not None:
        initial_epoch = epoch
    else:
        if start_from_best:
            file_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*.pth"))
            if file_list:
                assert len(file_list) == 1
                state_dict = torch.load(file_list[0], map_location="cpu")
                # 将 cdd 权重映射到 mdd
                if "cdd" in state_dict:
                    state_dict["mdd"] = state_dict.pop("cdd")
                model.load_state_dict(state_dict, strict=False)
                return eval(
                    file_list[0]
                    .split("/")[-1]
                    .rstrip(".pth")
                    .lstrip("net_epoch_bestval_at")
                ), model
        initial_epoch = findLastCheckpoint(saved_path)

    if initial_epoch > 0:
        print("resuming by loading epoch %d" % initial_epoch)

    state_dict_ = torch.load(
        os.path.join(saved_path, "net_epoch%d.pth" % initial_epoch),
        map_location="cuda:0",
    )
    state_dict = {}
    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith("module") and not k.startswith("module_list"):
            state_dict[k[7:]] = state_dict_[k]
        else:
            if k.startswith("cdd"):
                # rename cdd to mdd
                state_dict["m" + k[1:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(
                    "Skip loading parameter {}, required shape{}, "
                    "loaded shape{}.".format(
                        k, model_state_dict[k].shape, state_dict[k].shape
                    )
                )
                state_dict[k] = model_state_dict[k]
        else:
            print("Drop parameter {}.".format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print("No param {}.".format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    return initial_epoch, model


def setup_train(hypes):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes["name"]
    current_time = datetime.now()
    tag = hypes["tag"]
    time_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, "../logs")

    full_path = os.path.join(current_path, model_name, tag + "_" + time_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except FileExistsError:
                pass
        # save the yaml file
        save_name = os.path.join(full_path, "config.yaml")
        with open(save_name, "w") as outfile:
            yaml.dump(hypes, outfile)

    return full_path


def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes["model"]["core_method"]
    backbone_config = hypes["model"]["args"]

    model_filename = "opencood.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace("_", "")
    print("model_lib: ", model_lib)
    print("target_model_name: ", target_model_name)

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            print(name.lower(), cls)
            model = cls

    if model is None:
        print(
            "backbone not found in models folder. Please make sure you "
            "have a python file named %s and has a class "
            "called %s ignoring upper/lower case" % (model_filename, target_model_name)
        )
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    # modified by YH to support multi tasks
    if hypes.get("task") is None:
        loss_func_name = hypes["loss"]["core_method"]
        loss_func_config = hypes["loss"]["args"]
    else:
        task = hypes["task"]
        loss_func_name = hypes["loss"][task]["core_method"]
        loss_func_config = hypes["loss"][task]["args"]

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace("_", "")

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print(
            "loss function not found in loss folder. Please make sure you "
            "have a python file named %s and has a class "
            "called %s ignoring upper/lower case" % (loss_filename, target_loss_name)
        )
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes["optimizer"]
    optimizer_method = getattr(optim, method_dict["core_method"], None)
    if not optimizer_method:
        raise ValueError("{} is not supported".format(method_dict["name"]))
    if "args" in method_dict:
        return optimizer_method(
            model.parameters(), lr=method_dict["lr"], **method_dict["args"]
        )
    else:
        return optimizer_method(model.parameters(), lr=method_dict["lr"])


def setup_lr_schedular(hypes, optimizer, init_epoch=None, n_iter_per_epoch=None):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    """
    lr_schedule_config = hypes["lr_scheduler"]
    last_epoch = init_epoch if init_epoch is not None else 0

    if lr_schedule_config["core_method"] == "step":
        from torch.optim.lr_scheduler import StepLR

        step_size = lr_schedule_config["step_size"]
        gamma = lr_schedule_config["gamma"]
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config["core_method"] == "multistep":
        from torch.optim.lr_scheduler import MultiStepLR

        milestones = lr_schedule_config["step_size"]
        gamma = lr_schedule_config["gamma"]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif lr_schedule_config["core_method"] == "exponential":
        print("ExponentialLR is chosen for lr scheduler")
        from torch.optim.lr_scheduler import ExponentialLR

        gamma = lr_schedule_config["gamma"]
        scheduler = ExponentialLR(optimizer, gamma)

    elif lr_schedule_config["core_method"] == "cosineannealwarm":
        print("cosine annealing is chosen for lr scheduler")
        from timm.scheduler.cosine_lr import CosineLRScheduler

        num_steps = lr_schedule_config["epoches"] * n_iter_per_epoch
        warmup_lr = lr_schedule_config["warmup_lr"]
        warmup_steps = lr_schedule_config["warmup_epoches"] * n_iter_per_epoch
        lr_min = lr_schedule_config["lr_min"]

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        sys.exit("not supported lr schedular")

    for epoch in range(last_epoch):
        if lr_schedule_config["core_method"] == "cosineannealwarm":
            scheduler.step(epoch)
        else:
            scheduler.step()

    return scheduler


# def to_device(inputs, device):
#     if isinstance(inputs, list):
#         return [to_device(x, device) for x in inputs]
#     elif isinstance(inputs, dict):
#         return {k: to_device(v, device) for k, v in inputs.items()}
#     else:
#         if isinstance(inputs, int) or isinstance(inputs, float) \
#                 or isinstance(inputs, str):
#             return inputs
#         return inputs.to(device)
import numpy as np
import torch


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    elif isinstance(inputs, np.ndarray):
        # 将 numpy 数组转换为 PyTorch 张量
        return torch.tensor(inputs).to(device)
    elif isinstance(inputs, (int, float, str)):
        return inputs  # 对于基础类型，直接返回
    elif isinstance(inputs, np.integer):
        # 如果输入是 numpy 整数类型，转换为 Python 标量 int
        return int(inputs)
    elif isinstance(inputs, np.floating):
        # 如果输入是 numpy 浮点数类型，转换为 Python 标量 float
        return float(inputs)
    else:
        # 其他情况尝试调用 to 方法
        try:
            return inputs.to(device)
        except:
            pass

from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import cv2
import numpy as np
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common_modules.airv2x_encoder import LiftSplatShootEncoder
from opencood.models.common_modules.airv2x_pillar_vfe import PillarVFE


import torch
import matplotlib.pyplot as plt


class Airv2xBase(nn.Module):
    """
    Base class for Airv2x models.
    """

    def __init__(self, args):
        super(Airv2xBase, self).__init__()
        self.args = args
        self.collaborators = args["collaborators"]
        self.active_sensors = args["active_sensors"]

        self.veh_models = None
        self.rsu_models = None
        self.drone_models = None
        
        
    def init_encoders(self, args):
        self.veh_models = nn.ModuleList()
        self.rsu_models = nn.ModuleList()
        self.drone_models = nn.ModuleList()
        
        if "vehicle" in self.collaborators:
            agent_type = "vehicle"
            vehicle_modalities = args[agent_type]["modalities"]
            for m in vehicle_modalities:
                if m == "cam":
                    self.veh_models.append(LiftSplatShootEncoder(args[agent_type][m], agent_type=agent_type))
                elif m == "lidar":
                    self.veh_models.append(
                        nn.Sequential(PillarVFE(
                                        args[agent_type][m]["pillar_vfe"],
                                        num_point_features=4,
                                        voxel_size=args[agent_type][m]["voxel_size"],
                                        point_cloud_range=args[agent_type][m]["lidar_range"],
                                        agent_type=agent_type),
                                      PointPillarScatter(args[agent_type][m]["point_pillar_scatter"]))
                        
                        )
                else:
                    raise NotImplementedError(f"Modality {m} not supported for vehicle.")

        if "rsu" in self.collaborators:
            agent_type = "rsu"
            rsu_modalities = args[agent_type]["modalities"]
            for m in rsu_modalities:
                if m == "cam":
                    self.rsu_models.append(LiftSplatShootEncoder(args[agent_type][m], agent_type=agent_type))
                elif m == "lidar":
                    self.rsu_models.append(
                        nn.Sequential(PillarVFE(
                                        args[agent_type][m]["pillar_vfe"],
                                        num_point_features=4,
                                        voxel_size=args[agent_type][m]["voxel_size"],
                                        point_cloud_range=args[agent_type][m]["lidar_range"],
                                        agent_type=agent_type),
                                      PointPillarScatter(args[agent_type][m]["point_pillar_scatter"]))
                        
                        )
                else:
                    raise NotImplementedError(f"Modality {m} not supported for drone.")

        if "drone" in self.collaborators:
            agent_type = "drone"
            drone_modalities = args[agent_type]["modalities"]
            for m in drone_modalities:
                if m == "cam":
                    self.drone_models.append(LiftSplatShootEncoder(args[agent_type]["cam"], agent_type=agent_type))
                elif m == "lidar":
                    self.drone_models.append(
                        nn.Sequential(PillarVFE(
                                        args[agent_type][m]["pillar_vfe"],
                                        num_point_features=4,
                                        voxel_size=args[agent_type][m]["voxel_size"],
                                        point_cloud_range=args[agent_type][m]["lidar_range"],
                                        agent_type=agent_type),
                                        PointPillarScatter(args[agent_type][m]["point_pillar_scatter"]))
                        
                        )
                else:
                    raise NotImplementedError(f"Modality {m} not supported for drone.")

    def extract_features(self, data_dict):
        """
        Extract and aggregate features from each collaborating agent.

        This method processes the input `data_dict` by forwarding each agent's data
        through its respective sub-model (vehicle, RSU, drone). It gathers the
        intermediate outputs into a unified batch structure.

        Only the agents specified in `self.collaborators` and with non-empty `batch_idxs`
        will be processed. Each agent must have its corresponding sub-model initialized
        before feature extraction.

        Parameters
        ----------
        data_dict : dict
            A dictionary containing batched input data for each agent type. Each agent's
            data must include at least 'batch_idxs' and 'record_len' fields.

        Returns
        -------
        batch_output_dict : dict
            The merged feature outputs from all processed agents, structured batch-wise.

        batch_record_len : torch.Tensor
            A tensor indicating the number of records aggregated per batch sample.
        """
        batch_dicts = OrderedDict()
        if "vehicle" in self.collaborators and len(data_dict["vehicle"]["batch_idxs"]) > 0:
            assert self.veh_models is not None, "Vehicle model is not initialized."
            output_dict_veh = []
            for v_model in self.veh_models:
                output_dict_veh.append(v_model(data_dict))
            output_dict_veh = self.fuse_bev(output_dict_veh)
            batch_dicts["vehicle"] = output_dict_veh
            
        if "rsu" in self.collaborators and len(data_dict["rsu"]["batch_idxs"]) > 0:
            assert self.rsu_models is not None, "RSU model is not initialized."
            output_dict_rsu = []
            for r_model in self.rsu_models:
                output_dict_rsu.append(r_model(data_dict))
            output_dict_rsu = self.fuse_bev(output_dict_rsu)
            batch_dicts["rsu"] = output_dict_rsu
            
        if "drone" in self.collaborators and len(data_dict["drone"]["batch_idxs"]) > 0:
            assert self.drone_models is not None, "Drone model is not initialized."
            output_dict_drone = []
            for d_model in self.drone_models:
                output_dict_drone.append(d_model(data_dict))
            output_dict_drone = self.fuse_bev(output_dict_drone)
            batch_dicts["drone"] = output_dict_drone

        # Normally, all agent types has the same batch size, but if one type of agent is not in the collaborator list,
        # the batch size of that agent type will be 0. So we need to find the maximum batch size among all agent types.
        B = max(len(data_dict["vehicle"]["batch_idxs"]), 
                    len(data_dict["rsu"]["batch_idxs"]),
                    len(data_dict["drone"]["batch_idxs"]))

        batch_output_dict, batch_record_len = self.repack_batch(batch_dicts, data_dict, B)

        assert (
            batch_output_dict["spatial_features"].shape[0]
            == batch_record_len.sum().item()
        ), f"{batch_output_dict['spatial_features'].shape}, {batch_record_len}"
        return batch_output_dict, batch_record_len
    
    
    def fuse_bev(self, batch_dict_list: list):
        # For here, only "spatial_features" is used for the future module, so we only keep this one
        fused_batch_dict = {
            "spatial_features": 
                torch.mean(
                    torch.stack(
                        [batch_dict["spatial_features"] for batch_dict in batch_dict_list], 
                        dim=0), 
                    dim=0)
        }
        return fused_batch_dict

    def repack_batch(self, batch_dicts, data_dict, batch_size):
        """
        Repack extracted features into a unified batch format.

        This method reorganizes the output dictionaries from each agent into
        a batch-aligned structure. It processes each batch index separately,
        merging features across available agents while tracking the corresponding
        record lengths.

        Parameters
        ----------
        batch_dicts : dict
            A dictionary where each key corresponds to an agent type ('vehicle', 'rsu', 'drone')
            and the value is the output dictionary from the agent's model.

        data_dict : dict
            The original input dictionary containing the metadata and batch indexing
            information for each agent.

        batch_size : int
            The total number of samples in the current batch.

        Returns
        -------
        output_dict : dict
            A dictionary containing the merged feature outputs for the entire batch.

        record_len : torch.Tensor
            A tensor containing the total record length for each sample in the batch.
        """
        output_batch_dicts_list = []
        record_len_list = []
        # import pdb; pdb.set_trace()
        for idx in range(batch_size):
            output_dict_list = []
            record_len = 0
            for agent_type, output_dict in batch_dicts.items():
                batch_idxs = data_dict[agent_type]["batch_idxs"]
                if idx not in batch_idxs:
                    continue

                # if batch_idxs == [1]:
                #     # print("")
                #     pass
                tensor_idx = batch_idxs.index(idx)
                agent_record_len = data_dict[agent_type]["record_len"]
                # NOTE(YH): here we must filter record_len > 0, else it will cause error to regroup
                output_dict = self._regroup(
                    output_dict, tensor_idx, agent_record_len[agent_record_len > 0]
                )
                output_dict_list.append(output_dict)
                record_len += data_dict[agent_type]["record_len"][idx].reshape(-1)
            sample_output_dict = self.merge_output_dict_list(output_dict_list)
            output_batch_dicts_list.append(sample_output_dict)
            record_len_list.append(record_len)

        output_dict = self.merge_output_dict_list(output_batch_dicts_list)
        record_len = torch.cat(record_len_list, dim=0)
        
        try:
            assert (
                output_dict["spatial_features"].shape[0]
                == record_len.sum().item()
            ), f"{output_dict['spatial_features'].shape}, {record_len}"
        except:
            print("Error in assert statement")
            print(f"batch_output_dict['spatial_features'].shape: {output_dict['spatial_features'].shape}")
            import pdb; pdb.set_trace()

        return output_dict, record_len

    def merge_output_dict_list(self, output_dict_list):
        """
        Merge a list of output dictionaries by concatenating features.

        Given a list of dictionaries where each dictionary contains multiple feature
        tensors, this method groups tensors with the same feature name and concatenates
        them along the batch dimension.

        Parameters
        ----------
        output_dict_list : list of dict
            A list of dictionaries, each containing feature tensors with identical keys.

        Returns
        -------
        merged_output_dict : dict
            A single dictionary where features with the same name are concatenated
            into a single tensor along the batch dimension (dim=0).
            batch -> [N1+N2+...+Nk, C]
            where N1, N2, ..., Nk are the number agents of each sample.
        """
        accumulated_dict = defaultdict(list)

        # First pass: accumulate features
        for output_dict in output_dict_list:
            for feat_name, feat in output_dict.items():
                accumulated_dict[feat_name].append(feat)

        # Second pass: create the final merged output
        merged_output_dict = OrderedDict()
        for feat_name, feat_list in accumulated_dict.items():
            merged_output_dict[feat_name] = torch.cat(feat_list, dim=0)

        return merged_output_dict

    def _regroup(self, output_dict, idx, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        ret_output_dict = {}
        for k, v in output_dict.items():
            split_x = torch.tensor_split(v, cum_sum_len[:-1].cpu())[idx]
            ret_output_dict[k] = split_x
        return ret_output_dict

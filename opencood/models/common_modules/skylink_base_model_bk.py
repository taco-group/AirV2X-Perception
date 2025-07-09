from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import cv2
import numpy as np


class SkylinkBase(nn.Module):
    """
    Base class for Skylink models.
    """

    def __init__(self, args):
        super(SkylinkBase, self).__init__()
        self.args = args
        self.collaborators = args["collaborators"]
        self.active_sensors = args["active_sensors"]

        self.veh_model = None
        self.rsu_model = None
        self.drone_model = None

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
        if (
            "vehicle" in self.collaborators
            and len(data_dict["vehicle"]["batch_idxs"]) > 0
        ):
            assert self.veh_model is not None, "Vehicle model is not initialized."
            output_dict_veh = self.veh_model(data_dict)
            batch_dicts["vehicle"] = output_dict_veh
        if "rsu" in self.collaborators and len(data_dict["rsu"]["batch_idxs"]) > 0:
            assert self.rsu_model is not None, "RSU model is not initialized."
            output_dict_rsu = self.rsu_model(data_dict)
            batch_dicts["rsu"] = output_dict_rsu
        if "drone" in self.collaborators and len(data_dict["drone"]["batch_idxs"]) > 0:
            assert self.drone_model is not None, "Drone model is not initialized."
            output_dict_drone = self.drone_model(data_dict)
            batch_dicts["drone"] = output_dict_drone
            
            
        # feat = output_dict_veh['spatial_features'][0].mean(0).detach().cpu().numpy()
        # cv2.imwrite("/home/xiangbog/Folder/Research/SkyLink/skylink/debug/debug_image.png", ((feat - feat.min()) / (feat.max() - feat.min()) * 255).astype(np.uint8),)
        # # import pdb; pdb.set_trace()

        batch_output_dict, batch_record_len = self.repack_batch(
            batch_dicts, data_dict, len(data_dict["vehicle"]["batch_idxs"])
        )

        assert (
            batch_output_dict["spatial_features"].shape[0]
            == batch_record_len.sum().item()
        ), f"{batch_output_dict['spatial_features'].shape}, {batch_record_len}"
        return batch_output_dict, batch_record_len

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
        for idx in range(batch_size):
            output_dict_list = []
            record_len = 0
            for agent_type, output_dict in batch_dicts.items():
                batch_idxs = data_dict[agent_type]["batch_idxs"]
                if idx not in batch_idxs:
                    continue

                if batch_idxs == [1]:
                    print("")
                    pass
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

import os
import pickle

import yaml


def convert_yaml_to_pickle(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(("objects.yaml", ".yml")):
                yaml_path = os.path.join(dirpath, filename)

                # Load YAML
                with open(yaml_path, "r") as f:
                    try:
                        data = yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        print(f"Error parsing {yaml_path}: {e}")
                        continue

                # Save as pickle
                pickle_path = yaml_path.rsplit(".", 1)[0] + ".pkl"
                with open(pickle_path, "wb") as pf:
                    pickle.dump(data, pf)

                print(f"Converted: {yaml_path} -> {pickle_path}")


# Example usage
root_folder = "/code/dataset/skylink"
convert_yaml_to_pickle(root_folder)

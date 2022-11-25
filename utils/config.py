import argparse
from omegaconf import OmegaConf
def str2bool(v):
    return v.lower() in ('true', '1', 'yes', 'y', 't')

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default=None)
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--resume', type=str2bool, default=None)
    parser.add_argument('--save_weights', type=str2bool, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--mode', type=str, default=None)
    main_config = parser.parse_args()

    # get configs
    config = OmegaConf.load(main_config.yaml_path)
            
    # overwrite dataset dir
    if not main_config.dataset_dir is None:
        config.data.dataset.dir = main_config.dataset_dir
    # overwrite name
    if not main_config.name is None:
        config.metadata.name = main_config.name
    # overwrite resume and save weights
    if not main_config.resume is None:
        config.training.checkpoint.resume = main_config.resume
    if not main_config.save_weights is None:
        config.training.checkpoint.save_weights = main_config.save_weights
    # overwrite dirs
    if not main_config.log_dir is None:
        config.viz.log_dir = main_config.log_dir
    if not main_config.pretrained_path is None:
        config.training.loss.feature.recon.pretrained_path = main_config.pretrained_path
    if not main_config.model_dir is None:
        config.metadata.model_dir = main_config.model_dir
    if not main_config.mode is None:
        config.mode = main_config.mode

    return config



def print_config(config):
    print(OmegaConf.to_yaml(config))


def config_to_string(config):
    return OmegaConf.to_yaml(config)
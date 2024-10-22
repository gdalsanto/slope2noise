
import argparse
import yaml
import pickle 
import os 

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config.config import Config
from coupled_rooms_dataset import CommonSlopesRIR, sample_room_interior
from rir_synthesis import rir_synthesis

def gen_dataset(config_dict: Config):
    
    # get number of batches 
    n_batch = config_dict.n_rirs // config_dict.batch_size 
    if config_dict.n_rirs % config_dict.batch_size != 0:
        n_batch += 1
    
    room_dim_1 = [10, 7.5, 3.5] # x, y, z 
    # room_dim_2 = [15, 4.5, 3.5] 

    for i_batch in range(n_batch):
        # sample the source location within the room 
        source_locs = sample_room_interior(room_dim_1, n_locs=config_dict.batch_size)
        # sample the receiver location within the room 
        receiver_locs = sample_room_interior(room_dim_1, n_locs=config_dict.batch_size)
        # get the amplitudes of the slopes 
        a_vals = np.random.uniform(10**(-3/10), 
                                   10**(0/10),
                                   (config_dict.batch_size, config_dict.n_slopes))
        # get the decay times of the slopes
        t_vals = np.random.uniform(0.1, 
                                   3.5,
                                   (config_dict.batch_size, config_dict.n_slopes))
        # generate the shaped wgn give the slopes and the decay times
        _, rirs = rir_synthesis(t_vals, 
                                a_vals, 
                                config_dict.f_bands, 
                                config_dict.fs, 
                                config_dict.ir_len, 
                                type=config_dict.synthesis_type)
        # create instance of CommonSlopes dataclass
        RIRs = CommonSlopesRIR(source_locs=source_locs,
                            receiver_locs=receiver_locs,
                            n_slopes=config_dict.n_slopes,
                            a_vals=a_vals,
                            t_vals=t_vals,
                            rir=rirs, 
                            batch_id=i_batch)
        # save it to a pkl file 
        with open(os.path.join(config_dict.output_dir, f"bb_wgn_{i_batch:04}.pkl"), "wb") as f:
            pickle.dump(RIRs, f)
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config_file",
        default=None,
        help="Configuration file (YAML) \
        (if none provided the default parameters are loaded).",
    )

    args = parser.parse_args()

    if args.config_file:
        # Resolve the relative file path
        file_path = Path(args.config_file).resolve()

        # Read and parse the YAML file
        with open(file_path, 'r') as file:
            config_data = yaml.safe_load(file)
        config_dict = Config(**config_data)
    else:
        config_dict = Config()

    # make output directory if it does not exist
    Path(config_dict.output_dir).mkdir(parents=True, exist_ok=True)

    gen_dataset(config_dict)
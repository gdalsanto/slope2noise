import argparse
import yaml
import os
import pickle
import numpy as np

from pathlib import Path
from config.config import Config
from slope2noise.generate import *
from slope2noise.utils import *
from slope2noise.dataclass import CommonSlopesRIRSimple



def main(config_dict: Config):

    # detect whether the decay time values are broadband of frequency dependent
    if config_dict.f_bands is None:
        n_bands = 1
    else:
        n_bands = len(config_dict.f_bands)

    t_vals = np.random.uniform(
        0.5, 3,
        (config_dict.n_rirs, config_dict.n_slopes, n_bands))

    a_vals = np.random.uniform(
        10**(-3. / 10), 1,
        (config_dict.n_rirs, config_dict.n_slopes, n_bands))

    for i_batch in range(int(np.ceil(config_dict.n_rirs / config_dict.batch_size))):
        _, rirs = shaped_wgn(t_vals[:(i_batch + 1) * config_dict.batch_size, ...],
                            a_vals[:(i_batch + 1) * config_dict.batch_size, ...],
                            fs=config_dict.fs,
                            ir_len=config_dict.ir_len,
                            f_bands=config_dict.f_bands,
                            )
        
        RIRs = CommonSlopesRIRSimple(
            n_slopes = config_dict.n_slopes,
            a_vals = a_vals[:(i_batch + 1) * config_dict.batch_size, ...],
            t_vals = t_vals[:(i_batch + 1) * config_dict.batch_size, ...],
            rir = rirs,
            sample_rate = config_dict.fs,
            batch_id = i_batch,
            f_bands = config_dict.f_bands,
        )
        # save it to a pkl file
        with open(
                os.path.join(config_dict.output_dir,
                             f"bb_wgn_{i_batch:04}.pkl"), "wb") as f:
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
    
    # make output directory if it doesn't exist
    if not os.path.exists(config_dict.output_dir):
        os.makedirs(config_dict.output_dir)

    main(config_dict)
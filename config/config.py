import os
import time
from pydantic import BaseModel, model_validator
from typing import Dict, Optional, List

class Config(BaseModel):
    # number of IRs to generate 
    n_rirs: int = 1
    # size of a subset of rirs that will be saved in the same file 
    batch_size: int = 1
    # length of the IRs in samples 
    ir_len: int = 8192
    # sampling rate of the IRs
    fs: int = 48000
    # number of modes to synthesize
    n_modes: Optional[int] = 10
    # center frequnecy of the band. If none is provided, it is assumed to have homogeneous attenuation
    f_bands: Optional[List[float]] = None # [125, 250, 500, 1000, 2000, 4000, 8000]
    # decay time values limits 
    t_vals_lims: List[float] = [0.1, 3.5]
    # amplitude values limits (linear scale)
    a_vals_lims: List[float] = [10**(-3./10), 1]
    # number of common slopes
    n_slopes: int = 1
    # type of synthesis to use
    synthesis_type: str = "wgn" # 'modal' or 'wgn'
    # output directory
    output_dir: str = "output"

    @model_validator(mode="after")
    @classmethod
    def check_batch_size(cls, model):
        '''check if the batch size is smaller than the number of rirs'''
        n_rirs = model.n_rirs
        batch_size = model.batch_size
        if batch_size > n_rirs:
            raise ValueError(f"Batch size {batch_size} should be smaller than the number of rirs {n_rirs}")

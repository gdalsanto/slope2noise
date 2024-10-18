import os
import time
from pydantic import BaseModel, model_validator
from typing import Dict, Optional, List

class Config(BaseModel):
    # number of IRs to generate 
    n_rirs: int = 1
    # length of the IRs in samples 
    ir_len: int = 8192
    # sampling rate of the IRs
    fs: int = 48000
    # number of modes to synthesize
    n_modes: int = 10
    # center frequnecy of the band. If none is provided, it is assumed to have homogeneous attenuation
    f_bands: Optional[List[float]] = None # [125, 250, 500, 1000, 2000, 4000, 8000]
    # frequency distribution 
    f_dist: str = "log" # 'log' or 'linear'
    # decay time values limits 
    t_vals_lims: List[float] =[0.1, 3.0]
    # amplitude values limits
    a_vals_lims: List[float] = [0.1, 1.0]
    # number of common slopes
    n_slopes: int = 1
    # type of synthesis to use
    synthesis_type: str = "modal" # 'modal' or 'wgn'
    # output directory
    output_dir: str = "output"
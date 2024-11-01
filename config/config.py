import os
import time
from pydantic import BaseModel, model_validator, field_validator
from typing import Dict, Optional, List, Tuple
from pathlib import Path


class RoomGeometryConfig(BaseModel):
    num_rooms: int
    room_dims: List[List]
    start_coordinates: List[List]
    source_pos: List
    aperture_coords: Optional[List[List[Tuple]]] = None


class AmpGenerationConfig(BaseModel):
    # whether to generate amplitudes randomly, or based on geometry
    generate_amplitude_based_on_geometry: bool = False
    # path to pickle file containing amplitude distribution, analysed from the ThreeRoomDataset
    amp_dist_filepath: Optional[str] = None


class Config(BaseModel):
    # number of IRs to generate
    n_rirs: int = 1
    # size of a subset of rirs that will be saved in the same file
    batch_size: int = 1
    # length of the IRs in samples
    ir_len: int = 96000
    # sampling rate of the IRs
    fs: int = 48000
    # number of modes to synthesize
    n_modes: Optional[int] = None
    # center frequnecy of the band. If none is provided, it is assumed to have homogeneous attenuation
    f_bands: Optional[
        List[float]] = None  # [125, 250, 500, 1000, 2000, 4000, 8000]
    # decay time values limits, of num_freq_bands x num_rooms
    t_vals: List[List[float]] = [[0.1], [3.5]]
    # amplitude values limits (linear scale)
    a_vals_lims: List[float] = [10**(-3. / 10), 1]
    # number of common slopes
    n_slopes: int = 1
    # type of synthesis to use
    synthesis_type: str = "wgn"  # 'modal' or 'wgn'
    # output directory
    output_dir: str = "output"
    # if amplitudes will be generated based on geometry
    amp_gen_config: AmpGenerationConfig = AmpGenerationConfig()
    # room geometry of the coupled space
    room_geom_config: Optional[RoomGeometryConfig] = None

    @model_validator(mode="after")
    @classmethod
    def check_config_dict(cls, model):
        """Check of the room geometry makes sense"""
        num_rooms = model.room_geom_config.num_rooms
        if num_rooms != model.n_slopes:
            raise ValueError(
                "Number of rooms must be equal to the number of slopes")

        if len(model.t_vals) != model.n_slopes:
            raise ValueError(
                "Length of specified T60 must match number of slopes")

        if model.batch_size > model.n_rirs:
            raise ValueError(
                f"Batch size {model.batch_size} should be smaller than the number of rirs {model.n_rirs}"
            )

        num_dims = len(model.room_geom_config.room_dims)
        num_start_coords = len(model.room_geom_config.start_coordinates)
        assert num_dims == num_start_coords == num_rooms, \
                "Room dimensions and start coordinates must be equal to number of rooms"

        return model

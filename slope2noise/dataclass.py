from dataclasses import dataclass
from typing import List, Optional, Union
from numpy.typing import ArrayLike, NDArray


@dataclass
class CommonSlopesRIR():
    ''' 
    dataclass for the synthesized RIRs using shaped wgn and common slope parameters
    '''
    room_dims: List
    room_start_coords: List
    source_locs: Union[ArrayLike, NDArray]
    receiver_locs: NDArray
    n_slopes: int
    a_vals: NDArray
    t_vals: NDArray
    rir: NDArray
    sample_rate: float
    aperture_coords: Optional[List] = None
    batch_id: Optional[int] = None
    f_bands: Optional[ArrayLike] = None


@dataclass
class CommonSlopesRIRSimple():
    ''' 
    dataclass for the synthesized RIRs using shaped wgn and common slope parameters
    '''
    n_slopes: int
    a_vals: NDArray
    t_vals: NDArray
    rir: NDArray
    sample_rate: float
    batch_id: Optional[int] = None
    f_bands: Optional[ArrayLike] = None
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class CommonSlopesRIR():
    ''' 
    dataclass for the synthesized RIRs using shaped wgn and common slope parameters
    '''
    source_locs: tuple
    receiver_locs: tuple
    n_slopes: int
    a_vals: np.ndarray
    t_vals: np.ndarray
    rir: np.ndarray
    batch_id: Optional[int] = None

def sample_room_interior(room_dim, n_locs, lim=0.1):
    '''
    sample coordinates of a point inside the room, with a minimum distance lim from the walls
    '''
    x = np.random.uniform(lim, room_dim[0]-lim, size=(n_locs,1))
    y = np.random.uniform(lim, room_dim[1]-lim, size=(n_locs,1))
    z = np.random.uniform(lim, room_dim[2]-lim, size=(n_locs,1))
    coord = np.hstack((x, y, z))
    return np.round(np.multiply(coord,1000))/1000  # round to 3 decimal places

from abc import ABC
import numpy as np
from typing import List
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

@dataclass
class Meshgrid():
    xmesh: NDArray
    ymesh: NDArray
    zmesh: NDArray

class RoomGeometry(ABC):
    def __init__(
            self,
            sample_rate: int,
            num_rooms: int,
            room_dims: List,
            room_start_coord: List):

        self.sample_rate = sample_rate
        self.num_rooms = num_rooms
        self.room_dims = room_dims
        self.room_start_coord = room_start_coord
        self.room_meshgrid = self.get_3D_meshgrid(grid_spacing_m=0.1)

    def get_3D_meshgrid(self, grid_spacing_m: float) -> Meshgrid:
        """
        Return the 3D meshgrid of the room's geometry
        Args:
            grid_spacing_m: spacing for creating the meshgrid
        Returns:
            Tuple : tuple of x, y and z meshes in 3D
        """
        Xcombined = []
        Ycombined = []
        Zcombined = []
        for nroom in range(self.num_rooms):
            num_x_points = int(self.room_dims[nroom][0] / grid_spacing_m)
            num_y_points = int(self.room_dims[nroom][1] / grid_spacing_m)
            num_z_points = int(self.room_dims[nroom][2] / grid_spacing_m)
            x = np.linspace(
                self.room_start_coord[nroom][0],
                self.room_start_coord[nroom][0] + self.room_dims[nroom][0],
                num_x_points)
            y = np.linspace(
                self.room_start_coord[nroom][1],
                self.room_start_coord[nroom][1] + self.room_dims[nroom][1],
                num_y_points)
            z = np.linspace(
                self.room_start_coord[nroom][2],
                self.room_start_coord[nroom][2] + self.room_dims[nroom][2],
                num_z_points)
            (xm, ym, zm) = np.meshgrid(x, y, z)
            Xcombined = np.concatenate((Xcombined, xm.flatten()))
            Ycombined = np.concatenate((Ycombined, ym.flatten()))
            Zcombined = np.concatenate((Zcombined, zm.flatten()))

        return Meshgrid(Xcombined,
                        Ycombined,
                        Zcombined)

    def plot_3D_meshgrid(self):
        """Plot the 3D meshgrid to visualise the room geometry"""
        xmesh = self.room_meshgrid.xmesh
        ymesh = self.room_meshgrid.ymesh
        zmesh = self.room_meshgrid.zmesh

        x_flat = xmesh.flatten()
        y_flat = ymesh.flatten()
        z_flat = zmesh.flatten()

        # Plot using scatter without any additional data for color
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the X, Y, Z points
        ax.scatter(x_flat, y_flat, z_flat, color='b', marker='.')

        # Set the limits for all axes
        ax.set_xlim(0,
                    self.room_dims[-1][0] + self.room_start_coord[-1][0] + 0.5)
        ax.set_ylim(0,
                    self.room_dims[-1][1] + self.room_start_coord[-1][1] + 0.5)
        ax.set_zlim(0, self.room_dims[-1][-1] + 0.5)

        # Set the viewing angle so the origin is in the front bottom-left corner
        # ax.view_init(elev=90, azim=-90)

        # Labels and title
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('3D mesh grid of coupled space')

        # Show the plot
        plt.show()
        return fig
    
    def sample_interior_points(self, n_points=1, i_room = None, lim=0.1) -> List[float]:
        """
        Sample a random point in the interior of the room i_room with a minimum distance lim from the walls
        """
        if i_room is None:
            # sample a random room index 
            i_room = np.random.randint(0, self.num_rooms)
        # Randomly sample a point inside each room's dimensions
        random_x = np.random.uniform(self.room_start_coord[i_room][0] + lim,
                                        self.room_start_coord[i_room][0] + self.room_dims[i_room][0] - lim, (n_points, ))
        random_y = np.random.uniform(self.room_start_coord[i_room][1] + lim,
                                        self.room_start_coord[i_room][1] + self.room_dims[i_room][1] - lim, (n_points, ))
        random_z = np.random.uniform(self.room_start_coord[i_room][2] + lim,
                                        self.room_start_coord[i_room][2] + self.room_dims[i_room][2] - lim, (n_points, ))
        return np.stack((random_x, random_y, random_z), axis=-1)
    
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

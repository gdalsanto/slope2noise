import numpy as np
from numpy.typing import NDArray, ArrayLike
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from dataclasses import dataclass
from typing import Optional, List, Tuple

from .utils import db


@dataclass
class Meshgrid():
    xmesh: NDArray
    ymesh: NDArray
    zmesh: NDArray


class RoomGeometry():

    def __init__(self, sample_rate: int, num_rooms: int, room_dims: List,
                 room_start_coord: List):

        self.sample_rate = sample_rate
        self.num_rooms = num_rooms
        self.room_dims = room_dims
        self.room_start_coord = room_start_coord
        self.room_meshgrid = self.get_3D_meshgrid(grid_spacing_m=0.1)

    @property
    def room_boundaries_2D(self):
        # get (x, y) boundaries of each room
        return [[[self.room_start_coord[i][0], self.room_start_coord[i][1]],
                 [
                     self.room_start_coord[i][0] + self.room_dims[i][0],
                     self.room_start_coord[i][1]
                 ],
                 [
                     self.room_start_coord[i][0] + self.room_dims[i][0],
                     self.room_start_coord[i][1] + self.room_dims[i][1]
                 ],
                 [
                     self.room_start_coord[i][0],
                     self.room_start_coord[i][1] + self.room_dims[i][1]
                 ]] for i in range(self.num_rooms)]

    @property
    def room_midpoint_2D(self):
        # get (x, y) midpoint of each room
        return [(np.array([self.room_start_coord[i][0], self.room_start_coord[i][1]]) + \
                   np.array([self.room_start_coord[i][0] + self.room_dims[i][0], self.room_start_coord[i][1] + self.room_dims[i][1]]))/2.0 \
                   for i in range(self.num_rooms)]

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

        return Meshgrid(Xcombined, Ycombined, Zcombined)

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

    def sample_interior_points(self,
                               n_points=1,
                               i_room=None,
                               lim=0.1) -> List[float]:
        """
        Sample a random point in the interior of the room i_room with a minimum distance lim from the walls
        """
        if i_room is None:
            # sample a random room index
            i_room = np.random.randint(0, self.num_rooms, size=n_points)
        # Randomly sample a point inside each room's dimensions
        random_x = np.squeeze(
            np.array([
                np.random.uniform(self.room_start_coord[i_room[k]][0] + lim,
                                  self.room_start_coord[i_room[k]][0] +
                                  self.room_dims[i_room[k]][0] - lim,
                                  size=1) for k in range(n_points)
            ]))
        random_y = np.squeeze(
            np.array([
                np.random.uniform(self.room_start_coord[i_room[k]][1] + lim,
                                  self.room_start_coord[i_room[k]][1] +
                                  self.room_dims[i_room[k]][1] - lim,
                                  size=1) for k in range(n_points)
            ]))
        random_z = np.squeeze(
            np.array([
                np.random.uniform(self.room_start_coord[i_room[k]][2] + lim,
                                  self.room_start_coord[i_room[k]][2] +
                                  self.room_dims[i_room[k]][2] - lim,
                                  size=1) for k in range(n_points)
            ]))
        return np.stack((random_x, random_y, random_z), axis=-1)

    @staticmethod
    def get_euclidean_distance(point1: NDArray, point2: NDArray,
                               ax: int) -> ArrayLike:
        """Get the euclidean distance between a set of points"""
        return np.sqrt(np.sum((point1 - point2)**2, axis=ax))

    @staticmethod
    def point_to_room_distance(P: ArrayLike, room_bounds: List) -> float:
        """
        Get the distance from a point to a room edge, if the point
        is inside the room, the distance is 1e-3 to avoid overflow. Works only in 2D for now.
        Args:
            P (ArrayLike): 2D point
            room_bounds (List): [min_x, max_x, min_y, max_y] corresponding to 2D room bounds
        """
        Px, Py = P
        x_min, x_max, y_min, y_max = room_bounds

        # Distance in the x direction
        dx = max(x_min - Px, 0, Px - x_max)

        # Distance in the y direction
        dy = max(y_min - Py, 0, Py - y_max)

        # Overall distance (Euclidean distance)
        dist = np.sqrt(dx**2 + dy**2)
        return 1e-3 if dist <= 1e-3 else dist

    @staticmethod
    def is_point_in_room(corners: List, point: ArrayLike) -> bool:
        """
        Check if a point is inside a quadrilateral.

        Parameters:
        - corners (list of tuples): List of 4 (x, y) coordinates for the quadrilateral.
        - point (tuple): The (x, y) coordinate of the point to check.

        Returns:
        - bool: True if the point is inside the quadrilateral, False otherwise.
        """

        def cross_product(A, B, P):
            # Compute the 2D cross product of vectors AB and AP
            return (B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1]) * (P[0] -
                                                                    A[0])

        # Extract the four corners and the point
        A, B, C, D = corners
        P = point

        # Compute the cross products for all four edges
        cross1 = cross_product(A, B, P)
        cross2 = cross_product(B, C, P)
        cross3 = cross_product(C, D, P)
        cross4 = cross_product(D, A, P)

        # Check if all cross products have the same sign
        if (cross1 > 0 and cross2 > 0 and cross3 > 0 and cross4 > 0) or \
           (cross1 < 0 and cross2 < 0 and cross3 < 0 and cross4 < 0):
            return True
        else:
            return False

    def get_amplitude_based_on_position(self,
                                        rec_pos: NDArray,
                                        source_pos: ArrayLike,
                                        mean_amp: Optional[NDArray] = None):
        """
        Get amplitudes corresponding to K common slopes, given the room geometry and the mean 
        of the amplitude distribution
        Args:
            rec_pos (NDArray): Array of receiver positions of size N x 3
            source_pos (ArrayLike): source position of length 3
            mean_amp (Optional, NDArray): mean of the amplitude distribution, of size K x n_slopes
        """

        if mean_amp is None:
            mean_amp = np.ones(self.num_rooms)
        else:
            assert len(mean_amp) == self.num_rooms

        num_rec = rec_pos.shape[0]
        rec_pos_2D = np.stack((rec_pos[:, 0], rec_pos[:, 1]), axis=0)
        amplitudes = np.zeros((self.num_rooms, num_rec))

        dist_from_room = np.zeros((self.num_rooms, num_rec))
        dist_from_source = RoomGeometry.get_euclidean_distance(np.repeat(
            source_pos[np.newaxis, :], num_rec, axis=0),
                                                               rec_pos,
                                                               ax=1)

        point_in_room = np.empty((self.num_rooms, num_rec), dtype=bool)
        for k in range(self.num_rooms):
            # find out which room the receiver is in
            point_in_room[k, :] = np.array([
                RoomGeometry.is_point_in_room(self.room_boundaries_2D[k],
                                              rec_pos_2D[:, i])
                for i in range(num_rec)
            ])
            # distance of the receivers from the room
            dist_from_room[k,:] = np.array([RoomGeometry.point_to_room_distance(rec_pos_2D[:, i], \
                                                               [self.room_start_coord[k][0], self.room_start_coord[k][0] + self.room_dims[k][0],
                                                                self.room_start_coord[k][1], self.room_start_coord[k][1] + self.room_dims[k][1]]) \
                                        for i in range(num_rec)])
            # amplitudes depend on 1/r
            amplitudes[k, :] = 1.0 / (
                np.sqrt(dist_from_room[k, :] * dist_from_source) + 1e-12)

        # scale the amplitudes by the mean of the distributions
        for j in range(num_rec):
            which_room = np.argwhere(point_in_room[:, j])[0][0]
            amplitudes[:, j] *= mean_amp[which_room]

        return amplitudes

    def plot_amps_at_receiver_points(self,
                                     rec_pos: NDArray,
                                     source_pos: ArrayLike,
                                     amps: NDArray,
                                     scatter_plot: bool = True,
                                     title: Optional[str] = None,
                                     save_path: Optional[str] = None):
        """
        Plot the amplitudes of the different slopes at specified receiver points.
        Args:
            rec_pos (NDArray): N x 3 array of listener positions in cartesian coordinates
            amps (NDArray): N x num_rooms matrix of amplitudes at specified listener positions
            scatter_plot (bool): whether to plot the discrete amplitudes, 
                                 or interpolate them to be continuous functions of space
        """
        x_rec = rec_pos[:, 0]
        y_rec = rec_pos[:, 1]

        fig, ax = plt.subplots(self.num_rooms, 1, figsize=(6, 8))
        fig.tight_layout()

        if not scatter_plot:
            # Create a grid for the surface
            num_samps = 1000
            x_lin = np.linspace(
                0, self.room_dims[-1][0] + self.room_start_coord[-1][0],
                num_samps)
            y_lin = np.linspace(
                0, self.room_dims[-1][1] + self.room_start_coord[-1][1],
                num_samps)
            x_mesh, y_mesh = np.meshgrid(x_lin, y_lin)

            # Create a mask for values within the limits (so that outside the boundaries the amps are zero)
            mask = []
            combined_mask = np.array([])
            for i in range(self.num_rooms):
                cur_mask = (x_mesh >= self.room_start_coord[i][0]) & (x_mesh <= self.room_dims[i][0] + self.room_start_coord[i][0]) & \
                       (y_mesh >= self.room_start_coord[i][1]) & (y_mesh <= self.room_dims[i][1] + self.room_start_coord[i][1])
                if combined_mask.size == 0:
                    combined_mask = cur_mask
                else:
                    combined_mask = np.logical_or(combined_mask, cur_mask)

        # Plot the X, Y, Z points
        for i in range(self.num_rooms):
            if scatter_plot:
                im = ax[i].scatter(x_rec,
                                   y_rec,
                                   c=db(amps[i, :], is_squared=True))
                # Set the limits for all axes
                ax[i].set_xlim(
                    0,
                    self.room_dims[-1][0] + self.room_start_coord[-1][0] + 0.5)
                ax[i].set_ylim(
                    0,
                    self.room_dims[-1][1] + self.room_start_coord[-1][1] + 0.5)
            else:
                amps_interp = griddata((x_rec, y_rec),
                                       amps[i, :], (x_mesh, y_mesh),
                                       method='cubic')  # Interpolate z value
                # Set values outside the limits to 0
                amps_interp[~combined_mask] = 0  # Apply the mask
                im = ax[i].imshow(db(amps_interp, is_squared=True),
                                  extent=(0, self.room_dims[-1][0] +
                                          self.room_start_coord[-1][0], 0,
                                          self.room_dims[-1][1] +
                                          self.room_start_coord[-1][1]),
                                  origin='lower',
                                  cmap='viridis')
            fig.colorbar(im, ax=ax[i], orientation='vertical')
            ax[i].scatter(source_pos[0],
                          source_pos[1],
                          color='red',
                          marker='x',
                          s=50)
            # Labels and title

            ax[i].set_xlabel('X axis')
            ax[i].set_ylabel('Y axis')
            ax[i].set_title(f'Amplitudes for slope = {i+1} at receiver points')

        # Show the plot
        if title is not None:
            plt.suptitle(title)
        fig.subplots_adjust(hspace=0.2)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


@dataclass
class CommonSlopesRIR():
    ''' 
    dataclass for the synthesized RIRs using shaped wgn and common slope parameters
    '''
    room_dims: List
    room_start_coords: List
    source_locs: ArrayLike
    receiver_locs: NDArray
    n_slopes: int
    a_vals: NDArray
    t_vals: NDArray
    rir: NDArray
    sample_rate: float
    batch_id: Optional[int] = None

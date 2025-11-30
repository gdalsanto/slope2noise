import numpy as np
from numpy.typing import NDArray, ArrayLike
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal.windows import gaussian
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

from .utils import db

scale = 2
plt.rcParams.update({
    'font.size': scale * 8,  # base font size
    'axes.labelsize': scale * 9,  # x/y label
    'xtick.labelsize': scale * 8,
    'ytick.labelsize': scale * 8,
    'legend.fontsize': scale * 8,
    'axes.titlesize': scale * 10,  # usually unused in journal figures
})


@dataclass
class Meshgrid:
    xmesh: NDArray
    ymesh: NDArray
    zmesh: NDArray


class RoomGeometry:

    def __init__(
        self,
        sample_rate: int,
        num_rooms: int,
        room_dims: List,
        room_start_coord: List,
        aperture_coords: Optional[List[List[Tuple]]] = None,
    ):

        self.sample_rate = sample_rate
        self.num_rooms = num_rooms
        self.room_dims = room_dims
        self.room_start_coord = room_start_coord
        self.room_meshgrid = self.get_3D_meshgrid(grid_spacing_m=0.1)
        self.aperture_coords = aperture_coords

    @property
    def room_boundaries_2D(self):
        # get (x, y) boundaries of each room
        return [[
            [self.room_start_coord[i][0], self.room_start_coord[i][1]],
            [
                self.room_start_coord[i][0] + self.room_dims[i][0],
                self.room_start_coord[i][1],
            ],
            [
                self.room_start_coord[i][0] + self.room_dims[i][0],
                self.room_start_coord[i][1] + self.room_dims[i][1],
            ],
            [
                self.room_start_coord[i][0],
                self.room_start_coord[i][1] + self.room_dims[i][1],
            ],
        ] for i in range(self.num_rooms)]

    @property
    def room_midpoint_2D(self):
        # get (x, y) midpoint of each room
        return [(np.array(
            [self.room_start_coord[i][0], self.room_start_coord[i][1]]) +
                 np.array([
                     self.room_start_coord[i][0] + self.room_dims[i][0],
                     self.room_start_coord[i][1] + self.room_dims[i][1],
                 ])) / 2.0 for i in range(self.num_rooms)]

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
                num_x_points,
            )
            y = np.linspace(
                self.room_start_coord[nroom][1],
                self.room_start_coord[nroom][1] + self.room_dims[nroom][1],
                num_y_points,
            )
            z = np.linspace(
                self.room_start_coord[nroom][2],
                self.room_start_coord[nroom][2] + self.room_dims[nroom][2],
                num_z_points,
            )
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
        ax = fig.add_subplot(111, projection="3d")

        # Plot the X, Y, Z points
        ax.scatter(x_flat, y_flat, z_flat, color="b", marker=".")

        # Set the limits for all axes
        ax.set_xlim(0,
                    self.room_dims[-1][0] + self.room_start_coord[-1][0] + 0.5)
        ax.set_ylim(0,
                    self.room_dims[-1][1] + self.room_start_coord[-1][1] + 0.5)
        ax.set_zlim(0, self.room_dims[-1][-1] + 0.5)

        # Set the viewing angle so the origin is in the front bottom-left corner
        # ax.view_init(elev=90, azim=-90)

        # Labels and title
        ax.set_xlabel("X axis (m)")
        ax.set_ylabel("Y axis (m)")
        ax.set_zlabel("Z axis (m)")
        ax.set_title("3D mesh grid of coupled space")

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
                np.random.uniform(
                    self.room_start_coord[i_room[k]][0] + lim,
                    self.room_start_coord[i_room[k]][0] +
                    self.room_dims[i_room[k]][0] - lim,
                    size=1,
                ) for k in range(n_points)
            ]))
        random_y = np.squeeze(
            np.array([
                np.random.uniform(
                    self.room_start_coord[i_room[k]][1] + lim,
                    self.room_start_coord[i_room[k]][1] +
                    self.room_dims[i_room[k]][1] - lim,
                    size=1,
                ) for k in range(n_points)
            ]))
        random_z = np.squeeze(
            np.array([
                np.random.uniform(
                    self.room_start_coord[i_room[k]][2] + lim,
                    self.room_start_coord[i_room[k]][2] +
                    self.room_dims[i_room[k]][2] - lim,
                    size=1,
                ) for k in range(n_points)
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
        if (cross1 > 0 and cross2 > 0 and cross3 > 0
                and cross4 > 0) or (cross1 < 0 and cross2 < 0 and cross3 < 0
                                    and cross4 < 0):
            return True
        else:
            return False

    def get_amplitude_based_on_position(
        self,
        rec_pos: NDArray,
        source_pos: ArrayLike,
        mean_amp: Optional[NDArray] = None,
    ):
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
            dist_from_room[k, :] = np.array([
                RoomGeometry.point_to_room_distance(
                    rec_pos_2D[:, i],
                    [
                        self.room_start_coord[k][0],
                        self.room_start_coord[k][0] + self.room_dims[k][0],
                        self.room_start_coord[k][1],
                        self.room_start_coord[k][1] + self.room_dims[k][1],
                    ],
                ) for i in range(num_rec)
            ])
            # amplitudes depend on 1/r
            amplitudes[k, :] = 1.0 / (
                np.sqrt(dist_from_room[k, :] * dist_from_source) + 1e-12)

        # scale the amplitudes by the mean of the distributions
        for j in range(num_rec):
            which_room = np.argwhere(point_in_room[:, j])[0][0]
            amplitudes[:, j] *= mean_amp[which_room]

        return amplitudes

    def draw_boundaries(self, ax):
        """Draw room boundaries around the 2D amplitude plot"""
        for k in range(self.num_rooms):

            x_start = self.room_start_coord[k][0]
            x_end = self.room_dims[k][0] + self.room_start_coord[k][0]
            y_start = self.room_start_coord[k][1]
            y_end = self.room_dims[k][1] + self.room_start_coord[k][1]
            ax.plot([x_start, x_end], [y_start, y_start],
                    color="k",
                    linestyle="-")
            ax.plot([x_start, x_start], [y_start, y_end],
                    color="k",
                    linestyle="-")
            ax.plot([x_start, x_end], [y_end, y_end], color="k", linestyle="-")
            ax.plot([x_end, x_end], [y_start, y_end], color="k", linestyle="-")

        if self.aperture_coords is not None:

            num_apertures = len(self.aperture_coords)
            for k in range(num_apertures):
                cur_ap_coords = self.aperture_coords[k]
                start_xy = cur_ap_coords[0]
                end_xy = cur_ap_coords[1]
                ax.plot(
                    [start_xy[0], end_xy[0]],
                    [start_xy[1], start_xy[1]],
                    color="w",
                    linestyle="-",
                )
                ax.plot(
                    [start_xy[0], start_xy[0]],
                    [start_xy[1], end_xy[1]],
                    color="w",
                    linestyle="-",
                )
                ax.plot(
                    [start_xy[0], end_xy[0]],
                    [end_xy[1], end_xy[1]],
                    color="w",
                    linestyle="-",
                )
                ax.plot(
                    [end_xy[0], end_xy[0]],
                    [start_xy[1], end_xy[1]],
                    color="w",
                    linestyle="-",
                )
        return ax

    @staticmethod
    def apply_centered_window(matrix,
                              x,
                              y,
                              len_x,
                              len_y,
                              fade_len_samp: Optional[int] = None):
        """
        Applies a 2D window centered at (x, y) with size (len_x, len_y).

        Parameters:
            matrix (ndarray): 2D array where the Hanning window is applied.
            x (int): X-coordinate of the window center.
            y (int): Y-coordinate of the window center.
            len_x (int): Width of the window.
            len_y (int): Height of the  window.
            fade_len_samp (int, optional): how much fade to apply in samples
        Returns:
            ndarray: Matrix with the applied Hanning window.
        """
        h, w = matrix.shape  # Get full matrix size

        # Create fading windows
        if fade_len_samp is None:
            win_x = np.hanning(len_x)
            win_y = np.hanning(len_y)
            win_2D = np.sqrt(np.outer(win_y, win_x))
        else:
            n = np.linspace(start=0, stop=1, num=fade_len_samp)
            fade_out_win = 0.5 - 0.5 * np.cos(np.pi * (n + 1))
            fade_in_win = 0.5 - 0.5 * np.cos(np.pi * n)

            fade_out_win_2D = np.sqrt(np.outer(fade_out_win,
                                               fade_out_win))  # 2D window
            fade_in_win_2D = np.sqrt(np.outer(fade_in_win, fade_in_win))
            win_2D = np.ones((len_y, len_x))
            win_2D[:fade_len_samp, :fade_len_samp] *= fade_in_win_2D
            win_2D[:fade_len_samp, -fade_len_samp:] *= np.flip(fade_in_win_2D,
                                                               axis=1)
            win_2D[-fade_len_samp:, :fade_len_samp] *= np.flip(fade_out_win_2D,
                                                               axis=1)
            win_2D[-fade_len_samp:, -fade_len_samp:] *= fade_out_win_2D

        # Get window boundaries (ensuring they fit within matrix size)
        x_start = max(x - len_x // 2, 0)
        x_end = min(x_start + len_x, w)
        y_start = max(y - len_y // 2, 0)
        y_end = min(y_start + len_y, h)

        # Trim window if it goes out of bounds
        win_2D = win_2D[:y_end - y_start, :x_end - x_start]

        # Apply the Hanning window to the selected region
        matrix[y_start:y_end, x_start:x_end] += win_2D

        return matrix

    def get_2D_matrix_of_amplitudes(
        self,
        rec_pos: NDArray,
        amps: ArrayLike,
        num_samps_x: int = 1000,
        num_samps_y: int = 1000,
        smooth_edges: bool = False,
        plot: bool = False,
        boundary_limits: Optional[Tuple[float, float]] = None,
        grid_spacing_m: Optional[float] = None,
    ):
        """
        Create a uniform 2D grid, and place the amplitudes corresponding to a particular slope in it.
        Blur the edges to prevent high frequency artifacts.
        """
        x_rec = rec_pos[:, 0]
        y_rec = rec_pos[:, 1]

        if boundary_limits is None:
            # set axis limits
            boundaries_list = [[a + b for a, b in zip(sublist1, sublist2)]
                               for sublist1, sublist2 in zip(
                                   self.room_dims, self.room_start_coord)]

            x_lim = max(lst[0] for lst in boundaries_list)
            y_lim = max(lst[1] for lst in boundaries_list)
            num_samps_x = int(x_lim / grid_spacing_m)
            num_samps_y = int(y_lim / grid_spacing_m)
        else:
            (x_lim, y_lim) = boundary_limits

        x_lin = np.linspace(0, x_lim, num_samps_x)
        y_lin = np.linspace(0, y_lim, num_samps_y)
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

        amps_interp = griddata((x_rec, y_rec),
                               amps, (x_mesh, y_mesh),
                               method='cubic')  # Interpolate z value
        # Set values outside the limits to 0
        amps_interp[~combined_mask] = np.nan  # Apply the mask

        # ---- Apply 2D Hanning Window ----
        if smooth_edges:
            win_2D_matrix = np.zeros_like(amps_interp)
            for k in range(self.num_rooms):
                cur_room_midpoint = (np.array(self.room_midpoint_2D[k]) /
                                     grid_spacing_m).astype('int')
                # levaing some extra room for windows to overlap
                cur_room_dims = np.ceil((np.array(self.room_dims[k])) /
                                        grid_spacing_m).astype('int')
                hann_2D_matrix = self.apply_centered_window(
                    win_2D_matrix,
                    cur_room_midpoint[0],
                    cur_room_midpoint[1],
                    cur_room_dims[0],
                    cur_room_dims[1],
                )

            amps_interp *= win_2D_matrix

        if plot:
            fig, ax = plt.subplots(2, 1, figsize=(6, 6))
            fig.tight_layout()
            im = ax[0].imshow(amps_interp,
                              extent=(0, x_lim, 0, y_lim),
                              origin='lower',
                              cmap='viridis')
            fig.colorbar(im, ax=ax[0], orientation='vertical')
            # Labels and title

            ax[0].set_xlabel('X axis(m)')
            ax[0].set_ylabel('Y axis (m)')
            ax[0].set_title('Amplitudes')

            im = ax[1].imshow(win_2D_matrix,
                              extent=(0, x_lim, 0, y_lim),
                              origin='lower',
                              cmap='viridis')
            ax[1].set_title('Window shapes')
            fig.subplots_adjust(hspace=0.3)
            fig.colorbar(im, ax=ax[1], orientation='vertical')

        return (amps_interp, win_2D_matrix) if smooth_edges else amps_interp

    def plot_edc_error_at_receiver_points(
        self,
        rec_pos: NDArray,
        source_pos: ArrayLike,
        edc_error: NDArray,
        scatter_plot: bool = True,
        title: Optional[str] = None,
        cur_freq_hz: Optional[float] = 1000,
        save_path: Optional[str] = None,
    ):
        """Plot the MSE EDC error at different receiver points"""
        # set axis limits
        x_rec = rec_pos[:, 0]
        y_rec = rec_pos[:, 1]

        boundaries_list = [[
            a + b for a, b in zip(sublist1, sublist2)
        ] for sublist1, sublist2 in zip(self.room_dims, self.room_start_coord)]

        x_lim = max(lst[0] for lst in boundaries_list)
        y_lim = max(lst[1] for lst in boundaries_list)

        fig, cur_ax = plt.subplots(1, 1, figsize=(6, 6))
        fig.tight_layout()

        if scatter_plot:
            to_plot = db(edc_error, is_squared=True, min_value=0)
            im = cur_ax.scatter(x_rec, y_rec, c=to_plot, vmin=0, vmax=3.5)
            # Set the limits for all axes
            cur_ax.set_xlim(0, x_lim + 0.5)
            cur_ax.set_ylim(0, y_lim + 0.5)
        else:
            edc_error_interp = self.get_2D_matrix_of_amplitudes(
                rec_pos, edc_error, boundary_limits=(x_lim, y_lim))
            to_plot = db(edc_error_interp, is_squared=True)

            # set NaN values outside boundaries to white
            cmap = plt.cm.viridis.copy()
            cmap.set_bad('white')
            im = cur_ax.imshow(
                to_plot,
                extent=(0, x_lim, 0, y_lim),
                origin='lower',
                vmin=0,
                vmax=max(3.0, np.max(to_plot)),
                cmap=cmap,
            )
        cbar = fig.colorbar(im, ax=cur_ax, orientation='vertical')
        cbar.set_label("dB", fontsize=8 * scale)

        cur_ax.scatter(source_pos[0],
                       source_pos[1],
                       color='red',
                       marker='x',
                       s=50)

        # Labels and title
        cur_ax.set_xlabel('X axis (m)')
        cur_ax.set_ylabel('Y axis (m)')
        if cur_freq_hz is not None:
            cur_ax.set_title(f'{cur_freq_hz:.0f} Hz EDC error')

        cur_ax = self.draw_boundaries(cur_ax)

        # Show the plot
        if title is not None:
            plt.suptitle(title, x=0.5)
        fig.subplots_adjust(hspace=0.3)
        fig.subplots_adjust(left=0.2)
        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.close()
        return fig

    def plot_amps_at_receiver_points(self,
                                     rec_pos: NDArray,
                                     source_pos: ArrayLike,
                                     amps: NDArray,
                                     scatter_plot: bool = True,
                                     title: Optional[str] = None,
                                     save_path: Optional[str] = None,
                                     error_plot: bool = False,
                                     db_limits: Optional[Tuple] = None):
        """
        Plot the amplitudes of the different slopes at specified receiver points.
        Args:
            rec_pos (NDArray): N x 3 array of listener positions in cartesian coordinates
            amps (NDArray): num_rooms x N matrix of amplitudes at specified listener positions
            scatter_plot (bool): whether to plot the discrete amplitudes,
                                 or interpolate them to be continuous functions of space
            cur_freq_hz (optional (float)): band centre frequency in Hz
            error_plot (bool): whether we are plotting amplitudes or their mismatch error
            db_limits (tuple, optional): the limits of the plot in dB
        """
        # set axis limits
        x_rec = rec_pos[:, 0]
        y_rec = rec_pos[:, 1]

        boundaries_list = [[
            a + b for a, b in zip(sublist1, sublist2)
        ] for sublist1, sublist2 in zip(self.room_dims, self.room_start_coord)]

        x_lim = max(lst[0] for lst in boundaries_list)
        y_lim = max(lst[1] for lst in boundaries_list)

        fig, ax = plt.subplots(self.num_rooms,
                               1,
                               figsize=(6, 3 * self.num_rooms))
        fig.tight_layout()
        if db_limits is None:
            db_limits = np.zeros((2, self.num_rooms))
            db_limits[0, :] = np.min(db(amps, is_squared=True))
            db_limits[1, :] = np.max(db(amps, is_squared=True))

        # Plot the X, Y, Z points
        for i in range(self.num_rooms):
            cur_ax = ax if self.num_rooms == 1 else ax[i]
            if scatter_plot:
                im = cur_ax.scatter(x_rec,
                                    y_rec,
                                    c=db(amps[i, :], is_squared=True),
                                    vmin=db_limits[0, i],
                                    vmax=db_limits[0, i])
                # Set the limits for all axes
                cur_ax.set_xlim(0, x_lim + 0.5)
                cur_ax.set_ylim(0, y_lim + 0.5)
            else:
                amps_interp = self.get_2D_matrix_of_amplitudes(
                    rec_pos, amps[i, :], boundary_limits=(x_lim, y_lim))

                # set NaN values outside boundaries to white
                cmap = plt.cm.viridis.copy()
                cmap.set_bad('white')
                im = cur_ax.imshow(
                    db(amps_interp, is_squared=True),
                    extent=(0, x_lim, 0, y_lim),
                    origin='lower',
                    vmin=db_limits[0, i],
                    vmax=db_limits[1, i],
                    cmap=cmap,
                )
            cbar = fig.colorbar(im, ax=cur_ax, orientation='vertical')
            cbar.set_label("dB", fontsize=8 * scale)

            cur_ax.scatter(source_pos[0],
                           source_pos[1],
                           color='red',
                           marker='x',
                           s=50)
            # Labels and title

            cur_ax.set_xlabel('X axis (m)')
            cur_ax.set_ylabel('Y axis (m)')

            cur_ax = self.draw_boundaries(cur_ax)

        # Show the plot
        if title is not None:
            plt.suptitle(title)
        fig.subplots_adjust(hspace=0.3)
        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.close()
        return fig

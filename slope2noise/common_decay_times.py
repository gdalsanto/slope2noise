from typing import Sequence
import numpy as np
from tqdm import trange
from sklearn.cluster import KMeans
import DecayFitNet.python.toolbox.DecayFitNetToolbox as dfn

def compute_common_decay_times(
    srirs_array: np.ndarray,
    fs: int,
    f_bands: Sequence[int],
    model_order: int = 3,
    verbose: bool = False,
):
    """Compute estimated slope parameters and k-means cluster centers per band.

    Parameters
    ----------
    srirs_array : np.ndarray
        Array of SRIRs with shape (n_srirs, srir_length, channels).
    fs : int
        Sampling frequency of the SRIRs.
    f_bands : sequence of int
        List of centre frequencies / bands to analyse (e.g. [125,250,...]).
    model_order : int
        Number of slopes / clusters to estimate.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    dict
        A dictionary containing:
        - T_estimated: shape (N, len(f_bands), model_order)
        - A_estimated: shape (N, len(f_bands), model_order)
        - N_estimated: shape (N, len(f_bands), 1)
        - norm_vals: shape (N, 1, len(f_bands))
        - cluster_labels: shape (N, len(f_bands), model_order)
        - cluster_centers: shape (model_order, len(f_bands))
    """
    N = srirs_array.shape[0]    # number of SRIRs
    L = srirs_array.shape[1]    # length of RIRs

    # initialize toolbox
    dfn_band =  dfn.DecayFitNetToolbox(n_slopes=model_order, sample_rate=fs, filter_frequencies=f_bands)

    # containers
    T_estimated = np.empty((N, len(f_bands), model_order))
    A_estimated = np.empty((N, len(f_bands), model_order))
    N_estimated = np.empty((N, len(f_bands), 1))
    norm_vals = np.empty((N, 1, len(f_bands)))

    # estimate parameters for each RIR
    iterator = trange(N, desc="Estimating parameters") if not verbose else range(N)
    for i in iterator:
        ref_rir = srirs_array[i, :, 0]
        current_estimated_param, current_norm_vals = dfn_band.estimate_parameters(ref_rir, analyse_full_rir=True)
        T_estimated[i, :, :] = current_estimated_param[0]
        A_estimated[i, :, :] = current_estimated_param[1]
        N_estimated[i, :, :] = current_estimated_param[2]
        norm_vals[i, :, :] = current_norm_vals

    # clustering per band
    cluster_labels = np.empty((N, len(f_bands), model_order), dtype=int)
    cluster_centers = np.empty((model_order, len(f_bands)))

    for idx in range(len(f_bands)):
        kmeans = KMeans(n_clusters=model_order )#, random_state=50496)
        T_band = T_estimated[:, idx, :].reshape(-1, 1)
        kmeans.fit(T_band)
        cluster_labels[:, idx, :] = kmeans.labels_.reshape(N, model_order)
        cluster_centers[:, idx] = np.squeeze(kmeans.cluster_centers_)

    return {
        "T_estimated": T_estimated,
        "A_estimated": A_estimated,
        "N_estimated": N_estimated,
        "norm_vals": norm_vals,
        "cluster_labels": cluster_labels,
        "cluster_centers": cluster_centers,
    }

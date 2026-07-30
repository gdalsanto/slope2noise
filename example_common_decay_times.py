from slope2noise.common_decay_times import compute_common_decay_times
import matplotlib.pyplot as plt
import mat73
import numpy as np 

def main():
    # path to the MAT file used in the notebook (change to your location)
    srirs_mat = "/Users/dalsag1/Documents/datasets/Georg_3room_FDTD/srirs.mat"

    # load SRIRs
    if isinstance(srirs_mat, str):
        loaded = mat73.loadmat(srirs_mat)
        if "srirDataset" in loaded:
            srirs = loaded["srirDataset"]
        else:
            # try previous key names used in similar datasets
            srirs = loaded
    elif isinstance(srirs_mat, dict):
        srirs = srirs_mat
    else:
        raise ValueError("srirs_mat must be a path to a .mat file or a dict-like object")

    fs = srirs["fs"].item() if isinstance(srirs["fs"], np.ndarray) else srirs["fs"]
    srirs_array = srirs["srirs"]

    f_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    model_order = 3

    results = compute_common_decay_times(
        srirs_array=srirs_array,
        fs=fs,
        f_bands=f_bands,
        model_order=model_order,
        verbose=True,
    )

    print("Cluster centres shape:", results["cluster_centers"].shape)
    print("Cluster centres (per band):\n", results["cluster_centers"])

    fig, axes = plt.subplots(len(f_bands), 1, figsize=(10, 20))
    T_estimated = results["T_estimated"]
    cluster_centers = results["cluster_centers"]
    # plot histogram for each frequency band
    for idx, ax in enumerate(axes):
        ax.hist(T_estimated[:, idx, :].flatten(), bins=100, alpha=0.75)
        # found with kMeans
        ax.axvline(cluster_centers[0, idx], color='r', linestyle='--', label='kmeans')
        ax.axvline(cluster_centers[1, idx], color='r', linestyle='--')
        ax.axvline(cluster_centers[2, idx], color='r', linestyle='--')
        ax.set_title(f'Histogram of decay times for {f_bands[idx]} Hz')
        ax.set_xlabel('Decay time values')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

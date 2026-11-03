import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def get_roi_boundary(faces, roi_mask):
    """Find mesh edges forming the boundary of an ROI."""

    roi_faces = faces[np.all(roi_mask[faces], axis=1)]

    if len(roi_faces) == 0:
        return np.empty((0, 2), dtype=int)

    edges = np.concatenate([
        roi_faces[:, [0, 1]],
        roi_faces[:, [1, 2]],
        roi_faces[:, [2, 0]],
    ])

    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)

    return unique_edges[counts == 1]


def map_edges_to_flat(boundary_edges, flat_indices, n_vertices):
    """Map original surface edges to flattened surface indices."""

    if len(boundary_edges) == 0:
        return np.empty((0, 2), dtype=int)

    original_to_flat = np.full(n_vertices, -1, dtype=int)
    original_to_flat[flat_indices] = np.arange(len(flat_indices))

    local_edges = original_to_flat[boundary_edges]

    return local_edges[np.all(local_edges >= 0, axis=1)]


def rotate_flatmap(coords, angle):
    """Rotate flatmap coordinates around their centre."""

    angle = np.deg2rad(angle)

    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])

    xy = coords[:, :2]
    centre = xy.mean(axis=0)

    return (xy - centre) @ rotation_matrix.T + centre


def plot_numerosity_flatmaps(
    sgd_params,
    r_squared,
    valid_vertex_indices,
    roi_ids,
    roi_mapping,
    surf_dir,
    lh_roi_id,
    rh_roi_id,
    r2_threshold=0.1,
    num_min=1,
    num_max=7,
    sigma_max=3.5,
    color_vmin=1,
    color_vmax=8.5,
    cmap="hsv",
    lh_rotation=-100,
    rh_rotation=-80,
    hemi_gap=-171,
    background_grey=0.65,
):
    """Plot preferred numerosity on flattened cortical surfaces."""

    surf_dir = Path(surf_dir)

    # --------------------------------------------------
    # Load surfaces
    # --------------------------------------------------

    lh_flat = nib.load(surf_dir / "lh.flat.gii")
    rh_flat = nib.load(surf_dir / "rh.flat.gii")

    lh_coords = lh_flat.darrays[0].data
    lh_faces = lh_flat.darrays[1].data

    rh_coords = rh_flat.darrays[0].data
    rh_faces = rh_flat.darrays[1].data

    lh_flat_indices = np.load(surf_dir / "lh.flat.vertex_indices.npy")
    rh_flat_indices = np.load(surf_dir / "rh.flat.vertex_indices.npy")

    lh_original_coords, lh_original_faces = nib.freesurfer.read_geometry(
        surf_dir / "lh.white"
    )
    rh_original_coords, rh_original_faces = nib.freesurfer.read_geometry(
        surf_dir / "rh.white"
    )

    n_lh = len(lh_original_coords)
    n_rh = len(rh_original_coords)
    n_total = n_lh + n_rh

    # Check surface and ROI alignment
    assert len(roi_ids) == n_total, "ROI labels do not match surface vertices."
    assert len(valid_vertex_indices) == len(sgd_params), (
        "valid_vertex_indices must match the fitted parameter rows."
    )

    # --------------------------------------------------
    # Filter pRF estimates
    # --------------------------------------------------

    preferred_num = np.exp(sgd_params["mu"].to_numpy())
    r2 = np.asarray(r_squared)
    sigma = sgd_params["sigma"].to_numpy()

    valid_prf = (
        (r2 > r2_threshold)
        & (preferred_num >= num_min)
        & (preferred_num <= num_max)
        & (sigma < sigma_max)
        & np.isfinite(preferred_num)
        & np.isfinite(r2)
        & np.isfinite(sigma)
    )

    # --------------------------------------------------
    # Map pRF values to original surfaces
    # --------------------------------------------------

    preferred_num_full = np.full(n_total, np.nan)
    preferred_num_full[valid_vertex_indices[valid_prf]] = preferred_num[valid_prf]

    lh_num_full = preferred_num_full[:n_lh]
    rh_num_full = preferred_num_full[n_lh:]

    # Map values to flattened surfaces
    lh_num_flat = lh_num_full[lh_flat_indices]
    rh_num_flat = rh_num_full[rh_flat_indices]

    print("\nFlatmap coverage")
    print(f"LH: {np.isfinite(lh_num_flat).sum():,} / {len(lh_num_flat):,}")
    print(f"RH: {np.isfinite(rh_num_flat).sum():,} / {len(rh_num_flat):,}")

    # --------------------------------------------------
    # Extract ROI boundaries
    # --------------------------------------------------

    lh_roi_mask = roi_ids[:n_lh] == lh_roi_id
    rh_roi_mask = roi_ids[n_lh:] == rh_roi_id

    lh_boundary = get_roi_boundary(lh_original_faces, lh_roi_mask)
    rh_boundary = get_roi_boundary(rh_original_faces, rh_roi_mask)

    lh_flat_edges = map_edges_to_flat(lh_boundary, lh_flat_indices, n_lh)
    rh_flat_edges = map_edges_to_flat(rh_boundary, rh_flat_indices, n_rh)


    # --------------------------------------------------
    # Rotate and position flatmaps
    # --------------------------------------------------

    lh_rotated = rotate_flatmap(lh_coords, lh_rotation)
    rh_rotated = rotate_flatmap(rh_coords, rh_rotation)

    lh_rotated[:, 0] += hemi_gap
    rh_rotated[:, 0] -= hemi_gap

    # --------------------------------------------------
    # Plot both hemispheres
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(16, 9))

    norm = Normalize(vmin=color_vmin, vmax=color_vmax)
    color_map = plt.get_cmap(cmap)

    hemispheres = [
        (lh_rotated, lh_faces, lh_num_flat, lh_flat_edges),
        (rh_rotated, rh_faces, rh_num_flat, rh_flat_edges),
    ]

    for coords, faces, values, boundary_edges in hemispheres:
        x, y = coords[:, 0], coords[:, 1]

        # Background
        ax.tripcolor(
            x, y, faces,
            facecolors=np.full(len(faces), background_grey),
            cmap="Greys",
            vmin=0,
            vmax=1,
            edgecolors="none",
        )

        # Preferred numerosity
        ax.tripcolor(
            x, y, faces,
            values,
            cmap=color_map,
            norm=norm,
            shading="gouraud",
        )

        # ROI outline
        for v1, v2 in boundary_edges:
            ax.plot(
                [x[v1], x[v2]],
                [y[v1], y[v2]],
                color="black",
                linewidth=1.5,
                zorder=10,
            )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    ax.set_aspect("equal")
    ax.axis("off")

    sm = ScalarMappable(norm=norm, cmap=color_map)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        fraction=0.025,
        pad=0.01,
    )

    cbar.set_label("Preferred numerosity", fontsize=15, labelpad=12)
    cbar.set_ticks(np.arange(1, 8))
    cbar.ax.set_ylim(1, 7)

    # Main title
    fig.suptitle(
        "Preferred numerosity on flatmaps",
        fontsize=17,
        y=0.97,
    )

    # ROI names underneath the main title
    fig.text(
        0.5,
        0.925,
        f"{roi_mapping[lh_roi_id]} and {roi_mapping[rh_roi_id]} in black",
        ha="center",
        fontsize=11,
        color="black",
    )

    # Hemisphere labels
    ax.text(
        0.25, 0.98,
        "Left hemisphere",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
    )

    ax.text(
        0.72, 0.98,
        "Right hemisphere",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
    )
    plt.show()


import numpy as np

from keras import ops
from prfmodel.fitters import GridFitter, LeastSquaresFitter, SGDFitter
from prfmodel.fitters.adapter import Adapter, ParameterTransform
from prfmodel.utils import batched, calculate_r_squared


def detrend_responses(response, n_dct=0.5, chunk_rows=4096):
    """Apply cosine detrending to voxel responses."""

    n_timepoints = response.shape[1]

    time = np.linspace(0, 2 * np.pi, n_timepoints)
    frequencies = np.arange(0, n_dct + 0.25, 0.5)
    trends = np.cos(np.outer(time, frequencies))

    pinv_trends = np.linalg.pinv(trends)
    detrended = np.empty(response.shape, dtype=np.float32)

    for start in range(0, response.shape[0], chunk_rows):
        end = min(start + chunk_rows, response.shape[0])
        chunk = response[start:end].astype(np.float64, copy=False)

        beta = pinv_trends @ chunk.T
        fitted_trend = (trends @ beta).T

        detrended[start:end] = (chunk - fitted_trend).astype(np.float32)

    return detrended


def fit_condition_roi(
    dataset,
    condition,
    prf_model,
    stimulus,
    param_ranges,
    convert_psc,
    roi_index=4,
    grid_batch_size=20,
    ls_batch_size=200,
    prediction_batch_size=100,
):
    """Fit numerosity pRF models to vertices in a selected ROI."""

    print(f"\n{'=' * 50}")
    print(f"Fitting {condition} | ROI index {roi_index}")
    print(f"{'=' * 50}")

    # 1. Select valid volumetric responses
    response_raw = np.asarray(dataset.responses[condition])
    voxel_valid = np.asarray(dataset.valid[condition], dtype=bool).copy()

    voxel_valid &= np.isfinite(response_raw).all(axis=1)

    valid_voxel_indices = np.flatnonzero(voxel_valid)
    response_valid = response_raw[voxel_valid]

    print(f"Valid voxels: {len(valid_voxel_indices):,}")

    # 2. Convert to PSC and detrend
    response_psc = convert_psc(response_valid)

    finite_psc = np.isfinite(response_psc).all(axis=1)

    response_psc = response_psc[finite_psc]
    valid_voxel_indices = valid_voxel_indices[finite_psc]

    response_detrended = detrend_responses(response_psc)

    # Map original voxel indices to rows in the filtered response array
    original_to_valid = np.full(len(voxel_valid), -1, dtype=int)
    original_to_valid[valid_voxel_indices] = np.arange(len(valid_voxel_indices))

    # 3. Select ROI vertices and project voxel responses
    vertex_roi = np.asarray(dataset.vertex_roi_index)
    roi_vertex_indices = np.flatnonzero(vertex_roi == roi_index)

    offsets = dataset.method2_vertex_voxel_offsets
    voxel_rows = dataset.method2_vertex_voxel_rows

    roi_response = np.full(
        (len(roi_vertex_indices), response_detrended.shape[1]),
        np.nan,
        dtype=np.float32,
    )

    for i, vertex in enumerate(roi_vertex_indices):
        start, end = offsets[vertex:vertex + 2]
        rows = voxel_rows[start:end]

        response_rows = original_to_valid[rows]
        response_rows = response_rows[response_rows >= 0]

        if response_rows.size:
            roi_response[i] = response_detrended[response_rows].mean(axis=0)

    vertex_valid_roi = np.isfinite(roi_response).all(axis=1)

    vertex_indices = roi_vertex_indices[vertex_valid_roi]
    vertex_response_roi = roi_response[vertex_valid_roi]

    print(f"Valid ROI vertices: {len(vertex_indices):,}")

    if len(vertex_indices) == 0:
        raise ValueError(f"No valid vertices found for {condition}, ROI {roi_index}.")

    # 4. Grid search
    grid_fitter = GridFitter(
        model=prf_model,
        stimulus=stimulus,
        compile_step=True,
    )

    grid_history, grid_params = grid_fitter.fit(
        data=vertex_response_roi,
        parameter_values=param_ranges,
        batch_size=grid_batch_size,
    )

    # 5. Least-squares refinement
    ls_fitter = LeastSquaresFitter(
        model=prf_model,
        stimulus=stimulus,
    )

    ls_history, ls_params = ls_fitter.fit(
        data=vertex_response_roi,
        parameters=grid_params,
        slope_name="amplitude",
        intercept_name="baseline",
        batch_size=ls_batch_size,
    )

    # 6. SGD refinement
    adapter = Adapter([
        ParameterTransform(
            ["sigma"],
            transform_fun=ops.log,
            inverse_fun=ops.exp,
        )
    ])

    sgd_fitter = SGDFitter(
        model=prf_model,
        stimulus=stimulus,
        adapter=adapter,
        compile_step=True,
    )

    sgd_history, sgd_params = sgd_fitter.fit(
        data=vertex_response_roi,
        init_parameters=ls_params,
    )

    # 7. Generate predictions and calculate R²
    prf_model_batched = batched(prf_model)

    pred_response = np.asarray(
        prf_model_batched(
            stimulus,
            sgd_params,
            batch_size=prediction_batch_size,
        )
    )

    r_squared = calculate_r_squared(
        vertex_response_roi,
        pred_response,
    )

    return {
        "condition": condition,
        "roi_index": roi_index,
        "vertex_indices": vertex_indices,
        "vertex_response": vertex_response_roi,
        "grid_params": grid_params,
        "ls_params": ls_params,
        "sgd_params": sgd_params,
        "pred_response": pred_response,
        "r_squared": r_squared,
        "grid_history": grid_history,
        "ls_history": ls_history,
        "sgd_history": sgd_history,
    }


import numpy as np
import pandas as pd

from scipy.optimize import curve_fit


def bin_preferred_numerosity(data, bin_edges):
    """Calculate mean preferred numerosity and SEM within distance bins."""

    df = pd.DataFrame({
        "distance": data["distance"],
        "preferred_num": data["preferred_num"],
    })

    df["bin"] = pd.cut(
        df["distance"],
        bins=bin_edges,
        include_lowest=True,
    )

    grouped = df.groupby("bin", observed=True)["preferred_num"]

    result = grouped.agg(
        mean="mean",
        std="std",
        count="count",
    ).reset_index()

    result["sem"] = result["std"] / np.sqrt(result["count"])

    result["distance"] = result["bin"].apply(
        lambda interval: (interval.left + interval.right) / 2
    )

    return result.dropna(subset=["mean"])


def log_function(x, a, b, c):
    """Logarithmic function used to model preferred numerosity."""
    return a * np.log(x + c) + b


def fit_distance_numerosity(
    plot_data,
    bin_width=2.0,
    n_bootstrap=1000,
    random_state=42,
):
    """Bin preferred numerosity by distance, fit a logarithmic curve,
    and estimate its 95% bootstrap confidence interval.
    """

    # 1. Create shared distance bins
    max_distance = max(
        np.nanmax(data["distance"])
        for data in plot_data.values()
        if len(data["distance"]) > 0
    )

    bin_edges = np.arange(
        0,
        np.ceil(max_distance / bin_width) * bin_width + bin_width + 0.001,
        bin_width,
    )

    binned_data = {
        condition: bin_preferred_numerosity(data, bin_edges)
        for condition, data in plot_data.items()
    }

    # 2. Fit logarithmic function to Average
    df_average = binned_data["Average"]

    x_average = df_average["distance"].to_numpy()
    y_average = df_average["mean"].to_numpy()

    if len(x_average) < 3:
        raise ValueError("Not enough Average distance bins for a logarithmic fit.")

    fit_params, _ = curve_fit(
        log_function,
        x_average,
        y_average,
        p0=[1, 1, 1],
        bounds=(
            [-np.inf, -np.inf, 0.01],
            [np.inf, np.inf, 100],
        ),
        maxfev=10000,
    )


    # 3. Bootstrap confidence interval
    rng = np.random.default_rng(random_state)

    average = plot_data["Average"]

    x = np.asarray(average["distance"])
    y = np.asarray(average["preferred_num"])

    x_fit = np.linspace(x.min(), x.max(), 200)

    bootstrap_curves = []

    for _ in range(n_bootstrap):

        # Resample vertices with replacement
        indices = rng.integers(0, len(x), len(x))

        bootstrap_data = {
            "distance": x[indices],
            "preferred_num": y[indices],
        }

        bootstrap_binned = bin_preferred_numerosity(
            bootstrap_data,
            bin_edges,
        )

        bx = bootstrap_binned["distance"].to_numpy()
        by = bootstrap_binned["mean"].to_numpy()

        if len(bx) < 3:
            continue

        try:
            params, _ = curve_fit(
                log_function,
                bx,
                by,
                p0=fit_params,
                bounds=(
                    [-np.inf, -np.inf, 0.01],
                    [np.inf, np.inf, 100],
                ),
                maxfev=10000,
            )

            bootstrap_curves.append(log_function(x_fit, *params))

        except (RuntimeError, ValueError):
            continue

    bootstrap_curves = np.asarray(bootstrap_curves)

    if len(bootstrap_curves) == 0:
        raise ValueError("No successful bootstrap fits.")

    lower = np.percentile(bootstrap_curves, 2.5, axis=0)
    upper = np.percentile(bootstrap_curves, 97.5, axis=0)

    print(
        f"\nSuccessful bootstrap fits: "
        f"{len(bootstrap_curves)} / {n_bootstrap}"
    )

    return {
        "bin_edges": bin_edges,
        "binned_data": binned_data,
        "fit_params": fit_params,
        "x_fit": x_fit,
        "bootstrap_curves": bootstrap_curves,
        "lower": lower,
        "upper": upper,
    }

import matplotlib.pyplot as plt


def plot_distance_numerosity(
    binned_data,
    fit_params,
    x_fit,
    lower,
    upper,
    roi_name,
    y_limits=(1, 7)
):
    """Plot binned numerosity preferences, logarithmic fit and bootstrap CI."""

    fig, ax = plt.subplots(figsize=(11, 7))

    # Individual stimulus conditions
    colors = {
        "Area": "tab:blue",
        "Size": "tab:orange",
        "Dens": "tab:green",
        "Circ": "tab:red",
    }

    for condition, color in colors.items():
        df = binned_data[condition]

        ax.plot(
            df["distance"],
            df["mean"],
            marker="o",
            markersize=5,
            linewidth=1.8,
            color=color,
            label=condition,
        )

    # Average binned means and SEM
    df_average = binned_data["Average"]

    ax.errorbar(
        df_average["distance"],
        df_average["mean"],
        yerr=df_average["sem"],
        fmt="o",
        color="black",
        markersize=6,
        capsize=3,
        linewidth=1.5,
        label="Average",
        zorder=5,
    )

    # Logarithmic fit
    ax.plot(
        x_fit,
        log_function(x_fit, *fit_params),
        color="black",
        linewidth=2.5,
        label="Logarithmic fit",
        zorder=4,
    )

    # 95% bootstrap confidence interval
    ax.plot(
        x_fit,
        lower,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )

    ax.plot(
        x_fit,
        upper,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )

    # Formatting
    ax.set_xlabel("Geodesic distance (mm)", fontsize=12)
    ax.set_ylabel("Preferred numerosity", fontsize=12)
    ax.set_title(f"Numerosity Preference along {roi_name}", fontsize=14)
    ax.set_ylim(*y_limits)

    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()

    return fig, ax


import numpy as np


def prepare_distance_data(
    dataset,
    sgd_params,
    r_squared,
    vertex_valid,
    roi_index_valid,
    condition_results,
    roi_index,
    r2_threshold=0.3,
    num_range=(1, 7),
    sigma_max=2.8,
):
    """Prepare filtered pRF parameters and geodesic distances for each condition."""

    # Map fitted Average results to their original surface vertex indices
    valid_vertex_indices = np.flatnonzero(vertex_valid)

    assert len(sgd_params) == len(valid_vertex_indices)
    assert len(roi_index_valid) == len(valid_vertex_indices)
    assert len(r_squared) == len(valid_vertex_indices)

    plot_data = {}

    # Prepare Average and individual conditions using the same criteria
    all_results = {
        "Average": {
            "vertex_indices": valid_vertex_indices,
            "sgd_params": sgd_params,
            "r_squared": r_squared,
            "roi_indices": roi_index_valid,
        },
        **{
            condition: {
                "vertex_indices": result["vertex_indices"],
                "sgd_params": result["sgd_params"],
                "r_squared": result["r_squared"],
                "roi_indices": None,
            }
            for condition, result in condition_results.items()
        },
    }

    for condition, result in all_results.items():
        vertex_indices = result["vertex_indices"]
        params = result["sgd_params"]

        preferred_num = np.exp(params["mu"].to_numpy())
        sigma = params["sigma"].to_numpy()
        r2 = np.asarray(result["r_squared"])

        distances = dataset.roi_distances[roi_index][vertex_indices]

        # Apply the same validity criteria to every condition
        valid = (
            (r2 > r2_threshold)
            & (preferred_num >= num_range[0])
            & (preferred_num <= num_range[1])
            & (sigma < sigma_max)
            & np.isfinite(preferred_num)
            & np.isfinite(sigma)
            & np.isfinite(r2)
            & np.isfinite(distances)
        )

        # Restrict Average to the selected ROI
        if condition == "Average":
            valid &= result["roi_indices"] == roi_index

        plot_data[condition] = {
            "vertex_indices": vertex_indices[valid],
            "distance": distances[valid],
            "preferred_num": preferred_num[valid],
            "sigma": sigma[valid],
            "r_squared": r2[valid],
        }


    return plot_data
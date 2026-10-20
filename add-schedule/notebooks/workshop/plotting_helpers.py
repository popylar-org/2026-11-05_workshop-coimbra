"""Plotting helper functions that are shared by the practicals of the workshop.

The practicals import this module from the directory of the notebook. On Colab, the first cell of each notebook
downloads it next to the notebook.
"""

import warnings
from collections.abc import Callable

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from nilearn.image import new_img_like
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import InMemoryMesh, PolyMesh, vol_to_surf

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#8c8c8c"


# -- Interactive figures ------------------------------------------------------


def make_slider(
    name: str, value: float, min_value: float, max_value: float, step: float, log: bool = False,
) -> widgets.FloatSlider | widgets.FloatLogSlider:
    """Create a slider for a parameter; with `log=True`, the slider moves in steps of `step` in log10 space."""
    if log:
        return widgets.FloatLogSlider(
            value=value,
            min=np.log10(min_value),
            max=np.log10(max_value),
            step=step,
            description=name,
            continuous_update=False,  # Only update the figure when the slider is released
        )
    return widgets.FloatSlider(
        value=value, min=min_value, max=max_value, step=step, description=name, continuous_update=False,
    )


def show_interactive(plot_fn: Callable[..., None], sliders: dict[str, widgets.ValueWidget]) -> None:
    """Show sliders above a figure that `plot_fn` redraws whenever a slider changes."""
    controls = widgets.GridBox(
        list(sliders.values()),
        layout=widgets.Layout(grid_template_columns="repeat(3, max-content)", grid_gap="4px 24px"),
    )
    display(widgets.VBox([controls, widgets.interactive_output(plot_fn, sliders)]))


def to_frame(**params: float) -> pd.DataFrame:
    """Turn parameter values into a parameter DataFrame with a single row."""
    return pd.DataFrame({name: [value] for name, value in params.items()})


def style_axes(ax: plt.Axes, xlabel: str, ylabel: str, grid: bool = True) -> None:
    """Label the axes and make the frame and grid recede behind the data."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)


def add_legend(ax: plt.Axes) -> None:
    """Place the legend above the top right corner of the plot so that it does not cover the data."""
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, borderaxespad=0.2)


# -- Surface plots ------------------------------------------------------------


def make_surface_projection(dataset) -> Callable[[np.ndarray], np.ndarray]:
    """Create a function that projects one value per voxel in the mask of `dataset` onto the surface vertices.

    The projection samples the volume between the pial and the white matter surface with `nilearn.surface.vol_to_surf`.
    Voxels with NaN values are ignored: The function returns the weighted average of the valid voxels at each vertex
    (left hemisphere first) and NaN for vertices without valid voxels.
    """
    # The pial and white matter surfaces are always downloaded together with the dataset
    pial_mesh = PolyMesh(left=dataset.files["pia_lh"], right=dataset.files["pia_rh"])
    wm_mesh = PolyMesh(left=dataset.files["wm_lh"], right=dataset.files["wm_rh"])

    is_in_mask = np.asarray(dataset.mask.dataobj).astype(bool)

    def project_to_surface(voxel_values: np.ndarray) -> np.ndarray:
        """Project one value per voxel in the mask onto the vertices of both hemispheres (left first)."""
        voxel_values = np.asarray(voxel_values)
        is_valid = np.isfinite(voxel_values)
        # Stack the values (zero for invalid voxels) and the weights (one for valid voxels) into a 4D volume
        volume_data = np.zeros((*dataset.mask.shape, 2))
        volume_data[is_in_mask, 0] = np.where(is_valid, voxel_values, 0.0)
        volume_data[is_in_mask, 1] = is_valid
        volume = new_img_like(dataset.mask, volume_data)
        with warnings.catch_warnings():
            # Vertices without any samples inside the mask are NaN; nilearn warns about them
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            surf_values, surf_weights = np.concatenate([
                vol_to_surf(
                    volume,
                    pial_mesh.parts[hemi],
                    inner_mesh=wm_mesh.parts[hemi],
                    mask_img=dataset.mask,  # Ignore samples outside the gray matter mask
                )
                for hemi in ("left", "right")
            ]).T
        # Weighted average of valid voxels; NaN for vertices without valid voxels
        return np.divide(surf_values, surf_weights, out=np.full_like(surf_values, np.nan), where=surf_weights > 0)

    return project_to_surface


def separate_hemispheres(mesh: PolyMesh, gap: float = 10.0) -> PolyMesh:
    """Shift the hemispheres of a surface mesh apart along the left-right axis so that they do not overlap."""
    if set(mesh.parts) != {"left", "right"}:
        return mesh
    left, right = mesh.parts["left"], mesh.parts["right"]
    left_coords = left.coordinates.copy()
    right_coords = right.coordinates.copy()
    left_coords[:, 0] -= left_coords[:, 0].max() + gap / 2
    right_coords[:, 0] -= right_coords[:, 0].min() - gap / 2
    return PolyMesh(
        left=InMemoryMesh(left_coords, left.faces),
        right=InMemoryMesh(right_coords, right.faces),
    )


def downsample_mesh(mesh: InMemoryMesh, cell_size: float) -> tuple[InMemoryMesh, np.ndarray]:
    """Merge all vertices that fall into the same cube of a grid with spacing `cell_size` into a single vertex.

    Returns the downsampled mesh and, for every original vertex, the index of the vertex it was merged into.
    """
    cells = np.floor(mesh.coordinates / cell_size).astype(int)
    _, merged_index = np.unique(cells, axis=0, return_inverse=True)
    merged_index = merged_index.ravel()
    # Each merged vertex lies at the mean position of the vertices that were merged into it
    coords = np.stack(
        [np.bincount(merged_index, weights=dim_coords) for dim_coords in mesh.coordinates.T], axis=1,
    ) / np.bincount(merged_index)[:, np.newaxis]
    faces = merged_index[mesh.faces]
    # Remove faces whose corners were merged into fewer than three vertices and duplicate faces
    is_triangle = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[is_triangle]
    _, first_index = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    return InMemoryMesh(coords, faces[np.sort(first_index)]), merged_index


def downsample_stat_map(stat_map: np.ndarray, merged_index: np.ndarray, cyclic: bool = False) -> np.ndarray:
    """Average the values of the vertices that were merged into the same vertex, ignoring NaN values."""
    if cyclic:
        # Average angles as unit vectors so that angles close to -pi and pi do not cancel out
        return np.arctan2(
            downsample_stat_map(np.sin(stat_map), merged_index),
            downsample_stat_map(np.cos(stat_map), merged_index),
        )
    is_valid = np.isfinite(stat_map)
    num_merged = merged_index.max() + 1
    sums = np.bincount(merged_index[is_valid], weights=stat_map[is_valid], minlength=num_merged)
    counts = np.bincount(merged_index[is_valid], minlength=num_merged)
    return np.divide(sums, counts, out=np.full(num_merged, np.nan), where=counts > 0)


def make_surface_plot(mesh: PolyMesh, cell_size: float = 2.0) -> Callable[..., tuple[plt.Figure, np.ndarray]]:
    """Create a function that plots a stat map with one value per vertex of `mesh` on the surface.

    The hemispheres are moved apart so that they do not overlap, and the surface is downsampled for faster plotting:
    All vertices that fall into the same cube of a grid with spacing `cell_size` (in mm) are merged into a single
    vertex that gets the average value of the merged vertices.
    """
    mesh = separate_hemispheres(mesh)
    hemispheres = [hemi for hemi in ("left", "right") if hemi in mesh.parts]  # Rows of the response are in this order
    num_vertices = {hemi: mesh.parts[hemi].n_vertices for hemi in hemispheres}

    # Downsample each hemisphere for plotting
    downsampled = {hemi: downsample_mesh(mesh.parts[hemi], cell_size=cell_size) for hemi in hemispheres}
    plot_mesh = PolyMesh(**{hemi: plot_part for hemi, (plot_part, _) in downsampled.items()})
    merged_indices = {hemi: merged_index for hemi, (_, merged_index) in downsampled.items()}

    # Views as (hemisphere, elevation, azimuth); medial and lateral views are rotated slightly towards the back of the
    # brain and the posterior view looks slightly upward from below to show more of the occipital cortex
    if len(hemispheres) == 2:
        surf_views = {
            "Left medial": ("left", -10, 330),
            "Posterior": ("both", -20, 270),
            "Right medial": ("right", -10, 210),
        }
    else:
        hemi = hemispheres[0]
        surf_views = {
            f"{hemi.capitalize()} medial": (hemi, -10, 330 if hemi == "left" else 210),
            "Posterior": (hemi, -20, 270),
            f"{hemi.capitalize()} lateral": (hemi, -10, 210 if hemi == "left" else 330),
        }

    def plot_surf_stat_map_helper(
            stat_map: np.ndarray,
            vmin: float | None = None,
            vmax: float | None = None,
            title: str | None = None,
            cmap: str = "inferno",
            cyclic: bool = False,
        ) -> tuple[plt.Figure, np.ndarray]:
        """Helper function to plot a stat map with one value per vertex on the inflated surface.

        The stat map must have one value per vertex. Vertices with NaN values are not colored. Set `cyclic=True` for
        angles in radians (e.g., polar angle) to use the color range [-pi, pi] with the cyclic colormap "hsv" (like
        `prfmodel.plotting.plot_surface_stat_map`). The angles of merged vertices are then averaged as unit vectors.
        """
        stat_map = np.asarray(stat_map, dtype=float)

        if stat_map.shape != (sum(num_vertices.values()),):
            msg = f"Expected one value for each of the {sum(num_vertices.values())} vertices, got shape {stat_map.shape}"
            raise ValueError(msg)

        if cyclic:
            vmin, vmax, cmap = -np.pi, np.pi, "hsv"

        # Use the same color range for all views
        vmin = np.nanmin(stat_map) if vmin is None else vmin
        vmax = np.nanmax(stat_map) if vmax is None else vmax

        # Downsampled stat maps for each hemisphere (left first)
        stat_maps = {}
        start = 0
        for hemi in hemispheres:
            stop = start + num_vertices[hemi]
            stat_maps[hemi] = downsample_stat_map(stat_map[start:stop], merged_indices[hemi], cyclic)
            start = stop
        stat_maps["both"] = np.concatenate([stat_maps[hemi] for hemi in hemispheres])

        fig, axes = plt.subplots(
            1, len(surf_views), subplot_kw={"projection": "3d"}, figsize=(12, 4.5), layout="constrained",
        )

        for ax, (view_name, (hemi, elev, azim)) in zip(axes, surf_views.items()):
            plot_surf_stat_map(
                # nilearn centers the coordinates of a single hemisphere in place, which would undo
                # separate_hemispheres
                plot_mesh if hemi == "both" else InMemoryMesh(
                    plot_mesh.parts[hemi].coordinates.copy(), plot_mesh.parts[hemi].faces,
                ),
                stat_maps[hemi],
                vmin=vmin,
                vmax=vmax,
                symmetric_cbar=False,
                hemi=hemi,
                view=(elev, azim),
                cmap=cmap,
                colorbar=False,
                axes=ax,
                figure=fig,
            )
            ax.set_title(view_name)

        fig.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
            ax=axes,
            shrink=0.6,
        )

        if title is not None:
            fig.suptitle(title)

        return fig, axes

    return plot_surf_stat_map_helper

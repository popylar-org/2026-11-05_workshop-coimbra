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


def get_color_range(
    values: np.ndarray, vmin: float | None, vmax: float | None, cmap: str, cyclic: bool,
) -> tuple[float, float, str]:
    """Return the color range and colormap that all views of a plot share.

    With `cyclic=True`, angles in radians use the color range [-pi, pi] with the cyclic colormap "hsv". Otherwise, a
    missing `vmin` or `vmax` is set to the smallest or largest value that is not NaN.
    """
    if cyclic:
        return -np.pi, np.pi, "hsv"
    if (vmin is None or vmax is None) and np.all(np.isnan(values)):
        msg = "Cannot infer the color range because all values are NaN; pass vmin and vmax"
        raise ValueError(msg)
    vmin = np.nanmin(values) if vmin is None else vmin
    vmax = np.nanmax(values) if vmax is None else vmax
    return vmin, vmax, cmap


def plot_surface_views(
    surf_views: dict[str, dict],
    norm: mpl.colors.Normalize,
    cmap: str,
    title: str | None,
    panel_width: float,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot views of a surface side by side with one colorbar for `norm` and `cmap` next to them.

    `surf_views` maps the title of each view to the keyword arguments of `nilearn.plotting.plot_surf_stat_map` for
    this view (including `surf_mesh`).
    """
    fig, axes = plt.subplots(
        1, len(surf_views), subplot_kw={"projection": "3d"}, figsize=(panel_width * len(surf_views), 4.5),
        layout="constrained", squeeze=False,
    )
    axes = axes[0]

    for ax, (view_name, view_kwargs) in zip(axes, surf_views.items()):
        surf_mesh = view_kwargs["surf_mesh"]
        if isinstance(surf_mesh, InMemoryMesh):
            # nilearn centers the coordinates of a single mesh in place, which would, e.g., undo separate_hemispheres
            surf_mesh = InMemoryMesh(surf_mesh.coordinates.copy(), surf_mesh.faces)
        plot_surf_stat_map(**(view_kwargs | {"surf_mesh": surf_mesh}), colorbar=False, axes=ax, figure=fig)
        ax.set_title(view_name)

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.6)

    if title is not None:
        fig.suptitle(title)

    return fig, axes


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

        # Use the same color range for all views
        vmin, vmax, cmap = get_color_range(stat_map, vmin, vmax, cmap, cyclic)

        # Downsampled stat maps for each hemisphere (left first)
        stat_maps = {}
        start = 0
        for hemi in hemispheres:
            stop = start + num_vertices[hemi]
            stat_maps[hemi] = downsample_stat_map(stat_map[start:stop], merged_indices[hemi], cyclic)
            start = stop
        stat_maps["both"] = np.concatenate([stat_maps[hemi] for hemi in hemispheres])

        return plot_surface_views(
            {
                view_name: {
                    "surf_mesh": plot_mesh if hemi == "both" else plot_mesh.parts[hemi],
                    "stat_map": stat_maps[hemi],
                    "vmin": vmin,
                    "vmax": vmax,
                    "symmetric_cbar": False,
                    "hemi": hemi,
                    "view": (elev, azim),
                    "cmap": cmap,
                }
                for view_name, (hemi, elev, azim) in surf_views.items()
            },
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
            cmap=cmap,
            title=title,
            panel_width=4.0,
        )

    return plot_surf_stat_map_helper


# -- Electrode plots ----------------------------------------------------------


def compute_vertex_normals(mesh: InMemoryMesh) -> np.ndarray:
    """Compute the outward unit normal of every vertex as the area-weighted average of the normals of its faces.

    Vertices that do not belong to any face with a non-zero area get a zero vector.
    """
    coords, faces = mesh.coordinates, mesh.faces
    # The cross product of two edges is normal to the face and as long as twice the face area
    face_normals = np.cross(coords[faces[:, 1]] - coords[faces[:, 0]], coords[faces[:, 2]] - coords[faces[:, 0]])
    vertex_normals = np.zeros_like(coords, dtype=float)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    return np.divide(vertex_normals, lengths, out=np.zeros_like(vertex_normals), where=lengths > 0)


def crop_mesh(mesh: InMemoryMesh, center: np.ndarray, half_width: float) -> tuple[InMemoryMesh, np.ndarray]:
    """Only keep the faces of `mesh` whose corners all lie inside a cube around `center`.

    Returns the cropped mesh and, for every vertex of the cropped mesh, the index of the vertex in `mesh`.
    """
    is_inside = np.all(np.abs(mesh.coordinates - center) <= half_width, axis=1)
    faces = mesh.faces[np.all(is_inside[mesh.faces], axis=1)]
    kept_vertices = np.unique(faces)
    new_index = np.full(mesh.n_vertices, -1)
    new_index[kept_vertices] = np.arange(kept_vertices.size)
    return InMemoryMesh(mesh.coordinates[kept_vertices], new_index[faces]), kept_vertices


def make_electrode_plot(
    mesh: InMemoryMesh,
    electrode_vertices: np.ndarray,
    hemi: str,
    sulcal_depth: np.ndarray | None = None,
    zoom_electrodes: np.ndarray | None = None,
    cell_size: float = 2.0,
) -> Callable[..., tuple[plt.Figure, np.ndarray]]:
    """Create a function that plots one value per electrode as markers on the surface of a single hemisphere.

    `electrode_vertices` contains the index of the vertex of `mesh` that lies closest to each electrode. The first
    view shows the lateral side of the hemisphere on a surface that is downsampled like in `make_surface_plot`. If
    `zoom_electrodes` (a boolean mask over the electrodes) is given, a second view zooms in on these electrodes and
    looks at them from the direction that their part of the surface faces. If `sulcal_depth` is given, sulci are shown
    in a darker gray than gyri. Each view only shows the electrodes on the part of the surface that faces the viewer.
    """
    electrode_vertices = np.asarray(electrode_vertices)
    sulcal_depth = None if sulcal_depth is None else np.asarray(sulcal_depth, dtype=float)

    # Place the markers slightly above the surface
    normals = compute_vertex_normals(mesh)[electrode_vertices]
    has_no_normal = ~np.any(normals, axis=1)
    if np.any(has_no_normal):
        # A zero normal never faces the viewer
        msg = f"Not showing {has_no_normal.sum()} electrode(s) that lie on vertices without a surface normal"
        warnings.warn(msg, stacklevel=2)
    marker_coords = mesh.coordinates[electrode_vertices] + 1.5 * normals

    def to_background(depth: np.ndarray | None) -> np.ndarray | None:
        """Show sulci (positive sulcal depth) in a darker gray than gyri."""
        return None if depth is None else np.where(depth > 0, 0.45, 0.2)

    # Each view has an elevation and azimuth, a mesh, a background, the limits of the axes, and a marker size; the
    # lateral view is rotated slightly towards the back of the brain
    lateral_mesh, merged_index = downsample_mesh(mesh, cell_size=cell_size)
    surf_views = {
        "Lateral": {
            "view": (-10, 210 if hemi == "left" else 330),
            "mesh": lateral_mesh,
            "background": None if sulcal_depth is None else to_background(
                downsample_stat_map(sulcal_depth, merged_index),
            ),
            "limits": None,
            "marker_size": 12,
        },
    }

    if zoom_electrodes is not None:
        # Crop the surface at full resolution to a cube around the electrodes, leaving a margin of 5 mm
        zoom_electrodes = np.asarray(zoom_electrodes, dtype=bool)
        zoom_center = marker_coords[zoom_electrodes].mean(axis=0)
        half_width = np.abs(marker_coords[zoom_electrodes] - zoom_center).max() + 5.0
        zoom_mesh, kept_vertices = crop_mesh(mesh, zoom_center, half_width)
        zoom_normal = normals[zoom_electrodes].mean(axis=0)
        zoom_normal /= np.linalg.norm(zoom_normal)
        surf_views["Zoomed in"] = {
            "view": (np.rad2deg(np.arcsin(zoom_normal[2])), np.rad2deg(np.arctan2(zoom_normal[1], zoom_normal[0]))),
            "mesh": zoom_mesh,
            "background": None if sulcal_depth is None else to_background(sulcal_depth[kept_vertices]),
            "limits": np.stack([zoom_center - half_width, zoom_center + half_width], axis=1),
            "marker_size": 25,
        }

    def plot_electrodes(
        values: np.ndarray,
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = None,
        cmap: str = "inferno",
        cyclic: bool = False,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot one value per electrode as colored markers on the surface.

        Electrodes with NaN values are shown as small gray markers. Set `cyclic=True` for angles in radians (e.g.,
        polar angle) to use the color range [-pi, pi] with the cyclic colormap "hsv".
        """
        values = np.asarray(values, dtype=float)

        if values.shape != electrode_vertices.shape:
            msg = f"Expected one value for each of the {electrode_vertices.size} electrodes, got shape {values.shape}"
            raise ValueError(msg)

        # Use the same color range for all views
        vmin, vmax, cmap = get_color_range(values, vmin, vmax, cmap, cyclic)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        is_missing = np.isnan(values)

        with warnings.catch_warnings():
            # nilearn only shows the background together with a stat map, so we plot a stat map without values;
            # nilearn warns that it only contains NaN values
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            fig, axes = plot_surface_views(
                {
                    view_name: {
                        "surf_mesh": surf_view["mesh"],
                        "stat_map": np.full(surf_view["mesh"].n_vertices, np.nan),
                        "bg_map": surf_view["background"],
                        "vmin": 0.0,
                        "vmax": 1.0,
                        "hemi": hemi,
                        "view": surf_view["view"],
                    }
                    for view_name, surf_view in surf_views.items()
                },
                norm=norm,
                cmap=cmap,
                title=title,
                panel_width=5.5,
            )

        for ax, surf_view in zip(axes, surf_views.values()):
            # Move the markers and limits by the same amount that nilearn centered the coordinates by
            offset = surf_view["mesh"].coordinates.mean(axis=0)
            if surf_view["limits"] is not None:
                limits = surf_view["limits"] - offset[:, np.newaxis]
                ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
                ax.set_box_aspect((1, 1, 1), zoom=1.3)

            # Only show electrodes on the part of the surface that faces the viewer
            elev, azim = np.deg2rad(surf_view["view"])
            view_direction = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
            is_visible = normals @ view_direction > 0.0
            if surf_view["limits"] is not None:
                # Hide electrodes outside of the zoomed box, where the surface is cropped
                is_visible &= np.all(
                    (marker_coords >= surf_view["limits"][:, 0]) & (marker_coords <= surf_view["limits"][:, 1]), axis=1,
                )

            # Draw the markers on top of the surface instead of sorting them by depth
            ax.computed_zorder = False
            marker_size = surf_view["marker_size"]
            ax.scatter(
                *(marker_coords[is_visible & is_missing] - offset).T,
                s=marker_size / 3,
                color=GRAY,
                depthshade=False,
                zorder=2,
            )
            ax.scatter(
                *(marker_coords[is_visible & ~is_missing] - offset).T,
                c=values[is_visible & ~is_missing],
                s=marker_size,
                cmap=cmap,
                norm=norm,
                edgecolors="black",
                linewidths=0.3,
                depthshade=False,
                zorder=3,
            )

        return fig, axes

    return plot_electrodes

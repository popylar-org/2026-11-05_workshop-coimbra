---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: workshop-coimbra (3.12.3)
  language: python
  name: python3
---

# Practical 4: The canonical Gaussian connective field model

This is the notebook for the connective field practical of the workshop. This practical covers the canonical Gaussian connective field (CF) model, its different computational parts, and how to fit it to empirical data from a visual functional magnetic resonance imaging (fMRI) experiment.

**We recommend going through the [first practical](01-prf-visual-fmri.md) on the canonical Gaussian population receptive field (pRF) model first to get familiar with the prfmodel package before starting with this practical.**

The first part of this practical goes through the canonical Gaussian CF model step-by-step. The model input and output as well as every submodel is motivated and explained with interactive visualizations. This part only looks at model prediction but not at fitting the model to data. This is what the second part is about: Fitting the model to an empirical dataset obtained with a 7T fMRI scanner. This part covers data inspection and visualization on an inflated surface of the brain. It goes through two consecutive fitting stages (grid search, least-squares) and concludes with evaluating the model fit and interpreting its parameters.

+++

## Installation and Setup

First, we need to install prfmodel from PyPI. We install the package with the TensorFlow backend (for more information on the different backends, see the [Backends](https://popylar-org.github.io/prfmodel/development/backends.html) section in the package documentation).

**Please make sure to use a GPU instance when installing prfmodel and running this notebook in Colab.**

```{code-cell} ipython3
!pip install "prfmodel[tensorflow]==0.1.0"
```

We set the backend to TensorFlow and a default for printing.

```{code-cell} ipython3
import os

import pandas as pd

# Set keras backend to 'tensorflow' (this is normally the default)
os.environ["KERAS_BACKEND"] = "tensorflow"
# Print parameter DataFrames with three decimals
pd.set_option("display.precision", 3)
```

## The canonical Gaussian CF model

A connective field (CF) model maps the recorded activity of one neuron population (source) to the recorded activity of another
neuron population (target). Both neuron populations can be located in different brain areas (i.e., regions of interest; ROIs). In this example, we use the canonical CF model adapted from Haak et al. (2013) where the CF is defined in the ROI of the source population, for example, V1 in the visual cortex. It is centered on the location of the neuron population whose activity (and the activity of surrounding populations) predicts the activity in the target population best.

In this example, we use V1 as the ROI in which the source populations are located. We map the recorded blood oxygenation level-dependent (BOLD) response from an fMRI experiment of the source populations to *all* other neuron populations in the brain (i.e., one population per vertex on the surface). The BOLD response is recorded while the subject is exposed to an experimental stimulus. In this case, this is a series of bar apertures that move across the visual field on a computer screen. 

+++

### Input: Source region response and distance matrix

The canonical CF model requires two inputs: The recorded activity of all neuron populations in the source ROI and a matrix that contains the distance between the source populations on the surface. Whereas a pRF model is defined in the feature space of the experimental stimulus (e.g., visual space), a CF model is defined on the distance matrix of the source region neuron populations. In the canonical formulation by Haak et al. (2013), the center of a CF must always be located at the known position of a source neuron population on the surface. The distance matrix then indicates how far away other source populations are from the CF center. The distance is used to compute the strength of the CF for all other source populations. Thus, the distance matrix of a CF model has a similar function as the stimulus grid in a pRF model. Because the distance is measured in 2D space, canonical CF models always use a 2D Gaussian to model CFs.

The recorded neural activity in the source ROI acts as the stimulus design matrix of the CF model, which multiplies the CF with the source region activity and sums the product over all source region populations.

+++

In this practical, we use the BOLD response of a single subject who watched a moving bar stimulus in a 7 Tesla MRI scanner. The response was recorded with a repetition time (TR) of 1.5 seconds and is given for every vertex of a standardized cortical surface from the Human Connectome Project (HCP; Van Essen et al., 2012). We load the dataset with the tag `7t-retbar-visual` for both hemispheres together with the standardized surface with the tag `hcp-999999-surface`. The surface comes with an atlas that assigns every vertex to a brain region, which we need to select the source ROI.

```{code-cell} ipython3
from prfmodel.examples import load_dataset

# Downloads on first use and caches in a user data directory (see prfmodel.examples.get_data_dir)
surface = load_dataset("hcp-999999-surface", surface_type="inflated")
dataset = load_dataset("7t-retbar-visual", hemisphere="both")

dataset.response.shape  # One timecourse for every vertex (left hemisphere first)
```

The raw BOLD response has an arbitrary scale that differs between vertices. We convert it to percent signal change (PSC) relative to the mean of each timecourse, so that every timecourse fluctuates around zero.

```{code-cell} ipython3
import numpy as np

response_raw = dataset.response
response_psc = (response_raw / response_raw.mean(axis=1, keepdims=True) - 1.0) * 100.0
```

> **Exercise 1:** Visualize the experimental stimulus of the dataset. The dataset does not contain a ready-made stimulus object, only the raw stimulus design matrix (hint: load it with `scipy.io.loadmat(dataset.files["design"])["stim"]` and plot single time frames with `matplotlib.pyplot.imshow`). What do the dimensions of the design matrix refer to? How many bar passes in which directions does the stimulus design contain?

```{code-cell} ipython3
# Your code goes here
```

We select V1 as the source ROI. The atlas is a multi-modal parcellation of the cortex (Glasser et al., 2016) in which V1 has the label 1. The response of the vertices in V1 is the source region response.

```{code-cell} ipython3
# V1 has the label 1 in the atlas
label_v1 = 1

# Concatenate the labels of both hemispheres (left first)
is_v1 = np.concatenate([surface.atlas["left"] == label_v1, surface.atlas["right"] == label_v1])

source_response = response_psc[is_v1]

source_response.shape  # One timecourse for every V1 vertex
```

The distance matrix contains the distance between every pair of V1 vertices along the cortical surface. We compute it on the white matter surface of each hemisphere: The surface is a mesh of triangles, which we turn into a graph that connects neighboring vertices with edges as long as the distance between them. Dijkstra's algorithm then finds the shortest path between every pair of V1 vertices along the edges of the graph.

```{code-cell} ipython3
import nibabel as nib
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def calculate_distance_matrix(surface_path: str | os.PathLike, source_indices: np.ndarray) -> np.ndarray:
    """Calculate the shortest distance along a surface mesh between every pair of source vertices."""
    coords, triangles = nib.load(surface_path).agg_data()

    # Every triangle has three edges between its corners
    i, j, k = triangles.T
    edges = np.concatenate([np.column_stack((i, j)), np.column_stack((j, k)), np.column_stack((k, i))])
    edge_lengths = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)

    num_vertices = coords.shape[0]
    graph = csr_matrix((edge_lengths, (edges[:, 0], edges[:, 1])), shape=(num_vertices, num_vertices))

    # Only keep the edges between source vertices, so that the shortest paths stay inside the source region
    source_graph = graph[source_indices][:, source_indices]

    return dijkstra(source_graph, directed=False)
```

We compute the distance matrix for each hemisphere separately. A CF cannot extend across hemispheres, so we set the distance between vertices in different hemispheres to infinity.

```{code-cell} ipython3
# Indices of the V1 vertices in each hemisphere
v1_indices_left = np.where(surface.atlas["left"] == label_v1)[0]
v1_indices_right = np.where(surface.atlas["right"] == label_v1)[0]

distance_matrix_left = calculate_distance_matrix(surface.files["wm_lh"], v1_indices_left)
distance_matrix_right = calculate_distance_matrix(surface.files["wm_rh"], v1_indices_right)

# Vertices in different hemispheres are infinitely far apart
num_left, num_right = len(v1_indices_left), len(v1_indices_right)
distance_matrix = np.block([
    [distance_matrix_left, np.full((num_left, num_right), np.inf)],
    [np.full((num_right, num_left), np.inf), distance_matrix_right],
])

distance_matrix.shape  # One row and one column for every V1 vertex
```

We combine the source region response and the distance matrix into a `CFStimulus` object, which is the input of the CF model.

```{code-cell} ipython3
from prfmodel.stimuli import CFStimulus

stimulus = CFStimulus(distance_matrix=distance_matrix, source_response=source_response)

print(stimulus)
```

### Building the model step by step

The canonical Gaussian CF model gets from input (i.e., source region response and distance matrix) to output (i.e., predicted BOLD response) in several steps. Each step is performed by a *submodel* that does a dedicated computation:

1. The *tuning profile* describes which part of the source ROI predicts activity in the target population.
2. The *stimulus encoding* computes the overlap between the tuning profile and the recorded neural activity in the source ROI, which gives the predicted target neuron population activity.
3. The *scaling* adjusts the baseline and amplitude of the predicted BOLD response.

Importantly, the canonical CF model does not have an impulse model to account for the shape of the BOLD measurement. This is because the recorded response in the source region already has the same shape as the response in the target populations.

+++

> **Exercise 2:** Imagine the source response was recorded with a method that does not measure the BOLD response (e.g., electrocorticography). How would the CF model need to be adjusted to account for the difference in measurement?

+++

We build up the model one submodel at a time and then combine all submodels into the full model. To explore how the parameters of each submodel affect its predictions, we use interactive figures.

The first hidden cell defines helper functions for the interactive figures. Some of them are shared by all practicals and are imported from the module `plotting_helpers.py`, which lies next to this notebook.

The figures show V1 as a flat map. Because the cortical surface is curved, we cannot draw V1 on a plane without distorting it. Instead, we arrange the V1 vertices of each hemisphere on a plane so that the distances between them match the distance matrix as closely as possible (with classical multidimensional scaling). On this map, distances are approximately the same as along the cortical surface, so a CF with size `sigma` covers about a circle with radius `sigma`. The horizontal axis of the map runs from the back (posterior) to the front (anterior) of V1 and the vertical axis from its lower (ventral) to its upper (dorsal) part.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Helper functions for the interactive figures { display-mode: "form" }
import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
from plotting_helpers import BLUE, GRAY, ORANGE, add_legend, make_slider, show_interactive, style_axes, to_frame

# Vertices without tuning are light gray, so that the outline of V1 stays visible
TUNING_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "tuning", ["#e6e6e6", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)
SOURCE_CMAP = mpl.colors.LinearSegmentedColormap.from_list("source", ["#f2f2f2", "#000000"])

time_frames = np.arange(stimulus.source_response.shape[1])

# Starting value, minimum, maximum, and step size of the slider for each parameter
PARAM_SLIDERS = {
    "center_index": (448, 0, len(distance_matrix) - 1, 1),
    "sigma": (2.0, 0.5, 20.0, 0.5),
    "baseline": (0.0, -2.0, 2.0, 0.1),
    "amplitude": (1.0, 0.0, 3.0, 0.1),
}
START_VALUES = {name: value for name, (value, *_) in PARAM_SLIDERS.items()}


def make_sliders(*names: str) -> dict[str, widgets.ValueWidget]:
    """Create a slider for each parameter; `center_index` indexes a V1 vertex, so its slider only takes whole numbers."""
    sliders = {}
    for name in names:
        value, min_value, max_value, step = PARAM_SLIDERS[name]
        if name == "center_index":
            sliders[name] = widgets.IntSlider(
                value=value, min=min_value, max=max_value, step=step, description=name, continuous_update=False,
            )
        else:
            sliders[name] = make_slider(name, value, min_value, max_value, step)
    return sliders


def embed_in_plane(distances: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Place vertices on a plane so that the distances between them approximate `distances` (classical MDS).

    The first axis of the plane runs along the longest extent of the vertices. Both axes are flipped to point in the
    same direction as the anterior (y) and superior (z) axes of the vertex coordinates `coords` in the brain.
    """
    num_vertices = distances.shape[0]
    centering = np.eye(num_vertices) - 1.0 / num_vertices
    gram = -0.5 * centering @ np.square(distances) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)  # In ascending order of the eigenvalues
    plane_coords = eigenvectors[:, [-1, -2]] * np.sqrt(eigenvalues[[-1, -2]])
    for axis, brain_axis in enumerate([1, 2]):
        if np.corrcoef(plane_coords[:, axis], coords[:, brain_axis])[0, 1] < 0:
            plane_coords[:, axis] *= -1
    return plane_coords


# Map of V1 with one row of coordinates for every V1 vertex (left hemisphere first); the right hemisphere lies below
v1_maps = {
    "Left V1": embed_in_plane(distance_matrix_left, surface.mesh.parts["left"].coordinates[v1_indices_left]),
    "Right V1": embed_in_plane(distance_matrix_right, surface.mesh.parts["right"].coordinates[v1_indices_right]),
}
v1_maps["Right V1"][:, 1] += v1_maps["Left V1"][:, 1].min() - v1_maps["Right V1"][:, 1].max() - 15.0  # 15 mm gap
v1_map = np.concatenate(list(v1_maps.values()))


def draw_v1_map(ax: plt.Axes, **scatter_kwargs) -> mpl.collections.PathCollection:
    """Draw every V1 vertex as a dot on the map of V1; `scatter_kwargs` set the colors of the dots."""
    points = ax.scatter(v1_map[:, 0], v1_map[:, 1], s=6, linewidths=0, **scatter_kwargs)
    for name, hemi_map in v1_maps.items():
        ax.text(hemi_map[:, 0].min(), hemi_map[:, 1].max() + 3.0, name, color="#595959", fontsize=9)
    # The coordinates of the map have no meaning, only the distances between vertices; a scale bar shows them
    x_start, y_bar = v1_map[:, 0].min(), v1_map[:, 1].min() - 5.0
    ax.plot([x_start, x_start + 10.0], [y_bar, y_bar], color="#595959", linewidth=2)
    ax.text(x_start + 12.0, y_bar, "10 mm", color="#595959", fontsize=9, va="center")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_xlabel("Posterior → anterior")
    ax.set_ylabel("Ventral → dorsal")
    return points


def draw_cf_location(ax: plt.Axes, center_index: int, sigma: float, fill: bool = True) -> None:
    """Draw the CF center and size (a circle with radius `sigma`) on the map of V1."""
    center = v1_map[center_index]
    ax.add_patch(
        mpl.patches.Circle(
            center, sigma, facecolor=BLUE if fill else "none", alpha=0.25 if fill else 1.0, edgecolor=BLUE, linewidth=2,
        )
    )
    ax.plot(*center, "o", color=BLUE, markersize=6, markeredgecolor="white", markeredgewidth=1)
```

### Tuning profiles

In prfmodel, the core of every CF model is called *tuning profile*, just like in pRF models. In the canonical Gaussian CF model, the tuning profile is an isotropic Gaussian density on the cortical surface of the source ROI. It is centered on a source vertex and decreases with the distance from this center vertex along the surface. It describes how strongly the activity of every source neuron population contributes to the activity of the target neuron population.

+++

To illustrate this, we create an instance of a Gaussian CF tuning profile and evaluate it on the distance matrix. The Gaussian CF tuning profile only has two parameters, namely, the index of the source vertex at the CF center (`center_index`; i.e., a row of the distance matrix), and the size defined as the standard deviation (`sigma`, in mm along the surface).

```{code-cell} ipython3
from prfmodel.models.cf import GaussianCFTuning

cf_tuning = GaussianCFTuning()

tuning_params = pd.DataFrame(
    {
        "center_index": [448],
        "sigma": [2.0],
    },
)

# Predict the tuning profile
simulated_tuning_profile = cf_tuning(stimulus, tuning_params)

simulated_tuning_profile.shape  # One value for every V1 vertex
```

The predicted tuning profile assigns a value to every V1 vertex. The figure below shows it on the map of V1 (left) and against the distance of every vertex to the CF center (right). The darkest vertices lie at the CF center (`center_index`), and the spread of the profile is determined by its size (`sigma`). The `center_index` slider moves the CF center through V1: The first vertices lie in the left hemisphere, the remaining ones in the right hemisphere. You can also type a vertex index into the box next to the slider.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: Tuning profile { display-mode: "form" }
def plot_tuning(center_index: int, sigma: float) -> None:
    profile = cf_tuning(stimulus, to_frame(center_index=center_index, sigma=sigma))[0]
    distances = distance_matrix[center_index]
    is_same_hemi = np.isfinite(distances)  # Vertices in the other hemisphere are infinitely far away

    fig, (ax_map, ax_distance) = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1]}, layout="constrained",
    )

    points = draw_v1_map(ax_map, c=profile, cmap=TUNING_CMAP)
    fig.colorbar(points, ax=ax_map, label="Tuning")
    ax_map.set_title("Tuning profile on V1", loc="left")

    ax_distance.plot(distances[is_same_hemi], profile[is_same_hemi], "o", color=BLUE, markersize=3)
    ax_distance.set_xlim(0, 60)
    ax_distance.set_title("Tuning profile by distance", loc="left")
    style_axes(ax_distance, "Distance to the CF center (mm)", "Tuning")

    plt.show()


show_interactive(plot_tuning, make_sliders("center_index", "sigma"))
```

### Stimulus encoding

In CF models, the tuning profile is encoded with the source region response through a dot-product operation, just like the stimulus design matrix in pRF models. At every time frame, this operation multiplies the response of every source vertex with its value in the tuning profile and sums the products over all source vertices. The result is the predicted activity of the target neuron population.

+++

We illustrate this by creating an instance of the CF stimulus encoder.

```{code-cell} ipython3
from prfmodel.models.cf import CFStimulusEncoder

cf_encoder = CFStimulusEncoder()

# We first predict the tuning profile
simulated_tuning_profile = cf_tuning(stimulus, tuning_params)

# Then we stimulus-encode it
simulated_encoded_profile = cf_encoder(stimulus, simulated_tuning_profile, tuning_params)

simulated_encoded_profile.shape  # One value per time frame
```

The figure below shows the source region response at a single time frame on the map of V1 together with the CF (left; the circle has a radius of `sigma`) and the predicted target neuron population activity (right). The `time_frame` slider selects the time frame shown on the left, which is marked with a dot on the right.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: Stimulus encoding { display-mode: "form" }
def predict_encoded(params: pd.DataFrame) -> np.ndarray:
    """Predict the tuning profile and encode it with the source region response."""
    return cf_encoder(stimulus, cf_tuning(stimulus, params), params)[0]


# Use the same color range for all time frames
source_limits = np.percentile(stimulus.source_response, [1, 99])


def plot_encoding(time_frame: int, **params: float) -> None:
    encoded = predict_encoded(to_frame(**params))

    fig, (ax_map, ax_response) = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 2]}, layout="constrained",
    )

    # Vertices with a strong response are dark
    points = draw_v1_map(
        ax_map, c=stimulus.source_response[:, time_frame], cmap=SOURCE_CMAP, vmin=source_limits[0], vmax=source_limits[1],
    )
    draw_cf_location(ax_map, params["center_index"], params["sigma"], fill=False)
    fig.colorbar(points, ax=ax_map, label="BOLD response (PSC)")
    ax_map.set_title(f"Source region response and CF at time frame {time_frame}", loc="left")

    ax_response.plot(time_frames, encoded, color=BLUE, linewidth=2)
    ax_response.axvline(time_frame, color=GRAY, linestyle="--", linewidth=1)
    ax_response.plot(
        time_frame, encoded[time_frame], "o", color=BLUE, markersize=9, markeredgecolor="white", markeredgewidth=2,
    )
    ax_response.set_title("Predicted target neuron population activity", loc="left")
    style_axes(ax_response, "Time frame", "Predicted activity")

    plt.show()


# Start at the time frame where the starting CF predicts the highest activity
start_encoded = predict_encoded(to_frame(center_index=START_VALUES["center_index"], sigma=START_VALUES["sigma"]))
time_frame_slider = widgets.IntSlider(
    value=int(np.argmax(start_encoded)),
    min=0,
    max=len(time_frames) - 1,
    description="time_frame",
    continuous_update=False,
)

show_interactive(plot_encoding, {"time_frame": time_frame_slider, **make_sliders("center_index", "sigma")})
```

### Scaling models

The encoded response has a certain amplitude (i.e., scale) and a baseline. However, the observed BOLD responses of target neuron populations often differ in their amplitude and baseline from the source region response (e.g., due to differences in the vasculature or in the distance to the receiver coil). To account for these differences, a *scaling* submodel can modify the amplitude and baseline of the predicted response, just like in pRF models.

+++

We create an instance of a scaling model and modify the encoded response.

```{code-cell} ipython3
from prfmodel.scaling import BaselineAmplitude

scaling_model = BaselineAmplitude()

scaling_params = pd.DataFrame({"baseline": [0.0], "amplitude": [1.0]})

scaled_response = scaling_model(simulated_encoded_profile, scaling_params)

scaled_response.shape
```

The figure below shows the encoded response before (blue) and after (orange) scaling. The dashed gray line shows the scaled response for the starting parameters.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: Scaling { display-mode: "form" }
def predict_scaled(**params: float) -> tuple[np.ndarray, np.ndarray]:
    """Predict the encoded response before and after scaling."""
    params = to_frame(**{**START_VALUES, **params})
    encoded = predict_encoded(params)
    return encoded, scaling_model(encoded[np.newaxis], params)[0]


_, start_scaled = predict_scaled()


def plot_scaling(**params: float) -> None:
    encoded, scaled = predict_scaled(**params)

    fig, ax = plt.subplots(figsize=(12, 3.8), layout="constrained")

    ax.plot(time_frames, start_scaled, color=GRAY, linestyle="--", linewidth=1.5, label="Starting parameters (scaled)")
    ax.plot(time_frames, encoded, color=BLUE, linewidth=2, label="Before scaling")
    ax.plot(time_frames, scaled, color=ORANGE, linewidth=2, label="After scaling")
    add_legend(ax)
    ax.set_title("Encoded response before and after scaling", loc="left")
    style_axes(ax, "Time frame", "Predicted response")

    plt.show()


show_interactive(plot_scaling, make_sliders("baseline", "amplitude", "sigma"))
```

### The full canonical model

Now that we have seen every submodel, we can combine them into the full CF model. In prfmodel, we call this a *canonical* model because it performs all necessary steps to get from input (i.e., source region response and distance matrix) to output (i.e., predicted BOLD response): tuning profile, stimulus encoding, and scaling. We create an instance of the canonical Gaussian CF model, which uses the submodels from above by default, and make a prediction with all parameters at once.

```{code-cell} ipython3
from prfmodel.models.cf import GaussianCFModel

cf_model = GaussianCFModel()

start_params = pd.DataFrame(
    {
        "center_index": [448],
        "sigma": [2.0],
        "baseline": [0.0],
        "amplitude": [1.0],
    },
)

# Make prediction with CF model
simulated_response = cf_model(stimulus, start_params)

simulated_response.shape  # One predicted timecourse with one value per time frame
```

The figure below shows the CF on the map of V1 (left) and the predicted BOLD response of the full model (right). The dashed gray line shows the prediction for the starting parameters above. Use the sliders to see how all parameters together shape the predicted BOLD response.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: Full model { display-mode: "form" }
start_response = cf_model(stimulus, to_frame(**START_VALUES))[0]


def plot_prediction(**params: float) -> None:
    response = cf_model(stimulus, to_frame(**params))[0]

    fig, (ax_map, ax_response) = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 2]}, layout="constrained",
    )

    draw_v1_map(ax_map, color="#d9d9d9")
    draw_cf_location(ax_map, params["center_index"], params["sigma"])
    ax_map.set_title("CF on V1", loc="left")

    ax_response.plot(time_frames, start_response, color=GRAY, linestyle="--", linewidth=1.5, label="Starting parameters")
    ax_response.plot(time_frames, response, color=BLUE, linewidth=2, label="Current parameters")
    add_legend(ax_response)
    ax_response.set_title("Predicted BOLD response", loc="left")
    style_axes(ax_response, "Time frame", "Predicted BOLD response")

    plt.show()


show_interactive(plot_prediction, make_sliders("center_index", "sigma", "baseline", "amplitude"))
```

## Fitting the model to data

### Inspecting the data

Before we fit the canonical Gaussian CF model to empirical data, we inspect the BOLD response timecourses in percent signal change that we computed at the beginning (`response_psc`). The dataset contains one timecourse for every vertex of the standardized surface in both hemispheres. We start with the source region response, which is the input of the CF model.

+++

> **Exercise 3:** Plot the recorded BOLD response in V1 (hint: use `prfmodel.plotting.plot_response_heatmap`). How does V1 respond to the experimental stimulus?

```{code-cell} ipython3
# Your code goes here
```

To visualize the response on the inflated surface, we create a helper function with `make_surface_plot` from `plotting_helpers.py` like in the first practical. The helper shows three views that focus on the visual cortex in the occipital lobe: the inner (medial) side of each hemisphere, where the calcarine sulcus lies, and both hemispheres from behind (posterior). In contrast to the first practical, the response is already given for every vertex of the surface, so we do not need to project it from a volume onto the surface.

```{code-cell} ipython3
from plotting_helpers import make_surface_plot

plot_surf_stat_map_helper = make_surface_plot(surface.mesh)
```

We plot the standard deviation of each timecourse on the surface to get an overview of the variability in the BOLD timecourses across the surface. Vertices with missing values (`NaN`) are not colored.

```{code-cell} ipython3
# Calculate standard deviation of each timecourse
response_sd = response_psc.std(axis=1)

plot_surf_stat_map_helper(response_sd, vmax=5.0, title="BOLD response standard deviation (in PSC)");
```

To orient ourselves on the surface, we also plot the location of the early visual areas V1 to V4. In the atlas, V2, V3, and V4 have the labels 4, 5, and 6. We will use these areas later to interpret the estimated CF parameters.

```{code-cell} ipython3
# Atlas labels of the early visual areas
visual_area_labels = {"V1": 1, "V2": 4, "V3": 5, "V4": 6}

# Concatenate the labels of both hemispheres (left first)
atlas_labels = np.concatenate([surface.atlas["left"], surface.atlas["right"]])

# Name of the visual area of each vertex; empty for vertices outside of V1 to V4
visual_area = np.full(atlas_labels.shape, "", dtype=object)
for name, label in visual_area_labels.items():
    visual_area[atlas_labels == label] = name

# Number the areas from 1 (V1) to 4 (V4) for plotting
visual_area_number = np.full(atlas_labels.shape, np.nan)
for number, name in enumerate(visual_area_labels, start=1):
    visual_area_number[visual_area == name] = number

fig, _ = plot_surf_stat_map_helper(
    visual_area_number,
    vmin=0.5,
    vmax=4.5,
    cmap=mpl.colors.ListedColormap(["#2a78d6", "#eb6834", "#1baf7a", "#8c5bd6"]),
    title="Visual areas V1 to V4",
)
# Label the colors of the colorbar (the last axes of the figure) with the names of the areas
fig.axes[-1].set_yticks([1, 2, 3, 4], labels=list(visual_area_labels));
```

### Selecting valid vertices

Some vertices do not have a valid timecourse, that is, their timecourse contains missing values (`NaN`). We filter out all vertices with missing values.

```{code-cell} ipython3
response_is_valid = np.all(np.isfinite(response_psc), axis=1)

response_valid = response_psc[response_is_valid]

response_is_valid.mean()  # Fraction of valid vertices
```

### Grid search

We fit the canonical Gaussian CF model that we created above (`cf_model`) to the response of every valid vertex. In contrast to the pRF model in the first practical, we do not need to adjust the model to the TR of the recording: The CF model has no impulse response submodel, and its prediction has the same time frames as the source region response.

We fit the model in two stages. In the first stage, we perform a "brute-force" grid search to find good estimates for the parameters of the CF tuning profile (i.e., `center_index` and `sigma`). For `center_index`, we test every V1 vertex. Because `center_index` is a whole number that selects a row of the distance matrix, the grid search is the only way to estimate it in prfmodel (it cannot be optimized with stochastic gradient descent). The values for `sigma` are log-spaced between 0.5 and 20 mm. A CF with a `sigma` of 0.5 mm practically only includes the center vertex because neighboring vertices are about 1.5 mm apart.

For `baseline` and `amplitude`, we only provide a single value so that they will stay constant across the entire grid.

```{code-cell} ipython3
grid_param_ranges = {
    "center_index": np.arange(len(distance_matrix)),  # Every V1 vertex
    "sigma": np.geomspace(0.5, 20.0, 10),  # Log-spaced (in mm)
    "baseline": [0.0],
    "amplitude": [1.0],
}
```

To run the grid search, we construct the `prfmodel.fitters.GridFitter`. The grid has 29,390 parameter combinations (2,939 V1 vertices times 10 values for `sigma`). We set `batch_size=20` to let the fitter evaluate 20 parameter combinations at the same time which saves us a lot of memory. By default, the `loss` (i.e., the metric the fitter minimizes between model predictions and data) is the negative correlation coefficient, which ignores differences in baseline and amplitude between model predictions and observed data.

```{code-cell} ipython3
from prfmodel.fitters import GridFitter

# Create grid fitter object
grid_fitter = GridFitter(
    model=cf_model,
    stimulus=stimulus,
    compile_step=True,  # Setting 'compile_step=True' speeds up the fitting
)

# Run grid search
grid_history, grid_params = grid_fitter.fit(
    data=response_valid,
    parameter_values=grid_param_ranges,
    batch_size=20,
)
```

We can print the parameters estimated by the grid search.

```{code-cell} ipython3
grid_params
```

Because the grid search can only return values that are in the grid, it is worth checking how the estimates are distributed across the grid. The figure below shows how many vertices have their CF center at each V1 vertex on the map of V1 (left) and the distribution of the CF sizes with `prfmodel.plotting.plot_grid_parameter_distribution` (right). CF sizes at the smallest or largest value of the grid are highlighted in red.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Figure: Distribution of the grid search estimates { display-mode: "form" }
from prfmodel.plotting import plot_grid_parameter_distribution

fig, (ax_map, ax_sigma) = plt.subplots(
    1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.3]}, layout="constrained",
)

# Number of vertices with their CF center at each V1 vertex; a log scale also shows V1 vertices with few counts
center_counts = np.bincount(grid_params["center_index"].astype(int), minlength=len(distance_matrix))
points = draw_v1_map(ax_map, c=center_counts, cmap=TUNING_CMAP, norm=mpl.colors.SymLogNorm(linthresh=1.0, vmin=0.0))
fig.colorbar(points, ax=ax_map, label="Number of vertices")
ax_map.set_title("CF centers", loc="left")

plot_grid_parameter_distribution(grid_params, grid_param_ranges, "sigma", ax=ax_sigma)  # Detects log spacing
ax_sigma.set(xlabel="sigma (in mm)", title="CF sizes")

plt.show()
```

### Least-squares

In the grid search, we ignored the `baseline` and `amplitude` parameters. In prfmodel, we estimate them with least-squares by regressing the observed responses on the predicted CF model responses for each vertex. The intercept and slope of the least-squares regression become the `baseline` and `amplitude` parameter, respectively.

```{code-cell} ipython3
from prfmodel.fitters import LeastSquaresFitter

# Create least-squares fitter
ls_fitter = LeastSquaresFitter(
    model=cf_model,
    stimulus=stimulus,
)

# Run least squares fit
ls_history, ls_params = ls_fitter.fit(
    data=response_valid,
    parameters=grid_params,
    slope_name="amplitude",  # Names of parameters to be optimized with least squares
    intercept_name="baseline",
    batch_size=200,
)
```

We can print the parameter estimates from the grid search together with the least-squares parameter estimates.

```{code-cell} ipython3
ls_params
```

## Evaluating the model fit

Now that the CF model parameters are estimated, we can compare the model predictions against the observed responses. Because we want to make predictions for all vertices at once, we wrap our `cf_model` in the `prfmodel.utils.batched` function. The modifier changes the behavior of the model to make predictions for batches of vertices sequentially. This saves us a lot of memory at the expense of minimal runtime overhead.

```{code-cell} ipython3
from prfmodel.utils import batched

predict_batched = batched(cf_model)

# Make predictions with the estimated parameters
pred_response = np.asarray(predict_batched(stimulus, ls_params, batch_size=200))
```

We can quantify how well the predictions align with the observed timecourses using the R-squared metric, which we compute with `prfmodel.utils.calculate_r_squared`. This metric indicates the proportion of variance in the observed data explained by our model predictions.

```{code-cell} ipython3
from prfmodel.utils import calculate_r_squared

r_squared = calculate_r_squared(response_valid, pred_response)  # One score per vertex
r_squared.shape
```

> **Exercise 4:** Plot the distribution of R-squared values (hint: use `prfmodel.plotting.plot_r_squared_hist`). Why is there a large spike of R-squared values close to 1?

```{code-cell} ipython3
# Your code goes here
```

To plot the R-squared scores on the surface, we need one value for every vertex, including the invalid ones. We define a helper function that fills the values of the valid vertices into an array with one value per vertex and sets the invalid vertices to `NaN`.

```{code-cell} ipython3
def fill_valid_vertices(values: np.ndarray) -> np.ndarray:
    """Fill values for the valid vertices into an array with one value per vertex."""
    values_full = np.full((response_psc.shape[0],), fill_value=np.nan)
    values_full[response_is_valid] = values
    return values_full


plot_surf_stat_map_helper(fill_valid_vertices(r_squared), vmin=0.0, vmax=1.0, title="Variance explained (R-squared)");
```

Vertices far outside the visual cortex, for example, in the motor or auditory cortex, reach R-squared scores of about 0.3, although they do not respond to the visual stimulus. The grid search picks the best of 29,390 predictions for every vertex, so some prediction always matches the noise in a timecourse to some degree. Such vertices often have the smallest CF size in the grid because a CF that only includes a single (noisy) V1 vertex can follow the noise more closely than a larger CF that averages over many V1 vertices.

We also compare the predicted against the observed timecourses for a few vertices with `prfmodel.plotting.plot_observed_predicted`. We pick the vertex with the median R-squared score in each of V2, V3, and V4.

```{code-cell} ipython3
from prfmodel.plotting import plot_observed_predicted

visual_area_valid = visual_area[response_is_valid]

fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, layout="constrained")

for ax, area in zip(axes, ["V2", "V3", "V4"]):
    # Index of the valid vertex with the median R-squared score in the area
    area_indices = np.flatnonzero(visual_area_valid == area)
    vertex_index = area_indices[np.argsort(r_squared[area_indices])[len(area_indices) // 2]]
    plot_observed_predicted(response_valid[vertex_index], pred_response[vertex_index], ax=ax)
    ax.set(xlabel="", ylabel="BOLD response (in PSC)", title=f"{area}: R-squared = {r_squared[vertex_index]:.2f}")

axes[-1].set_xlabel("Time frame (in TR)");
```

> **Exercise 5:** Why does the fit for the vertex in V2 look so different from the fits in V3 and V4? Think about what the response of V1 to the experimental stimulus looks like and what a response in V2-V4 *should* look like.

+++

## Interpreting the model parameters

To analyze and interpret the CF parameters, we zoom in on vertices for which we can trust the CF estimates. We select vertices that fulfill three criteria:

- The R-squared is above 0.5. This threshold is somewhat arbitrary, but it excludes most vertices whose R-squared scores only reflect noise (see above).
- The vertex lies outside V1. V1 vertices are perfectly predicted by their own response (see Exercise 4).
- The CF size is below the largest `sigma` value in the grid. An estimate at the ceiling of the grid suggests that the best fitting CF size lies outside the grid.

```{code-cell} ipython3
center_index = ls_params["center_index"].to_numpy().astype(int)
sigma = ls_params["sigma"].to_numpy()

r_squared_threshold = 0.5

selection_criteria = pd.DataFrame({
    f"r_squared > {r_squared_threshold}": r_squared > r_squared_threshold,
    "outside V1": ~is_v1[response_is_valid],
    # Least squares does not change sigma, so it is always one of the grid values
    "sigma below grid ceiling": ~np.isclose(sigma, np.max(grid_param_ranges["sigma"])),
})

is_selected = selection_criteria.all(axis=1).to_numpy()


def fill_selected(values: np.ndarray) -> np.ndarray:
    """Fill values of the selected vertices into an array with one value per vertex; all others are NaN."""
    return fill_valid_vertices(np.where(is_selected, values, np.nan))


# Fraction of valid vertices that fulfill each criterion and all criteria together
selection_criteria.assign(all_criteria=is_selected).mean()
```

First, we look at the CF size indicated by `sigma` and plot it on the surface.

```{code-cell} ipython3
plot_surf_stat_map_helper(fill_selected(sigma), vmin=0.0, vmax=10.0, title="CF size (sigma, in mm)");
```

Second, to compare the CF sizes between the visual areas, we plot the average `sigma` of the selected vertices in V2, V3, and V4 with `prfmodel.plotting.plot_parameter_by_roi`. The error bars show the standard error of the mean.

```{code-cell} ipython3
from prfmodel.plotting import plot_parameter_by_roi

plot_parameter_by_roi(
    ls_params[is_selected],
    "sigma",
    visual_area_valid[is_selected],
    order=["V2", "V3", "V4"],
    error="sem",
    xlabel="Visual area",
    ylabel="CF size (sigma, in mm)",
    figsize=(4, 4),
);
```

> **Exercise 6:** Interpret the CF size surface plot and target ROI comparison. What does it mean when higher-order visual areas have larger CFs in V1?

+++

> **Exercise 7:** CF models weight the response of all neuron populations in the source region. When these responses substantially differ in scale, this can distort the CF estimates. Instead of expressing the source response in PSC, we can also express it in z-scores. Does this change the CF estimates for this example dataset? Does z-scoring source responses change the interpretation of the CF model parameters in general?

```{code-cell} ipython3
response_z = (response_raw - np.mean(response_raw, axis=1, keepdims=True)) / np.std(response_raw, axis=1, keepdims=True)
response_z.shape
```

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 8:** Are CF models symmetric? Would the CF for V1 $\rightarrow$ V2 be the same as the CF for V2 $\rightarrow$ V1?

+++

> **Exercise 9:** Fit a CF model with V2 as the source ROI and V3 and V4 as the target ROIs. Do you see a similar relationship between CF size and visual hierarchy?

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 10 (advanced):** In their study, Haak et al. (2013) correlate the pRF eccentricity and polar angle of source and target neuron populations in the CF model. Fit a canonical Gaussian 2D pRF model to V1-V4 and correlate the pRF eccentricity and polar angle associated with the CF center of every target population in V2-V4 with its actual pRF eccentricity and polar angle. Do you see similar correlations as reported by Haak et al. (2013)? Hint: The dataset does not contain a ready-made pRF stimulus, so you first need to create a `prfmodel.stimuli.PRFStimulus` from the raw design matrix (see Exercise 1).

```{code-cell} ipython3
# Your code goes here
```

## References

Glasser, M. F., Coalson, T. S., Robinson, E. C., Hacker, C. D., Harwell, J., Yacoub, E., Ugurbil, K., Andersson, J., Beckmann, C. F., Jenkinson, M., Smith, S. M., & Van Essen, D. C. (2016). A multi-modal parcellation of human cerebral cortex. *Nature*, *536*(7615), 171-178. https://doi.org/10.1038/nature18933

Haak, K. V., Winawer, J., Harvey, B. M., Renken, R., Dumoulin, S. O., Wandell, B. A., & Cornelissen, F. W. (2013). Connective field modeling. *NeuroImage*, *66*, 376-384. https://doi.org/10.1016/j.neuroimage.2012.10.037

Van Essen, D. C., Ugurbil, K., Auerbach, E., Barch, D., Behrens, T. E. J., Bucholz, R., Chang, A., Chen, L., Corbetta, M., Curtiss, S. W., Della Penna, S., Feinberg, D., Glasser, M. F., Harel, N., Heath, A. C., Larson-Prior, L., Marcus, D., Michalareas, G., Moeller, S., ... WU-Minn HCP Consortium. (2012). The Human Connectome Project: A data acquisition perspective. *NeuroImage*, *62*(4), 2222-2231. https://doi.org/10.1016/j.neuroimage.2012.02.018

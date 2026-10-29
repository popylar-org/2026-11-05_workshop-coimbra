---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: workshop-coimbra (3.12.3)
  language: python
  name: python3
---

# Practical 5: Population receptive field models for electrocorticography data

This is the notebook for the fifth practical of the workshop. This practical covers how to fit population receptive field (pRF) models to electrocorticography (ECoG) data from a visual experiment.

**We recommend first going through the [first practical](01-prf-visual-fmri.md) because this practical builds on the canonical Gaussian 2D pRF model and the fitting workflow that are covered there. For the last part of this practical, you also need the [second](02-prf-visual-fmri-cv.md) and [third practical](03-prf-visual-fmri-comparison.md) on cross-validation, model comparison, and model selection.**

In the previous practicals, we fitted pRF models to the blood oxygenation level-dependent (BOLD) response measured with functional magnetic resonance imaging (fMRI). The BOLD response is an indirect measure of neural activity because it reflects the change in blood oxygenation that follows neural activity, and it rises and falls slowly over several seconds. ECoG measures the electrical activity of the brain directly with electrodes that are placed on the surface of the cortex. The power of the ECoG signal in high frequencies (broadband power) is closely linked to the spiking activity of the neuron population near an electrode and follows it almost instantly.

The pRF concept does not depend on how we measure neural activity: A neuron population near an electrode still responds to stimulation in a particular part of the visual field. However, the differences between ECoG and fMRI require us to adjust the pRF model and the way we look at the data. This practical first goes through an ECoG example dataset and its experimental stimulus. Then, it adjusts the canonical Gaussian 2D pRF model to ECoG data, fits it with the same stages as in the first practical (grid search, least-squares, and SGD), evaluates the model fit, and interprets the parameters. Finally, it prepares you to compare different pRF models on the ECoG data yourself, following the third practical.

+++

## Contents

- [Installation and Setup](#installation-and-setup)
- [Loading the dataset](#loading-the-dataset)
  - [The stimulus](#the-stimulus)
  - [The broadband response](#the-broadband-response)
  - [The electrodes](#the-electrodes)
- [Adjusting the pRF model to ECoG data](#adjusting-the-prf-model-to-ecog-data)
- [Fitting the model to data](#fitting-the-model-to-data)
  - [Grid search](#grid-search)
  - [Least-squares](#least-squares)
  - [Stochastic gradient descent](#stochastic-gradient-descent)
- [Evaluating the model fit](#evaluating-the-model-fit)
- [Interpreting the model parameters](#interpreting-the-model-parameters)
- [Comparing pRF models on ECoG data](#comparing-prf-models-on-ecog-data)
- [Summary](#summary)
- [References](#references)

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

import numpy as np
import pandas as pd

# Set keras backend to 'tensorflow' (this is normally the default)
os.environ["KERAS_BACKEND"] = "tensorflow"
# Print parameter DataFrames with three decimals
pd.set_option("display.precision", 3)
```

## Loading the dataset

In this practical, we use ECoG data of a single subject from the openly available Visual ECoG dataset (Groen et al., 2024). The data were preprocessed by Yuasa et al. (2025), who compared the spatial tuning of alpha oscillations (8-13 Hz) and broadband power in the visual cortex. We only use the broadband response, which is the mean power between 70 and 180 Hz. The dataset has the tag `visual-ecog-broadband`. We can get a description of the dataset with `prfmodel.examples.describe_dataset`.

```{code-cell} ipython3
from prfmodel.examples import describe_dataset, load_dataset

print(describe_dataset("visual-ecog-broadband"))
```

```{code-cell} ipython3
# Downloads on first use and caches in a user data directory (see prfmodel.examples.get_data_dir)
dataset = load_dataset("visual-ecog-broadband")

print(dataset)
```

The dataset contains the stimulus, the broadband response, and a table that describes the electrodes (`units`). We look at each of them in turn.

+++

### The stimulus

The stimulus can be accessed through the `dataset.stimulus` attribute.

```{code-cell} ipython3
stimulus = dataset.stimulus

print(stimulus)
```

Like in the first practical, the stimulus design matrix has the shape `(num_frames, num_y, num_x)` and indicates which part of the visual field is covered by a bar at each time frame. The grid contains the coordinates of the visual field in degrees of visual angle. The stimulus covers a larger part of the visual field than in the first practical: about $\pm 8.2$ degrees in both directions. We visualize the stimulus as a video.

```{code-cell} ipython3
from IPython.display import HTML
from prfmodel.plotting import animate_2d_prf_stimulus

ani = animate_2d_prf_stimulus(stimulus, interval=50)  # Pause 50 ms between time frames

HTML(ani.to_html5_video())
```

The design of this stimulus differs from the stimulus in the first practical. It consists of four blocks with 56 time frames each. Every block starts with a bar that sweeps across the entire visual field along a cardinal direction (28 time frames). Next, a bar sweeps diagonally from one of the corners towards the center of the visual field, but the sweep stops after 12 time frames before the bar reaches the center. A blank screen is shown for the remaining 16 time frames of the block.

| Block | Time frames | Cardinal sweep | Diagonal sweep starts at |
|---|---|---|---|
| 1 | 0–55 | Vertical bar, left to right | Bottom right |
| 2 | 56–111 | Horizontal bar, top to bottom | Bottom left |
| 3 | 112–167 | Vertical bar, right to left | Top left |
| 4 | 168–223 | Horizontal bar, bottom to top | Top right |

We can see the blocks when we plot at which time frames a bar is visible.

```{code-cell} ipython3
import matplotlib.pyplot as plt

num_frames = stimulus.design.shape[0]
time_frames = np.arange(num_frames)

# A frame shows the bar if any pixel of the design matrix is non-zero
is_bar_visible = stimulus.design.reshape(num_frames, -1).max(axis=1) > 0

fig, ax = plt.subplots(figsize=(10, 2), layout="constrained")
ax.step(time_frames, is_bar_visible, where="mid")
ax.set(xlabel="Time frame", yticks=[0, 1], yticklabels=["Blank", "Bar"]);
```

In an ECoG experiment, a time frame is not tied to a repetition time (TR) of the scanner. Here, each time frame corresponds to one position of the bar, and the response at each time frame summarizes the broadband power while the bar was at this position.

+++

### The broadband response

The broadband response can be accessed through the `dataset.response` attribute. Instead of one timecourse per voxel, it contains one timecourse per electrode (also called *channel*).

```{code-cell} ipython3
response = dataset.response

response.shape  # One timecourse per channel
```

The dataset contains timecourses for 136 channels and 224 time frames, which are averaged over four runs of the experiment. The response is the change in broadband power relative to the power during the blank screen in percent. Thus, the response fluctuates around zero during blank frames and increases when the bar stimulates the pRF of the neuron population near an electrode. Unlike for the BOLD response in the first practical, we do not need to subtract a baseline or convert the response to percent signal change.

+++

> **Exercise 1:** Visualize the broadband response across all channels (hint: use `prfmodel.plotting.plot_response_heatmap`). Can you see the response pattern evoked by the experimental stimulus? What does the response pattern corresponding to the diagonal bar sweeps look like?

<details class="solution">
<summary>Solution</summary>

```python
from prfmodel.plotting import plot_response_heatmap

fig, ax = plot_response_heatmap(
    response,
    vmin=-50,
    vmax=200,
    xlabel="Time frame",
    ylabel="Channel index",
    colorbar_label="Broadband power change (in %)",
    figsize=(8, 5),
)

# Mark the start of each block
for block_start in [56, 112, 168]:
    ax.axvline(block_start, color="white", linestyle="--", linewidth=1)
```

Many channels, especially those of the high-density grid (channel indices 21 and higher), show the same pattern in all four blocks: Short bursts of broadband power during the cardinal sweep (the first 28 time frames of each block), when the bar crosses the pRF of the channel, and no response during the blank period at the end of each block.

The diagonal sweeps (time frames 28–39 of each block) evoke a response that grows towards the end of the sweep, when the bar approaches the center of the visual field, where the pRFs of most channels lie. This response is strongest for the diagonal sweeps that start in the lower corners (blocks 1 and 2), weaker for the sweep that starts in the top left corner (block 3), and almost absent for the sweep that starts in the top right corner (block 4). The electrodes lie on the right hemisphere and mostly over dorsal parts of the visual cortex, which represent the lower left visual field.

</details>

+++

> **Exercise 2:** Why do some channels not respond to the experimental stimulus?

<details class="solution">
<summary>Solution</summary>

An ECoG electrode only records the activity of the neuron population in the cortex directly below it. A channel only responds to the bar if this population responds to visual stimulation in the part of the visual field covered by the stimulus. Channels may not respond because:

- The electrode does not lie over visual cortex. Most of the standard grids and strips lie further to the front and top of the brain, and many channels are labeled `none` in the atlas.
- The pRF of the neuron population lies outside the stimulated part of the visual field (beyond about 8 degrees), or the population prefers other stimuli than a contrast pattern in a moving bar (e.g., objects, faces, or motion in higher visual areas).
- The recording is noisy, for example, because the electrode has poor contact with the cortex.

</details>

+++

> **Exercise 3:** Visualize a single channel with a strong response to the stimulus. Compared to the BOLD response in previous practicals, how does the broadband ECoG response differ?

<details class="solution">
<summary>Solution</summary>

We plot channel 113, which is also used in the manual fit below, and shade the time frames at which the bar is visible.

```python
channel = 113

fig, ax = plt.subplots(figsize=(10, 3), layout="constrained")
ax.plot(time_frames, response[channel], color="black", linewidth=1)
ax.fill_between(
    time_frames,
    0,
    1,
    where=is_bar_visible,
    step="mid",
    color="gray",
    alpha=0.2,
    transform=ax.get_xaxis_transform(),  # Shade the full height of the plot
    label="Bar visible",
)
ax.axhline(0.0, color="gray", linewidth=0.8)
ax.set(xlabel="Time frame", ylabel="Broadband power change (in %)", title=f"Channel {channel}")
ax.legend(frameon=False);
```

Compared to the BOLD response:

- There is no delay: The response peaks while the bar crosses the pRF and not several seconds later.
- The peaks are narrow and only last a few time frames. The response returns to baseline right after the bar has left the pRF, without the slow decay and undershoot of the BOLD response.
- The response is much larger: The broadband power changes by several hundred percent, whereas the BOLD signal changes by a few percent.
- The response during the blank periods is flat and close to zero, so the response to the stimulus stands out clearly from the noise.

</details>

+++

### The electrodes

The table in `dataset.units` describes the electrode of each channel. It has one row per channel in the same order as the rows of `response`.

```{code-cell} ipython3
units = dataset.units

units[["name", "group", "hemisphere", "x", "y", "z", "wangarea"]].head()
```

The column `group` tells us the type of electrode array: Most channels belong to a high-density grid (`HDgrid`) with closely spaced electrodes. The other channels belong to standard grids (`grid`) and strips (`strip`) with larger spacing between electrodes. All electrodes lie on the right hemisphere.

```{code-cell} ipython3
units["group"].value_counts()
```

The columns `x`, `y`, and `z` contain the position of each electrode in millimeters, where `x` points to the right, `y` to the front, and `z` to the top of the head. In the first practical, we projected values per voxel onto an inflated surface of the brain. In ECoG, we only have measurements at the positions of the electrodes, so we plot the value of each channel as a marker at the position of its electrode on the inflated surface.

The Visual ECoG dataset also contains the cortical surfaces of each subject, which were reconstructed with FreeSurfer from an anatomical MRI scan. We download the inflated surface of the right hemisphere (`rh.inflated`) and the sulcal depth (`rh.sulc`), which tells us whether a vertex lies in a sulcus or on a gyrus. Both files together are about 5 MB. The subject has the label `p10` in the dataset (see the column `subject_name`).

```{code-cell} ipython3
from urllib.request import urlretrieve

from prfmodel.examples import get_data_dir

subject = units["subject_name"].iloc[0]

# OpenNeuro addresses a file of a dataset snapshot by its path with the separators replaced by colons
surface_url = (
    "https://openneuro.org/crn/datasets/ds004194/snapshots/3.0.0/files/"
    f"derivatives:freesurfer:sub-{subject}:surf:"
)

# Download on first use and cache next to the prfmodel datasets
surface_dir = get_data_dir() / "visual-ecog-surfaces"
surface_dir.mkdir(parents=True, exist_ok=True)

surface_files = {}
for name in ["rh.inflated", "rh.sulc"]:
    surface_files[name] = surface_dir / name
    if not surface_files[name].exists():
        # Download to a temporary file first so that an interrupted download is not mistaken for a complete one
        partial_file = surface_dir / f"{name}.part"
        urlretrieve(surface_url + name, partial_file)
        partial_file.replace(surface_files[name])

surface_files
```

The column `node_idx` contains the index of the vertex of the cortical surface that lies closest to each electrode. All surfaces of a subject (e.g., the pial and the inflated surface) have the same vertices in the same order, so we can use the index to place each electrode on the inflated surface. The dataset was created with MATLAB, where indices start at 1 instead of 0, so we subtract 1 from the index.

We create the function `plot_electrodes` with `make_electrode_plot` from `plotting_helpers.py`. It shows two views of the right hemisphere: The lateral side of the whole hemisphere and a view that zooms in on the high-density grid. Sulci are shown in a darker gray than gyri, and each view only shows the electrodes on the part of the surface that faces the viewer.

```{code-cell} ipython3
from nilearn.surface import load_surf_data, load_surf_mesh
from plotting_helpers import make_electrode_plot

# Indices in MATLAB start at 1
electrode_vertices = units["node_idx"].to_numpy().astype(int) - 1

plot_electrodes = make_electrode_plot(
    load_surf_mesh(surface_files["rh.inflated"]),
    electrode_vertices,
    hemi="right",
    sulcal_depth=load_surf_data(surface_files["rh.sulc"]),
    zoom_electrodes=(units["group"] == "HDgrid").to_numpy(),  # Zoom in on the high-density grid
)
```

We plot the standard deviation of each timecourse at the electrode positions to see which electrodes record a variable response.

```{code-cell} ipython3
response_sd = response.std(axis=1)

plot_electrodes(response_sd, vmin=0.0, vmax=150.0, title="Broadband response standard deviation");
```

The high-density grid covers the lateral occipital cortex, where many channels have a variable response. The standard grids and strips lie further to the front and top, where the response varies much less. In the zoomed view, the rows of the high-density grid are pulled apart: The electrodes lie on top of the gyri, and the inflated surface unfolds the sulci between them.

The column `wangarea` assigns each electrode to a visual area of the maximum probability atlas by Wang et al. (2015). The atlas predicts the location of visual areas from the anatomy of the brain. Electrodes that lie outside of the areas of the atlas are labeled `none`. The dataset groups the areas of the atlas into regions of interest (ROIs), for example, by merging the dorsal and ventral parts of V2 and V3. The ROI of each channel is stored as an index in `dataset.roi_index` together with a mapping from index to ROI name in `dataset.roi_mapping`.

```{code-cell} ipython3
roi = pd.Series(dataset.roi_index).map(dataset.roi_mapping).to_numpy()

# Number of channels per ROI
pd.Series(roi).value_counts()
```

Most of the channels that lie in visual areas belong to V2, V3, V3b, and LO1. Note that the ROIs are based on an atlas and not on measurements in this subject, so the assignment of individual electrodes can be wrong, especially near the borders between areas.

+++

## Adjusting the pRF model to ECoG data

In the first practical, the canonical Gaussian 2D pRF model made its prediction in four steps: tuning profile, stimulus encoding, impulse response, and scaling. The impulse response accounts for the slow and delayed BOLD response that follows the neural activity. For ECoG data, this step is not needed because the broadband power follows the activity of the neuron population almost instantly, much faster than the duration of a single bar position. Therefore, we skip the impulse response, and the predicted broadband response at time frame $t$ is the scaled stimulus-encoded tuning profile:

$$
\hat{y}(t) = \beta \, P(t; \mu_x, \mu_y, \sigma) + \alpha
$$

Here, $P(t; \mu_x, \mu_y, \sigma)$ is the overlap between the Gaussian tuning profile and the stimulus at time frame $t$ (see the third practical for the equations), $\beta$ is the `amplitude`, and $\alpha$ is the `baseline`. Without the impulse response, the model does not have the `weight_deriv` parameter.

We create the model by setting `impulse_model=None`.

```{code-cell} ipython3
from prfmodel.models.prf import Gaussian2DPRFModel

prf_model = Gaussian2DPRFModel(impulse_model=None)

prf_model.parameter_names
```

Because the model does not have an impulse response, the prediction at each time frame only depends on the stimulus at the same time frame. This means that the order of the time frames does not matter for the prediction, which will be useful at the end of the practical.

Before we let prfmodel fit the model, you can try to fit it yourself. The figure below shows the observed broadband response of channel 113 (gray) together with the prediction of the pRF model (blue). The left panel shows the location and size of the pRF in the visual field. The R-squared in the title shows how much of the variance in the observed response the prediction explains (1 is a perfect fit). The code that creates the figure is hidden. Run the hidden cell to show the figure and move the sliders to change the parameters.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Manual fit? { display-mode: "form" }
import matplotlib as mpl
from plotting_helpers import BLUE, GRAY, add_legend, make_slider, show_interactive, style_axes, to_frame

channel_index = 113  # A channel with a clear response to the stimulus
channel_response = response[channel_index]

# Visual field coordinates of the stimulus grid (dimensions are ordered as y, x)
x_coords = stimulus.grid[0, :, 1]
y_coords = stimulus.grid[:, 0, 0]


def draw_prf_location(ax: plt.Axes, mu_x: float, mu_y: float, sigma: float) -> None:
    """Draw the pRF center and size (a circle with radius `sigma`) in the visual field."""
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    ax.add_patch(
        mpl.patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min, facecolor="#f2f2f2", edgecolor=GRAY, linewidth=1,
        )
    )
    ax.text(x_min, y_max + 0.5, "Stimulus area", color="#595959", fontsize=9)
    ax.add_patch(mpl.patches.Circle((mu_x, mu_y), sigma, facecolor=BLUE, alpha=0.25, edgecolor=BLUE, linewidth=2))
    ax.plot(mu_x, mu_y, "o", color=BLUE, markersize=6)
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal")
    style_axes(ax, "x (degrees visual angle)", "y (degrees visual angle)")


def plot_manual_fit(**params: float) -> None:
    prediction = np.asarray(prf_model(stimulus, to_frame(**params))[0])
    residuals = channel_response - prediction
    r_squared = 1.0 - np.sum(residuals**2) / np.sum((channel_response - channel_response.mean()) ** 2)

    fig, (ax_field, ax_response) = plt.subplots(
        1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 3]}, layout="constrained",
    )

    draw_prf_location(ax_field, params["mu_x"], params["mu_y"], params["sigma"])
    ax_field.set_title("pRF in the visual field")

    ax_response.plot(time_frames, channel_response, color=GRAY, linewidth=1, label="Observed")
    ax_response.plot(time_frames, prediction, color=BLUE, linewidth=2, label="Predicted")
    add_legend(ax_response)
    ax_response.set_title(f"Channel {channel_index}: R-squared = {r_squared:.2f}", loc="left")
    style_axes(ax_response, "Time frame", "Broadband power change (in %)")

    plt.show()


show_interactive(
    plot_manual_fit,
    {
        "mu_x": make_slider("mu_x", 0.0, -8.0, 8.0, 0.1),
        "mu_y": make_slider("mu_y", 0.0, -8.0, 8.0, 0.1),
        "sigma": make_slider("sigma", 1.0, 0.1, 8.0, 0.1),
        "baseline": make_slider("baseline", 0.0, -50.0, 50.0, 1.0),
        "amplitude": make_slider("amplitude", 10.0, 0.0, 50.0, 0.5),
    },
)
```

Note that the amplitude of the broadband response is much larger than the amplitude of the BOLD response in the first practical, so the `amplitude` slider covers a larger range.

+++

## Fitting the model to data

We fit the model in the same three stages as in the first practical: grid search, least-squares, and stochastic gradient descent (SGD). The dataset only contains 136 channels, so fitting is much faster than for the whole-brain fMRI data and we do not need to worry about memory.

+++

### Grid search

We create the values for the grid search with `prfmodel.fitters.grid_values_2d_prf`. Like in the first practical, the values for `mu_x` and `mu_y` span twice the visual field covered by the stimulus (`mu_extent=2.0`), and we use 21 values so that the center and the edges of the stimulus land on grid values. The values for `sigma` are log-spaced between 0.1 and 8 degrees. For `baseline` and `amplitude`, we only provide a single value so that they stay constant across the grid.

```{code-cell} ipython3
from prfmodel.fitters import GridFitter, grid_values_2d_prf

grid_param_ranges = grid_values_2d_prf(
    stimulus,
    num_mu=21,
    mu_extent=2.0,  # Span twice the visual field covered by the stimulus
    num_sigma=15,
    sigma_range=(0.1, 8.0),  # Log-spaced by default
) | {  # Add constant parameters
    "baseline": [0.0],
    "amplitude": [1.0],
}

grid_fitter = GridFitter(
    model=prf_model,
    stimulus=stimulus,
    compile_step=True,  # Setting 'compile_step=True' speeds up the fitting
)

grid_history, grid_params = grid_fitter.fit(
    data=response,
    parameter_values=grid_param_ranges,
    batch_size=20,
)

grid_params
```

The grid search uses the negative correlation as loss, which ignores the scale of the predictions. Thus, `baseline` and `amplitude` keep their constant values.

+++

> **Exercise 4:** Visualize the parameter estimates from the grid search. Do the estimated pRF centers lie within the stimulus grid (hint: use `prfmodel.plotting.plot_2d_prf_centers`)? On which side of the stimulus grid do the estimated pRF centers lie and why? How many channels have parameter estimates sitting on the edge of the parameter grid (hint: use `prfmodel.plotting.plot_grid_parameter_distribution`)?

<details class="solution">
<summary>Solution</summary>

```python
from prfmodel.plotting import plot_2d_prf_centers, plot_grid_parameter_distribution

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")

plot_2d_prf_centers(grid_params, grid_param_ranges, stimulus=stimulus, ax=axes[0])
axes[0].set(xlabel="mu_x (in degrees)", ylabel="mu_y (in degrees)", title="pRF centers")

plot_grid_parameter_distribution(grid_params, grid_param_ranges, "sigma", ax=axes[1])
axes[1].set(xlabel="sigma (in degrees)", title="pRF sizes");
```

```python
# Channels with a parameter at the smallest or largest value of its grid (constant parameters are skipped)
is_on_edge = pd.DataFrame({
    name: np.isclose(grid_params[name], np.min(values)) | np.isclose(grid_params[name], np.max(values))
    for name, values in grid_param_ranges.items()
    if len(values) > 1
})

print(is_on_edge.sum())
print("Channels with at least one parameter on the edge:", is_on_edge.any(axis=1).sum())
```

Most pRF centers lie within the stimulus grid, close to the center of the visual field. Almost all of them lie in the left half and in the lower half of the visual field: All electrodes lie on the right hemisphere, which represents the left (contralateral) visual field, and most of them lie over dorsal visual areas, which represent the lower visual field.

The centers outside the stimulus grid mostly belong to channels that do not respond to the stimulus, for which the grid search fits noise. In total, 29 channels have at least one parameter on the edge of the grid: 22 channels have a `sigma` at the smallest (10) or largest (12) value of the grid, and a few channels have a center on the outer edge of the grid of `mu_x` or `mu_y`.

</details>

+++

### Least-squares

We estimate `baseline` and `amplitude` with least-squares by regressing the observed responses on the predictions with the grid search estimates.

```{code-cell} ipython3
from prfmodel.fitters import LeastSquaresFitter

ls_fitter = LeastSquaresFitter(
    model=prf_model,
    stimulus=stimulus,
)

ls_history, ls_params = ls_fitter.fit(
    data=response,
    parameters=grid_params,
    slope_name="amplitude",  # Names of parameters to be optimized with least squares
    intercept_name="baseline",
)

ls_params
```

### Stochastic gradient descent

Finally, we use SGD to finetune all parameters at once, starting from the least-squares estimates. Like in the first practical, we optimize `sigma` in log space so that it stays positive. Because the dataset is small, we can afford the default number of SGD steps (`num_steps=1000`) and do not need to fix any parameters.

```{code-cell} ipython3
from keras import ops
from prfmodel.fitters import SGDFitter
from prfmodel.fitters.adapter import Adapter, ParameterTransform

sgd_fitter = SGDFitter(
    model=prf_model,
    stimulus=stimulus,
    adapter=Adapter([ParameterTransform(["sigma"], transform_fun=ops.log, inverse_fun=ops.exp)]),
    compile_step=True,
)

sgd_history, sgd_params = sgd_fitter.fit(
    data=response,
    init_parameters=ls_params,
)

sgd_params
```

## Evaluating the model fit

We make predictions with the least-squares and SGD estimates and compute the R-squared of each channel with `prfmodel.utils.calculate_r_squared`. The dataset is small, so we can make predictions for all channels at once.

```{code-cell} ipython3
from prfmodel.utils import calculate_r_squared

pred_response_ls = np.asarray(prf_model(stimulus, ls_params))
pred_response_sgd = np.asarray(prf_model(stimulus, sgd_params))

r_squared_ls = calculate_r_squared(response, pred_response_ls)
r_squared_sgd = calculate_r_squared(response, pred_response_sgd)  # One score per channel
```

> **Exercise 5:** Compare the R-squared between the least-squares and SGD stage (hint: use `prfmodel.plotting.plot_r_squared_comparison`). Does SGD improve the model fit?

<details class="solution">
<summary>Solution</summary>

```python
from prfmodel.plotting import plot_r_squared_comparison

plot_r_squared_comparison(
    r_squared_ls,
    r_squared_sgd,
    value_range=(-0.1, 1.0),
    xlabel="R-squared (least-squares)",
    ylabel="R-squared (SGD)",
)

print("Mean R-squared least-squares:", r_squared_ls.mean())
print("Mean R-squared SGD:", r_squared_sgd.mean())
print("Fraction of channels improved by SGD:", np.mean(r_squared_sgd > r_squared_ls))
```

Yes. SGD improves the R-squared of almost every channel, and it never makes the fit worse: All channels lie on or above the identity line. For most channels the improvement is small (about 0.01), but for a few channels R-squared increases by up to 0.2. The grid search can only return pRF centers and sizes on the grid (the centers are about 1.6 degrees apart), and least-squares only adjusts `baseline` and `amplitude`. SGD can move the center and size between the grid values and adjust all parameters at the same time.

</details>

+++

> **Exercise 6:** Plot the R-squared at the electrode positions on the inflated surface (hint: use `plot_electrodes`). In which brain region does the model fit the observed data the best?

<details class="solution">
<summary>Solution</summary>

```python
plot_electrodes(r_squared_sgd, vmin=0.0, vmax=1.0, title="Variance explained (R-squared)");
```

The model fits best for the channels of the high-density grid at the back of the lateral occipital cortex, which lie in the visual areas V3, V3b, and LO1 (and V2). Here, the model explains up to 90% of the variance. The fit becomes worse for the more anterior channels of the high-density grid (e.g., in TO and IPS), and the model hardly explains any variance for the standard grids and strips outside of the visual cortex.

</details>

+++

We can also compare the R-squared between ROIs with `prfmodel.plotting.plot_parameter_by_roi`, which shows the mean and standard deviation across the channels in each ROI. The function expects a DataFrame, so we wrap the R-squared in one.

```{code-cell} ipython3
from prfmodel.plotting import plot_parameter_by_roi

# Visual areas ordered roughly from early to higher visual cortex
roi_order = ["V2", "V3", "V3a", "V3b", "LO1", "LO2", "TO", "IPS", "none"]

fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
plot_parameter_by_roi(pd.DataFrame({"R-squared": r_squared_sgd}), "R-squared", roi, order=roi_order, ax=ax)
ax.set_ylim(-0.1, 1.0);
```

> **Exercise 7:** Plot the predicted vs. the observed broadband response for channels with a high, medium, and low R-squared (hint: use `prfmodel.plotting.plot_observed_predicted`). Can you see any patterns in the observed response that the Gaussian 2D pRF model cannot account for?

<details class="solution">
<summary>Solution</summary>

We sort the channels by their R-squared and pick the best channel, the channel at the upper quartile, and the channel at the median. About half of the channels hardly respond to the stimulus, so the median channel already has a low R-squared.

```python
from prfmodel.plotting import plot_observed_predicted

# Channels sorted from the highest to the lowest R-squared
order = np.argsort(r_squared_sgd)[::-1]
examples = {"High": order[0], "Medium": order[len(order) // 4], "Low": order[len(order) // 2]}

fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, layout="constrained")

for ax, (label, channel) in zip(axes, examples.items()):
    plot_observed_predicted(response[channel], pred_response_sgd[channel], ax=ax)
    ax.set(
        xlabel="",
        ylabel="Broadband power\nchange (in %)",
        title=f"{label}: channel {channel} ({roi[channel]}), R-squared = {r_squared_sgd[channel]:.2f}",
    )

axes[-1].set_xlabel("Time frame");
```

For the channel with a high R-squared, the model predicts the timing of the peaks very well. However, there are patterns that the model cannot account for:

- The observed response dips below zero right before and after some of the peaks. With a positive amplitude, the Gaussian model can only predict responses above its baseline. A dip below baseline is a sign of a suppressive surround around the pRF center.
- Some peaks are narrower and higher than the predictions, and others are lower. The Gaussian model assumes that the response grows linearly with the stimulated part of the pRF, which fixes the shape and the relative height of all peaks.
- Some channels show brief transients that do not depend on the position of the bar, for example, at the very first time frame when the stimulus appears.

For the channel with a medium R-squared, the response is noisier and the model predicts a broad response with a baseline that lies above the observed response during the blank periods. For the channel with a low R-squared, the response to the stimulus is weak compared to the noise.

</details>

+++

## Interpreting the model parameters

Like in the first practical, we only interpret the parameters of channels for which we can trust the estimates. We select channels that fulfill three criteria:

- The R-squared is above 0.3 (note that this threshold is somewhat arbitrary).
- The pRF center lies inside the visual field covered by the stimulus.
- The pRF size is below the largest `sigma` value in the grid.

```{code-cell} ipython3
from prfmodel.utils import calculate_eccentricity, calculate_polar_angle

mu_x = sgd_params["mu_x"].to_numpy()
mu_y = sgd_params["mu_y"].to_numpy()
sigma = sgd_params["sigma"].to_numpy()

r_squared_threshold = 0.3

grid_min_xy, grid_max_xy = stimulus.grid.min(axis=(0, 1)), stimulus.grid.max(axis=(0, 1))

selection_criteria = pd.DataFrame({
    f"r_squared > {r_squared_threshold}": r_squared_sgd > r_squared_threshold,
    "center inside stimulus": (
        (mu_x >= grid_min_xy[1]) & (mu_x <= grid_max_xy[1]) & (mu_y >= grid_min_xy[0]) & (mu_y <= grid_max_xy[0])
    ),
    "sigma below grid ceiling": sigma < np.max(grid_param_ranges["sigma"]),
})

is_selected = selection_criteria.all(axis=1).to_numpy()

# Number of channels that fulfill each criterion and all criteria together
selection_criteria.assign(all_criteria=is_selected).sum()
```

We plot the pRFs of the selected channels in the visual field. Each circle shows the center and size (`sigma`) of one pRF, and the color shows the ROI of the channel.

```{code-cell} ipython3
# Imported again so that this cell also works if the hidden manual fit cell was not run
import matplotlib as mpl
from plotting_helpers import GRAY, style_axes

roi_colors = dict(zip(roi_order, plt.get_cmap("tab10").colors))

# ROIs that contain at least one selected channel
selected_rois = [name for name in roi_order if np.any(is_selected & (roi == name))]

fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")

# Mark the visual field covered by the stimulus
ax.add_patch(
    mpl.patches.Rectangle(
        grid_min_xy[::-1], *(grid_max_xy - grid_min_xy)[::-1], facecolor="#f2f2f2", edgecolor=GRAY, linewidth=1,
    )
)

for channel in np.flatnonzero(is_selected):
    color = roi_colors[roi[channel]]
    ax.add_patch(
        mpl.patches.Circle((mu_x[channel], mu_y[channel]), sigma[channel], fill=False, color=color, alpha=0.6),
    )
    ax.plot(mu_x[channel], mu_y[channel], "o", color=color, markersize=4)

ax.axhline(0.0, color=GRAY, linewidth=0.8)
ax.axvline(0.0, color=GRAY, linewidth=0.8)
ax.legend(
    handles=[
        mpl.lines.Line2D([], [], marker="o", linestyle="", color=roi_colors[name], label=name) for name in selected_rois
    ],
    frameon=False,
    loc="upper right",
)
ax.set(xlim=(-9, 9), ylim=(-9, 9), aspect="equal", title="pRFs of the selected channels")
style_axes(ax, "x (degrees visual angle)", "y (degrees visual angle)", grid=False)
```

We compute the eccentricity and polar angle of the pRF centers with `prfmodel.utils.calculate_eccentricity` and `prfmodel.utils.calculate_polar_angle` and plot them for the selected channels at the electrode positions. We use a cyclic colormap for the polar angle (`cyclic=True`) because it wraps around from $-\pi$ to $\pi$. Channels that are not selected are shown as small gray markers.

```{code-cell} ipython3
eccentricity = calculate_eccentricity(mu_x, mu_y)
polar_angle = calculate_polar_angle(mu_x, mu_y)

# Most selected pRF centers lie within 4 degrees of the fovea; the few further out get the brightest color
plot_electrodes(np.where(is_selected, eccentricity, np.nan), vmin=0.0, vmax=4.0, title="Eccentricity (in degrees)")
plot_electrodes(np.where(is_selected, polar_angle, np.nan), cyclic=True, title="Polar angle (in radians)");
```

Finally, we look at the pRF size. We plot the pRF size against the eccentricity and the mean pRF size in each ROI for the selected channels.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")

for name in selected_rois:
    is_roi = is_selected & (roi == name)
    axes[0].scatter(eccentricity[is_roi], sigma[is_roi], color=roi_colors[name], label=name)

axes[0].legend(frameon=False)
style_axes(axes[0], "Eccentricity (in degrees)", "pRF size (sigma, in degrees)")

plot_parameter_by_roi(sgd_params[is_selected], "sigma", roi[is_selected], order=selected_rois, ax=axes[1])
style_axes(axes[1], "ROI", "pRF size (sigma, in degrees)")
```

> **Exercise 8:** Looking at the plots with pRF size, polar angle, and eccentricity, do the displayed results align with the results you have seen in the first practical based on BOLD fMRI responses? Do they align with the results reported by Dumoulin and Wandell (2008)?

<details class="solution">
<summary>Solution</summary>

Yes, the results show the same organization of the visual cortex as in the first practical and in Dumoulin and Wandell (2008):

- **Polar angle:** Almost all selected pRFs lie in the left visual field, contralateral to the right hemisphere with the electrodes. Most of them lie in the lower left quadrant because the electrodes cover dorsal visual areas, which represent the lower visual field.
- **Eccentricity:** Most selected pRFs lie within 4 degrees of the center of the visual field. The high-density grid lies on the lateral occipital cortex close to the occipital pole, where the central visual field is represented (the foveal confluence of V1–V3).
- **pRF size:** The pRF size increases with eccentricity and from early to higher visual areas, from V2 over V3 to V3b, LO1, and IPS. Dumoulin and Wandell (2008) reported the same two trends, and we saw them on the surface in the first practical.

The main difference to fMRI is the coverage: ECoG only measures at a few positions on the cortex. Thus, we cannot see complete maps of polar angle and eccentricity, only the small part of the visual field that the pRFs of the channels cover.

</details>

+++

## Comparing pRF models on ECoG data

When we looked at the timecourses of individual channels, we saw that the response of some channels dips below zero before and after a peak. The canonical Gaussian pRF model cannot predict a response below its baseline. A dip below baseline is a sign of a suppressive surround around the pRF center, which the Difference of Gaussians (DoG) model can capture (Zuiderbaan et al., 2012). Moreover, the model did not capture the shape of the peaks well for some channels. This could be a sign that the response grows less than linearly with the stimulated part of the pRF (compression), which the Compressive Spatial Summation (CSS) model can capture (Kay et al., 2013). The Divisive Normalization (DN) model can produce both suppression and compression (Aqil et al., 2021).

Now it is your turn: Compare the Gaussian, CSS, DoG, and DN models on the ECoG data by following the steps of the [third practical](03-prf-visual-fmri-comparison.md). Most of the code from the third practical can be reused, but there are a few differences to keep in mind:

- **Models without impulse response:** Create all models with `impulse_model=None`. The models then do not have the `weight_deriv` parameter, so remove it from the grid values and from the `fixed_parameters` of the SGD fitters.
- **Splitting the data:** The two halves of the experiment (time frames 0–111 and 112–223) both contain a horizontal and a vertical cardinal sweep. Without an impulse response, the response at a time frame does not depend on previous time frames. Thus, we do not need to wait for the response to return to baseline and can split the data directly between the two halves at time frame 112.
- **Estimating the noise ceiling:** The cardinal sweeps in the second half show the same bar positions as the cardinal sweeps in the first half, but in reverse order (e.g., right to left instead of left to right). Without an impulse response, reversing the response to a sweep in time gives the response to the same bar positions in the same order. Thus, you can estimate the noise ceiling by correlating the responses to the cardinal sweeps in the first half (time frames 0–27 and 56–83) with the reversed responses to the cardinal sweeps in the second half (time frames 112–139 and 168–195). The diagonal sweeps are not repeated in the other half, so they cannot be used for the noise ceiling.
- **Selecting a model per channel:** Instead of projecting the selected models onto the surface, you can plot whether each model is selected for a channel at its electrode with `plot_electrodes`, or compare the selected models between ROIs.
- **Fitting time:** The dataset only contains 136 channels, so all models can be fitted in a few minutes. This makes it feasible to run the full two-fold cross-validation (fitting on the second half and evaluating on the first half as well).

+++

## Summary

- The pRF concept does not depend on the measurement method, but the pRF model must match how the measured signal relates to neural activity.
- ECoG broadband power follows neural activity almost instantly. Therefore, pRF models for ECoG data do not need an impulse response, and the predicted response at each time frame only depends on the stimulus at the same time frame.
- ECoG only measures neural activity at the positions of the electrodes. The coverage of the visual cortex is therefore sparse and differs between subjects, and visual areas are assigned to electrodes with an atlas.
- For channels with a reliable response, the pRF estimates show the expected organization of the visual cortex: pRFs in the contralateral visual field and pRF sizes that increase with eccentricity and from early to higher visual areas.

## References

Aqil, M., Knapen, T., & Dumoulin, S. O. (2021). Divisive normalization unifies disparate response signatures throughout the human visual hierarchy. *Proceedings of the National Academy of Sciences*, *118*(46), e2108713118. https://doi.org/10.1073/pnas.2108713118

Dumoulin, S. O., & Wandell, B. A. (2008). Population receptive field estimates in human visual cortex. *NeuroImage*, *39*(2), 647-660. https://doi.org/10.1016/j.neuroimage.2007.09.034

Groen, I., Yuasa, K., Brands, A., Piantoni, G., Montenegro, S., Flinker, A., Devore, S., Devinsky, O., Doyle, W., Dugan, P., Friedman, D., Ramsey, N., Petridou, N., & Winawer, J. (2024). Visual ECoG dataset [Dataset]. OpenNeuro. https://doi.org/10.18112/openneuro.ds004194.v3.0.0

Kay, K. N., Winawer, J., Mezer, A., & Wandell, B. A. (2013). Compressive spatial summation in human visual cortex. *Journal of Neurophysiology*, *110*(2), 481-494. https://doi.org/10.1152/jn.00105.2013

Wang, L., Mruczek, R. E. B., Arcaro, M. J., & Kastner, S. (2015). Probabilistic maps of visual topography in human cortex. *Cerebral Cortex*, *25*(10), 3911-3931. https://doi.org/10.1093/cercor/bhu277

Yuasa, K., Groen, I. I. A., Piantoni, G., Montenegro, S., Flinker, A., Devore, S., Devinsky, O., Doyle, W., Dugan, P., Friedman, D., Ramsey, N. F., Petridou, N., & Winawer, J. (2025). Precise spatial tuning of visually driven alpha oscillations in human visual cortex. *eLife*, *12*, RP90387. https://doi.org/10.7554/eLife.90387

Zuiderbaan, W., Harvey, B. M., & Dumoulin, S. O. (2012). Modeling center-surround configurations in population receptive fields using fMRI. *Journal of Vision*, *12*(3), 10. https://doi.org/10.1167/12.3.10

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

# Practical 6: The Gaussian 1D population receptive field model for numerosity

This is the notebook for the numerosity practical of the workshop. This practical covers the Gaussian one-dimensional (1D) population receptive field (pRF) model for numerosity and how to fit it to empirical data from a functional magnetic resonance imaging (fMRI) experiment in which a subject viewed displays with different numbers of dots.

**We recommend going through the [first practical](01-prf-visual-fmri.md) on the canonical Gaussian 2D pRF model first to get familiar with the prfmodel package before starting with this practical.**

The first part of this practical introduces the numerosity stimulus and the dataset. It prepares the recorded BOLD responses for model fitting by converting them to percent signal change, removing slow trends, and projecting them from the volume onto the cortical surface. The second part fits the model to the data. It adapts the model predictions to the preprocessing of the BOLD responses, goes through multiple consecutive fitting stages (grid search, least-squares, and SGD), and concludes with evaluating the model fit and interpreting its parameters across numerosity maps on the cortical surface. Finally, we fit the model separately to four stimulus conditions that control for different visual properties of the dots and investigate how the preferred numerosity changes along a numerosity map.

+++

## Contents

- [Installation and Setup](#installation-and-setup)
- [The numerosity stimulus](#the-numerosity-stimulus)
- [Preparing the data](#preparing-the-data)
  - [Loading the dataset](#loading-the-dataset)
  - [Stimulus conditions](#stimulus-conditions)
  - [Selecting valid voxels](#selecting-valid-voxels)
  - [Converting to percent signal change](#converting-to-percent-signal-change)
  - [Removing slow trends](#removing-slow-trends)
  - [Projecting the responses onto the cortical surface](#projecting-the-responses-onto-the-cortical-surface)
  - [Numerosity maps](#numerosity-maps)
- [Fitting the model to data](#fitting-the-model-to-data)
  - [Inspecting the data](#inspecting-the-data)
  - [Defining the pRF model](#defining-the-prf-model)
  - [Matching the predictions to the preprocessed responses](#matching-the-predictions-to-the-preprocessed-responses)
  - [Grid search](#grid-search)
  - [Least-squares](#least-squares)
  - [Stochastic gradient descent](#stochastic-gradient-descent)
- [Evaluating the model fit](#evaluating-the-model-fit)
- [Interpreting the model parameters](#interpreting-the-model-parameters)
- [Preferred numerosity on the cortical surface](#preferred-numerosity-on-the-cortical-surface)
- [Preferred numerosity along a numerosity map](#preferred-numerosity-along-a-numerosity-map)
  - [Fitting the stimulus conditions](#fitting-the-stimulus-conditions)
  - [Comparing the stimulus conditions](#comparing-the-stimulus-conditions)
- [Summary](#summary)
- [References](#references)

+++

## Installation and Setup

First, we need to install prfmodel from GitHub. We install the package with the TensorFlow backend (for more information on the different backends, see the [Backends](https://popylar-org.github.io/prfmodel/development/backends.html) section in the package documentation).

**Please make sure to use a GPU instance when installing prfmodel and running this notebook in Colab.**

```{code-cell} ipython3
!pip install "prfmodel[tensorflow] @ git+https://github.com/popylar-org/prfmodel.git"
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

## The numerosity stimulus

A pRF model maps the recorded activity of neuron populations to an experimental stimulus that is presented during the recording. In the first practical, the pRF was defined in the two-dimensional visual field. In this practical, we use the pRF model by Harvey et al. (2013), in which the pRF is defined in the *numerosity* space of the stimulus: Some neuron populations respond most strongly to a particular number of items, their *preferred numerosity*, and less strongly to numerosities that are further away from it. Numerosity is a single number, so the feature space of the stimulus and the pRF are one-dimensional. Following Harvey et al. (2013), we define the pRF on the logarithm of numerosity (*log numerosity*).

In this practical, we use data from Hendrikx et al. (2024) that is publicly available on [figshare](https://doi.org/10.6084/m9.figshare.24126663). The numerosity stimulus of this dataset is included in prfmodel and can be loaded with `prfmodel.examples.load_1d_prf_lognumerosity_stimulus`.

```{code-cell} ipython3
from prfmodel.examples import load_1d_prf_lognumerosity_stimulus

stimulus = load_1d_prf_lognumerosity_stimulus()

print(stimulus)
```

The stimulus design matrix indicates which numerosity is shown at which time frame of the experiment. It has one column for each of the eight numerosities in the stimulus and one-hot encodes the numerosity at every time frame (i.e., the column of the shown numerosity is one and all other columns are zero). The stimulus grid contains the log numerosity of each column. We can look at the numerosities on the natural scale.

```{code-cell} ipython3
# Log numerosity of each column of the design matrix
unique_log_numerosities = stimulus.grid.squeeze()
# Numerosity on the natural scale
unique_numerosities = np.round(np.exp(unique_log_numerosities))

unique_numerosities
```

We plot the stimulus design matrix with `prfmodel.plotting.plot_1d_prf_stimulus` to see how numerosity changes over time.

```{code-cell} ipython3
from prfmodel.plotting import plot_1d_prf_stimulus

plot_1d_prf_stimulus(
    stimulus,
    tick_labels=unique_numerosities.astype(int),
    secondary_tick_labels=np.round(unique_log_numerosities, 2),
    ylabel="Numerosity (natural scale)",
    secondary_ylabel="Numerosity (log scale)",
);
```

The stimulus consists of a cycle that is repeated four times: The numerosities 1 to 7 are shown in ascending order, followed by numerosity 20, then the numerosities 7 to 1 in descending order, followed by numerosity 20 again. Each numerosity from 1 to 7 is shown for two time frames and numerosity 20 for eight time frames, so one cycle lasts 44 time frames. Numerosity 20 serves as a baseline. Before the first cycle, there is a pre-scan interval of six time frames that also shows numerosity 20, which gives $6 + 4 \times 44 = 182$ time frames in total. We take both the cycles and the pre-scan interval into account when fitting the pRF model.

+++

> **Exercise 1:** What do you expect the BOLD response of a neuron population that prefers numerosity 2 to look like during one cycle of the stimulus? How would it differ from the response of a neuron population that prefers numerosity 6? Why do you think the stimulus shows numerosity 20 between the sequences instead of, for example, a blank screen?

+++

## Preparing the data

### Loading the dataset

Now that we have the stimulus, we load the recorded BOLD responses of a single subject. We use a preprocessed version of the dataset (Mauro, 2026) that has the tag `numerosity-four-conditions`. Besides the responses, the dataset contains flattened surfaces of both hemispheres that we use later for visualization. The responses are stored in a pickle file that contains a `Dataset` object. To load the file, Python needs to find the module `dataset.py` that defines this object and lies next to this notebook.

```{code-cell} ipython3
import pickle

from prfmodel.examples import load_dataset

# Downloads on first use and caches in a user data directory (see prfmodel.examples.get_data_dir)
example_data = load_dataset("numerosity-four-conditions")

# The responses are stored as a Dataset object, which is defined in the module dataset.py
with example_data.files["responses"].open("rb") as f:
    dataset = pickle.load(f)

{condition: response.shape for condition, response in dataset.responses.items()}  # One timecourse per voxel
```

The dataset contains the BOLD responses in a volume of voxels with one timecourse per voxel, as well as a mapping between the voxels and the vertices of the cortical surface. The responses were already preprocessed: The six time frames of the pre-scan interval were removed and the four cycles were averaged, which gives a timecourse with 44 time frames for each voxel.

+++

### Stimulus conditions

Changing the number of dots also changes other visual properties of the stimulus, for example, the total area covered by the dots. Consequently, a neuron population whose response appears to be tuned to numerosity might instead be tuned to one of these properties. To control for this, the dataset contains responses to four stimulus conditions, each of which keeps a different visual property constant across numerosities:

- **Area (constant area):** The total area of all dots is the same for all numerosities.
- **Size (constant dot size):** The size of each dot is the same for all numerosities.
- **Dens (high density):** The same dots as in the constant area condition are shown in a smaller region of the screen, which increases their density.
- **Circ (constant circumference):** The total circumference of all dots is the same for all numerosities.

The dataset also contains the **Average** condition, which averages the responses across the four conditions. We first fit the pRF model to the Average condition to get an overall estimate of the preferred numerosity. Later, we fit the model separately to each condition to compare their estimates.

+++

> **Exercise 2:** Imagine a neuron population that does not respond to the number of dots but to the total area covered by them. How would its response change with numerosity in each of the four stimulus conditions?

+++

### Selecting valid voxels

Some voxels fall outside the scanned volume at some time frames and have a value of zero at these time frames. The dataset marks voxels with zeros in their timecourse as invalid. We select the responses of the Average condition and only keep the voxels that are valid in this condition.

```{code-cell} ipython3
average_response = dataset.responses["Average"]
average_valid = dataset.valid["Average"]

average_response_valid = average_response[average_valid]

average_valid.mean()  # Fraction of valid voxels
```

### Converting to percent signal change

The raw BOLD response has an arbitrary scale that differs between voxels. We convert it to percent signal change (PSC) relative to the mean of each timecourse, so that every timecourse fluctuates around zero and the response amplitudes are comparable across voxels. We define the conversion as a function because we will apply it again to the four stimulus conditions later.

```{code-cell} ipython3
def convert_psc(response_raw: np.ndarray) -> np.ndarray:
    """Convert raw responses into percent signal change relative to the mean of each timecourse."""
    response_mean = response_raw.mean(axis=1, keepdims=True)

    # Timecourses with a mean of zero become NaN or infinite
    with np.errstate(divide="ignore", invalid="ignore"):
        response_psc = (response_raw / response_mean - 1.0) * 100.0

    return response_psc


average_response_psc = convert_psc(average_response_valid)

average_response_psc.shape
```

### Removing slow trends

BOLD responses often contain slow fluctuations that are not related to the stimulus. We remove these slow trends from each timecourse with a basis of cosine functions (i.e., a discrete cosine transform basis): We fit the cosine functions to each timecourse with least-squares (using the pseudoinverse of the basis) and subtract the fitted trend from the timecourse. `N_DCT` sets the highest frequency of the cosine functions in cycles per timecourse. With `N_DCT = 0.5`, the basis contains a constant and a cosine that completes half a cycle over the 44 time frames.

Later, we apply the same detrending to the model predictions, so that the observed and predicted responses are processed in the same way.

```{code-cell} ipython3
N_DCT = 0.5

n_timepoints = average_response_psc.shape[1]

# Cosine basis with frequencies from 0 (a constant) to N_DCT in steps of half a cycle
time = np.linspace(0, 2 * np.pi, n_timepoints)
frequencies = np.arange(0, N_DCT + 0.25, 0.5)
trends = np.cos(np.outer(time, frequencies))

# Fit the trends to each timecourse and subtract them
pinv_trends = np.linalg.pinv(trends)
beta = pinv_trends @ average_response_psc.T
fitted_trend = (trends @ beta).T

average_response_detrended = average_response_psc - fitted_trend

average_response_detrended.shape
```

We compare the timecourse of a single voxel before and after detrending. For the figure, we use the colors and helper functions from the module `plotting_helpers.py`, which is shared by all practicals and lies next to this notebook.

```{code-cell} ipython3
import matplotlib.pyplot as plt
from plotting_helpers import BLUE, ORANGE, add_legend, style_axes

voxel_index = 100  # Index of a valid voxel

time_frames = np.arange(n_timepoints)

fig, ax = plt.subplots(figsize=(12, 3.8), layout="constrained")

ax.axhline(0.0, color="#bdbdbd", linewidth=1)
ax.plot(time_frames, average_response_psc[voxel_index], color=BLUE, linewidth=2, label="Before detrending")
ax.plot(time_frames, average_response_detrended[voxel_index], color=ORANGE, linewidth=2, label="After detrending")
add_legend(ax)
ax.set_title(f"Voxel {voxel_index}: BOLD response before and after detrending", loc="left")
style_axes(ax, "Time frame (in TR)", "BOLD response (in PSC)")

plt.show()
```

Detrending removes the slow downward drift of the response. This drift is not related to the stimulus, so removing it makes the fluctuations that are related to the stimulus easier to see.

+++

> **Exercise 3:** Why do we only remove a constant and half a cosine cycle from the timecourses? Detrend the responses with `N_DCT = 1.0` and plot the result for the same voxel (hint: store the results in new variables so that the rest of the notebook still uses `N_DCT = 0.5`). What could go wrong when the cosine basis contains frequencies that are similar to the frequency of the response to the stimulus?

```{code-cell} ipython3
# Your code goes here
```

### Projecting the responses onto the cortical surface

The numerosity maps that we analyze later are defined on the cortical surface, so we project the detrended voxel responses onto the surface. The dataset contains a mapping between surface vertices and voxels that was created with a nearest-neighbor projection. For each vertex, we average the timecourses of the valid voxels that are assigned to it. Vertices without any valid voxels are excluded. We keep track of the indices of the remaining vertices on the surface, so that we can later match their responses to their ROI labels and geodesic distances.

```{code-cell} ipython3
# The voxels assigned to vertex v are in vertex_voxel_rows[vertex_voxel_offsets[v]:vertex_voxel_offsets[v + 1]]
vertex_voxel_offsets = dataset.method2_vertex_voxel_offsets
vertex_voxel_rows = dataset.method2_vertex_voxel_rows

n_vertices = len(dataset.vertex_coordinates)

# Map the voxel indices of the dataset to rows of the valid responses; invalid voxels get -1
original_to_valid = np.full(len(average_valid), -1, dtype=int)
original_to_valid[np.flatnonzero(average_valid)] = np.arange(average_valid.sum())

# Average the responses of the valid voxels assigned to each vertex
vertex_response = np.zeros((n_vertices, n_timepoints))
vertex_valid = np.zeros(n_vertices, dtype=bool)

for v in range(n_vertices):
    rows = original_to_valid[vertex_voxel_rows[vertex_voxel_offsets[v]:vertex_voxel_offsets[v + 1]]]
    rows = rows[rows >= 0]

    if len(rows):
        vertex_response[v] = average_response_detrended[rows].mean(axis=0)
        vertex_valid[v] = True

vertex_valid.mean()  # Fraction of vertices with valid voxels
```

We only keep the vertices with valid voxels and store their indices on the surface.

```{code-cell} ipython3
valid_vertex_indices = np.flatnonzero(vertex_valid)

vertex_response = vertex_response[vertex_valid]

vertex_response.shape  # One timecourse for every valid vertex
```

### Numerosity maps

Next, we look up the region of interest (ROI) of each valid vertex. The dataset contains an ROI label for every vertex on the surface (vertices outside of the ROIs have the label -1) and a mapping between the labels and the names of the ROIs.

```{code-cell} ipython3
# ROI label of each valid vertex
roi_index_valid = dataset.vertex_roi_index[vertex_valid]

roi_mapping = dataset.roi_names
roi_mapping
```

The names of the ROIs start with their hemisphere (`lh` for left and `rh` for right), followed by `Num` to indicate that they are numerosity maps. NumTO (temporal-occipital) and NumPO (parietal-occipital) lie in the occipital cortex, NumPC1, NumPC2, and NumPC3 in the parietal cortex, and NumF in the frontal cortex (see Hendrikx et al., 2024, for details). We count the valid vertices in each ROI.

```{code-cell} ipython3
# Number of valid vertices for each ROI label (-1 for vertices outside of the ROIs)
pd.Series(roi_index_valid).value_counts().sort_index().rename(index=roi_mapping)
```

## Fitting the model to data

### Inspecting the data

Before we fit the pRF model, we inspect the BOLD response timecourses of the vertices in one numerosity map, the left NumTO. The figure below shows the timecourse of every tenth vertex in the map; use the slider or the play button to go through them. Note that the unit of the time frames is the repetition time (TR), which is 2.1 seconds for this dataset.

```{code-cell} ipython3
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "notebook_connected"  # Requires an internet connection
pio.templates.default = "simple_white"

# Responses of the vertices in the left NumTO (ROI label 1)
roi_response = vertex_response[roi_index_valid == 1]

fig = px.line(
    roi_response[::10].T,  # Every tenth vertex
    animation_frame="variable",
    range_x=(0, 44),
    range_y=(-1.5, 1.5),
    labels={
        "index": "Time frame (in TR)",
        "value": "BOLD response (in PSC)",
        "variable": "Vertex",
    },
    title="BOLD response of vertices in lh_NumTO",
)

# Show each vertex for 900 ms with a transition of 400 ms
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 900
fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 400

fig.update_layout(showlegend=False, height=450)
fig.show()
```

> **Exercise 4:** Go through the timecourses of the vertices. Which patterns do you see and how do they relate to the stimulus? Do they match your expectations from Exercise 1?

+++

> **Exercise 5:** Plot the timecourses of the vertices in another numerosity map, for example, the left NumF (ROI label 6). How do they compare to the timecourses in the left NumTO?

```{code-cell} ipython3
# Your code goes here
```

### Defining the pRF model

Now that the stimulus and the BOLD response are in place, we can define a pRF model that *predicts* the response to the stimulus. The goal is to predict the observed timecourses as closely as possible: By comparing the predictions against the observed timecourses, we can identify vertices that respond to numerosity and estimate their preferred numerosity.

We use the Gaussian 1D pRF model proposed by Harvey et al. (2013), which is implemented in the `prfmodel.models.prf.Gaussian1DPRFModel` class. It works in the same way as the canonical Gaussian 2D pRF model from the first practical, except that its tuning profile is a 1D Gaussian in log-numerosity space with a center `mu` (the preferred log numerosity) and a size `sigma`. The model computes the overlap between the tuning profile and the stimulus at every time frame, convolves the result with an impulse response that accounts for the shape of the BOLD response, and scales the prediction with a `baseline` and `amplitude` parameter.

Like in the first practical, we need to slightly customize the model: The stimulus and the BOLD response were recorded with a TR of 2.1 seconds, whereas pRF models in prfmodel assume a TR of 1 second by default. The TR is set through the resolution of the impulse response model, so we create an impulse response model with a resolution of 2.1 seconds. The stimulus stores the TR as its `frame_duration` attribute. Matching the resolution of the impulse response model with the TR ensures that the predicted pRF model response has the same sampling rate as the stimulus and the observed BOLD timecourses. For details, see the [Important Details](https://popylar-org.github.io/prfmodel/important_details.html) section in the online documentation of prfmodel.

```{code-cell} ipython3
from prfmodel.impulse import DerivativeTwoGammaImpulse

# Create custom impulse model with the TR of the stimulus as resolution
impulse_model = DerivativeTwoGammaImpulse(resolution=stimulus.frame_duration)

stimulus.frame_duration  # TR in seconds
```

The parameters of the two gamma densities of the impulse response (`delay`, `dispersion`, `undershoot`, `u_dispersion`, and `ratio`) default to the values of the Glover hemodynamic response function (see `prfmodel.impulse.defaults.default_two_gamma_impulse_glover_hrf`), so we only need to set `weight_deriv`. This parameter weights the derivative of the impulse response and accounts for small differences in the timing of the BOLD response between vertices: Negative values shift the impulse response earlier in time and positive values later. We start with `weight_deriv = -0.5` and estimate it for each vertex later.

```{code-cell} ipython3
# Only weight_deriv is set; the parameters of the two gamma densities use their default values
impulse_params = pd.DataFrame({"weight_deriv": [-0.5]})

impulse_response = impulse_model(impulse_params)

impulse_response.shape  # One value per time frame of the impulse response
```

We plot the predicted impulse response.

```{code-cell} ipython3
fig = px.line(
    x=np.arange(impulse_response.shape[1]),
    y=impulse_response[0],
    labels={"x": "Time frame (in TR)", "y": "Predicted impulse response"},
    title="Impulse response",
)

fig.update_layout(height=450)
fig.show()
```

We insert the impulse model into the pRF model.

```{code-cell} ipython3
from prfmodel.models.prf import Gaussian1DPRFModel

# Define pRF model with custom impulse response submodel
prf_model = Gaussian1DPRFModel(
    impulse_model=impulse_model,
)
```

### Matching the predictions to the preprocessed responses

The pRF model predicts the response to the full stimulus with 182 time frames, but the observed responses only contain the 44 time frames of a single (averaged) cycle. To compare them, we process the predictions in the same way as the observed responses:

1. We remove the six time frames of the pre-scan interval. We do not remove them from the stimulus itself: The subject already saw the stimulus during the pre-scan interval, and because the BOLD response is slow, this affects the response at the beginning of the first cycle.
2. We average the four cycles in the remaining 176 time frames.
3. We detrend the averaged predictions with the same cosine basis as the observed responses.

We implement these steps in a function that wraps the prediction function of the model (i.e., its `call` method) and replace the original prediction function with the wrapped one. This way, the fitters below automatically compare the processed predictions against the observed responses.

```{code-cell} ipython3
from collections.abc import Callable

from keras import ops
from prfmodel.stimuli import PRFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame


def average_cycles_and_detrend(
    fn: Callable,
    num_prescan_frames: int = 6,
    cycle_length: int = 44,
    num_cycles: int = 4,
    n_dct: float = 0.5,
) -> Callable:
    """Wrap a prediction function to remove the pre-scan frames, average the cycles, and detrend the predictions."""
    # Use the same cosine basis as for the observed responses
    time = np.linspace(0, 2 * np.pi, cycle_length)
    frequencies = np.arange(0, n_dct + 0.25, 0.5)
    trends = np.cos(np.outer(time, frequencies))
    pinv_trends = np.linalg.pinv(trends)

    def wrapper(stimulus: PRFStimulus, parameters: TensorFrame, *args, **kwargs) -> Tensor:
        predictions = fn(stimulus, parameters, *args, **kwargs)

        # 1. Remove the pre-scan frames
        predictions = predictions[:, num_prescan_frames:]

        # 2. Average the repeated stimulus cycles
        predictions = ops.mean(ops.reshape(predictions, (-1, num_cycles, cycle_length)), axis=1)

        # 3. Fit the trends to each prediction and subtract them
        beta = ops.matmul(predictions, ops.transpose(ops.cast(pinv_trends, predictions.dtype)))
        fitted_trend = ops.matmul(beta, ops.transpose(ops.cast(trends, predictions.dtype)))

        return predictions - fitted_trend

    return wrapper


# Replace the prediction function of the model with the wrapped one
prf_model.call = average_cycles_and_detrend(prf_model.call)
```

To see what the processed predictions look like, we predict the response for each numerosity in the stimulus as the preferred numerosity (i.e., `mu` is the log numerosity) with the same pRF size `sigma`.

```{code-cell} ipython3
num_numerosities = len(unique_log_numerosities)

# One row of parameters for every numerosity in the stimulus
start_params = pd.DataFrame(
    {
        "mu": unique_log_numerosities,
        "sigma": [1.0] * num_numerosities,
        "weight_deriv": [-0.5] * num_numerosities,
        "baseline": [0.0] * num_numerosities,
        "amplitude": [1.0] * num_numerosities,
    },
)

# Make prediction with pRF model
simulated_response = np.asarray(prf_model(stimulus, start_params))

simulated_response.shape  # One predicted timecourse of one cycle for every numerosity
```

The figure below shows the predicted response for one preferred numerosity at a time; use the slider or the play button to go through them.

```{code-cell} ipython3
fig = px.line(
    pd.DataFrame(simulated_response.T, columns=unique_numerosities.astype(int)),
    animation_frame="variable",
    range_x=(0, 44),
    labels={
        "index": "Time frame (in TR)",
        "value": "Predicted BOLD response",
        "variable": "Preferred numerosity",
    },
    title="Predicted BOLD response for different preferred numerosities",
)

# Show each preferred numerosity for 900 ms with a transition of 400 ms
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 900
fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 400

fig.update_layout(showlegend=False, height=450)
fig.show()
```

> **Exercise 6:** How does the predicted response change with the preferred numerosity? Relate the timing of the peaks in the predicted response to the stimulus design.

+++

> **Exercise 7:** Predict the responses again with a smaller (e.g., `sigma = 0.3`) and a larger pRF size (e.g., `sigma = 2.0`). How do the predictions change and why?

```{code-cell} ipython3
# Your code goes here
```

### Grid search

We fit the pRF model to the detrended responses of the valid vertices in three stages, just like in the first practical. In the first stage, we perform a "brute-force" grid search to find good estimates for the parameters of the pRF tuning profile (i.e., `mu` and `sigma`).

For the grid search, we define ranges of `mu` and `sigma` that we want to construct a grid of parameter values from. We create them from the stimulus with `prfmodel.fitters.grid_values_1d_prf`. By default, the function creates 49 values for `mu` that span twice the range of the stimulus around its midpoint, and 50 log-spaced values for `sigma` that range from the smallest spacing between the stimulus coordinates to their full range. This results in a grid with $49 \times 50 = 2,450$ parameter combinations. This is still a relatively small grid, and we recommend using finer grids in practice.

For `baseline`, `amplitude`, and `weight_deriv`, we only provide a single value so that they will stay constant across the entire grid. The parameters of the two gamma densities of the impulse response are not part of the grid, so the model uses their default values. If we wanted to change them, we could also add ranges for them here.

```{code-cell} ipython3
from prfmodel.fitters import grid_values_1d_prf

grid_param_ranges = grid_values_1d_prf(stimulus) | {  # Add constant parameters
    "weight_deriv": [-0.5],
    "baseline": [0.0],
    "amplitude": [1.0],
}

{name: len(values) for name, values in grid_param_ranges.items()}  # Number of values for each parameter
```

Two properties of the stimulus limit what we can learn from these ranges. First, the values for `mu` extend beyond the numerosities in the stimulus, from about 0.2 to 90 on the natural scale. Extending the grid beyond the stimulus prevents vertices from accumulating at the edges of the grid. However, estimates outside the numerosities 1 to 7 are only weakly constrained by the data (numerosity 20 is only shown as a baseline), so we will exclude them when we interpret the pRF parameters.

Second, the stimulus samples log-numerosity space at only eight points. The closest two points are $\log(7) - \log(6) \approx 0.15$ apart, which is the smallest value for `sigma` in the grid. A pRF that is much narrower than this spacing only responds to a single numerosity in the stimulus, no matter how much smaller `sigma` gets. Therefore, the data cannot distinguish between small values of `sigma`, and estimates close to the smallest value should be interpreted as "no wider than the spacing between numerosities".

To run the grid search, we construct the `prfmodel.fitters.GridFitter`. Note that we set `batch_size=20` to let the fitter evaluate 20 parameter combinations at the same time which saves us a lot of memory. By default, the `loss` (i.e., the metric the fitter minimizes between model predictions and data) is the negative correlation coefficient, which ignores differences in baseline and amplitude between model predictions and observed data. Therefore, the grid search cannot estimate `baseline` and `amplitude`, so we estimate them with least-squares in the next stage.

```{code-cell} ipython3
from prfmodel.fitters import GridFitter

# Create grid fitter object
grid_fitter = GridFitter(
    model=prf_model,
    stimulus=stimulus,
    compile_step=True,  # Setting 'compile_step=True' speeds up the fitting
)

# Run grid search
grid_history, grid_params = grid_fitter.fit(
    data=vertex_response,
    parameter_values=grid_param_ranges,
    batch_size=20,
)
```

We can print the parameters estimated by the grid search.

```{code-cell} ipython3
grid_params
```

> **Exercise 8:** Plot how the estimates of `mu` and `sigma` are distributed across the values of the grid (hint: use `prfmodel.plotting.plot_grid_parameter_distribution`). Do many vertices have estimates at the smallest or largest value of the grid? What could this mean?

```{code-cell} ipython3
# Your code goes here
```

### Least-squares

In the grid search, we ignored the `baseline` and `amplitude` parameters. In prfmodel, we estimate them with least-squares by regressing the observed responses on the predicted pRF model responses for each vertex. The intercept and slope of the least-squares regression become the `baseline` and `amplitude` parameter, respectively. Because the observed and predicted responses are both detrended, which also removes their mean, the `baseline` estimates are close to zero.

```{code-cell} ipython3
from prfmodel.fitters import LeastSquaresFitter

# Create least-squares fitter
ls_fitter = LeastSquaresFitter(
    model=prf_model,
    stimulus=stimulus,
)

# Run least squares fit
ls_history, ls_params = ls_fitter.fit(
    data=vertex_response,
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

### Stochastic gradient descent

After least-squares estimation, the estimates for `mu` and `sigma` are still lying on one of our grid points. To finetune all parameters at once, we use stochastic gradient descent (SGD) with the least-squares estimates as starting values (see the first practical for details). Because `sigma` is a strictly positive parameter, we create an adapter with a parameter transformation that allows SGD to optimize `sigma` in an unconstrained log space while converting it to a strictly positive space when making model predictions.

By default, `prfmodel.fitters.SGDFitter` optimizes every parameter it is given, so it also estimates `weight_deriv` for every vertex. This allows the timing of the BOLD response to differ slightly between vertices, but it also means that the shape of the impulse response is no longer fixed while `mu` and `sigma` are estimated.

```{code-cell} ipython3
from keras import ops
from prfmodel.fitters import SGDFitter
from prfmodel.fitters.adapter import Adapter, ParameterTransform

adapter = Adapter([ParameterTransform(["sigma"], transform_fun=ops.log, inverse_fun=ops.exp)])

sgd_fitter = SGDFitter(
    model=prf_model,
    stimulus=stimulus,
    adapter=adapter,
    compile_step=True,  # Setting 'compile_step=True' speeds up the fitting
)

sgd_history, sgd_params = sgd_fitter.fit(
    data=vertex_response,
    init_parameters=ls_params,
)
```

We can print the SGD parameter estimates.

```{code-cell} ipython3
sgd_params
```

> **Exercise 9:** Compare the SGD parameter estimates against the least-squares estimates. How do they differ? (hint: you can use `plot_grid_parameter_distribution` again)

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 10:** Run SGD again, but fix `weight_deriv` to its starting value with `fixed_parameters=["weight_deriv"]` (like in the first practical). How does this change the estimates of `mu` and `sigma`? Store the results in new variables so that the rest of the notebook still uses the estimates from above.

```{code-cell} ipython3
# Your code goes here
```

## Evaluating the model fit

Now that the pRF model parameters are estimated, we can compare the model predictions against the observed responses. Because we want to make predictions for all vertices at once, we wrap our `prf_model` in the `prfmodel.utils.batched` function. The modifier changes the behavior of the model to make predictions for batches of vertices sequentially. This saves us a lot of memory at the expense of minimal runtime overhead.

```{code-cell} ipython3
from prfmodel.utils import batched

predict_batched = batched(prf_model)

# Make predictions with the estimated parameters
pred_response = np.asarray(predict_batched(stimulus, sgd_params, batch_size=100))
```

We can quantify how well the predictions align with the observed timecourses using the R-squared metric, which we compute with `prfmodel.utils.calculate_r_squared`. This metric indicates the proportion of variance in the observed data explained by our model predictions. Note that we compare the predictions against the same responses that we fitted the model to, so the R-squared scores describe the in-sample model fit (see the [second practical](02-prf-visual-fmri-cv.md) on cross-validation).

```{code-cell} ipython3
from prfmodel.utils import calculate_r_squared

r_squared = calculate_r_squared(vertex_response, pred_response)  # One score per vertex
r_squared.shape
```

> **Exercise 11:** Plot the distribution of R-squared scores across vertices (hint: use `prfmodel.plotting.plot_r_squared_hist`). For how many vertices does the model fit the data well?

```{code-cell} ipython3
# Your code goes here
```

In addition to computing the R-squared scores, we compare the predicted against the observed timecourses for individual vertices. We select the vertices in the left NumTO and sort them by their R-squared score. The figure below shows the observed and predicted timecourses of the ten vertices with the highest scores, together with their R-squared score, pRF size (`sigma`), and preferred numerosity. The function `plot_vertex_fits` that creates the figure is defined in the module `helpers.py`, which lies next to this notebook.

```{code-cell} ipython3
from helpers import plot_vertex_fits

roi_index = 1  # lh_NumTO
roi_mask = roi_index_valid == roi_index

# Observed and predicted responses, R-squared scores, and parameters of the vertices in the ROI
vertex_response_roi = vertex_response[roi_mask]
pred_response_roi = pred_response[roi_mask]
r_squared_roi = r_squared[roi_mask]
sgd_params_roi = sgd_params.loc[roi_mask].reset_index(drop=True)

# Indices of the ten vertices with the highest R-squared scores (highest first)
best_vertices = np.argsort(r_squared_roi)[::-1][:10]

plot_vertex_fits(
    observed=vertex_response_roi[best_vertices],
    predicted=pred_response_roi[best_vertices],
    r_squared=r_squared_roi[best_vertices],
    sigma=sgd_params_roi["sigma"].to_numpy()[best_vertices],
    preferred_numerosity=np.exp(sgd_params_roi["mu"].to_numpy()[best_vertices]),
    roi_name=roi_mapping[roi_index],
)
```

> **Exercise 12:** Plot the observed and predicted timecourses of the ten vertices in the left NumTO with the *lowest* R-squared scores. Why does the model not fit well for these vertices?

```{code-cell} ipython3
# Your code goes here
```

## Interpreting the model parameters

To analyze and interpret the pRF parameters, we zoom in on vertices for which we can trust the pRF estimates. We select vertices that fulfill three criteria:

- The R-squared is above 0.2 (note that this threshold is somewhat arbitrary).
- The preferred numerosity lies between 1 and 7. Numerosity 20 is only shown as a baseline, and estimates outside the numerosities in the stimulus are only weakly constrained by the data.
- The pRF size (`sigma`) is below 2.8, which is close to the range of the log numerosities in the stimulus ($\log(20) \approx 3.0$). Such a wide Gaussian predicts an almost flat response, so its center `mu` cannot be identified. Note that SGD optimizes `sigma` in an unconstrained log space, so the estimates are no longer limited by the values of the grid.

```{code-cell} ipython3
numerosity = np.exp(sgd_params["mu"].to_numpy())
sigma = sgd_params["sigma"].to_numpy()

r_squared_threshold = 0.2

selection_criteria = pd.DataFrame({
    f"r_squared > {r_squared_threshold}": r_squared > r_squared_threshold,
    "numerosity between 1 and 7": (numerosity >= 1) & (numerosity <= 7),
    "sigma < 2.8": sigma < 2.8,
})

is_selected = selection_criteria.all(axis=1).to_numpy()

# Fraction of valid vertices that fulfill each criterion and all criteria together
selection_criteria.assign(all_criteria=is_selected).mean()
```

We add the name of the ROI to the parameters of each vertex, convert `mu` into the preferred numerosity on the natural scale, and only keep the selected vertices inside the numerosity maps.

```{code-cell} ipython3
final_params = sgd_params.copy()

# Name of the ROI of each vertex; NaN for vertices outside of the ROIs
final_params["roi"] = [roi_mapping.get(int(idx), np.nan) for idx in roi_index_valid]

# Preferred numerosity on the natural scale
final_params["numerosity"] = np.exp(final_params["mu"])

params_selected = final_params.loc[is_selected].dropna(subset=["roi"]).copy()

# Keep the order of the ROIs from the mapping
roi_order = list(roi_mapping.values())

params_selected["roi"] = pd.Categorical(params_selected["roi"], categories=roi_order, ordered=True)
```

Now, we can compare the average preferred numerosity and pRF size between the numerosity maps. We also count the selected vertices in each map, because their number varies by an order of magnitude and tells us how many vertices go into each average.

```{code-cell} ipython3
params_agg_roi = params_selected.groupby("roi", observed=False)[["numerosity", "sigma"]].agg(["mean", "std", "count"])

params_agg_roi.round(2)
```

We plot the average preferred numerosity of each map with `prfmodel.plotting.plot_parameter_by_roi`. The error bars show the standard deviation across vertices (not the standard error of the mean).

```{code-cell} ipython3
from prfmodel.plotting import plot_parameter_by_roi

fig, ax = plot_parameter_by_roi(
    params_selected,
    "numerosity",
    params_selected["roi"],
    order=roi_order,
    xlabel="Numerosity map",
    ylabel="Preferred numerosity",
    figsize=(12, 5),
)

# Rotate the ROI names so that they do not overlap
ax.tick_params(axis="x", labelrotation=45)
plt.setp(ax.get_xticklabels(), ha="right");
```

We also plot the average pRF size of each map.

```{code-cell} ipython3
fig, ax = plot_parameter_by_roi(
    params_selected,
    "sigma",
    params_selected["roi"],
    order=roi_order,
    xlabel="Numerosity map",
    ylabel="pRF size (sigma, in log numerosity)",
    figsize=(12, 5),
)

# Rotate the ROI names so that they do not overlap
ax.tick_params(axis="x", labelrotation=45)
plt.setp(ax.get_xticklabels(), ha="right");
```

> **Exercise 13:** How do the preferred numerosity and the pRF size differ between the numerosity maps? Keep in mind that the variation within each map is large and that the number of selected vertices differs strongly between maps (see the table above).

+++

> **Exercise 14:** Change the R-squared threshold (`r_squared_threshold`) and recreate the table and the plots. Do the differences between the maps change?

```{code-cell} ipython3
# Your code goes here
```

## Preferred numerosity on the cortical surface

We can also look at the preferred numerosity on the cortical surface. In contrast to the first practical, we plot it on flattened surfaces (flatmaps) of both hemispheres, which show the entire cortex at once. The function `plot_numerosity_flatmaps` from `helpers.py` maps the preferred numerosity of the vertices onto the flatmaps and outlines one numerosity map in each hemisphere. As above, it only shows vertices with an R-squared above a threshold (here, 0.2) and a preferred numerosity between 1 and 7. Here, we outline NumPC1 in both hemispheres.

The outlines of the numerosity maps were delineated with other software and transformed into the coordinate space of this surface in several steps, so they may not align exactly with the vertices. Therefore, the outlines should be interpreted as an anatomical reference rather than an exact border of the maps.

```{code-cell} ipython3
from helpers import plot_numerosity_flatmaps

# The flatmaps of both hemispheres lie in the same directory
surf_dir = example_data.files["lh_flat"].parent

plot_numerosity_flatmaps(
    sgd_params=sgd_params,
    r_squared=r_squared,
    valid_vertex_indices=valid_vertex_indices,
    roi_ids=dataset.vertex_roi_index,
    roi_mapping=roi_mapping,
    surf_dir=surf_dir,
    lh_roi_id=3,  # lh_NumPC1
    rh_roi_id=9,  # rh_NumPC1
    r2_threshold=0.2,
)
```

> **Exercise 15:** Plot the flatmaps with the outline of another numerosity map, for example, NumTO (ROI labels 1 and 7). Does the preferred numerosity change gradually within the maps?

```{code-cell} ipython3
# Your code goes here
```

## Preferred numerosity along a numerosity map

So far, we have compared the pRF parameters between numerosity maps. Now, we investigate how the preferred numerosity changes with the position of a vertex within a single map, the right NumTO.

To do this, we use the geodesic distance of each vertex in the map, which is stored in the dataset. The geodesic distance is measured along the cortical surface rather than in a straight line through the brain. Two subregions at the ends of each map were delineated beforehand: one with low and one with high preferred numerosities. For each vertex, the dataset contains its relative position between the low end (0) and the high end (1) of the map, scaled by the average geodesic distance between both ends (in mm). Thus, the distance is 0 mm at the low end of the map and increases towards the high end.

+++

### Fitting the stimulus conditions

We first fit the pRF model separately to the four stimulus conditions (Area, Size, Dens, and Circ). The function `fit_condition_roi` from `helpers.py` goes through the same steps as above for the vertices in a single ROI: It selects the valid voxels of a condition, converts their responses to PSC, detrends them, projects them onto the surface, fits the model with grid search, least-squares, and SGD, and computes the R-squared scores.

```{code-cell} ipython3
from helpers import fit_condition_roi

conditions = ["Area", "Size", "Dens", "Circ"]

roi_index = 7  # rh_NumTO

condition_results = {}

for condition in conditions:
    condition_results[condition] = fit_condition_roi(
        dataset=dataset,
        condition=condition,
        prf_model=prf_model,
        stimulus=stimulus,
        param_ranges=grid_param_ranges,
        convert_psc=convert_psc,
        roi_index=roi_index,
    )
```

### Comparing the stimulus conditions

Before comparing the conditions, we select vertices with the function `prepare_distance_data` from `helpers.py`. For the Average condition and each of the four stimulus conditions, it selects the vertices that fulfill similar criteria as above:

- The R-squared is above 0.3.
- The preferred numerosity lies between 1 and 7.
- The pRF size (`sigma`) is below 2.8.

The function also restricts the Average results to the vertices in the right NumTO (the fits for the four stimulus conditions only contain vertices in this map) and looks up the geodesic distance of each selected vertex. The number of selected vertices can differ between conditions because the criteria are applied to each condition separately.

```{code-cell} ipython3
from helpers import prepare_distance_data

plot_data = prepare_distance_data(
    dataset=dataset,
    sgd_params=sgd_params,
    r_squared=r_squared,
    vertex_valid=vertex_valid,
    roi_index_valid=roi_index_valid,
    condition_results=condition_results,
    roi_index=roi_index,
    r2_threshold=0.3,
    num_range=(1, 7),
    sigma_max=2.8,
)
```

We group the vertices into distance bins of 2 mm and compute the mean preferred numerosity of the vertices in each bin, separately for each condition. For the Average condition, we additionally fit a logarithmic function to the mean preferred numerosity of the bins. To estimate the uncertainty of this fit, we use bootstrap resampling: We sample vertices with replacement, recompute the bins, and fit the logarithmic function again. We repeat this 1,000 times. The 2.5th and 97.5th percentiles of the resulting curves give a 95% confidence interval. The function `fit_distance_numerosity` from `helpers.py` implements these steps.

```{code-cell} ipython3
from helpers import fit_distance_numerosity

distance_results = fit_distance_numerosity(
    plot_data=plot_data,
    bin_width=2.0,  # in mm
    n_bootstrap=1000,
    random_state=22,
)
```

Finally, we plot the mean preferred numerosity of the bins for the four stimulus conditions together with the Average condition. The error bars of the Average condition show the standard error of the mean (SEM) within each bin. The solid black line shows the logarithmic fit and the dashed lines its 95% bootstrap confidence interval.

```{code-cell} ipython3
from helpers import plot_distance_numerosity

plot_distance_numerosity(
    binned_data=distance_results["binned_data"],
    fit_params=distance_results["fit_params"],
    x_fit=distance_results["x_fit"],
    lower=distance_results["lower"],
    upper=distance_results["upper"],
    roi_name=roi_mapping[roi_index],
);
```

> **Exercise 16:** How does the preferred numerosity change along the right NumTO? Do the four stimulus conditions show the same change? What does this tell us about whether the map represents numerosity rather than other visual properties of the dots, such as their size, total area, density, or circumference (see Exercise 2)?

+++

> **Exercise 17 (optional):** Repeat the analysis for another numerosity map, for example, the left NumPC1 (ROI label 3). Does the preferred numerosity change in a similar way along this map?

```{code-cell} ipython3
# Your code goes here
```

## Summary

- In the 1D pRF model for numerosity, the pRF is a Gaussian in log-numerosity space. Its center (`mu`) is the preferred (log) numerosity of a neuron population and its size (`sigma`) describes how broadly the population is tuned.
- When the observed responses are preprocessed (e.g., by averaging repeated cycles or removing slow trends), the model predictions need to be processed in the same way before we can compare them. In prfmodel, we can do this by wrapping the prediction function of the model.
- The stimulus only samples a few numerosities. Estimates of the preferred numerosity outside the shown numerosities and of pRF sizes below the spacing between them are only weakly constrained by the data.
- Changing numerosity also changes other visual properties of the stimulus. Fitting the model separately to stimulus conditions that keep different visual properties constant tests whether the estimated preferred numerosity reflects numerosity rather than these properties.

## References

Harvey, B. M., Klein, B. P., Petridou, N., & Dumoulin, S. O. (2013). Topographic representation of numerosity in the human parietal cortex. *Science*, *341*(6150), 1123-1126. https://doi.org/10.1126/science.1239052

Hendrikx, E., Paul, J. M., van Ackooij, M., van der Stoep, N., & Harvey, B. M. (2024). Cortical quantity representations of visual numerosity and timing overlap increasingly into superior cortices but remain distinct. *NeuroImage*, *286*, 120515. https://doi.org/10.1016/j.neuroimage.2024.120515

Mauro, K. (2026). Four condition numerosity dataset. *figshare*. https://doi.org/10.6084/m9.figshare.34898940

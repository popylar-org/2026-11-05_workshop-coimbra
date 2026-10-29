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

# Practical 1: The canonical Gaussian 2D population receptive field model

This is the notebook for the first practical of the workshop. This practical covers the canonical Gaussian 2D population receptive field (pRF) model, its different computational parts, and how to fit it to empirical data from a visual functional magnetic resonance imaging (fMRI) experiment.

**We recommend going through this notebook first to get familiar with the prfmodel package before starting with the notebooks of the other practicals.**

The first part of this practical goes through the canonical Gaussian 2D pRF model step-by-step. The model input and output as well as every submodel is motivated and explained with interavtive visualizations. This part only looks at model prediction but not at fitting the model to data. This is what the second part is about: Fitting the model to an empirical dataset obtained with a 7T fMRI scanner. This part covers data inspection and visualization on an inflated surface of the brain. It goes through multiple consecutive fitting stages (grid search, least-squares, and SGD) and concludes with evaluation the model fit and interpreting its parameters. 

+++

## Contents

- [Installation and Setup](#installation-and-setup)
- [The canonical Gaussian 2D pRF model](#the-canonical-gaussian-2d-prf-model)
  - [Input: Stimulus design matrix and grid](#input-stimulus-design-matrix-and-grid)
  - [Building the model step by step](#building-the-model-step-by-step)
  - [Tuning profiles](#tuning-profiles)
  - [Stimulus encoding](#stimulus-encoding)
  - [Impulse response models for BOLD measurements](#impulse-response-models-for-bold-measurements)
  - [Scaling models](#scaling-models)
  - [The full canonical model](#the-full-canonical-model)
- [Fitting the model to data](#fitting-the-model-to-data)
  - [Inspecting the data](#inspecting-the-data)
  - [Selecting valid voxels](#selecting-valid-voxels)
  - [Defining the pRF model](#defining-the-prf-model)
  - [Grid search](#grid-search)
  - [Least-squares](#least-squares)
  - [Stochastic gradient descent](#stochastic-gradient-descent)
- [Evaluating the model fit](#evaluating-the-model-fit)
- [Interpreting the model parameters](#interpreting-the-model-parameters)
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

import pandas as pd

# Set keras backend to 'tensorflow' (this is normally the default)
os.environ["KERAS_BACKEND"] = "tensorflow"
# Print parameter DataFrames with three decimals
pd.set_option("display.precision", 3)
```

> **Exercise 1:** Import the prfmodel package and check whether the TensorFlow backend has been installed with GPU support (hint: use `tensorflow.config.list_physical_devices('GPU')` and `tensorflow.test.is_built_with_cuda()`).

<details class="solution">
<summary>Solution</summary>

```python
import prfmodel
import tensorflow as tf

print("prfmodel version:", prfmodel.__version__)
print("TensorFlow built with CUDA:", tf.test.is_built_with_cuda())
print("Available GPUs:", tf.config.list_physical_devices("GPU"))
```

On a Colab GPU instance, TensorFlow is built with CUDA and the list contains at least one GPU, for example, `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`. If the list is empty, TensorFlow runs on the CPU. The notebook still works, but fitting the models takes much longer. On Colab, you can switch to a GPU instance under *Runtime > Change runtime type*.

</details>

```{code-cell} ipython3
# Your code goes here
```

## The canonical Gaussian 2D pRF model

A population receptive field (pRF) model maps the recorded activity of neuron populations to an experimental stimulus
that is presented during the recording. In this example, we use the canonical Gaussian 2D pRF model adapted from Dumoulin and Wandell (2008) to map the recorded blood oxygenation level-dependent (BOLD) response from an fMRI experiment to a series of bar apertures that move through the visual field of the recorded subject (i.e., the bar moves on a computer screen in front of the subject).

+++

### Input: Stimulus design matrix and grid

A pRF is defined in the *feature space* of the experimental stimulus. In this example, the stimulus has a two-dimensional visual feature space. How the position of the bar on the screen changes during the experiment is recorded as a stimulus *design matrix*. The design matrix indicates which cell in feature space is stimulated by the bar at which *time frame* (i.e., recording sample) of the experiment. The stimulus *grid* contains the feature space coordinates of every
cell in the stimulus design matrix. In this example, the unit of the coordinates is degree of visual angle (but it could also be, e.g., screen pixels).

+++

We can look at the experimental stimulus that is attached to the dataset that we will use in this example. The dataset is
called the Arrow-of-Time (AOT) dataset and the BOLD response was recorded with a 7 Tesla MRI scanner. Therefore, the
dataset has the tag `7t-aot-visual`. We also specify that we want to load an inflated surface of the subject that we will
use later for visualization.

```{code-cell} ipython3
from prfmodel.examples import load_dataset

# Downloads on first use and caches in a user data directory (see prfmodel.examples.get_data_dir)
dataset = load_dataset("7t-aot-visual", surface_type="inflated")
```

The stimulus can be accessed through the `dataset.stimulus` attribute.

```{code-cell} ipython3
stimulus = dataset.stimulus
```

> **Exercise 2:** Inspect the stimulus design matrix and grid. What do the dimensions and values refer to? Hint: You can
> get an overview of the stimulus with `print(stimulus)`.

<details class="solution">
<summary>Solution</summary>

```python
print(stimulus)

# Design matrix: one image of the visual field per time frame
print(stimulus.design.shape)  # (time frames, y, x)
print(stimulus.design.min(), stimulus.design.max())

# Grid: coordinates of every cell of the design matrix
print(stimulus.grid.shape)  # (y, x, coordinates)
print(stimulus.grid[0, 0], stimulus.grid[-1, -1])  # Coordinates (y, x) of two opposite corners
```

The design matrix has the shape (340, 128, 128). The first dimension contains the 340 time frames of the experiment (one frame per 0.9 seconds, see `frame_duration`). The other two dimensions are the 128 × 128 cells of the visual field in the order y, x (see `dimension_labels`). The values lie between 0 and 1 and indicate for how much of a time frame a cell is covered by the bar (0: not covered, 1: covered during the whole time frame). The design matrix averages the images shown on the screen over each time frame. Because the bar moves by about 6 cells per time frame, cells at its leading and trailing edges are covered only for part of the frame, which creates gradual ramps at the edges. In some time frames, the whole bar has a value of about 0.93 because it was not shown during a short part of the frame.

The grid has the shape (128, 128, 2). For every cell of the design matrix, it contains the y- and x-coordinate in degrees of visual angle. The coordinates run from about -4 to 4 degrees in both directions, so the stimulus covers 8 × 8 degrees of the visual field.

</details>

```{code-cell} ipython3
# Your code goes here
```

To get an impression of the experimental stimulus, we can also visualize the stimulus design matrix projected into grid
coordinates as a video.

```{code-cell} ipython3
from IPython.display import HTML
from prfmodel.plotting import animate_2d_prf_stimulus

ani = animate_2d_prf_stimulus(stimulus, interval=25)  # Pause 25 ms between time frames

HTML(ani.to_html5_video())
```

> **Exercise 3:** Going back to Exercise 2, does the video confirm your assumptions about what the stimulus design and grid dimensions refer to?

<details class="solution">
<summary>Solution</summary>

Every frame of the video shows one time frame of the design matrix (its first dimension), and the image shows the cells of the visual field (its second and third dimension: y and x). The axes of the video show the grid coordinates, which range from about -4 to 4 degrees of visual angle in both directions. Bright cells are covered by the bar. The bar sweeps across the visual field in four directions (top to bottom, left to right, bottom to top, right to left), with blank periods in between. After a long blank period in the middle of the run, the four sweeps are repeated in the reverse order.

</details>

+++

> **Exercise 4:** What do you expect the activity of a neuron population that responds to this stimulus to look like?

<details class="solution">
<summary>Solution</summary>

A neuron population only responds when the bar covers the part of the visual field that it is sensitive to (its pRF). We therefore expect a short burst of activity every time the bar passes over the pRF, that is, one burst per sweep, and no activity during the blank periods. The timing of the bursts within the sweeps depends on the location of the pRF. For example, a pRF in the upper visual field is crossed early in the top-to-bottom sweep and late in the bottom-to-top sweep. The duration of the bursts depends on the size of the pRF: A large pRF overlaps with the bar for longer than a small one.

</details>

+++

### Building the model step by step

The pRF is the part of the stimulus feature space that stimulates activity of the neuron population. The pRF model takes
the stimulus design matrix and grid as input and predicts the neuron population activity given a set of parameters. By comparing the predicted against observed activity and optimizing the model parameters, the pRF model can estimate the
pRF properties for the neuron population of the observed activity. Typically, pRFs are defined through a location (e.g.,
a center x- and y-coordinate in visual space) and a size or tuning width.

The canonical Gaussian 2D pRF model gets from input (i.e., stimulus design matrix and grid) to output (i.e., predicted BOLD response) in several steps. Each step is performed by a *submodel* that does a dedicated computation:

1. The *tuning profile* describes which part of the visual field stimulates the neuron population.
2. The *stimulus encoding* computes the overlap between the tuning profile and the stimulus at every time frame, which gives the predicted neuron population activity.
3. The *impulse response* turns the neuron population activity into a predicted BOLD response.
4. The *scaling* adjusts the baseline and amplitude of the predicted BOLD response.

We build up the model one submodel at a time and then combine all submodels into the full model.

To explore how the parameters of each submodel affect its predictions, we use interactive figures. The code that creates the figures is hidden: Run a hidden cell to show its figure and move the sliders to change the parameters. The figure updates when you release a slider. You can expand a hidden cell if you are curious about its code.

The first hidden cell defines helper functions for the interactive figures. Some of them are shared by all practicals and are imported from the module `plotting_helpers.py`, which lies next to this notebook.

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
import numpy as np
from plotting_helpers import BLUE, GRAY, ORANGE, add_legend, make_slider, show_interactive, style_axes, to_frame

TUNING_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "tuning", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)

# Visual field coordinates of the stimulus grid (dimensions are ordered as y, x)
x_coords = stimulus.grid[0, :, 1]
y_coords = stimulus.grid[:, 0, 0]
FIELD_EXTENT = (x_coords[0], x_coords[-1], y_coords[0], y_coords[-1])

time_frames = np.arange(stimulus.design.shape[0])

# Starting value, minimum, maximum, and step size of the slider for each parameter
PARAM_SLIDERS = {
    "mu_x": (0.0, -4.0, 4.0, 0.1),
    "mu_y": (0.0, -4.0, 4.0, 0.1),
    "sigma": (1.0, 0.1, 4.0, 0.1),
    "weight_deriv": (-0.5, -2.0, 2.0, 0.1),
    "baseline": (0.0, -100.0, 100.0, 5.0),
    "amplitude": (1.0, 0.0, 3.0, 0.1),
}
START_VALUES = {name: value for name, (value, *_) in PARAM_SLIDERS.items()}


def make_sliders(*names: str, **custom_sliders: tuple[float, float, float, float]) -> dict[str, widgets.FloatSlider]:
    """Create a slider for each parameter; `custom_sliders` replace the values from `PARAM_SLIDERS` for a parameter."""
    return {name: make_slider(name, *custom_sliders.get(name, PARAM_SLIDERS[name])) for name in names}


def draw_prf_location(ax: plt.Axes, mu_x: float, mu_y: float, sigma: float) -> None:
    """Draw the pRF center and size (a circle with radius `sigma`) in the visual field."""
    x_min, x_max, y_min, y_max = FIELD_EXTENT
    ax.add_patch(
        mpl.patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min, facecolor="#f2f2f2", edgecolor=GRAY, linewidth=1,
        )
    )
    ax.text(x_min, y_max + 0.3, "Stimulus area", color="#595959", fontsize=9)
    ax.add_patch(mpl.patches.Circle((mu_x, mu_y), sigma, facecolor=BLUE, alpha=0.25, edgecolor=BLUE, linewidth=2))
    ax.plot(mu_x, mu_y, "o", color=BLUE, markersize=6)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect("equal")
    style_axes(ax, "x (degrees visual angle)", "y (degrees visual angle)")
```

### Tuning profiles

In prfmodel, the core of every pRF model is called *tuning profile*. In the canonical Gaussian 2D pRF model, the tuning
profile is an isotropic multivariate Gaussian density. This means that the Gaussian has the same standard deviation in all dimensions and no correlation between them (i.e., its covariance matrix is the identity matrix multiplied by the variance). It describes the stimulation of the neuron population at every input from the visual field.

+++

To illustrate this, we create an instance of a Gaussian tuning profile and evaluate it on the stimulus grid. The Gaussian 2D tuning profile only has three parameters, namely, the x- and y-coordinate of its center, and the size defined as the standard deviation. Parameters in prfmodel live in `pandas.DataFrame` objects where each row corresponds to a different unit (e.g., a vertex or voxel in fMRI) and each column to a different model parameter.

```{code-cell} ipython3
from prfmodel.models.prf import Gaussian2DPRFTuning

prf_tuning = Gaussian2DPRFTuning()

tuning_params = pd.DataFrame(
    {
        "mu_x": [0.0],
        "mu_y": [0.0],
        "sigma": [1.0],
    },
)

# Predict the tuning profile
simulated_tuning_profile = prf_tuning(stimulus, tuning_params)

simulated_tuning_profile.shape  # One value for every cell of the stimulus grid
```

The predicted tuning profile assigns a value to every cell of the stimulus grid. We can visualize it as a heatmap in visual field coordinates: the darkest cells lie at the pRF center (`mu_x`, `mu_y`), and the spread of the profile is determined by its size (`sigma`).

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: Tuning profile { display-mode: "form" }
def plot_tuning(**params: float) -> None:
    profile = prf_tuning(stimulus, to_frame(**params))[0]

    fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")

    image = ax.imshow(profile, extent=FIELD_EXTENT, origin="lower", cmap=TUNING_CMAP)  # Positive y-coordinates point up
    fig.colorbar(image, ax=ax, label="Tuning")
    ax.set_title("Tuning profile")
    style_axes(ax, "x (degrees visual angle)", "y (degrees visual angle)", grid=False)

    plt.show()


show_interactive(plot_tuning, make_sliders("mu_x", "mu_y", "sigma"))
```

> **Exercise 5:** Use the sliders to change the tuning parameters to the following combinations:
>
> - `mu_x=2.1`, `mu_y=1.4`, `sigma=2.0`
> - `mu_x=-2.1`, `mu_y=-1.4`, `sigma=0.7`
>
> How does the predicted tuning profile change and why?

<details class="solution">
<summary>Solution</summary>

`mu_x` and `mu_y` move the peak of the tuning profile to the given location in the visual field. That is to the upper right for the first combination and to the lower left for the second. `sigma` changes the spread of the profile: With `sigma=2.0`, the profile is broad and parts of it extend beyond the area covered by the stimulus while it is narrow with `sigma=0.7`.

The height of the peak also changes. It is much lower for `sigma=2.0` than for `sigma=0.7` (note the colorbar). The tuning profile is a Gaussian density, which always integrates to one. A broader profile spreads the same total over a larger area, so its peak (which is proportional to $1 / \sigma^2$) must be lower.

</details>

+++

### Stimulus encoding

In pRF models, the tuning profile is typically encoded with the stimulus design matrix through a dot-product operation. This operation computes the overlap between tuning profile and stimulus design matrix at every time frame and returns the predicted neuron population activity.

+++

We illustrate this by creating an instance of the pRF stimulus encoder.

```{code-cell} ipython3
from prfmodel.models.prf import PRFStimulusEncoder

prf_encoder = PRFStimulusEncoder()

# We first predict the tuning profile
simulated_tuning_profile = prf_tuning(stimulus, tuning_params)

# Then we stimulus-encode it
simulated_encoded_profile = prf_encoder(stimulus, simulated_tuning_profile, tuning_params)

simulated_encoded_profile.shape  # One value per time frame
```

The figure below shows the stimulus at a single time frame together with the contours of the tuning profile (left) and the predicted neuron population activity (right). The `time_frame` slider selects the time frame shown on the left, which is marked with a dot on the right.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: Stimulus encoding { display-mode: "form" }
def predict_encoded(params: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predict the tuning profile and the stimulus-encoded response."""
    profile = prf_tuning(stimulus, params)
    return profile[0], prf_encoder(stimulus, profile, params)[0]


def plot_encoding(time_frame: int, **params: float) -> None:
    profile, encoded = predict_encoded(to_frame(**params))

    fig, (ax_field, ax_response) = plt.subplots(
        1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 3]}, layout="constrained",
    )

    # Stimulated cells are black
    ax_field.imshow(stimulus.design[time_frame], extent=FIELD_EXTENT, origin="lower", cmap="Greys", vmin=0, vmax=1)
    if profile.max() > 0:
        ax_field.contour(
            x_coords, y_coords, profile, levels=profile.max() * np.array([0.25, 0.5, 0.75]), colors=BLUE, linewidths=1.5,
        )
    ax_field.set_title(f"Stimulus and tuning profile\nat time frame {time_frame}")
    style_axes(ax_field, "x (degrees visual angle)", "y (degrees visual angle)", grid=False)

    ax_response.plot(time_frames, encoded, color=BLUE, linewidth=2)
    ax_response.axvline(time_frame, color=GRAY, linestyle="--", linewidth=1)
    ax_response.plot(
        time_frame, encoded[time_frame], "o", color=BLUE, markersize=9, markeredgecolor="white", markeredgewidth=2,
    )
    ax_response.set_title("Predicted neuron population activity")
    style_axes(ax_response, "Time frame", "Predicted activity")

    plt.show()


# Start at the time frame where the stimulus overlaps most with the starting tuning profile
_, start_encoded = predict_encoded(to_frame(mu_x=0.0, mu_y=0.0, sigma=1.0))
time_frame_slider = widgets.IntSlider(
    value=int(np.argmax(start_encoded)),
    min=0,
    max=len(time_frames) - 1,
    description="time_frame",
    continuous_update=False,
)

show_interactive(plot_encoding, {"time_frame": time_frame_slider, **make_sliders("mu_x", "mu_y", "sigma")})
```

> **Exercise 6:** Why does the predicted neuron population activity have this particular shape? How does it relate to the experimental stimulus (compare it to the video and move the `time_frame` slider)?

<details class="solution">
<summary>Solution</summary>

The predicted activity at a time frame is the overlap between the bar and the tuning profile, that is, the sum of the tuning profile over all cells covered by the bar. The pRF in the center of the visual field is crossed by every sweep, so there is one peak per sweep (eight in total). The activity rises when the bar approaches the pRF, reaches its maximum when the bar covers the center of the pRF, and falls when the bar moves away. Thus, each peak has the shape of the Gaussian profile, smoothed by the width of the bar. During the blank periods (including the long one in the middle of the run), nothing overlaps with the pRF and the activity is zero.

</details>

+++

> **Exercise 7:** Change the tuning parameters to the same combinations as in Exercise 5:
>
> - `mu_x=2.1`, `mu_y=1.4`, `sigma=2.0`
> - `mu_x=-2.1`, `mu_y=-1.4`, `sigma=0.7`
>
> How does the predicted neuron population activity change and why? 

<details class="solution">
<summary>Solution</summary>

Moving the pRF center shifts the peaks in time because the bar crosses the pRF earlier or later in each sweep. For the pRF in the upper right (`mu_x=2.1`, `mu_y=1.4`), the peaks come early in the top-to-bottom and right-to-left sweeps and late in the bottom-to-top and left-to-right sweeps. For the pRF in the lower left, it is the other way around.

The size changes the width and height of the peaks. With `sigma=2.0`, the bar overlaps with the pRF for a longer time, so the peaks are broader. They are also lower because the bar only covers a small part of the broad profile at any time frame and because part of the profile lies outside the stimulus area. Some peaks are cut off abruptly because the sweep ends while the bar still overlaps with the pRF. With `sigma=0.7`, the peaks are narrow and high because the bar covers almost the entire profile when it passes over the center.

</details>

+++

> **Exercise 8:** What happens to the predicted neuron population activity when you move the sliders to their extremes (e.g., a very small `sigma` or a pRF center at the edge of the stimulus)? Why?

<details class="solution">
<summary>Solution</summary>

- **Very small `sigma`:** The pRF is much narrower than the bar, so the bar either covers the entire pRF or nothing of it. The peaks become box-shaped with a flat top: The activity jumps up when the bar reaches the pRF center, stays constant while the bar covers it, and drops to zero when the bar has passed. The peak duration then only depends on the width and speed of the bar.
- **Center at the edge of the stimulus:** Only part of the profile lies inside the area that the bar can cover, so the peaks are lower. Sweeps that move towards the edge reach the pRF at their very end, and sweeps that move away from the edge reach it at their very beginning, so these peaks are cut off.
- **Large `sigma`:** The peaks become broad and low and overlap between consecutive sweeps.

In all cases, the reason is the same: The activity is the overlap between stimulus and tuning profile. Parts of the profile outside the stimulus area never contribute to the activity, and the profile cannot resolve details that are smaller than the bar.

</details>

+++

### Impulse response models for BOLD measurements

The predicted neuron population activity does not yet account for the characteristic shape of the BOLD recording. By default, the canonical Gaussian 2D pRF model in prfmodel approximates the shape of the BOLD response with the weighted difference of two gamma densities. From this difference, we then also subtract its weighted derivative. While the parameters for the difference weight and the gamma densities are typically fixed at default values, we estimate the weight of the derivative. This is the `weight_deriv` parameter.

+++

We create an instance of the default impulse response model, define a set of parameters, and predict the impulse response.

```{code-cell} ipython3
from prfmodel.impulse import DerivativeTwoGammaImpulse

impulse_model = DerivativeTwoGammaImpulse()

impulse_params = pd.DataFrame({"weight_deriv": [-0.5]})

simulated_impulse_response = impulse_model(impulse_params)

simulated_impulse_response.shape  # One value per time frame of the impulse response
```

The figure below shows the impulse response for the `weight_deriv` that you set with the slider. The dashed gray line shows the impulse response without the derivative (i.e., `weight_deriv = 0`).

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: Impulse response { display-mode: "form" }
no_deriv_impulse = impulse_model(to_frame(weight_deriv=0.0))[0]
impulse_frames = np.arange(len(no_deriv_impulse))


def plot_impulse(weight_deriv: float) -> None:
    impulse = impulse_model(to_frame(weight_deriv=weight_deriv))[0]

    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")

    ax.axhline(0.0, color="#bdbdbd", linewidth=1)
    ax.plot(impulse_frames, no_deriv_impulse, color=GRAY, linestyle="--", linewidth=1.5, label="weight_deriv = 0")
    ax.plot(impulse_frames, impulse, color=BLUE, linewidth=2, label=f"weight_deriv = {weight_deriv:.1f}")
    add_legend(ax)
    ax.set_title("Impulse response", loc="left")
    style_axes(ax, "Time frame", "Predicted impulse response")

    plt.show()


show_interactive(plot_impulse, make_sliders("weight_deriv"))
```

To predict the response that accounts for the shape of the BOLD measurement, the canonical Gaussian 2D pRF model convolves the predicted neuron population activity with the impulse response.

```{code-cell} ipython3
from prfmodel.impulse import convolve_prf_impulse_response

convolution_params = pd.DataFrame(
    {
        "mu_x": [0.0],
        "mu_y": [0.0],
        "sigma": [1.0],
        "weight_deriv": [-0.5],
    },
)

# We first predict the tuning profile
simulated_tuning_profile = prf_tuning(stimulus, convolution_params)

# We stimulus-encode the tuning profile
simulated_encoded_profile = prf_encoder(stimulus, simulated_tuning_profile, convolution_params)

# We predict the impulse response
simulated_impulse_response = impulse_model(convolution_params)

# We convolve the encoded tuning profile with the impulse response
convolved_response = convolve_prf_impulse_response(simulated_encoded_profile, simulated_impulse_response)

convolved_response.shape
```

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: Convolution with the impulse response { display-mode: "form" }
def predict_convolved(params: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predict the neuron population activity and its convolution with the impulse response."""
    _, encoded = predict_encoded(params)
    convolved = convolve_prf_impulse_response(encoded[np.newaxis], impulse_model(params))[0]
    return encoded, np.asarray(convolved)


def plot_convolution(**params: float) -> None:
    encoded, convolved = predict_convolved(to_frame(**params))

    fig, ax = plt.subplots(figsize=(12, 3.8), layout="constrained")

    ax.plot(time_frames, encoded, color=BLUE, linewidth=2, label="Neuron population activity")
    ax.plot(time_frames, convolved, color=ORANGE, linewidth=2, label="Convolved with impulse response")
    add_legend(ax)
    ax.set_title("Predicted response before and after convolution", loc="left")
    style_axes(ax, "Time frame", "Predicted response")

    plt.show()


show_interactive(plot_convolution, make_sliders("mu_x", "mu_y", "sigma", "weight_deriv"))
```

> **Exercise 9:** Move the `weight_deriv` slider to change the impulse model parameter. How does the convolved response change and why?

<details class="solution">
<summary>Solution</summary>

First, note how the convolution changes the predicted activity in general: The convolved response is delayed by several seconds, smoother, and dips below zero after each peak (the undershoot), just like the impulse response.

Subtracting a weighted derivative from a function approximately shifts it in time: $g(t) - w \, g'(t) \approx g(t - w)$. Thus, a negative `weight_deriv` moves the peak of the impulse response to earlier time frames and a positive `weight_deriv` to later ones. The convolved response is shifted in the same direction, so `weight_deriv` adjusts the delay of the predicted BOLD response, which differs between brain regions and subjects. For large absolute values, the approximation breaks down, and the shape of the impulse response (e.g., the height of its peak and the depth of its undershoot) changes as well.

</details>

+++

> **Exercise 10:** Imagine a pRF model that is not applied to BOLD, but for example, electrocorticography (ECoG) measurements. Would the model still require a convolution of the predicted neuron population activity with an impulse response?

<details class="solution">
<summary>Solution</summary>

No, at least not with the impulse response of the BOLD signal. The impulse response models the slow hemodynamic response that follows neural activity. ECoG measures the electrical activity of the neuron population directly, and the broadband power follows the activity almost instantly. Thus, the predicted neuron population activity can be scaled and compared to the measurements directly. In prfmodel, we can create such a model with `Gaussian2DPRFModel(impulse_model=None)`, which is what we do in the practical on ECoG data.

</details>

+++

### Scaling models

The predicted convolved pRF model response has a certain amplitude (i.e., scale) and a baseline of zero (i.e., no activity when the screen is blank). However, observed BOLD responses of different neuron populations often differ in their amplitude and baseline from the predictions of the canonical Gaussian 2D pRF model (e.g., due to differences in the vasculature or in the distance to the receiver coil). To account for these differences, a *scaling* submodel can modify the amplitude and baseline of the predicted response.

+++

We create an instance of a scaling model and modify the predicted convolved model response.

```{code-cell} ipython3
from prfmodel.scaling import BaselineAmplitude

scaling_model = BaselineAmplitude()

scaling_params = pd.DataFrame({"baseline": [0.0], "amplitude": [1.0]})

scaled_response = scaling_model(convolved_response, scaling_params)

scaled_response.shape
```

The figure below shows the convolved response before (blue) and after (orange) scaling. The dashed gray line shows the scaled response for the starting parameters.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: Scaling { display-mode: "form" }
def predict_scaled(**params: float) -> tuple[np.ndarray, np.ndarray]:
    """Predict the convolved response before and after scaling."""
    params = to_frame(**{**START_VALUES, **params})
    _, convolved = predict_convolved(params)
    return convolved, scaling_model(convolved[np.newaxis], params)[0]


_, start_scaled = predict_scaled()


def plot_scaling(**params: float) -> None:
    convolved, scaled = predict_scaled(**params)

    fig, ax = plt.subplots(figsize=(12, 3.8), layout="constrained")

    ax.plot(time_frames, start_scaled, color=GRAY, linestyle="--", linewidth=1.5, label="Starting parameters (scaled)")
    ax.plot(time_frames, convolved, color=BLUE, linewidth=2, label="Before scaling")
    ax.plot(time_frames, scaled, color=ORANGE, linewidth=2, label="After scaling")
    add_legend(ax)
    ax.set_title("Convolved response before and after scaling", loc="left")
    style_axes(ax, "Time frame", "Predicted response")

    plt.show()


show_interactive(plot_scaling, make_sliders("baseline", "amplitude", "sigma"))
```

> **Exercise 11:** Move the `baseline` and `amplitude` sliders. How does the predicted response change and why?

<details class="solution">
<summary>Solution</summary>

The scaling model computes `amplitude * response + baseline`. `baseline` shifts the entire predicted response up or down, including the blank periods. `amplitude` stretches or compresses the response around zero: The peaks and the undershoots become larger or smaller, but the timing and the shape of the response stay the same. With `amplitude=0`, the prediction is a flat line at `baseline`. These two parameters allow the model to match the arbitrary scale and offset of the observed BOLD response of each voxel without changing what the model says about the pRF.

</details>

+++

> **Exercise 12:** Change both the `amplitude` and `sigma` parameter at the same time. How does this affect the scale of the predicted response?

<details class="solution">
<summary>Solution</summary>

Both parameters change the scale of the predicted response. Increasing `sigma` lowers the peaks (see Exercise 7), and increasing `amplitude` raises them. Thus, we can increase `sigma` and compensate the lower peaks with a larger `amplitude` (or vice versa) and get a predicted response with a similar scale. Only the shape of the response, for example, the width of the peaks, still tells the two apart. This trade-off makes it harder to estimate `sigma` and `amplitude` independently from noisy data because different combinations of the two can fit the data almost equally well. See the [Important Details](https://popylar-org.github.io/prfmodel/important_details.html) section in the package documentation for details.

</details>

+++

### The full canonical model

Now that we have seen every submodel, we can combine them into the full pRF model. In prfmodel, we call this a *canonical* model because it performs all necessary steps to get from input (i.e., stimulus design matrix and grid) to output (i.e., predicted BOLD response): tuning profile, stimulus encoding, convolution with the impulse response, and scaling. We create an instance of the canonical Gaussian 2D pRF model, which uses the submodels from above by default, and make a prediction with all parameters at once.

```{code-cell} ipython3
from prfmodel.models.prf import Gaussian2DPRFModel

prf_model = Gaussian2DPRFModel()

start_params = pd.DataFrame(
    {
        "mu_x": [0.0],
        "mu_y": [0.0],
        "sigma": [1.0],
        "weight_deriv": [-0.5],
        "baseline": [0.0],
        "amplitude": [1.0],
    },
)

# Make prediction with pRF model
simulated_response = prf_model(stimulus, start_params)

simulated_response.shape  # One predicted timecourse with one value per time frame
```

The figure below shows the pRF in the visual field (left) and the predicted BOLD response of the full model (right). The dashed gray line shows the prediction for the starting parameters above. Use the sliders to see how all parameters together shape the predicted BOLD response.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: Full model { display-mode: "form" }
start_response = prf_model(stimulus, to_frame(**START_VALUES))[0]


def plot_prediction(**params: float) -> None:
    response = prf_model(stimulus, to_frame(**params))[0]

    fig, (ax_field, ax_response) = plt.subplots(
        1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 3]}, layout="constrained",
    )

    draw_prf_location(ax_field, params["mu_x"], params["mu_y"], params["sigma"])
    ax_field.set_title("pRF in the visual field")

    ax_response.plot(time_frames, start_response, color=GRAY, linestyle="--", linewidth=1.5, label="Starting parameters")
    ax_response.plot(time_frames, response, color=BLUE, linewidth=2, label="Current parameters")
    add_legend(ax_response)
    ax_response.set_title("Predicted BOLD response", loc="left")
    style_axes(ax_response, "Time frame", "Predicted BOLD response")

    plt.show()


show_interactive(plot_prediction, make_sliders("mu_x", "mu_y", "sigma", "weight_deriv", "baseline", "amplitude"))
```

## Fitting the model to data

### Inspecting the data

Before we fit the canonical Gaussian 2D pRF model to empirical data, we first load and inspect the recorded BOLD response timecourses. The BOLD response was recorded in a volume of voxels with a size of 2 mm. The dataset only contains the voxels that
fall inside a (dilated) gray matter mask. We can access the timecourses from the `dataset.response` attribute. Note that the raw response includes a non-zero baseline to accommodate pRF models that cannot predict a zero baseline (e.g., the divisive normalization pRF model; Aqil et al., 2020).

```{code-cell} ipython3
response_raw = dataset.response
response_raw.shape 
```

> **Exercise 13:** What do the dimensions of `response_raw` refer to? Visualize one or multiple timecourses with matplotlib or plotly. What is the unit of the BOLD response? Do you see any patterns in the timecourses and if so, how do they relate to the stimulus?

<details class="solution">
<summary>Solution</summary>

```python
print(response_raw.shape)  # (number of voxels, number of time frames)

# Time frames at which the bar is visible (any cell of the design matrix is non-zero)
is_bar_visible = stimulus.design.reshape(stimulus.design.shape[0], -1).max(axis=1) > 0

fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")

# Two voxels in the visual cortex and a voxel with a typical (small) variability
for voxel_index in [60795, 40141, 95109]:
    ax.plot(response_raw[voxel_index], label=f"Voxel {voxel_index}")

# Shade the time frames during which the bar is visible
ax.fill_between(
    np.arange(len(is_bar_visible)),
    0,
    1,
    where=is_bar_visible,
    color="gray",
    alpha=0.2,
    transform=ax.get_xaxis_transform(),
    label="Bar visible",
)
ax.set(xlabel="Time frame (in TR)", ylabel="BOLD response")
ax.legend(loc="upper right");
```

`response_raw` has the shape (105867, 340): one timecourse for each of the 105,867 voxels in the gray matter mask, with one value for each of the 340 time frames of the stimulus.

The timecourses are scaled relative to their baseline: During the blank periods, they fluctuate around 100. After subtracting 100 (as we do below), the unit is percent signal change.

The two voxels in the visual cortex (60795 and 40141) respond to each of the eight bar sweeps with a peak. A peak appears a few seconds after the bar crosses the pRF of the voxel and is followed by a short dip below baseline (the undershoot of the BOLD response). During the blank periods, the response returns to baseline. The second half of the run shows the same four sweeps in the reverse order, so the voxels respond with four peaks again, in a different order. The two voxels respond at different times within the sweeps because their pRFs lie at different locations in the visual field. Most voxels, such as voxel 95109, do not respond to the stimulus at all and only contain noise.

</details>

```{code-cell} ipython3
# Your code goes here
```

We transform the raw BOLD response timecourses by subtracting their baseline. With a `baseline` parameter of zero, our pRF model predicts a response of zero at blank stimulus frames, so setting the baseline to zero matches this default and makes the numbers a bit easier to work with.

```{code-cell} ipython3
response_no_baseline = response_raw - 100.0
```

To get a more complete overview of the BOLD timecourses, we can plot them for all voxels at once in a heatmap with `prfmodel.plotting.plot_response_heatmap`.

```{code-cell} ipython3
from prfmodel.plotting import plot_response_heatmap

# Uses matplotlib because plotly cannot handle this many voxels
plot_response_heatmap(
    response_no_baseline,
    vmin=-2,
    vmax=5,
    xlabel="Time frame (in TR)",
    ylabel="Voxel index",
    colorbar_label="BOLD response",
    figsize=(6, 6),
);
```

> **Exercise 14:** Which patterns do you see in the heatmap and how do they relate to the stimulus?

<details class="solution">
<summary>Solution</summary>

- Most rows only show noise around zero: Most voxels in the gray matter mask do not respond to the stimulus.
- Some rows are bright during the bar sweeps (frames 30–150 and 186–307) and dark at the start of the run and during the long blank period in the middle (frames 151–185). These are voxels in the visual cortex that respond to the bar. The rows are ordered by the position of the voxels in the volume, so neighboring rows often contain neighboring voxels. Neighboring voxels have similar pRFs and respond at similar times, so the responding voxels form bands of several rows.
- Some rows are bright or dark during the entire run. These voxels have an offset from zero that is unrelated to the stimulus.

</details>

+++

> **Exercise 15:** Why do BOLD response patterns differ between voxels?

<details class="solution">
<summary>Solution</summary>

- **Location of the pRF:** Visual cortex is organized retinotopically: Neighboring voxels represent neighboring parts of the visual field. The bar crosses the pRFs of different voxels at different times, so their responses peak at different times.
- **Size of the pRF:** Voxels with larger pRFs respond for longer and with broader peaks.
- **Responsiveness:** The mask contains the gray matter of the entire brain. Most voxels lie outside the visual cortex or represent parts of the visual field that the stimulus does not cover, so they do not respond to the bar at all and only contain noise.
- **Measurement:** The amplitude, delay, and noise level of the BOLD response differ between voxels, for example, because of differences in the vasculature (voxels close to large veins have large and delayed responses), the fraction of gray matter in the voxel, and the distance to the receiver coil.

</details>

+++

To visualize the response on the cortical surface, we need to project it from the volume onto the surface. The volume and the surfaces of the subject share the same coordinate system, so we can use `nilearn.surface.vol_to_surf` for the projection. The function samples the volume at several depths between the pial surface and the white matter surface, so we load these surfaces as well. To map a value per voxel back into the volume, we fill the values into the voxels inside the gray matter mask from the dataset.

The projection linearly interpolates between neighboring voxels. To prevent voxels outside the mask or voxels with `NaN` values from affecting the result, we set them to zero and additionally project a volume that is one for all other (valid) voxels and zero otherwise. Dividing the first projection by the second gives a weighted average of only the valid voxels at each vertex. Vertices without any valid voxels are `NaN`. The function `make_surface_projection` from `plotting_helpers.py` implements these steps and creates the function `project_to_surface` for our dataset.

```{code-cell} ipython3
from plotting_helpers import make_surface_projection

project_to_surface = make_surface_projection(dataset)
```

We also create a helper function with `make_surface_plot` from `plotting_helpers.py` to plot a per-voxel statistic on the inflated surface. The inflated surfaces of both hemispheres overlap in space, so we first move them apart. The helper shows three views that focus on the visual cortex in the occipital lobe: the inner (medial) side of each hemisphere, where the calcarine sulcus lies, and both hemispheres from behind (posterior).

Each hemisphere of the surface has about 150,000 vertices, which makes plotting slow (especially on Colab). Because the figures are too small to show that much detail, the helper plots a downsampled surface: All vertices that fall into the same cube of a 2 mm grid are merged into a single vertex that gets the average value of the merged vertices. This reduces the surface to about 27,000 vertices per hemisphere.

```{code-cell} ipython3
from plotting_helpers import make_surface_plot

plot_surf_stat_map_helper = make_surface_plot(dataset.mesh)
```

We can use the helper to plot, for example, the standard deviation of each timecourse on the surface to get an overview of the variability in the BOLD timecourses across the surface.

```{code-cell} ipython3
# Calculate standard deviation of each timecourse
response_sd = response_no_baseline.std(axis=1)

plot_surf_stat_map_helper(project_to_surface(response_sd), vmax=5.0, title="BOLD response standard deviation");
```

> **Exercise 16:** What can the response standard deviation tell us about neuron activity in different brain regions? Can you find the primary visual cortex (i.e., V1)?

<details class="solution">
<summary>Solution</summary>

The standard deviation measures how much a timecourse varies over time, whatever the cause. In voxels that respond to the stimulus, part of this variability is the stimulus-driven response, so the standard deviation is elevated in the occipital lobe. However, noise from breathing, the heartbeat, head motion, or large veins also contributes to the variability. Therefore, some regions far from the visual cortex have an even larger standard deviation, for example, the bright spots next to the front end of the medial wall (the gray area in the medial views) and at the lower edge of the surface. A large standard deviation alone does not tell us that the neurons in a region respond to the stimulus.

</details>

+++

> **Exercise 17:** Plot the average response of each timecourse on the surface. Do you see any interesting patterns?

<details class="solution">
<summary>Solution</summary>

```python
# Average of each timecourse (after subtracting the baseline of 100)
response_mean = response_no_baseline.mean(axis=1)

plot_surf_stat_map_helper(
    project_to_surface(response_mean), vmin=-2.0, vmax=2.0, cmap="RdBu_r", title="Average BOLD response",
);
```

Because we subtracted the baseline, the average is close to zero for most of the surface. It is positive (red) in patches of the occipital lobe, especially on the posterior surface. The voxels there respond to the bar with peaks above baseline, which raises their average. A few patches have a negative average (blue), that is, their response lies below baseline for long parts of the run. This can be caused, for example, by slow drifts of the signal or by suppression of activity during the stimulation. The average is a less clear indicator of stimulus-driven activity than the standard deviation, because a response with peaks and undershoots partly averages out.

</details>

```{code-cell} ipython3
# Your code goes here
```

### Selecting valid voxels

Some voxels in the mask do not have valid timecourses, that is, their timecourses are constant and do not contain any signal. We filter out all voxels that have a constant timecourse.

```{code-cell} ipython3
response_is_valid = response_no_baseline.std(axis=1) > 0.0

response_valid = response_no_baseline[response_is_valid]

response_is_valid.mean()  # Fraction of valid voxels
```

### Defining the pRF model

Now that the stimulus and the BOLD response are in place, we can fit the canonical Gaussian 2D pRF model. However, we need to slightly customize the model: The stimulus design matrix and the BOLD response were recorded with a repetition time (TR) of 0.9 seconds. By default, pRF models in prfmodel assume a TR of 1 second. The TR is set through the resolution of the impulse response model, so we replace the default with a custom impulse response model that has a resolution of 0.9 seconds. Matching the resolution of the impulse response model with the TR ensures that the predicted pRF model response has the same sampling rate as the stimulus and the observed BOLD timecourses. For details, see the [Important Details](https://popylar-org.github.io/prfmodel/important_details.html) section in the online documentation of prfmodel.

```{code-cell} ipython3
# Define repetition time (TR)
tr = 0.9

# Create custom impulse model
impulse_model = DerivativeTwoGammaImpulse(resolution=tr)

# Define pRF model with custom impulse response submodel
prf_model = Gaussian2DPRFModel(
    impulse_model=impulse_model,
)
```

Before we let prfmodel fit the model, you can try to fit it yourself. The figure below shows the observed BOLD response of a voxel with a clear signal (gray) together with the prediction of the pRF model (blue). The R-squared in the title shows how much of the variance in the observed response the prediction explains (1 is a perfect fit). Note that the amplitude of the observed response is much smaller than in the simulations above, so the `amplitude` slider covers a smaller range.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Manual fit? { display-mode: "form" }
voxel_index = 60686  # A valid voxel with a clear response to the stimulus
voxel_response = response_valid[voxel_index]

# Show the full range of the observed response, even when the prediction is much larger
response_margin = 0.1 * np.ptp(voxel_response)
response_limits = (voxel_response.min() - response_margin, voxel_response.max() + response_margin)


def plot_manual_fit(**params: float) -> None:
    prediction = np.asarray(prf_model(stimulus, to_frame(**params))[0])
    r_squared = 1.0 - np.sum((voxel_response - prediction) ** 2) / np.sum((voxel_response - voxel_response.mean()) ** 2)

    fig, (ax_field, ax_response) = plt.subplots(
        1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 3]}, layout="constrained",
    )

    draw_prf_location(ax_field, params["mu_x"], params["mu_y"], params["sigma"])
    ax_field.set_title("pRF in the visual field")

    ax_response.plot(time_frames, voxel_response, color=GRAY, linewidth=1, label="Observed")
    ax_response.plot(time_frames, prediction, color=BLUE, linewidth=2, label="Predicted")
    ax_response.set_ylim(response_limits)
    add_legend(ax_response)
    ax_response.set_title(f"Voxel {voxel_index}: R-squared = {r_squared:.2f}", loc="left")
    style_axes(ax_response, "Time frame", "BOLD response")

    plt.show()


show_interactive(
    plot_manual_fit,
    make_sliders(
        "mu_x",
        "mu_y",
        "sigma",
        "weight_deriv",
        "baseline",
        "amplitude",
        baseline=(0.0, -2.0, 2.0, 0.05),
        amplitude=(0.05, 0.0, 0.5, 0.01),
    ),
)
```

> **Exercise 18:** Move the sliders until the prediction matches the observed response as well as you can. Which R-squared can you reach? Which parameters are easy to find and which are hard? Why?

<details class="solution">
<summary>Solution</summary>

You can reach an R-squared of about 0.85, for example, with `mu_x` = 0.7, `mu_y` = 2.3, `sigma` = 0.9, `weight_deriv` = -0.5, `baseline` = 0.1, and `amplitude` = 0.11. With `weight_deriv` = 0, the R-squared increases to about 0.9. For comparison, the grid search and least-squares below reach an R-squared of 0.84 for this voxel.

- **Easy:** The pRF center determines *when* the peaks occur. The horizontal bar (top to bottom and bottom to top) crosses the pRF at a time that depends on `mu_y`, and the vertical bar at a time that depends on `mu_x`. Because each bar orientation moves in both directions, moving the center shifts the peaks of the two opposite sweeps in opposite directions in time. `baseline` and `amplitude` are also easy because they only shift and scale the prediction.
- **Hard:** `sigma` and `amplitude` trade off against each other (Exercise 12): A larger `sigma` makes the peaks broader and lower, which a larger `amplitude` can compensate. Only the width of the peaks distinguishes the two, and the noise makes the width hard to judge. `weight_deriv` shifts all peaks slightly in the same direction in time and changes their shape, which is easy to confuse with small changes of the center and size.

</details>

+++

### Grid search

We fit the canonical Gaussian 2D pRF model in two stages. In the first stage, we perform a "brute-force" grid search to find good estimates for the parameters of the pRF tuning profile (i.e., `mu_x`, `mu_y`, and `sigma`).

For the grid search, we define ranges of `mu_x`, `mu_y`, and `sigma` that we want to construct a grid of parameter values from. We create them from the stimulus with `prfmodel.fitters.grid_values_2d_prf`. Here, the values for `mu_x` and `mu_y` span twice the visual field covered by the stimulus (`mu_extent=2.0`). Later, we will filter out voxels for which the pRF center lies outside the stimulus grid because their responses cannot be reliably measured by the experiment. The values for `sigma` are log-spaced between 0.05 and 4 degrees. We use 21 values for `mu_x` and `mu_y` so that the center (0 degrees) and edges ($\pm 4$ degrees) of the stimulus land on grid values. If they did not, `grid_values_2d_prf` would warn us and suggest a number of values that aligns them.

For `baseline`, `amplitude`, and `weight_deriv`, we only provide a single value so that they will stay constant across the entire grid.

```{code-cell} ipython3
from prfmodel.fitters import grid_values_2d_prf

grid_param_ranges = grid_values_2d_prf(
    stimulus,
    num_mu=21,
    mu_extent=2.0,  # Span twice the visual field covered by the stimulus
    num_sigma=10,
    sigma_range=(0.05, 4.0),  # Log-spaced by default
) | {  # Add constant parameters
    "weight_deriv": [-0.5],
    "baseline": [0.0],
    "amplitude": [1.0],
}

grid_param_ranges
```

To run the grid search, we construct the `prfmodel.fitters.GridFitter`. Note that we set `batch_size=20` to let the fitter evaluate 20 parameter combinations at the same time which saves us a lot of memory. By default, the `loss` (i.e., the metric the fitter minimizes between model predictions and data) is the negative correlation coefficient, which ignores differences in baseline and amplitude between model predictions and observed data.

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
    data=response_valid,
    parameter_values=grid_param_ranges,
    batch_size=20,
)
```

We can print the parameters estimated by the grid search.

```{code-cell} ipython3
grid_params
```

Because the grid search can only return values that are in the grid, it is worth checking how the estimates are distributed across the grid. We plot the distribution of the pRF centers with `prfmodel.plotting.plot_2d_prf_centers` and the distribution of the pRF sizes with `prfmodel.plotting.plot_grid_parameter_distribution`. The rectangle marks the part of the visual field covered by the stimulus, and pRF sizes at the smallest or largest value of the grid are highlighted in red.

```{code-cell} ipython3
from prfmodel.plotting import plot_2d_prf_centers, plot_grid_parameter_distribution

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")

# Count estimates for each combination of pRF center coordinates in the grid
plot_2d_prf_centers(grid_params, grid_param_ranges, stimulus=stimulus, ax=axes[0])
axes[0].set(xlabel="mu_x (in degrees)", ylabel="mu_y (in degrees)", title="pRF centers")

# Count estimates for each pRF size in the grid
plot_grid_parameter_distribution(grid_params, grid_param_ranges, "sigma", ax=axes[1])  # Detects log spacing
axes[1].set(xlabel="sigma (in degrees)", title="pRF sizes");
```

> **Exercise 19:** Compare the parameter estimates to the ranges of the grid search. What do you observe in the distributions of pRF centers and sizes? Change the grid ranges with the arguments of `grid_values_2d_prf` (e.g., `mu_extent`, `num_mu`, `sigma_range`; you can also create a range for `weight_deriv`). Can you get to a lower (i.e., better) mean loss across voxels (shown in the progress bar under `loss`)?

<details class="solution">
<summary>Solution</summary>

- **pRF centers:** Many centers lie at the edges and corners of the stimulus or in its center. About a quarter of the voxels have a center outside the stimulus, some of them even at the outer edge of the grid (±8 degrees). Most of these voxels only contain noise, or their pRF lies (partly) outside the stimulus, so the data hardly constrain their center.
- **pRF sizes:** The sizes pile up at both ends of the grid: About 19% of the voxels have the smallest size (0.05 degrees) and 13% the largest size (4 degrees). Very small pRFs are typical for voxels that only contain noise: A tiny pRF predicts sharp peaks that can correlate with random fluctuations. More importantly, almost half of the voxels with an R-squared above 0.3 have the largest size of the grid. The best fitting size of these voxels probably lies above 4 degrees, so the range of `sigma` is too small.

We extend the range of `sigma` to 10 degrees and add a range for `weight_deriv`. Note that the grid search takes longer because the grid has 3.6 times as many parameter combinations.

```python
# Extend the range of sigma and add a range for weight_deriv
grid_param_ranges_new = grid_values_2d_prf(
    stimulus,
    num_mu=21,
    mu_extent=2.0,
    num_sigma=12,
    sigma_range=(0.05, 10.0),
) | {
    "weight_deriv": [-1.0, -0.5, 0.0],
    "baseline": [0.0],
    "amplitude": [1.0],
}

grid_history_new, grid_params_new = grid_fitter.fit(
    data=response_valid,
    parameter_values=grid_param_ranges_new,
    batch_size=20,
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")

plot_2d_prf_centers(grid_params_new, grid_param_ranges_new, stimulus=stimulus, ax=axes[0])
axes[0].set(xlabel="mu_x (in degrees)", ylabel="mu_y (in degrees)", title="pRF centers")

plot_grid_parameter_distribution(grid_params_new, grid_param_ranges_new, "sigma", ax=axes[1])
axes[1].set(xlabel="sigma (in degrees)", title="pRF sizes")

plot_grid_parameter_distribution(grid_params_new, grid_param_ranges_new, "weight_deriv", ax=axes[2])
axes[2].set(xlabel="weight_deriv", title="Weight of the derivative")

# Mean loss across voxels for the original and the new grid (lower is better)
print(f"Original grid: {np.mean(grid_history.history['loss']):.4f}, new grid: {np.mean(grid_history_new.history['loss']):.4f}")
```

On a random subset of 10,000 voxels, the mean loss decreases from -0.222 to -0.233. The two changes help in different ways:

- Extending `sigma` alone hardly changes the mean loss (-0.223) because most voxels only contain noise. However, it removes the ceiling for most of the well-fitting voxels: Only 5% of them have the new largest size of 10 degrees.
- The range for `weight_deriv` gives most of the improvement. Most of the well-fitting voxels choose `weight_deriv` = 0, which is again the edge of the grid, so the range could be extended further.
- A finer grid for the pRF centers (e.g., `num_mu=41`) also lowers the loss slightly (-0.224), but it takes about four times as long.

Keep in mind that a larger grid also fits the noise better, so a lower mean loss does not always mean that the estimates are more accurate.

</details>

```{code-cell} ipython3
# Your code goes here
```

### Least-squares

In the grid search, we ignored `baseline` and `amplitude` parameters. In prfmodel, we estimate them with least-squares by regressing the observed responses on the predicted pRF model responses for each voxel. The intercept and slope of the least-squares regression become the `baseline` and `amplitude` parameter, respectively.

```{code-cell} ipython3
from prfmodel.fitters import LeastSquaresFitter

# Create least-squares fitter
ls_fitter = LeastSquaresFitter(
    model=prf_model,
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

### Stochastic gradient descent

After least-squares estimation, the estimates for the tuning profile parameters are still lying on one of our grid points. Moreover, least-squares estimates `baseline` and `amplitude` *given* the other parameters, but it cannot adjust these other parameters to achieve a better fit. To finetune all parameters (or a subset) at once, we use stochastic gradient descent (SGD). It calculates the gradient of each parameter with respect to a loss function between model predictions and observed data and adjusts the parameters accordingly in small steps to minimize the loss. Importantly, the default mean squared error loss function in SGD is scale-sensitive, meaning that large differences in scale between model predictions and observed data can impair SGD estimation or make it inefficient.  Thus, SGD requires good starting values for the parameters, so that it quickly converges to a good minimum of the loss function instead of getting stuck in a poor local minimum. Therefore, we *first* perform the grid search *and* least-squares optimization and use their parameter estimates as starting values for SGD.

Without any adjustments, SGD moves parameter estimates in an unconstrained space, which can potentially lead to implausible values for parameters that are strictly positive, such as `sigma`. Therefore, and to make the estimation more efficient, we create an adapter with a parameter transformation that allows SGD to optimize `sigma` in an unconstrained log space while converting it to a strictly positive space when making model predictions. This way, `sigma` always has plausible values when making predictions.

We limit the flexibility of the estimation by fixing `weight_deriv` to its starting value. This forces SGD to focus on the other parameters.

**Note:** SGD is computationally very heavy for BOLD fMRI timecourses from an entire brain. Therefore, we only use a small number of steps (`num_steps=100`) to illustrate the procedure. In practice, more steps should be used (the default is `num_steps=1000`).

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
    data=response_valid,
    init_parameters=ls_params,
    fixed_parameters=["weight_deriv"],  # Fix weight_deriv to starting value
    batch_size=20_000,
    num_steps=100,
)
```

We can print the SGD parameter estimates.

```{code-cell} ipython3
sgd_params
```

> **Exercise 20:** Compare the SGD parameter estimates against the grid search and least-squares estimates. How do they differ? (hint: you can use `plot_2d_prf_centers` and `plot_grid_parameter_distribution`)

<details class="solution">
<summary>Solution</summary>

```python
# Absolute change of each parameter from the least-squares to the SGD estimates
print((sgd_params - ls_params).abs().describe())

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")

# The SGD estimates are not restricted to the grid values, so the histogram uses 50 bins in each dimension
plot_2d_prf_centers(sgd_params, stimulus=stimulus, ax=axes[0])
axes[0].set(xlabel="mu_x (in degrees)", ylabel="mu_y (in degrees)", title="pRF centers (SGD)")

# Compare the pRF sizes before and after SGD
axes[1].scatter(ls_params["sigma"], sgd_params["sigma"], s=1, alpha=0.2)
axes[1].axline((1.0, 1.0), slope=1.0, color="black", linestyle="--")
axes[1].set(xscale="log", yscale="log", xlabel="sigma (least-squares)", ylabel="sigma (SGD)", title="pRF sizes");
```

The SGD estimates are no longer restricted to the values of the grid, but they stay very close to the least-squares estimates. The pRF centers move by at most about 0.1 degrees, and `sigma` changes by at most about 10%. The histogram of the centers still shows the pattern of the grid, and the sizes form short vertical lines around the grid values. `weight_deriv` does not change because we fixed it.

The changes are small because we only used 100 steps. The `Adam` optimizer changes each parameter by at most about its learning rate per step (0.001 by default), so after 100 steps, a parameter can have changed by only about 0.1. For `sigma`, which SGD optimizes in log space, this corresponds to a factor of about 1.1. To let SGD move the estimates further, we could use more steps (the default is 1000) or a larger learning rate (e.g., `optimizer=keras.optimizers.Adam(learning_rate=0.01)`; we do this in the third practical).

</details>

```{code-cell} ipython3
# Your code goes here
```

## Evaluating the model fit

Now that the pRF model parameters are estimated, we can compare the model predictions against the observed responses. Because we want to make predictions for all voxels at once, we wrap our `prf_model` in the `prfmodel.utils.batched` function. The modifier changes the behavior of the model to make predictions for batches of voxels sequentially. This saves us a lot of memory at the expense of minimal runtime overhead.

We first evaluate the fit of the pRF model using the grid search and least-squares parameter estimates. 

```{code-cell} ipython3
from prfmodel.utils import batched

predict_batched = batched(prf_model)

# Make predictions with the estimated parameters
pred_response = np.asarray(predict_batched(stimulus, ls_params, batch_size=200))
```

We can quantify how well the predictions align with the observed timecourses using the R-squared metric, which we compute with `prfmodel.utils.calculate_r_squared`. This metric indicates the proportion of variance in the observed data explained by our model predictions.

```{code-cell} ipython3
from prfmodel.utils import calculate_r_squared

r_squared = calculate_r_squared(response_valid, pred_response)  # One score per voxel
r_squared.shape
```

We can look at the distribution of R-squared scores across voxels with `prfmodel.plotting.plot_r_squared_hist`.

```{code-cell} ipython3
from prfmodel.plotting import plot_r_squared_hist

# Many voxels have scores close to zero; a log scale also shows the voxels with higher scores
plot_r_squared_hist(r_squared, log=True);
```

> **Exercise 21:** Plot the R-squared score on the inflated surface. You can use the helper function below. For which brain regions does the model fit the data well?

```{code-cell} ipython3
def fill_valid_voxels(values: np.ndarray) -> np.ndarray:
    """Fill values for the valid voxels into an array with one value per voxel in the mask."""
    values_full = np.full((response_no_baseline.shape[0],), fill_value=np.nan)
    values_full[response_is_valid] = values
    return values_full
```

<details class="solution">
<summary>Solution</summary>

```python
r_squared_surf = project_to_surface(fill_valid_voxels(r_squared))

plot_surf_stat_map_helper(r_squared_surf, vmin=0.0, vmax=0.8, title="R-squared");
```

The model fits the data well only in the occipital lobe, i.e., around the occipital pole and the calcarine sulcus on the medial side (V1), and in the surrounding regions on the dorsal, ventral, and lateral surface of the occipital lobe (V2, V3, and higher visual areas). There, the R-squared reaches values of 0.6 to 0.9. A few small patches with a lower R-squared lie further up in the parietal lobe. In the rest of the brain, the R-squared is close to zero because these regions do not respond to the bar stimulus (or not in a way that the pRF model can describe).

</details>

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 22:** In addition to computing model fit with R-squared, we also recommend comparing model predictions against observed timecourses for individual voxels. Plot predictions against observed responses for voxels with different fit quality, for example, by sorting them according to R-squared (e.g., with `np.argsort(r_squared)`; hint: use `prfmodel.plotting.plot_observed_predicted`). Does the plot confirm the impression you get from the R-squared scores? Why does the model not fit well for some voxels?

<details class="solution">
<summary>Solution</summary>

```python
from prfmodel.plotting import plot_observed_predicted

# Voxel indices sorted from the lowest to the highest R-squared
order = np.argsort(r_squared)
num_voxels = len(order)

fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True, layout="constrained")

# The best voxel, the voxels at the top 1% and 10%, and the median voxel
for ax, rank in zip(axes, [num_voxels - 1, int(0.99 * num_voxels), int(0.9 * num_voxels), num_voxels // 2]):
    voxel_index = order[rank]
    plot_observed_predicted(response_valid[voxel_index], pred_response[voxel_index], ylabel="BOLD response", ax=ax)
    ax.set_title(f"Voxel {voxel_index}: R-squared = {r_squared[voxel_index]:.2f}", loc="left")
```

Yes, the plots confirm the R-squared scores:

- **Best voxel (R-squared of about 0.9):** The response has eight clear peaks, and the prediction follows them closely. The remaining differences are mostly noise, but there are also systematic differences: After some peaks, the observed response stays below baseline for longer than the predicted undershoot.
- **Top 1% (about 0.73):** The peaks are still clear, but the noise is larger.
- **Top 10% (about 0.2):** The peaks are hard to see in the noise, and the prediction only roughly follows the response.
- **Median voxel (about 0.02):** The response only contains noise, and the prediction is almost flat.

The model fits poorly when a voxel does not respond to the stimulus (because it lies outside the visual cortex or its pRF lies outside the stimulus), when the noise is large compared to the response, or when the response contains slow drifts that the model does not predict. In addition, the assumptions of the model may not hold for some voxels, for example, the fixed shape of the impulse response (`weight_deriv`), a linear response without compression (see Exercise 28), or the absence of a suppressive surround.

</details>

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 23:** Evaluate the model fit based on the SGD parameter estimates and compare it against the grid search and least-squares fit. Does SGD improve the fit and if not, why? (hint: use `prfmodel.plotting.plot_r_squared_comparison` to compare R-squared scores).

<details class="solution">
<summary>Solution</summary>

```python
from prfmodel.plotting import plot_r_squared_comparison

pred_response_sgd = np.asarray(predict_batched(stimulus, sgd_params, batch_size=200))
r_squared_sgd = calculate_r_squared(response_valid, pred_response_sgd)

plot_r_squared_comparison(
    r_squared,
    r_squared_sgd,
    value_range=(0.0, 1.0),
    xlabel="R-squared (grid search + least-squares)",
    ylabel="R-squared (SGD)",
)

print(f"Mean R-squared: {r_squared.mean():.3f} (least-squares), {r_squared_sgd.mean():.3f} (SGD)")
print(f"Fraction of voxels with a higher R-squared after SGD: {(r_squared_sgd > r_squared).mean():.2f}")
```

SGD improves the R-squared for almost all voxels (about 97% in our check on a subset of 4,000 voxels), but only very slightly: All points lie almost on the identity line. For voxels with an R-squared above 0.3, the R-squared increases by about 0.002 on average and by at most about 0.02. There are several reasons:

- The grid search and least-squares estimates are already close to the best parameters.
- With 100 steps and the default learning rate, SGD can only change the parameters a little (see Exercise 20). More steps or a larger learning rate would let it improve the fit further.
- `weight_deriv` is fixed, although many voxels would fit better with a different value (see Exercise 19).
- Most voxels only contain noise, which no parameter values can explain.

</details>

```{code-cell} ipython3
# Your code goes here
```

## Interpreting the model parameters

To analyze and interpret the pRF parameters, we zoom in on voxels for which we can trust the pRF estimates. We select
voxels that fulfill three criteria:

- The R-squared is above 0.3 (note that this threshold is somewhat arbitrary).
- The pRF center lies inside the visual field covered by the stimulus. We extended the grid for `mu_x` and `mu_y`
  beyond the stimulus, but pRF centers outside the stimulus are only weakly constrained by the data.
- The pRF size is below the largest `sigma` value in the grid. An estimate at the ceiling of the grid suggests
  that the best fitting pRF size lies outside the grid.

We select voxels instead of surface vertices because we fitted the pRF model to the timecourse of each voxel. The
projection onto the surface averages neighboring voxels, so selecting vertices would mix the parameters of poorly fitting
voxels into the selected vertices. Instead, we set the parameters of voxels that are not selected to `NaN` before
the projection, so that they are ignored.

```{code-cell} ipython3
mu_x = ls_params["mu_x"].to_numpy()
mu_y = ls_params["mu_y"].to_numpy()
sigma = ls_params["sigma"].to_numpy()

r_squared_threshold = 0.3
edge_tolerance = 1e-3  # in degrees

grid_min_xy, grid_max_xy = stimulus.grid.min(axis=(0, 1)), stimulus.grid.max(axis=(0, 1))

selection_criteria = pd.DataFrame({
    f"r_squared > {r_squared_threshold}": r_squared > r_squared_threshold,
    # The tolerance includes the grid values at the stimulus edges, which can be off due to rounding
    "center inside stimulus": (
        (mu_x >= grid_min_xy[1] - edge_tolerance)
        & (mu_x <= grid_max_xy[1] + edge_tolerance)
        & (mu_y >= grid_min_xy[0] - edge_tolerance)
        & (mu_y <= grid_max_xy[0] + edge_tolerance)
    ),
    # Least squares does not change sigma, so it is always one of the grid values
    "sigma below grid ceiling": ~np.isclose(sigma, np.max(grid_param_ranges["sigma"])),
})

is_selected = selection_criteria.all(axis=1).to_numpy()


def project_selected(voxel_values: np.ndarray) -> np.ndarray:
    """Project values of the selected voxels onto the surface, ignoring all other voxels."""
    return project_to_surface(fill_valid_voxels(np.where(is_selected, voxel_values, np.nan)))


# Fraction of valid voxels that fulfill each criterion and all criteria together
selection_criteria.assign(all_criteria=is_selected).mean()
```

First, we look at the pRF size indicated by `sigma` and plot it on the surface.

```{code-cell} ipython3
size_surf = project_selected(sigma)

plot_surf_stat_map_helper(size_surf, vmin=0.0, vmax=5.0, title="pRF size (sigma)");
```

> **Exercise 24:** How does the pRF size change across the inflated surface? Change the R-squared threshold (`r_squared_threshold`) and recreate the plot. Does the pattern change?

<details class="solution">
<summary>Solution</summary>

```python
# All criteria except the R-squared threshold (center inside stimulus and sigma below grid ceiling)
other_criteria = selection_criteria.iloc[:, 1:].all(axis=1).to_numpy()

for threshold in [0.1, 0.5]:
    is_selected_threshold = other_criteria & (r_squared > threshold)
    size_surf_threshold = project_to_surface(fill_valid_voxels(np.where(is_selected_threshold, sigma, np.nan)))
    plot_surf_stat_map_helper(
        size_surf_threshold, vmin=0.0, vmax=5.0, title=f"pRF size (sigma), R-squared > {threshold}",
    )
```

The pRFs are smallest (`sigma` below 1 degree) around the occipital pole and in the calcarine sulcus, that is, in early visual cortex. Further away from the pole, on the dorsal, ventral, and lateral surface of the occipital lobe, where the higher visual areas lie, the pRFs are larger (2 to 3 degrees).

With a lower threshold of 0.1, about twice as many voxels are selected (6,718 instead of 3,484). The additional voxels extend into higher visual areas and the parietal lobe and mostly have large pRFs, but they also include noisy voxels with very small or arbitrary sizes. With a higher threshold of 0.5, only 2,006 voxels remain, mostly in early visual cortex. The overall pattern stays the same, but the threshold determines how much of the cortex we see and how noisy the map is.

</details>

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 25:** Compute the polar angle of the pRF center with `prfmodel.utils.calculate_polar_angle`. Plot the polar angle of the selected voxels on the inflated surface (hint: use `cyclic=True` in the plotting helper to get a cyclic colormap). How does polar angle change on the inflated surface? Note that the projection onto the surface averages neighboring voxels, but angles cannot be averaged directly because they wrap around from $-\pi$ to $\pi$. Instead, project `mu_x` and `mu_y` onto the surface separately and compute the polar angle afterwards.

<details class="solution">
<summary>Solution</summary>

```python
from prfmodel.utils import calculate_polar_angle

# Project mu_x and mu_y onto the surface separately and compute the polar angle afterwards
mu_x_surf = project_selected(mu_x)
mu_y_surf = project_selected(mu_y)

polar_angle_surf = calculate_polar_angle(mu_x_surf, mu_y_surf)

plot_surf_stat_map_helper(polar_angle_surf, cyclic=True, title="Polar angle");
```

Each hemisphere represents the opposite (contralateral) half of the visual field: The left hemisphere has polar angles between $-\pi/2$ and $\pi/2$ (right visual field; green, cyan, and blue), and the right hemisphere has angles beyond $\pm\pi/2$ (left visual field; yellow, red, and magenta). Within each hemisphere, the upper visual field (positive angles) is represented on the ventral side, below the calcarine sulcus, and the lower visual field (negative angles) on the dorsal side, above it. Moving away from the calcarine sulcus, the polar angle changes until it reaches the vertical meridian and then reverses its direction at the border between V1 and V2, and again at the border between V2 and V3. In the posterior view, these reversals show up as alternating bands of color.

</details>

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 26:** Compute the eccentricity (i.e., the distance of the pRF center from the center of the screen) with `prfmodel.utils.calculate_eccentricity`. Plot it for the selected voxels on the inflated surface. How does eccentricity change across the inflated surface?

<details class="solution">
<summary>Solution</summary>

```python
from prfmodel.utils import calculate_eccentricity

# Use the projections of mu_x and mu_y from Exercise 25
eccentricity_surf = calculate_eccentricity(mu_x_surf, mu_y_surf)

plot_surf_stat_map_helper(eccentricity_surf, vmin=0.0, vmax=5.0, title="Eccentricity");
```

The eccentricity is lowest (center of the visual field) at the occipital pole and on the posterior surface close to it. It increases with the distance from the pole: forward along the calcarine sulcus in the medial views, and outward on the posterior surface. The stimulus only covers the central 4 degrees (up to about 5.7 degrees in the corners), so we only see the central part of the eccentricity map: Voxels that represent the periphery of the visual field further forward in the calcarine sulcus are not stimulated and therefore not selected.

</details>

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 27 (optional):** Compare the polar angle and eccentricity patterns you observed to the ones reported in the paper by Dumoulin and Wandell (2008). Do your results agree with theirs?

<details class="solution">
<summary>Solution</summary>

Dumoulin and Wandell (2008) found the classical retinotopic organization of the early visual cortex: Each hemisphere represents the contralateral visual field. In V1, the upper visual field is represented below and the lower visual field above the calcarine sulcus, and the polar angle reverses at the vertical and horizontal meridians, which form the borders between V1, V2, and V3. The eccentricity increases from the occipital pole forward.

Our polar angle and eccentricity maps agree with these patterns, although our stimulus only covers the central 4 degrees of the visual field, so we can only compare the central part of the maps.

Dumoulin and Wandell (2008) also found that the pRF size increases with eccentricity within each visual area and from V1 to the higher visual areas. Our size map shows larger pRFs in the higher visual areas (Exercise 24). However, among the selected voxels, the size hardly changes with eccentricity (the mean `sigma` is about 1.7 degrees in all eccentricity bins). This is a consequence of our fitting procedure rather than of the data: The grid only contains 10 sizes, least-squares does not change them, and we excluded the voxels with a size at the ceiling of the grid (4 degrees), which are almost half of the well-fitting voxels. With a grid that extends `sigma` to 10 degrees (Exercise 19), the size increases with eccentricity.

</details>

+++

> **Exercise 28 (optional, advanced):** Fit the compressive spatial summation (CSS) pRF model (Kay et al., 2013) to the dataset using the same workflow. The CSS pRF model is implemented in the `prfmodel.models.prf.Gaussian2DCSSPRFModel` class. Look at the API documentation of the model class to get more information about its parameters. You can also take a look at the paper by Kay et al. (2013) for more background information.

<details class="solution">
<summary>Solution</summary>

The CSS model raises the stimulus-encoded response to the power of an exponent `n` before the convolution with the impulse response. We add a range of values for `n` to the grid and then follow the same steps as for the Gaussian model. Note that the grid search takes four times as long as before.

```python
from prfmodel.models.prf import Gaussian2DCSSPRFModel

css_model = Gaussian2DCSSPRFModel(impulse_model=impulse_model)

# Grid search: same grid as for the Gaussian model plus a range of values for the compression exponent n
css_grid_param_ranges = grid_param_ranges | {"n": [0.25, 0.5, 0.75, 1.0]}

css_grid_fitter = GridFitter(model=css_model, stimulus=stimulus, compile_step=True)

css_grid_history, css_grid_params = css_grid_fitter.fit(
    data=response_valid,
    parameter_values=css_grid_param_ranges,
    batch_size=20,
)

# Least-squares for amplitude and baseline
css_ls_fitter = LeastSquaresFitter(model=css_model, stimulus=stimulus)

css_ls_history, css_ls_params = css_ls_fitter.fit(
    data=response_valid,
    parameters=css_grid_params,
    slope_name="amplitude",
    intercept_name="baseline",
    batch_size=200,
)

# SGD with sigma and n in log space to keep them positive
css_sgd_fitter = SGDFitter(
    model=css_model,
    stimulus=stimulus,
    adapter=Adapter([ParameterTransform(["sigma", "n"], transform_fun=ops.log, inverse_fun=ops.exp)]),
    compile_step=True,
)

css_sgd_history, css_sgd_params = css_sgd_fitter.fit(
    data=response_valid,
    init_parameters=css_ls_params,
    fixed_parameters=["weight_deriv"],
    batch_size=20_000,
    num_steps=100,
)

# Compare the fit with the Gaussian model (SGD estimates from Exercise 23)
pred_response_css = np.asarray(batched(css_model)(stimulus, css_sgd_params, batch_size=200))
r_squared_css = calculate_r_squared(response_valid, pred_response_css)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")

plot_r_squared_comparison(
    r_squared_sgd,
    r_squared_css,
    value_range=(0.0, 1.0),
    xlabel="R-squared Gaussian",
    ylabel="R-squared CSS",
    ax=axes[0],
)

# Compression exponent of the voxels that the CSS model fits well
axes[1].hist(css_sgd_params.loc[r_squared_css > 0.3, "n"], bins=30)
axes[1].set(xlabel="n", ylabel="Number of voxels", title="Compression exponent (R-squared > 0.3)");
```

Instead of the grid search over `n`, you can also start from the Gaussian estimates with `prfmodel.models.prf.init_css_from_gaussian` (see the third practical).

Keep in mind that the CSS model has one more parameter and contains the Gaussian model as a special case (`n` = 1), so it can always fit the data at least as well. Whether it also predicts new data better is the topic of the third practical.

</details>

```{code-cell} ipython3
# Your code goes here
```

## References

Aqil, M., Knapen, T., & Dumoulin, S. O. (2021). Divisive normalization unifies disparate response signatures throughout the human visual hierarchy. *Proceedings of the National Academy of Sciences*, *118*(46), e2108713118. https://doi.org/10.1073/pnas.2108713118

Dumoulin, S. O., & Wandell, B. A. (2008). Population receptive field estimates in human visual cortex. *NeuroImage*, *39*(2), 647-660. https://doi.org/10.1016/j.neuroimage.2007.09.034

Kay, K. N., Winawer, J., Mezer, A., & Wandell, B. A. (2013). Compressive spatial summation in human visual cortex. *Journal of Neurophysiology*, *110*(2), 481-494. https://doi.org/10.1152/jn.00105.2013

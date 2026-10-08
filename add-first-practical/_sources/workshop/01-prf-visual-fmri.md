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

import pandas as pd

# Set keras backend to 'tensorflow' (this is normally the default)
os.environ["KERAS_BACKEND"] = "tensorflow"
# Print parameter DataFrames with three decimals
pd.set_option("display.precision", 3)
```

> **Exercise 1:** Import the prfmodel package and check whether the TensorFlow backend has been installed with GPU support (hint: use `tensorflow.config.list_physical_devices('GPU')` and `tensorflow.test.is_built_with_cuda()`).

+++

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

+++

To get an impression of the experimental stimulus, we can also visualize the stimulus design matrix projected into grid
coordinates as a video.

```{code-cell} ipython3
from IPython.display import HTML
from prfmodel.plotting import animate_2d_prf_stimulus

ani = animate_2d_prf_stimulus(stimulus, interval=25)  # Pause 25 ms between time frames

HTML(ani.to_html5_video())
```

> **Exercise 3:** Going back to Exercise 2, does the video confirm your assumptions about what the stimulus design and grid dimensions refer to?

+++

> **Exercise 4:** What do you expect the activity of a neuron population that responds to this stimulus to look like?

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

The first hidden cell defines helper functions for the interactive figures.

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
from IPython.display import display

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#8c8c8c"
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


def make_sliders(*names: str) -> dict[str, widgets.FloatSlider]:
    """Create a slider for each parameter."""
    sliders = {}
    for name in names:
        value, min_value, max_value, step = PARAM_SLIDERS[name]
        sliders[name] = widgets.FloatSlider(
            value=value,
            min=min_value,
            max=max_value,
            step=step,
            description=name,
            continuous_update=False,  # Only update the figure when the slider is released
        )
    return sliders


def show_interactive(plot_fn, sliders: dict[str, widgets.ValueWidget]) -> None:
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

+++

> **Exercise 7:** Change the tuning parameters to the same combinations as in Exercise 5:
>
> - `mu_x=2.1`, `mu_y=1.4`, `sigma=2.0`
> - `mu_x=-2.1`, `mu_y=-1.4`, `sigma=0.7`
>
> How does the predicted neuron population activity change and why? 

+++

> **Exercise 8:** What happens to the predicted neuron population activity when you move the sliders to their extremes (e.g., a very small `sigma` or a pRF center at the edge of the stimulus)? Why?

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

+++

> **Exercise 10:** Imagine a pRF model that is not applied to BOLD, but for example, electrocorticography (ECoG) measurements. Would the model still require a convolution of the predicted neuron population activity with an impulse response?

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

+++

> **Exercise 12:** Change both the `amplitude` and `sigma` parameter at the same time. How does this affect the scale of the predicted response?

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
fall inside a (dilated) gray matter mask. We can access the timecourses from the `dataset.response` attribute.

```{code-cell} ipython3
response_raw = dataset.response
response_raw.shape 
```

> **Exercise 13:** What do the dimensions of `response_raw` refer to? Visualize one or multiple timecourses with matplotlib or plotly. What is the unit of the BOLD response? Do you see any patterns in the timecourses and if so, how do they relate to the stimulus?

+++

<!-- The `response_raw` object contains the BOLD response timecourse for each voxel inside the mask. It has shape `(num_voxels, num_frames)` where `num_voxels` is the number of voxels in the mask and `num_frames` the number of time frames of the recording. The BOLD response timecourses have 340 time frames. They have already been converted to percent signal change (PSC) relative to a baseline of 100, so we subtract 100 to center each timecourse around zero. -->

+++

We transform the raw BOLD response timecourses by subtracting their baseline. This makes fitting the pRF model easier because it does not need to account for baseline differences between model predictions and observed data.

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

+++

> **Exercise 15:** Why do BOLD response patterns differ between voxels?

+++

To visualize the response on the cortical surface, we need to project it from the volume onto the surface. The volume and the surfaces of the subject share the same coordinate system, so we can use `nilearn.surface.vol_to_surf` for the projection. The function samples the volume at several depths between the pial surface and the white matter surface, so we load these surfaces as well. To map a value per voxel back into the volume, we fill the values into the voxels inside the gray matter mask from the dataset.

The projection linearly interpolates between neighboring voxels. To prevent voxels outside the mask or voxels with `NaN` values from affecting the result, we set them to zero and additionally project a volume that is one for all other (valid) voxels and zero otherwise. Dividing the first projection by the second gives a weighted average of only the valid voxels at each vertex. Vertices without any valid voxels are `NaN`.

```{code-cell} ipython3
import warnings

import numpy as np
from nilearn.image import new_img_like
from nilearn.surface import PolyMesh, vol_to_surf

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
```

We also define a helper function to plot a per-voxel statistic on the inflated surface. The inflated surfaces of both hemispheres overlap in space, so we first move them apart. The helper shows three views that focus on the visual cortex in the occipital lobe: the inner (medial) side of each hemisphere, where the calcarine sulcus lies, and both hemispheres from behind (posterior).

```{code-cell} ipython3
import matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import InMemoryMesh


def separate_hemispheres(mesh: PolyMesh, gap: float = 10.0) -> PolyMesh:
    """Shift the hemispheres of a surface mesh apart along the left-right axis so that they do not overlap."""
    left, right = mesh.parts["left"], mesh.parts["right"]
    left_coords = left.coordinates.copy()
    right_coords = right.coordinates.copy()
    left_coords[:, 0] -= left_coords[:, 0].max() + gap / 2
    right_coords[:, 0] -= right_coords[:, 0].min() - gap / 2
    return PolyMesh(
        left=InMemoryMesh(left_coords, left.faces),
        right=InMemoryMesh(right_coords, right.faces),
    )


mesh = separate_hemispheres(dataset.mesh)
num_vertices_left = mesh.parts["left"].n_vertices

# Views as (hemisphere, elevation, azimuth); medial views are rotated slightly towards the back of the brain
SURF_VIEWS = {
    "Left medial": ("left", -10, 330),
    "Posterior": ("both", 0, 270),
    "Right medial": ("right", -10, 210),
}


def plot_surf_stat_map_helper(
        stat_map: np.ndarray,
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = None,
        cmap: str = "inferno",
        cyclic: bool = False,
    ) -> tuple[plt.Figure, np.ndarray]:
    """Helper function to plot a surface with a stat map from medial and posterior views.

    Set `cyclic=True` for angles in radians (e.g., polar angle) to use the color range [-pi, pi] with the cyclic
    colormap "hsv" (like `prfmodel.plotting.plot_surface_stat_map`).
    """
    if cyclic:
        vmin, vmax, cmap = -np.pi, np.pi, "hsv"

    # Use the same color range for all views
    vmin = np.nanmin(stat_map) if vmin is None else vmin
    vmax = np.nanmax(stat_map) if vmax is None else vmax

    # Stat maps for each hemisphere (left first)
    stat_maps = {
        "left": stat_map[:num_vertices_left],
        "right": stat_map[num_vertices_left:],
        "both": stat_map,
    }

    fig, axes = plt.subplots(
        1, len(SURF_VIEWS), subplot_kw={"projection": "3d"}, figsize=(12, 4.5), layout="constrained",
    )

    for ax, (view_name, (hemi, elev, azim)) in zip(axes, SURF_VIEWS.items()):
        plot_surf_stat_map(
            mesh if hemi == "both" else mesh.parts[hemi],
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
```

We can use the helper to plot, for example, the standard deviation of each timecourse on the surface to get an overview of the variability in the BOLD timecourses across the surface.

```{code-cell} ipython3
# Calculate standard deviation of each timecourse
response_sd = response_no_baseline.std(axis=1)

plot_surf_stat_map_helper(project_to_surface(response_sd), vmax=5.0, title="BOLD response standard deviation");
```

> **Exercise 16:** What can the response standard deviation tell us about neuron activity in different brain regions? Can you find the primary visual cortex (i.e., V1)?

+++

> **Exercise 17:** Plot the average response of each timecourse on the surface. Do you see any interesting patterns?

+++

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

> **Exercise 18:** Compare the parameter estimates to the ranges of the grid search. What do you observe in the distributions of pRF centers and sizes? Change the grid ranges with the arguments of `grid_values_2d_prf` (e.g., `mu_extent`, `num_mu`, `sigma_range`; you can also create a range for `weight_deriv`). Can you get to a lower (i.e., better) mean loss across voxels (shown in the progress bar under `loss`)?

+++

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

> **Exercise 19:** Compare the SGD parameter estimates against the grid search and least-squares estimates. How do they differ? (hint: you can use `plot_2d_prf_centers` and `plot_grid_parameter_distribution`)

+++

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

> **Exercise 20:** Plot the R-squared score on the inflated surface. You can use the helper function below. For which brain regions does the model fit the data well?

```{code-cell} ipython3
def fill_valid_voxels(values: np.ndarray) -> np.ndarray:
    """Fill values for the valid voxels into an array with one value per voxel in the mask."""
    values_full = np.full((response_no_baseline.shape[0],), fill_value=np.nan)
    values_full[response_is_valid] = values
    return values_full
```

> **Exercise 21:** In addition to computing model fit with R-squared, we also recommend comparing model predictions against observed timecourses for individual voxels. Plot predictions against observed responses for voxels with different fit quality, for example, by sorting them according to R-squared (e.g., with `np.argsort(r_squared)`; hint: use `prfmodel.plotting.plot_observed_predicted`). Does the plot confirm the impression you get from the R-squared scores? Why does the model not fit well for some voxels?

+++

> **Exercise 22:** Evaluate the model fit based on the SGD parameter estimates and compare it against the grid search and least-squares fit. Does SGD improve the fit and if not, why? (hint: use `prfmodel.plotting.plot_r_squared_comparison` to compare R-squared scores).

+++

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

> **Exercise 23:** How does the pRF size change across the inflated surface? Change the R-squared threshold (`r_squared_threshold`) and recreate the plot. Does the pattern change?

+++

> **Exercise 24:** Compute the polar angle of the pRF center with `prfmodel.utils.calculate_polar_angle`. Plot the polar angle of the selected voxels on the inflated surface (hint: use `cyclic=True` in the plotting helper to get a cyclic colormap). How does polar angle change on the inflated surface? Note that the projection onto the surface averages neighboring voxels, but angles cannot be averaged directly because they wrap around from $-\pi$ to $\pi$. Instead, project `mu_x` and `mu_y` onto the surface separately and compute the polar angle afterwards.

+++

> **Exercise 25:** Compute the eccentricity (i.e., the distance of the pRF center from the center of the screen) with `prfmodel.utils.calculate_eccentricity`. Plot it for the selected voxels on the inflated surface. How does eccentricity change across the inflated surface?

+++

> **Exercise 26 (optional):** Compare the polar angle and eccentricity patterns you observed to the ones reported in the paper by Dumoulin and Wandell (2008). Do your results agree with theirs?

+++

> **Exercise 27 (optional, advanced):** Fit the compressive spatial summation (CSS) pRF model (Kay et al., 2013) to the dataset using the same workflow. The CSS pRF model is implemented in the `prfmodel.models.prf.Gaussian2DCSSPRFModel` class. Look at the API documentation of the model class to get more information about its parameters. You can also take a look at the paper by Kay et al. (2013) for more background information.

+++

## References

Dumoulin, S. O., & Wandell, B. A. (2008). Population receptive field estimates in human visual cortex. *NeuroImage*, *39*(2), 647-660. https://doi.org/10.1016/j.neuroimage.2007.09.034

Kay, K. N., Winawer, J., Mezer, A., & Wandell, B. A. (2013). Compressive spatial summation in human visual cortex. *Journal of Neurophysiology*, *110*(2), 481-494. https://doi.org/10.1152/jn.00105.2013

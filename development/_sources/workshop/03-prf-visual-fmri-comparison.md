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

# Practical 3: Model comparison and selection

This is the notebook for the third practical of the workshop. This practical covers how to compare and select between different population receptive field (pRF) models on empirical data from visual functional magnetic resonance imaging (fMRI) experiments.

**We recommend first going through the [first practical](01-prf-visual-fmri.md) and [second practical](02-prf-visual-fmri-cv.md) because this practical builds on package-specific basics and general concepts that are covered there.**

The typical goal in computational modelling is to build a model that explains and predicts a target pattern in empirical data. That is, the model should not only predict the pattern in one set of observations but generalize to all observations that stem from the same data-generating process. The [second practical](02-prf-visual-fmri-cv.md) introduced cross-validation to assess how well a pRF model can generalize its predictions from training data to new observations. Cross-validation is used to *compare* models that vary in complexity, for example, in the number of free parameters, and select the model that shows the best generalization. More complex models can fit the training data more precisely, meaning that they are in theory better able to capture the target pattern. That is, they are less likely to *underfit* the training data. However, this means that they are also more prone to fitting aspects of the data that are not part of the data-generating process for the target pattern. This is called *overfitting* the training data.

Cross-validation is a means to assess whether a pRF model is over- or underfitting. When underfitting, it will show relatively low prediction performance on the training *and* test data compared to other models (or a noise ceiling). This is also called *bias*. Conversely, when overfitting, the model will show high prediction performance on the training but relatively low prediction performance on the test data. Moreover, the prediction performance will be more variable on different test datasets the more a model overfits the training data. This test *variance* occurs because the model test prediction performance depends on variable characteristics of the data that do not belong to the data-generating process of the target pattern. The tension between bias (underfitting) and variance (overfitting) is called *bias-variance trade-off*. Given fixed training and test data, the complexity of a model typically navigates this trade-off and the goal is to find the model complexity that sits on the optimal point between under- and overfitting.

Different pRF models make different assumptions about how a neuron population responds to visual stimulation. The canonical Gaussian 2D pRF model from the first practical assumes that the response grows linearly with the part of the pRF that is covered by the stimulus. Other models add, for example, a suppressive surround around the pRF center or a compression of the response when a larger part of the pRF is stimulated. Each model is a hypothesis about the response, and comparing the models on empirical data tells us which hypothesis describes the data best.

In this practical, we compare four models: The canonical Gaussian 2D pRF model, the Compressive Spatial Summation (CSS) model, the Difference of Gaussians (DoG) model, and the Divisive Normalization (DN) model. The latter three models can mimic the Gaussian model and have more free parameters, so in principle they always fit the training data at least as well as the Gaussian model. Whether they also predict new data better is what we test with cross-validation. We also estimate the *noise ceiling* of each voxel, which tells us how well any model could possibly predict the data given the noise in it. Finally, the best model can differ between brain regions, so we select a model for each voxel separately.

+++

## Contents

- [Installation and Setup](#installation-and-setup)
- [Loading the dataset](#loading-the-dataset)
- [Splitting the run into two halves](#splitting-the-run-into-two-halves)
- [Selecting voxels with a high noise ceiling](#selecting-voxels-with-a-high-noise-ceiling)
- [Candidate models](#candidate-models)
- [Fitting the models to the first half](#fitting-the-models-to-the-first-half)
  - [Gaussian model](#gaussian-model)
  - [CSS model](#css-model)
  - [DoG model](#dog-model)
  - [DN model](#dn-model)
- [Comparing the in-sample fit](#comparing-the-in-sample-fit)
- [Comparing the out-of-sample fit](#comparing-the-out-of-sample-fit)
- [Selecting a model per voxel](#selecting-a-model-per-voxel)
- [Fitting and interpreting the selected models](#fitting-and-interpreting-the-selected-models)
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

In this practical, we use the Arrow-of-Time (AOT) dataset from the first practical. The BOLD response was recorded with a 7 Tesla MRI scanner while the subject watched a moving bar stimulus. The dataset contains a single timecourse per voxel, which is the median over several runs of the experiment. We load the dataset together with the inflated surface of the subject.

```{code-cell} ipython3
from prfmodel.examples import load_dataset

# Downloads on first use and caches in a user data directory (see prfmodel.examples.get_data_dir)
dataset = load_dataset("7t-aot-visual", surface_type="inflated")

stimulus = dataset.stimulus
```

Like in the first practical, we subtract the baseline from the raw BOLD response and only keep voxels with a response that varies over time.

```{code-cell} ipython3
response_no_baseline = dataset.response - 100.0

# Voxels with a response that varies over time
response_is_valid = response_no_baseline.std(axis=1) > 0.0

response_valid = response_no_baseline[response_is_valid]


def fill_valid_voxels(values: np.ndarray) -> np.ndarray:
    """Fill values for the valid voxels into an array with one value per voxel in the mask."""
    values_full = np.full((response_no_baseline.shape[0],), fill_value=np.nan)
    values_full[response_is_valid] = values
    return values_full


response_valid.shape
```

We reuse the helper functions from the first practical to project values for each voxel onto the surface (`project_to_surface`) and to plot them on the inflated surface (`plot_surf_stat_map_helper`). We create them with the functions from the module `plotting_helpers.py`, which lies next to this notebook.

```{code-cell} ipython3
from plotting_helpers import make_surface_plot, make_surface_projection

project_to_surface = make_surface_projection(dataset)
plot_surf_stat_map_helper = make_surface_plot(dataset.mesh)
```

## Splitting the run into two halves

For cross-validation, we need data to fit the models to (training data) and data to evaluate their predictions on (test data). The AOT dataset only contains a single (median) run, so we split the run into two halves. The stimulus contains eight bar sweeps that are separated by blank periods. The first and the second half of the run contain the same four bar sweeps, but in a different order:

| Bar sweep | First half (frames) | Second half (frames) |
|---|---|---|
| Horizontal bar, top to bottom | 30–49 | 287–307 |
| Vertical bar, left to right | 53–79 | 258–284 |
| Horizontal bar, bottom to top | 101–120 | 216–236 |
| Vertical bar, right to left | 124–150 | 186–212 |

Between the two halves lies a long blank period from frame 151 to frame 185. We split the run in the middle of this blank period.

```{code-cell} ipython3
import matplotlib.pyplot as plt

num_frames = stimulus.design.shape[0]

# A frame shows the bar if any pixel of the design matrix is non-zero
is_bar_visible = stimulus.design.reshape(num_frames, -1).max(axis=1) > 0

split = 168  # Middle of the blank period between frames 151 and 185

fig, ax = plt.subplots(figsize=(10, 2), layout="constrained")
ax.step(np.arange(num_frames), is_bar_visible, where="mid")
ax.axvline(split, color="black", linestyle="--", label="Split")
ax.set(xlabel="Time frame", yticks=[0, 1], yticklabels=["Blank", "Bar"])
ax.legend(loc="upper right");
```

> **Exercise 1:** Using what you learned in the first practical, create a video of the stimulus (hint: use `prfmodel.plotting.animate_2d_prf_stimulus`). Can you confirm the bar sweeps in the table above?

```{code-cell} ipython3
# Your code goes here
```

We create a separate stimulus for each half that only contains the frames of that half. The stimulus grid stays the same.

```{code-cell} ipython3
from prfmodel.stimuli import PRFStimulus

first_half = slice(None, split)  # Frames 0 to 167
second_half = slice(split, None)  # Frames 168 to 339

stimulus_first = PRFStimulus(design=stimulus.design[first_half], grid=stimulus.grid, dimension_labels=["y", "x"])
stimulus_second = PRFStimulus(design=stimulus.design[second_half], grid=stimulus.grid, dimension_labels=["y", "x"])

stimulus_first.design.shape, stimulus_second.design.shape
```

> **Exercise 2:** Why do we split the run in the middle of the blank period and not directly after the last bar sweep of the first half (i.e., at frame 151)? Think about how long the BOLD response to a stimulus lasts (hint: look at the impulse response in the first practical).

+++

Both halves were recorded in the same session and the noise in both halves comes from the same runs. However, the two halves contain different time frames with different noise, and the bar sweeps appear in a different order and with blank periods of different lengths in between. Thus, the model must predict the response to the same bar sweeps in a new sequence, which is similar to the test with a new stimulus in the second practical. In the following, we use the first half for training and the second half for testing.

+++

## Selecting voxels with a high noise ceiling

Not all voxels in the gray matter mask respond to the stimulus, and among those that do, the noise level varies a lot. Comparing models on voxels that are dominated by noise is not informative because no model can predict noise. Therefore, we focus on voxels with a reliable response to the stimulus.

How reliable a response is can be estimated without any model when the same stimulus is shown twice: The correlation between the two responses tells us how well the response can be reproduced. Suppose that each response is the sum of the same stimulus-driven signal and independent noise with the same variance. Then, this correlation is the fraction of the variance of one response that is caused by the stimulus. Even a perfect model cannot explain more variance than that, so the correlation estimates the *noise ceiling*: The highest R-squared that a model can reach on one of the responses.

It is important that we select voxels with a criterion that does not depend on any of the candidate models. If we selected voxels by the R-squared of the Gaussian model, we would favor voxels that the Gaussian model already describes well and could miss voxels where other models make a difference.

The AOT dataset does not contain repeated runs, but both halves of the run contain the same four bar sweeps. We cut out the frames of each sweep from both halves and correlate them. To account for the delay of the BOLD response, we shift each sweep by 5 frames (4.5 seconds). Some sweeps last one frame longer in the second half, so we use the shorter duration for both halves.

```{code-cell} ipython3
# Frame at which each bar sweep starts in the first and second half, and the number of frames we use for it
sweeps = pd.DataFrame(
    {
        "onset_first": [30, 53, 101, 124],
        "onset_second": [287, 258, 216, 186],
        "num_frames": [20, 27, 20, 27],
    },
    index=[
        "Horizontal bar, top to bottom",
        "Vertical bar, left to right",
        "Horizontal bar, bottom to top",
        "Vertical bar, right to left",
    ],
)

lag = 5  # Delay of the BOLD response in frames


def cut_sweeps(onsets: pd.Series) -> np.ndarray:
    """Concatenate the responses of all valid voxels to the bar sweeps, shifted by the delay of the BOLD response."""
    return np.concatenate(
        [response_valid[:, onset + lag:onset + lag + n] for onset, n in zip(onsets, sweeps["num_frames"])],
        axis=1,
    )


sweep_response_first = cut_sweeps(sweeps["onset_first"])
sweep_response_second = cut_sweeps(sweeps["onset_second"])

sweep_response_first.shape
```

For each voxel, we compute the correlation between the responses to the sweeps in both halves.

```{code-cell} ipython3
def correlate_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation between each row of x and the same row of y."""
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    return (x * y).sum(axis=1) / np.sqrt((x**2).sum(axis=1) * (y**2).sum(axis=1))


# Voxels with a constant response during the sweeps get a NaN correlation
with np.errstate(invalid="ignore"):
    noise_ceiling = correlate_rows(sweep_response_first, sweep_response_second)

noise_ceiling.shape
```

We look at the distribution of the noise ceiling across voxels with `prfmodel.plotting.plot_r_squared_hist` and plot the noise ceiling on the surface.

```{code-cell} ipython3
from prfmodel.plotting import plot_r_squared_hist

plot_r_squared_hist(noise_ceiling[np.isfinite(noise_ceiling)], value_range=(-1.0, 1.0), log=True, xlabel="Noise ceiling");
```

```{code-cell} ipython3
plot_surf_stat_map_helper(
    project_to_surface(fill_valid_voxels(noise_ceiling)), vmin=0.0, vmax=1.0, title="Noise ceiling",
);
```

> **Exercise 3:** Where on the surface do the voxels with a high noise ceiling lie? Compare the map with the map of the response standard deviation from the first practical (hint: compute the standard deviation of `response_valid` and project it onto the surface). Are there voxels with a large standard deviation but a low noise ceiling? What could cause their variability?

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 4:** The two halves do not show exactly the same stimulus: The same sweep follows a different sweep or blank period in each half, and some sweeps last one frame longer in one of the halves. Why does this make our estimate of the noise ceiling lower than the noise ceiling we would get from two runs with the identical stimulus?

+++

We select voxels with a noise ceiling above 0.7. Note that this threshold is somewhat arbitrary. We split the responses of the selected voxels into the two halves.

```{code-cell} ipython3
noise_ceiling_threshold = 0.7

# NaN values are never above the threshold
is_selected = noise_ceiling > noise_ceiling_threshold

response_first = response_valid[is_selected][:, first_half]
response_second = response_valid[is_selected][:, second_half]

noise_ceiling_selected = noise_ceiling[is_selected]


def fill_selected_voxels(values: np.ndarray) -> np.ndarray:
    """Fill values for the selected voxels into an array with one value per voxel in the mask, NaN for all others."""
    values_valid = np.full((response_valid.shape[0],), fill_value=np.nan)
    values_valid[is_selected] = values
    return fill_valid_voxels(values_valid)


response_first.shape, response_second.shape
```

> **Exercise 5:** How many voxels would you select with a threshold of 0.5 or 0.9? What are the advantages and disadvantages of a lower and a higher threshold for comparing models?

```{code-cell} ipython3
# Your code goes here
```

## Candidate models

We compare four pRF models that make different assumptions about how the neuron population responds to the stimulus:

| Model | Class in `prfmodel.models.prf` | Assumption | Additional parameters | Free parameters |
|---|---|---|---|---|
| Gaussian (Dumoulin & Wandell, 2008) | `Gaussian2DPRFModel` | The response grows linearly with the stimulated part of the pRF | – | 5 |
| CSS (Kay et al., 2013) | `Gaussian2DCSSPRFModel` | The response grows less than linearly (compressive) with the stimulated part of the pRF | `n` (compression exponent) | 6 |
| DoG (Zuiderbaan et al., 2012) | `DoG2DPRFModel` | Stimulation around the pRF center suppresses the response (suppressive surround) | `sigma_surround`, `amplitude_surround` | 7 |
| DN (Aqil et al., 2021) | `DivNormGaussian2DPRFModel` | The response to the pRF center is divided by the response to a larger normalization pool, which produces both suppression and compression | `sigma_normalization`, `baseline_activation`, `baseline_normalization` | 8 |

The free parameters are the parameters that we estimate for each voxel (we fix `weight_deriv` for all models). The models are ordered by their number of free parameters. They can (approximately) mimic each other: The CSS model with `n` equal to 1 and the DoG model without a surround (`amplitude_surround` equal to 0) are equivalent to the Gaussian model, and the DN model behaves like the DoG model when normalization is weak and produces compression like the CSS model when it is strong (Aqil et al., 2021).

Fitting models with many parameters is difficult because the loss function has many local minima. We therefore fit the models in sequence and use the estimates of simpler models as starting values for more complex ones. prfmodel provides helper functions for this, for example, `prfmodel.models.prf.init_css_from_gaussian`. After initializing a model, we estimate its amplitude and baseline parameters with least-squares and then fine-tune all free parameters with stochastic gradient descent (SGD).

For a fair comparison, we use the same SGD settings for all models, including the Gaussian model. Otherwise, a model could fit better only because it was optimized for longer. We use a larger learning rate than the default of the `Adam` optimizer (0.001) so that 500 steps are enough to reach a stable fit.

```{code-cell} ipython3
from keras import ops
from keras.optimizers import Adam
from prfmodel.fitters import GridFitter, LeastSquaresFitter, SGDFitter, grid_values_2d_prf
from prfmodel.fitters.adapter import Adapter, ParameterTransform
from prfmodel.impulse import DerivativeTwoGammaImpulse

# Same SGD settings for all models
num_steps = 500
learning_rate = 0.01

# Define repetition time (TR)
tr = 0.9

# Create custom impulse model that is shared by all models
impulse_model = DerivativeTwoGammaImpulse(resolution=tr)
```

To see how the additional parameters of the CSS, DoG, and DN models change their predictions, we use interactive figures like in the first practical. The code that creates the figures is hidden: Run a hidden cell to show its figure and move the sliders to change the parameters.

Each figure shows a pRF in the center of the visual field with a size of 1 degree. The left panel shows the predicted neuron population response $r$ (before the convolution with the impulse response) while a vertical bar like the one in the experiment is at different horizontal positions in the visual field. The right panel shows the predicted BOLD response to the first half of the run. The dashed gray lines show the predictions of the Gaussian model with the same center and size. Because the additional parameters also change the scale of the predictions, we scale all predictions to a maximum of 1 so that we can compare their shapes.

The next hidden cell defines helper functions for the interactive figures. Like in the first practical, some of them are imported from the module `plotting_helpers.py`.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Helper functions for the interactive figures { display-mode: "form" }
from plotting_helpers import BLUE, GRAY, add_legend, make_slider, show_interactive, style_axes, to_frame
from prfmodel.models.prf import Gaussian2DPRFModel

# All figures show a pRF in the center of the visual field
CENTER_PARAMS = {"mu_x": 0.0, "mu_y": 0.0, "weight_deriv": -0.5, "baseline": 0.0}
GAUSSIAN_PARAMS = {"sigma": 1.0, "amplitude": 1.0}

# Stimulus with one vertical bar per frame; the bar moves from left to right over frames
bar_width = 1.7  # Width of the bar in the experiment (degrees visual angle)
bar_positions = np.linspace(-4.0, 4.0, 81)
bar_stimulus = PRFStimulus(
    design=(np.abs(stimulus.grid[..., 1] - bar_positions[:, None, None]) <= bar_width / 2).astype("float32"),
    grid=stimulus.grid,
    dimension_labels=["y", "x"],
)

time_frames_first = np.arange(stimulus_first.design.shape[0])


def scale_to_max(values: np.ndarray) -> np.ndarray:
    """Scale values so that their largest absolute value is 1."""
    values = np.asarray(values)
    max_value = np.abs(values).max()
    return values / max_value if max_value > 0 else values


def plot_model_comparison(model_class: type, model_name: str, **params: float) -> None:
    """Plot the bar response and the predicted BOLD response of a model next to those of the Gaussian model."""
    fig, (ax_bar, ax_response) = plt.subplots(
        1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 3]}, layout="constrained",
    )

    for current_class, current_name, current_params, style in [
        (Gaussian2DPRFModel, "Gaussian", GAUSSIAN_PARAMS, {"color": GRAY, "linestyle": "--", "linewidth": 1.5}),
        (model_class, model_name, params, {"color": BLUE, "linewidth": 2}),
    ]:
        param_frame = to_frame(**CENTER_PARAMS, **current_params)
        # Without an impulse response, every frame of the bar stimulus gives the response to the bar at one position
        bar_response = current_class(impulse_model=None)(bar_stimulus, param_frame)[0]
        bold_response = current_class(impulse_model=impulse_model)(stimulus_first, param_frame)[0]
        ax_bar.plot(bar_positions, scale_to_max(bar_response), label=current_name, **style)
        ax_response.plot(time_frames_first, scale_to_max(bold_response), label=current_name, **style)

    ax_bar.set_title("Response to the bar", loc="left")
    style_axes(ax_bar, "Bar position (degrees visual angle)", "Scaled response")

    add_legend(ax_response)
    ax_response.set_title("Predicted BOLD response", loc="left")
    style_axes(ax_response, "Time frame", "Scaled BOLD response")

    plt.show()
```

## Fitting the models to the first half

### Gaussian model

All four models build on the Gaussian model, so we first recall how it makes a prediction. Its tuning profile is the isotropic 2D Gaussian density centered at $(\mu_x, \mu_y)$ with standard deviation $\sigma$:

$$
G(x, y; \mu_x, \mu_y, \sigma) = \frac{1}{2 \pi \sigma^2} \exp\left(-\frac{(x - \mu_x)^2 + (y - \mu_y)^2}{2 \sigma^2}\right)
$$

The tuning profile is encoded with the stimulus design $S$ by summing the overlap over all points $(x, y)$ of the stimulus grid at every time frame $t$:

$$
P(t; \mu_x, \mu_y, \sigma) = \sum_{x, y} S(t, x, y) \, G(x, y; \mu_x, \mu_y, \sigma)
$$

For the Gaussian model, this encoded response is the predicted neuron population response $r(t) = P(t; \mu_x, \mu_y, \sigma)$. It is convolved with the impulse response $h$ and scaled with the parameters `amplitude` ($\beta$) and `baseline` ($\alpha$) to give the predicted BOLD response:

$$
\hat{y}(t) = \beta \, (h * r)(t) + \alpha
$$

The other models only differ in how they compute $r(t)$ from one or two encoded Gaussians. All Gaussians of a model share the same center $(\mu_x, \mu_y)$.

We fit the Gaussian model in the same way as in the first practical: grid search, least-squares, and SGD. We only use the stimulus and the responses of the first half.

Unlike in the first practical, the grid only contains pRF centers within the visual field covered by the stimulus. The grid search compares predictions and responses with a correlation, which does not depend on the scale of the prediction. For a small pRF with a center far outside the stimulus, the prediction only comes from the outer tail of the Gaussian and is tiny. Its correlation with the response can still be high, but the prediction is so small that least-squares cannot estimate its amplitude, and the fit fails. Starting the pRF centers within the stimulus avoids this, and SGD can still move them outside.

```{code-cell} ipython3
from prfmodel.models.prf import Gaussian2DPRFModel

gaussian_model = Gaussian2DPRFModel(impulse_model=impulse_model)

grid_param_ranges = grid_values_2d_prf(
    stimulus_first,
    num_mu=11,
    mu_extent=1.0,  # Only span the visual field covered by the stimulus
    num_sigma=10,
    sigma_range=(0.05, 4.0),  # Log-spaced by default
) | {  # Add constant parameters
    "weight_deriv": [-0.5],
    "baseline": [0.0],
    "amplitude": [1.0],
}

grid_fitter = GridFitter(
    model=gaussian_model,
    stimulus=stimulus_first,
    compile_step=True,  # Setting 'compile_step=True' speeds up the fitting
)

grid_history, grid_params = grid_fitter.fit(
    data=response_first,
    parameter_values=grid_param_ranges,
    batch_size=20,
)

ls_fitter = LeastSquaresFitter(
    model=gaussian_model,
    stimulus=stimulus_first,
)

ls_history, ls_params = ls_fitter.fit(
    data=response_first,
    parameters=grid_params,
    slope_name="amplitude",
    intercept_name="baseline",
    batch_size=200,
)
```

```{code-cell} ipython3
gaussian_sgd_fitter = SGDFitter(
    model=gaussian_model,
    stimulus=stimulus_first,
    adapter=Adapter([ParameterTransform(["sigma"], transform_fun=ops.log, inverse_fun=ops.exp)]),
    optimizer=Adam(learning_rate=learning_rate),
    compile_step=True,
)

gaussian_history, gaussian_params = gaussian_sgd_fitter.fit(
    data=response_first,
    init_parameters=ls_params,
    fixed_parameters=["weight_deriv"],
    num_steps=num_steps,
)

gaussian_params
```

### CSS model

The CSS model raises the stimulus-encoded pRF response to the power of the compression exponent `n` before it is convolved with the impulse response:

$$
r(t) = g \, P(t; \mu_x, \mu_y, \sigma)^n
$$

With `n` below 1, the response grows less than linearly with the part of the pRF that is covered by the stimulus. The model also has a `gain` parameter $g$ that multiplies the compressed response. It has the same effect as `amplitude`, so we fix it to its starting value of 1. The predicted BOLD response $\hat{y}(t)$ is computed in the same way as for the Gaussian model.

Move the `n` slider in the figure below to see how the exponent changes the response to the bar and the predicted BOLD response. With `n` equal to 1, the CSS model makes the same predictions as the Gaussian model.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: CSS model { display-mode: "form" }
from prfmodel.models.prf import Gaussian2DCSSPRFModel


def plot_css(n: float) -> None:
    plot_model_comparison(Gaussian2DCSSPRFModel, "CSS", sigma=1.0, gain=1.0, n=n, amplitude=1.0)


show_interactive(plot_css, {"n": make_slider("n", 0.5, 0.05, 1.5, 0.05)})
```

We start from the Gaussian estimates and an exponent of 0.5. Because the exponent changes the scale of the prediction, we re-estimate `amplitude` and `baseline` with least-squares. In SGD, we optimize `sigma` and `n` in log space to keep them positive.

```{code-cell} ipython3
from prfmodel.models.prf import Gaussian2DCSSPRFModel, init_css_from_gaussian

css_model = Gaussian2DCSSPRFModel(impulse_model=impulse_model)

# Start from the Gaussian estimates with n = 0.5 and gain = 1
css_init_params = init_css_from_gaussian(gaussian_params)

_, css_init_params = LeastSquaresFitter(model=css_model, stimulus=stimulus_first).fit(
    data=response_first,
    parameters=css_init_params,
    slope_name="amplitude",
    intercept_name="baseline",
    batch_size=200,
)

css_sgd_fitter = SGDFitter(
    model=css_model,
    stimulus=stimulus_first,
    adapter=Adapter([ParameterTransform(["sigma", "n"], transform_fun=ops.log, inverse_fun=ops.exp)]),
    optimizer=Adam(learning_rate=learning_rate),
    compile_step=True,
)

css_history, css_params = css_sgd_fitter.fit(
    data=response_first,
    init_parameters=css_init_params,
    fixed_parameters=["weight_deriv", "gain"],
    num_steps=num_steps,
)

css_params
```

### DoG model

The DoG model subtracts the response of a larger surround Gaussian from the response of a center Gaussian:

$$
r(t) = \beta_c \, P(t; \mu_x, \mu_y, \sigma_c) - \beta_s \, P(t; \mu_x, \mu_y, \sigma_s)
$$

The parameters $\beta_c$, $\beta_s$, $\sigma_c$, and $\sigma_s$ are called `amplitude_center`, `amplitude_surround`, `sigma_center`, and `sigma_surround`. A positive `amplitude_surround` means that stimulation of the surround suppresses the response. Because the two amplitudes already scale the response, the DoG model has no separate `amplitude` parameter, and the predicted BOLD response is $\hat{y}(t) = (h * r)(t) + \alpha$.

In the figure below, the center Gaussian has `amplitude_center` and `sigma_center` equal to 1. Move the sliders to see how the size and the strength of the surround change the response to the bar and the predicted BOLD response.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: DoG model { display-mode: "form" }
from prfmodel.models.prf import DoG2DPRFModel


def plot_dog(sigma_surround: float, amplitude_surround: float) -> None:
    plot_model_comparison(
        DoG2DPRFModel,
        "DoG",
        sigma_center=1.0,
        sigma_surround=sigma_surround,
        amplitude_center=1.0,
        amplitude_surround=amplitude_surround,
    )


show_interactive(
    plot_dog,
    {
        "sigma_surround": make_slider("sigma_surround", 2.0, 1.0, 6.0, 0.1),
        "amplitude_surround": make_slider("amplitude_surround", 0.5, 0.0, 1.0, 0.05),
    },
)
```

We start from the Gaussian estimates with a surround that is three times as large as the center and very weak (`amplitude_surround` equal to 0.001). Both amplitudes enter the prediction linearly, so we can estimate them together with the baseline with least-squares. For some voxels, least-squares returns a negative `amplitude_surround`, that is, a surround that increases the response. In SGD, we optimize `amplitude_surround` in log space to keep the surround suppressive, so we only use the least-squares estimates for voxels with a suppressive surround. The other voxels keep the starting values from the Gaussian estimates with the very weak surround.

```{code-cell} ipython3
from prfmodel.models.prf import DoG2DPRFModel, init_dog_from_gaussian

dog_model = DoG2DPRFModel(impulse_model=impulse_model)

# Start from the Gaussian estimates with a weak surround that is three times as large as the center
dog_init_params = init_dog_from_gaussian(gaussian_params, sigma_ratio=3.0)

_, dog_ls_params = LeastSquaresFitter(model=dog_model, stimulus=stimulus_first).fit(
    data=response_first,
    parameters=dog_init_params,
    slope_name=["amplitude_center", "amplitude_surround"],
    intercept_name="baseline",
    batch_size=200,
)

# Only use the least-squares estimates for voxels with a suppressive surround
is_suppressive = dog_ls_params["amplitude_surround"] > 0.0
dog_init_params.loc[is_suppressive] = dog_ls_params.loc[is_suppressive]

is_suppressive.mean()

dog_sgd_fitter = SGDFitter(
    model=dog_model,
    stimulus=stimulus_first,
    adapter=Adapter([
        ParameterTransform(
            ["sigma_center", "sigma_surround", "amplitude_surround"], transform_fun=ops.log, inverse_fun=ops.exp,
        ),
    ]),
    optimizer=Adam(learning_rate=learning_rate),
    compile_step=True,
)

dog_history, dog_params = dog_sgd_fitter.fit(
    data=response_first,
    init_parameters=dog_init_params,
    fixed_parameters=["weight_deriv"],
    num_steps=num_steps,
)

dog_params
```

### DN model

The DN model combines an activation Gaussian and a larger normalization Gaussian. The encoded response of the activation Gaussian $A(t) = P(t; \mu_x, \mu_y, \sigma_A)$ is divided by the encoded response of the normalization Gaussian $N(t) = P(t; \mu_x, \mu_y, \sigma_N)$:

$$
r(t) = \frac{a A(t) + b}{c N(t) + d} - \frac{b}{d}
$$

The standard deviations $\sigma_A$ and $\sigma_N$ are called `sigma_activation` and `sigma_normalization`.

The parameters $a$, $b$, $c$, and $d$ are called `amplitude_activation`, `baseline_activation`, `amplitude_normalization`, and `baseline_normalization`. Multiplying all four parameters by the same factor does not change the predicted response, meaning that different combinations of the four parameters can lead to the same predicted response and model fit. The consequence is that the DN model with four free parameters is *non-identifiable* which makes estimating the parameters difficult. To make parameter estimation easier and the model identifiable, we only estimate three of the four parameters and fix $c$ to 1.

The subtraction of $b/d$ makes sure that the response is zero when there is no stimulus. Like for the DoG model, $a$ already scales the response, so the predicted BOLD response is $\hat{y}(t) = (h * r)(t) + \alpha$.

In the figure below, $a$, $c$, and `sigma_activation` are equal to 1. Move the sliders to see how the size of the normalization Gaussian and the two baselines change the response to the bar and the predicted BOLD response. Because the baselines can be very small or very large, their sliders change them on a logarithmic scale.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: DN model { display-mode: "form" }
from prfmodel.models.prf import DivNormGaussian2DPRFModel


def plot_dn(sigma_normalization: float, baseline_activation: float, baseline_normalization: float) -> None:
    plot_model_comparison(
        DivNormGaussian2DPRFModel,
        "DN",
        sigma_activation=1.0,
        sigma_normalization=sigma_normalization,
        amplitude_activation=1.0,
        amplitude_normalization=1.0,
        baseline_activation=baseline_activation,
        baseline_normalization=baseline_normalization,
    )


show_interactive(
    plot_dn,
    {
        "sigma_normalization": make_slider("sigma_normalization", 2.0, 1.0, 6.0, 0.1),
        "baseline_activation": make_slider("baseline_activation", 1.0, 0.01, 100.0, 0.1, log=True),
        "baseline_normalization": make_slider("baseline_normalization", 1.0, 0.01, 100.0, 0.1, log=True),
    },
)
```

> **Exercise 6:** Check that the DN model is non-identifiable: Create a `DivNormGaussian2DPRFModel` and parameters for a single voxel (e.g., with the values from the interactive figure above) and predict the response to the first half of the run. Then multiply `amplitude_activation`, `baseline_activation`, `amplitude_normalization`, and `baseline_normalization` by the same constant (e.g., 2 or 10) and predict the response again. How does the prediction change? What happens when you only multiply some of the four parameters by the constant?

```{code-cell} ipython3
# Your code goes here
```

Good starting values for the DN model are hard to find. The helper function `prfmodel.models.prf.init_div_norm_from_dog_css` approximates them from the DoG estimates (suppression) and the CSS estimates (compression). It expects a CSS exponent below 1, so we limit the exponent to 0.9 for the few voxels where the CSS model estimated no compression. It also warns if the DoG surround of a voxel ended up smaller than its center. This can happen for a few voxels and is not a problem because SGD can still adjust the sizes of both Gaussians. The response is linear in $a$ and $b$, so we estimate them with least-squares. Like for the DoG model, least-squares returns negative values for some voxels, which we replace with very small positive values because we optimize $a$ and $b$ in log space.

```{code-cell} ipython3
from prfmodel.models.prf import DivNormGaussian2DPRFModel, init_div_norm_from_dog_css

dn_model = DivNormGaussian2DPRFModel(impulse_model=impulse_model)

# Approximate suppression from the DoG estimates and compression from the CSS estimates (n must be below 1)
dn_init_params = init_div_norm_from_dog_css(dog_params, css_n=css_params["n"].clip(upper=0.9), stimulus=stimulus_first)

_, dn_init_params = LeastSquaresFitter(model=dn_model, stimulus=stimulus_first).fit(
    data=response_first,
    parameters=dn_init_params,
    slope_name=["amplitude_activation", "baseline_activation"],
    intercept_name="baseline",
    batch_size=200,
)

# Keep a and b positive
dn_init_params[["amplitude_activation", "baseline_activation"]] = (
    dn_init_params[["amplitude_activation", "baseline_activation"]].clip(lower=1e-3)
)

dn_sgd_fitter = SGDFitter(
    model=dn_model,
    stimulus=stimulus_first,
    adapter=Adapter([
        ParameterTransform(
            [
                "sigma_activation",
                "sigma_normalization",
                "amplitude_activation",
                "baseline_activation",
                "baseline_normalization",
            ],
            transform_fun=ops.log,
            inverse_fun=ops.exp,
        ),
    ]),
    optimizer=Adam(learning_rate=learning_rate),
    compile_step=True,
)

dn_history, dn_params = dn_sgd_fitter.fit(
    data=response_first,
    init_parameters=dn_init_params,
    fixed_parameters=["weight_deriv", "amplitude_normalization"],
    num_steps=num_steps,
)

dn_params
```

> **Exercise 7:** Print the parameter names of each model (e.g., `css_model.parameter_names`) and compare them with the table above. Then, look at the distributions of the CSS exponent `n` and the ratio between the DoG surround and center sizes (`sigma_surround / sigma_center`). What do they tell you about compression and suppression in the selected voxels?

```{code-cell} ipython3
# Your code goes here
```

## Comparing the in-sample fit

We first compare how well the models fit the training data. We collect the models and their parameter estimates in dictionaries and compute the in-sample R-squared of each model.

```{code-cell} ipython3
from prfmodel.utils import batched, calculate_r_squared

model_names = ["Gaussian", "CSS", "DoG", "DN"]  # From the simplest to the most complex model

models = dict(zip(model_names, [gaussian_model, css_model, dog_model, dn_model]))
model_params = dict(zip(model_names, [gaussian_params, css_params, dog_params, dn_params]))

# Make predictions for batches of voxels to save memory
predict_batched = {name: batched(model) for name, model in models.items()}

pred_response_first = {
    name: np.asarray(predict_batched[name](stimulus_first, model_params[name], batch_size=200)) for name in model_names
}

# One column per model and one row per selected voxel
r_squared_train = pd.DataFrame(
    {name: calculate_r_squared(response_first, pred_response_first[name]) for name in model_names},
)

r_squared_train.agg(["mean", "median"])
```

We compare the in-sample R-squared of each model against the Gaussian model for each voxel.

```{code-cell} ipython3
from prfmodel.plotting import plot_r_squared_comparison

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")

for ax, name in zip(axes, model_names[1:]):
    plot_r_squared_comparison(
        r_squared_train["Gaussian"],
        r_squared_train[name],
        value_range=(0.0, 1.0),
        xlabel="R-squared Gaussian (training set)",
        ylabel=f"R-squared {name} (training set)",
        ax=ax,
    )
```

> **Exercise 8:** Which model fits the training data best? Why do the CSS, DoG, and DN models almost never fit worse than the Gaussian model? What would it mean if a more complex model fit a voxel clearly worse than the Gaussian model? Does the in-sample fit tell us which model is the best description of the data?

```{code-cell} ipython3
# Your code goes here
```

## Comparing the out-of-sample fit

Now, we compare the predictions of the models for the second half, which none of the models were fitted to. We make predictions with the stimulus of the second half and the parameters estimated on the first half.

```{code-cell} ipython3
pred_response_second = {
    name: np.asarray(predict_batched[name](stimulus_second, model_params[name], batch_size=200)) for name in model_names
}

r_squared_test = pd.DataFrame(
    {name: calculate_r_squared(response_second, pred_response_second[name]) for name in model_names},
)

pd.DataFrame({"training set": r_squared_train.mean(), "test set": r_squared_test.mean()})
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")

for ax, name in zip(axes, model_names[1:]):
    plot_r_squared_comparison(
        r_squared_test["Gaussian"],
        r_squared_test[name],
        value_range=(0.0, 1.0),
        xlabel="R-squared Gaussian (test set)",
        ylabel=f"R-squared {name} (test set)",
        ax=ax,
    )
```

> **Exercise 9:** Compare the out-of-sample R-squared of the models. Is the ranking of the models the same as for the in-sample fit? Which model generalizes best on average? How large is the advantage of the more complex models compared to the in-sample fit?

```{code-cell} ipython3
# Your code goes here
```

We compare the out-of-sample R-squared of each model with the noise ceiling of each voxel. Voxels on the identity line are predicted as well as the noise allows.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), layout="constrained")

for ax, name in zip(axes, model_names):
    plot_r_squared_comparison(
        noise_ceiling_selected,
        r_squared_test[name],
        value_range=(0.0, 1.0),
        xlabel="Noise ceiling",
        ylabel=f"R-squared {name} (test set)",
        ax=ax,
    )
```

> **Exercise 10:** How close do the models get to the noise ceiling? Are there voxels for which a model predicts the test data better than the noise ceiling? Considering how we estimated the noise ceiling, how is this possible? There are also voxels with a high noise ceiling that none of the models predicts well. Plot the responses of some of these voxels: What do they have in common, and why can't our models describe them?

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 11:** Find the voxels for which the out-of-sample R-squared of the CSS, DoG, or DN model is much higher than for the Gaussian model. Plot the observed test timecourses together with the predictions of all four models for some of these voxels (hint: use `prfmodel.plotting.plot_observed_predicted` with a dictionary of predictions). Which features of the response does the Gaussian model miss?

```{code-cell} ipython3
# Your code goes here
```

So far, we compared the models across all selected voxels. However, neuron populations in different parts of the visual cortex can respond differently to stimulation: For example, compression is stronger in higher visual areas than in V1 (Kay et al., 2013). A model that describes one region best may not be the best model for another region. Therefore, we now select a model for each voxel separately.

+++

## Selecting a model per voxel

For each voxel, we select the model with the highest out-of-sample R-squared. However, the out-of-sample R-squared is itself an estimate that contains noise. When two models predict the test data almost equally well, which of them comes out on top is partly chance. In such cases, we prefer the simpler model because it explains the data with fewer assumptions. We therefore only select a more complex model if its out-of-sample R-squared is higher than that of *every* simpler model by more than a minimum gain. Note that the value of the minimum gain is somewhat arbitrary.

```{code-cell} ipython3
def select_models(r_squared: pd.DataFrame, min_gain: float) -> np.ndarray:
    """Select the most complex model that beats all simpler models by more than min_gain for each voxel.

    The columns of r_squared must be ordered from the simplest to the most complex model.
    """
    scores = r_squared.to_numpy()
    selected = np.zeros(scores.shape[0], dtype=int)  # Start with the simplest model for all voxels
    for i in range(1, scores.shape[1]):
        beats_simpler = np.all(scores[:, [i]] > scores[:, :i] + min_gain, axis=1)
        selected[beats_simpler] = i
    return r_squared.columns.to_numpy()[selected]


min_gain = 0.01

best_model = select_models(r_squared_test[model_names], min_gain=min_gain)

# Fraction of selected voxels for which each model is the best model
pd.Series(best_model).value_counts(normalize=True).reindex(model_names, fill_value=0.0)
```

Selecting the model with the highest score among several noisy scores has a catch: The score of the selected model overestimates how well it will predict new data. Some models win only because the noise in the test data happens to favor them. This is the same problem as with the in-sample fit in the second practical, only one level higher: We used the test data to *select* the model, so we cannot use the same test data to fairly *evaluate* the selection. For a fair evaluation, we need data that was not used for the selection, for example, the other fold of the cross-validation (see the optional exercise below).

+++

To see where on the surface each model is selected, we cannot simply project the model names (or numbers) onto the surface because the projection averages neighboring voxels. Instead, we project for each model whether it was selected (1) or not (0). After the projection, each vertex shows the fraction of voxels around it for which the model was selected.

```{code-cell} ipython3
for name in model_names:
    is_best = (best_model == name).astype(float)
    plot_surf_stat_map_helper(
        project_to_surface(fill_selected_voxels(is_best)),
        vmin=0.0,
        vmax=1.0,
        title=f"Fraction of voxels for which {name} is selected",
    )
```

> **Exercise 12:** Where on the surface is each model selected? Compare the maps with the map of the noise ceiling. Do you see systematic differences between regions?

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 13:** Select the models with a minimum gain of 0 and of 0.05. How do the fractions of selected models change and why?

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 14:** Compute how much the out-of-sample R-squared of each model improves over the Gaussian model for each voxel. Plot the improvement against the eccentricity and the size (`sigma`) of the pRF estimated with the Gaussian model (hint: use `prfmodel.utils.calculate_eccentricity`; averaging the improvement within bins of eccentricity or size makes the pattern easier to see). You can also plot the improvement on the surface with a diverging colormap (e.g., `cmap="RdBu_r"` with a symmetric color range). For which pRFs do the more complex models make the largest difference? Keep in mind that the stimulus only covers the visual field up to 4 degrees from the center horizontally and vertically: How well can the models describe pRFs with centers outside of this region?

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 15 (optional):** Repeat the fitting with the halves swapped: Fit all four models to the second half and evaluate them on the first half. Select the models for this fold with `select_models`. For which fraction of the voxels is the same model selected in both folds? What does this tell you about how reliable the selection for individual voxels is? Also average the out-of-sample R-squared of each model across both folds and compare the models again.

```{code-cell} ipython3
# Your code goes here
```

## Fitting and interpreting the selected models

Once we have selected a model for each voxel, we want to estimate its parameters as precisely as possible so that we can interpret them. Like in the second practical, we now use all data, that is, the full run with both halves. The in-sample R-squared of these fits is optimistic, but we accept this because cross-validation has already told us how well the models generalize.

When we interpret the parameters of the selected models, we need to keep in mind that the parameters of different models do not always mean the same thing. All four models have a pRF center (`mu_x` and `mu_y`), so we can compare the polar angle and eccentricity of the pRF across voxels with different selected models. In contrast, the size parameters differ between models: For example, the `sigma` of the CSS model describes the Gaussian before the compression, which makes the pRF appear larger than its `sigma` suggests (Kay et al., 2013). The size of the center Gaussian of the DoG model (`sigma_center`) is also not the same as the size of a Gaussian pRF without a surround. A map of pRF sizes that mixes the size parameters of different models would therefore mix different definitions of size.

+++

> **Exercise 16:** Fit all four models to the full run (`stimulus` and `response_valid[is_selected]`) using the same steps as above. For each voxel, take the pRF center of its selected model (hint: use `np.select` with one condition per model) and plot the polar angle and eccentricity on the surface (hint: use `prfmodel.utils.calculate_polar_angle` and `prfmodel.utils.calculate_eccentricity`, and project `mu_x` and `mu_y` onto the surface separately like in the first practical). Do the maps differ from the maps you get from the Gaussian model alone?

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 17:** Interpret the additional parameters of the more complex models, but only for the voxels for which the respective model is selected. For example, plot the CSS exponent `n` and the ratio between the DoG surround and center sizes against the eccentricity of the pRF. How do compression and suppression change with eccentricity?

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 18:** Plot the surround and normalization pRFs of the DoG and DN model on the stimulus grid. How much of these pRFs is covered by the stimulus? Does the stimulus design provide enough information to estimate these pRFs? How does this relate to whether the DoG and DN model fit a voxel better than the Gaussian and CSS model?

+++

## Summary

- Models with more free parameters that can mimic simpler models always fit the training data at least as well. Therefore, we cannot use the in-sample fit to compare models.
- We compare models by their out-of-sample fit using cross-validation. When a dataset only contains a single run, we can split the run into two halves, as long as both halves sample the stimulus space well and the split leaves enough time for the BOLD response to return to baseline.
- The noise ceiling estimates how well any model could possibly predict a voxel given the noise in its response. Selecting voxels by a criterion that does not depend on any of the candidate models avoids favoring one of them.
- The best model can differ between voxels. When we select a model for each voxel, we should prefer simpler models when the difference is small, and keep in mind that the score of the selected model is optimistic. A fair evaluation of the selection requires data that was not used for it.
- Parameters of different models are only directly comparable when they mean the same thing, such as the pRF center.

## References

Aqil, M., Knapen, T., & Dumoulin, S. O. (2021). Divisive normalization unifies disparate response signatures throughout the human visual hierarchy. *Proceedings of the National Academy of Sciences*, *118*(46), e2108713118. https://doi.org/10.1073/pnas.2108713118

Dumoulin, S. O., & Wandell, B. A. (2008). Population receptive field estimates in human visual cortex. *NeuroImage*, *39*(2), 647-660. https://doi.org/10.1016/j.neuroimage.2007.09.034

Kay, K. N., Winawer, J., Mezer, A., & Wandell, B. A. (2013). Compressive spatial summation in human visual cortex. *Journal of Neurophysiology*, *110*(2), 481-494. https://doi.org/10.1152/jn.00105.2013

Zuiderbaan, W., Harvey, B. M., & Dumoulin, S. O. (2012). Modeling center-surround configurations in population receptive fields using fMRI. *Journal of Vision*, *12*(3), 10. https://doi.org/10.1167/12.3.10

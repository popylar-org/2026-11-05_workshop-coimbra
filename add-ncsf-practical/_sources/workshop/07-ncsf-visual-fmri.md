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

# Practical 7: The neural contrast sensitivity function model

This is the notebook for the neural contrast sensitivity function practical of the workshop. This practical covers the neural contrast sensitivity function (nCSF) model by Roelofzen et al. (2025), its different computational parts, and how to fit it to empirical data from a visual functional magnetic resonance imaging (fMRI) experiment.

**We recommend going through the [first practical](01-prf-visual-fmri.md) on the canonical Gaussian population receptive field (pRF) model first to get familiar with the prfmodel package before starting with this practical.**

The first part of this practical goes through the nCSF model step-by-step. The model input and output as well as every submodel is motivated and explained with interactive visualizations. The second part is about fitting the model to an empirical dataset obtained with a 7T fMRI scanner. This part covers data inspection and visualization on an inflated surface of the brain. It goes through multiple consecutive fitting stages (grid search, least-squares, and SGD) and concludes with evaluating the model fit and interpreting its parameters.

+++

## Contents

- [Installation and Setup](#installation-and-setup)
- [The neural contrast sensitivity function model](#the-neural-contrast-sensitivity-function-model)
  - [Input: Spatial frequency and contrast](#input-spatial-frequency-and-contrast)
  - [Building the model step by step](#building-the-model-step-by-step)
  - [Contrast sensitivity function](#contrast-sensitivity-function)
  - [Contrast response function](#contrast-response-function)
  - [The CSF tuning model](#the-csf-tuning-model)
  - [Impulse response and scaling](#impulse-response-and-scaling)
  - [The full canonical model](#the-full-canonical-model)
- [Fitting the model to data](#fitting-the-model-to-data)
  - [Inspecting the data](#inspecting-the-data)
  - [Selecting valid vertices](#selecting-valid-vertices)
  - [Grid search](#grid-search)
  - [Least-squares](#least-squares)
  - [Stochastic gradient descent](#stochastic-gradient-descent)
- [Evaluating the model fit](#evaluating-the-model-fit)
- [Interpreting the model parameters](#interpreting-the-model-parameters)
- [Summary](#summary)
- [References](#references)

+++

## Installation and Setup

First, we need to install prfmodel from PyPI. We install the package with the TensorFlow backend (for more information on the different backends, see the [Backends](https://popylar-org.github.io/prfmodel/development/backends.html) section in the package documentation).

**Please make sure to use a GPU instance when installing prfmodel and running this notebook in Colab.**

```{code-cell} ipython3
!pip install "prfmodel[tensorflow]==0.2.0"
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

## The neural contrast sensitivity function model

The contrast sensitivity function (CSF) is a classic measure of visual function from psychophysics. It describes the lowest contrast of a grating that an observer can perceive (the *contrast threshold*) as a function of the *spatial frequency* (SF) of the grating, that is, how many pairs of light and dark stripes fit into one degree of visual angle. Instead of the threshold, the CSF is usually expressed as the *contrast sensitivity*, which is the inverse of the threshold: When the contrast is given in percent, a threshold of 1% corresponds to a sensitivity of 100. Humans are most sensitive to SFs of a few cycles per degree, and their sensitivity falls off towards lower and, more steeply, towards higher SFs. In ophthalmology, the CSF is used, for example, to detect deficits in visual function.

Roelofzen et al. (2025) translated the CSF into a model of the blood oxygenation level-dependent (BOLD) response that is similar to a pRF model. The neural CSF (nCSF) model estimates a CSF for the neuron population of every vertex: To which SFs does the population respond, and how much contrast does it need to respond? Whereas a pRF model describes *where* in the visual field a neuron population responds, the nCSF model describes *which* visual patterns it responds to.

+++

### Input: Spatial frequency and contrast

In this practical, we use the BOLD response of a single subject from Roelofzen et al. (2025), which was recorded with a 7 Tesla MRI scanner. The dataset has the tag `7t-ncsf-visual`. We can get a description of the dataset with `prfmodel.examples.describe_dataset`.

```{code-cell} ipython3
from prfmodel.examples import describe_dataset, load_dataset

print(describe_dataset("7t-ncsf-visual"))
```

The dataset contains two tasks: In the `csf` task, the subject saw gratings with different SFs and contrasts, which we use to fit the nCSF model. In the `prf` task, the subject saw bars that moved through the visual field, which we only need for an optional exercise at the end of this practical. We load the `csf` task for both hemispheres (the default). The response is given for every vertex of a standardized cortical surface from the Human Connectome Project (HCP; Van Essen et al., 2012). Like in the [fourth practical](04-cf-visual-fmri.md), we also load this surface with the tag `hcp-999999-surface` to visualize the response and the model parameters later.

```{code-cell} ipython3
# Downloads on first use and caches in a user data directory (see prfmodel.examples.get_data_dir)
dataset = load_dataset("7t-ncsf-visual", task="csf")
surface = load_dataset("hcp-999999-surface", surface_type="inflated")

print(dataset)
```

The stimulus can be accessed through the `dataset.stimulus` attribute.

```{code-cell} ipython3
stimulus = dataset.stimulus

print(stimulus)
```

Unlike the stimulus of a pRF model, the stimulus of a CSF model has no design matrix and no grid. Every time frame shows a grating that covers the entire screen, so we do not need to describe which part of the visual field is stimulated. Instead, the stimulus describes the *features* of the grating at every time frame: its SF (`stimulus.sf`, in cycles per degree) and its Michelson contrast (`stimulus.contrast`, in percent). The `frame_duration` is the duration of a time frame in seconds, which is the repetition time (TR) of 1.5 seconds. During blank frames, the screen does not show a grating. Blank frames have a contrast and an SF of zero.

We plot the contrast of the grating at every time frame with `prfmodel.plotting.plot_csf_stimulus_design`. The color shows the SF of the grating (in cycles per degree). The function leaves out the blank frames, so they show up as gaps between the lines. Because the contrasts span more than two orders of magnitude, we use a logarithmic y-axis.

```{code-cell} ipython3
import matplotlib.pyplot as plt
from plotting_helpers import style_axes
from prfmodel.plotting import plot_csf_stimulus_design

time_frames = np.arange(stimulus.sf.shape[0])
is_blank = stimulus.contrast == 0.0

# SFs of the gratings (in cycles per degree)
sf_levels = np.unique(stimulus.sf[~is_blank])

fig, ax = plot_csf_stimulus_design(stimulus, figsize=(12, 3.5), layout="constrained")

ax.set_yscale("log")
style_axes(ax, "Time frame (in TR)", "Contrast (in %)")
```

> **Exercise 1:** How is the experiment structured? How many blocks with gratings are there, and how is each block built up? How often is each SF shown, and at how many contrast levels (hint: use `np.unique` on `stimulus.sf` and `stimulus.contrast`)?

```{code-cell} ipython3
# Your code goes here
```

### Building the model step by step

The nCSF model gets from input (i.e., the SF and contrast at every time frame) to output (i.e., the predicted BOLD response) in several steps. Each step is performed by a *submodel* that does a dedicated computation:

1. The *contrast sensitivity function* (CSF) describes how sensitive the neuron population is to every SF.
2. The *contrast response function* (CRF) turns the contrast of the grating and the sensitivity at its SF into the predicted neuron population activity at every time frame.
3. The *impulse response* turns the neuron population activity into a predicted BOLD response.
4. The *scaling* adjusts the baseline and amplitude of the predicted BOLD response.

In prfmodel, the first two steps together form the *tuning model* of the nCSF model. Note that, compared to the canonical Gaussian 2D pRF model from the first practical, the nCSF model has no stimulus encoding step: The tuning profile of a pRF model is defined on the stimulus grid, so it must be combined with the design matrix to predict the activity at every time frame. The CSF tuning model directly predicts the activity at every time frame from the SF and contrast of the grating at that frame.

We build up the model one submodel at a time and then combine all submodels into the full model. To explore how the parameters of each submodel affect its predictions, we use interactive figures. The code that creates the figures is hidden: Run a hidden cell to show its figure and move the sliders to change the parameters. The figure updates when you release a slider. You can expand a hidden cell if you are curious about its code. Note that the sliders of `cs_peak`, `sf_peak`, and `slope_crf` move on a logarithmic scale.

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
from plotting_helpers import BLUE, GRAY, make_slider, show_interactive, style_axes, to_frame
from prfmodel.models.csf import predict_contrast_sensitivity

# Starting value, minimum, maximum, step size, and logarithmic scale of the slider for each parameter
PARAM_SLIDERS = {
    "cs_peak": (50.0, 1.0, 1000.0, 0.05, True),
    "sf_peak": (2.0, 0.25, 20.0, 0.05, True),
    "width_r": (1.0, 0.1, 3.0, 0.05, False),
    "slope_crf": (2.0, 0.5, 10.0, 0.05, True),
    "weight_deriv": (-0.5, -2.0, 2.0, 0.1, False),
    "baseline": (0.0, -1.0, 1.0, 0.05, False),
    "amplitude": (1.0, 0.0, 3.0, 0.1, False),
}
START_VALUES = {name: value for name, (value, *_) in PARAM_SLIDERS.items()}

# Spatial frequencies for drawing smooth CSF curves (in cycles per degree)
sf_values = np.geomspace(0.25, 40.0, 200)

# One color for every SF of the gratings, the same as in plot_csf_stimulus_design (the default matplotlib colors)
sf_colors = dict(zip(sf_levels, plt.rcParams["axes.prop_cycle"].by_key()["color"]))

# Contrast sensitivity that corresponds to the contrast of every time frame (NaN for blank frames)
stimulus_sensitivity = 100.0 / np.where(is_blank, np.nan, stimulus.contrast)

# First and last time frame and SF of every sweep through the contrasts at a single SF
sweeps = []
sweep_start = 0
for frame in range(1, len(time_frames) + 1):
    if frame == len(time_frames) or stimulus.sf[frame] != stimulus.sf[sweep_start] or is_blank[frame] != is_blank[sweep_start]:
        if not is_blank[sweep_start]:
            sweeps.append((sweep_start, frame - 1, stimulus.sf[sweep_start]))
        sweep_start = frame


def make_sliders(*names: str, **custom_sliders: tuple) -> dict[str, widgets.ValueWidget]:
    """Create a slider for each parameter; `custom_sliders` replace the values from `PARAM_SLIDERS` for a parameter."""
    sliders = {}
    for name in names:
        value, min_value, max_value, step, log = custom_sliders.get(name, PARAM_SLIDERS[name])
        sliders[name] = make_slider(name, value, min_value, max_value, step, log=log)
    return sliders


def predict_sensitivity(sf: np.ndarray, cs_peak: float, sf_peak: float, width_r: float) -> np.ndarray:
    """Predict the contrast sensitivity of a single CSF at the spatial frequencies `sf`."""
    params = to_frame(cs_peak=cs_peak, sf_peak=sf_peak, width_r=width_r)
    return np.asarray(
        predict_contrast_sensitivity(
            sf=sf,
            cs_peak=params[["cs_peak"]],
            sf_peak=params[["sf_peak"]],
            width_l=0.68,
            width_r=params[["width_r"]],
        )
    )[0]


def draw_csf(ax: plt.Axes, cs_peak: float, sf_peak: float, width_r: float, **_: float) -> None:
    """Draw the CSF over the stimulus; time frames below the curve have a contrast above the threshold."""
    frame_sensitivity = predict_sensitivity(stimulus.sf, cs_peak, sf_peak, width_r)
    is_above = ~is_blank & (stimulus_sensitivity < frame_sensitivity)
    is_below = ~is_blank & ~is_above
    ax.scatter(
        stimulus.sf[is_below], stimulus_sensitivity[is_below], s=18, facecolors="none", edgecolors=GRAY,
        label="Below threshold",
    )
    ax.scatter(stimulus.sf[is_above], stimulus_sensitivity[is_above], s=18, color=BLUE, label="Above threshold")
    ax.plot(sf_values, predict_sensitivity(sf_values, cs_peak, sf_peak, width_r), color=BLUE, linewidth=2)
    ax.set(xscale="log", yscale="log", xlim=(sf_values[0], sf_values[-1]), ylim=(0.5, 2000.0))
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    style_axes(ax, "Spatial frequency (in cycles/degree)", "Contrast sensitivity")


def mark_sweeps(ax: plt.Axes) -> None:
    """Shade the sweeps through the contrasts and label them with their SF (in cycles per degree)."""
    for first_frame, last_frame, sf in sweeps:
        ax.axvspan(first_frame - 0.5, last_frame + 0.5, color="#f2f2f2", linewidth=0, zorder=0)
        ax.text(
            (first_frame + last_frame) / 2, 1.0, f"{sf:g}", transform=ax.get_xaxis_transform(), ha="center",
            va="bottom", fontsize=8, color="#595959",
        )
```

### Contrast sensitivity function

The nCSF model describes the CSF with an asymmetric log-parabola: With logarithmic axes, the CSF is a downward-opening parabola that can have different curvatures to the left and to the right of its peak. The CSF $S(f)$ at the SF $f$ has four parameters: the peak contrast sensitivity $CS_p$ (`cs_peak`), the SF at the peak $f_p$ (`sf_peak`), and the curvatures of the left branch $w_L$ (`width_l`) and of the right branch $w_R$ (`width_r`):

$$
\log_{10} S(f) = \log_{10} CS_p - \left(\log_{10} f - \log_{10} f_p\right)^2 \cdot
\begin{cases}
w_L^2 & \text{if } f < f_p \\
w_R^2 & \text{if } f \geq f_p
\end{cases}
$$

Note that despite their names, larger values of `width_l` and `width_r` make the CSF *narrower*. Like Roelofzen et al. (2025), prfmodel fixes `width_l` at 0.68 by default, so that only `cs_peak`, `sf_peak`, and `width_r` are estimated.

We can predict the contrast sensitivity at the SF of every time frame with `prfmodel.models.csf.predict_contrast_sensitivity`. Like all model parameters in prfmodel, we store the parameters in a `pandas.DataFrame`, where each row corresponds to a different unit (e.g., a vertex) and each column to a different parameter. The function expects every parameter as a DataFrame with a single column, so we select the columns with double brackets.

```{code-cell} ipython3
csf_params = pd.DataFrame(
    {
        "cs_peak": [50.0],
        "sf_peak": [2.0],
        "width_r": [1.0],
    },
)

simulated_sensitivity = predict_contrast_sensitivity(
    sf=stimulus.sf,
    cs_peak=csf_params[["cs_peak"]],
    sf_peak=csf_params[["sf_peak"]],
    width_l=0.68,
    width_r=csf_params[["width_r"]],
)

simulated_sensitivity.shape  # One row per unit with one value for every time frame
```

To relate the CSF to the stimulus, we plot every time frame with a grating as a point in the space of the CSF with `prfmodel.plotting.plot_csf_stimulus_curve`: The SF of the grating is on the x-axis, and the contrast sensitivity that corresponds to its contrast (100 / contrast) is on the y-axis. Both axes are logarithmic. The function draws the predicted CSF on top of the points (black line), connecting the sensitivities at the SFs in the stimulus. A neuron population responds to the grating at a time frame if the contrast lies above its threshold, that is, if the point lies *below* its CSF. Points at the top of the plot have a low contrast and only drive neuron populations with a high sensitivity.

```{code-cell} ipython3
from prfmodel.plotting import plot_csf_stimulus_curve

fig, ax = plot_csf_stimulus_curve(stimulus, np.asarray(simulated_sensitivity), figsize=(5.5, 4.5), layout="constrained")

ax.set_title("Stimulus and CSF in CSF space")
style_axes(ax, "Spatial frequency (in cycles/degree)", "Contrast sensitivity (100 / contrast)")
```

The interactive figure below draws the CSF as a smooth curve over a wider range of SFs (blue line). Time frames with a contrast above the threshold of the CSF (filled blue circles) lie below the curve, and time frames with a contrast below the threshold (empty gray circles) lie above it.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: Contrast sensitivity function { display-mode: "form" }
def plot_csf(**params: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")

    draw_csf(ax, **params)
    ax.set_title("Contrast sensitivity function")

    plt.show()


show_interactive(plot_csf, make_sliders("cs_peak", "sf_peak", "width_r"))
```

> **Exercise 2:** Use the sliders to change the CSF parameters to the following combinations:
>
> - `cs_peak=300`, `sf_peak=1.0`, `width_r=0.7`
> - `cs_peak=20`, `sf_peak=6.0`, `width_r=1.5`
>
> How does the CSF change? Which time frames have a contrast above the threshold of the neuron population?

+++

> **Exercise 3:** Why does it make sense to fix `width_l` instead of estimating it from the data? Think about which SFs the stimulus contains and move `sf_peak` to values around 1 cycle per degree.

+++

### Contrast response function

The CSF gives us the contrast threshold of the neuron population at every SF: $Q(f) = 100 / S(f)$ (in percent). The contrast response function (CRF) turns the contrast $C(t)$ of the grating at time frame $t$ into the predicted activity of the neuron population. The nCSF model uses a Naka-Rushton function with the threshold at the SF of the grating $f(t)$ as the semi-saturation constant:

$$
R(t) = \frac{C(t)^q}{C(t)^q + Q(f(t))^q}
$$

The predicted activity is zero when there is no contrast, reaches half of its maximum when the contrast equals the threshold, and approaches one when the contrast lies far above the threshold. The exponent $q$ (`slope_crf`) determines how steeply the activity rises around the threshold.

We can predict the contrast response with `prfmodel.models.csf.predict_contrast_response`. It takes the contrast at every time frame and the contrast sensitivity at the SF of every time frame. We first predict the sensitivity for the SFs of the stimulus and then the response to its contrasts.

```{code-cell} ipython3
from prfmodel.models.csf import predict_contrast_response

crf_params = pd.DataFrame(
    {
        "cs_peak": [50.0],
        "sf_peak": [2.0],
        "width_r": [1.0],
        "slope_crf": [2.0],
    },
)

# Contrast sensitivity at the SF of every time frame
frame_sensitivity = predict_contrast_sensitivity(
    sf=stimulus.sf,
    cs_peak=crf_params[["cs_peak"]],
    sf_peak=crf_params[["sf_peak"]],
    width_l=0.68,
    width_r=crf_params[["width_r"]],
)

simulated_activity = predict_contrast_response(
    contrast=stimulus.contrast,
    sensitivity=frame_sensitivity,
    slope_crf=crf_params[["slope_crf"]],
)

simulated_activity.shape  # One row per unit with one value for every time frame
```

The figure below shows the CRF for each SF in the stimulus: the predicted activity as a function of the contrast. The dot on each curve marks the contrast threshold of the population at this SF, where the activity reaches half of its maximum.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: Contrast response function { display-mode: "form" }
# Contrasts to predict the response for (in percent)
contrast_values = np.geomspace(0.1, 100.0, 200)


def plot_crf(cs_peak: float, sf_peak: float, width_r: float, slope_crf: float) -> None:
    sensitivity = predict_sensitivity(sf_levels, cs_peak, sf_peak, width_r)
    # One row with the sensitivity at a single SF for every contrast
    sensitivity_per_contrast = np.repeat(sensitivity[:, np.newaxis], len(contrast_values), axis=1)
    activity = np.asarray(
        predict_contrast_response(contrast=contrast_values, sensitivity=sensitivity_per_contrast, slope_crf=[[slope_crf]])
    )

    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")

    for sf, sf_sensitivity, sf_activity in zip(sf_levels, sensitivity, activity):
        ax.plot(contrast_values, sf_activity, color=sf_colors[sf], linewidth=2, label=f"{sf:g}")
        threshold = 100.0 / sf_sensitivity
        if contrast_values[0] <= threshold <= contrast_values[-1]:
            ax.plot(threshold, 0.5, "o", color=sf_colors[sf], markersize=7, markeredgecolor="white")

    ax.set(xscale="log", ylim=(-0.02, 1.02))
    ax.legend(title="SF (cycles/degree)", frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Contrast response function", loc="left")
    style_axes(ax, "Contrast (in %)", "Predicted activity")

    plt.show()


show_interactive(plot_crf, make_sliders("cs_peak", "sf_peak", "width_r", "slope_crf"))
```

> **Exercise 4:** Move the `slope_crf` slider. How does the CRF change? What happens for very large values of `slope_crf`? What does the neuron population then "compute"?

+++

### The CSF tuning model

In prfmodel, `prfmodel.models.csf.CSFTuning` combines the CSF and the CRF into the tuning model of the nCSF model. It predicts the neuron population activity at every time frame of the stimulus in the same two steps as above.

```{code-cell} ipython3
from prfmodel.models.csf import CSFTuning

csf_tuning = CSFTuning()  # Fixes width_l at 0.68 by default

simulated_tuning_response = csf_tuning(stimulus, crf_params)

simulated_tuning_response.shape  # One row per unit with one value for every time frame
```

The figure below shows the CSF (left) and the predicted neuron population activity at every time frame (right). The shaded areas mark the sweeps through the contrasts, and the number above each sweep is its SF (in cycles per degree).

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: CSF tuning model { display-mode: "form" }
def plot_tuning(**params: float) -> None:
    activity = np.asarray(csf_tuning(stimulus, to_frame(**params)))[0]

    fig, (ax_csf, ax_activity) = plt.subplots(
        1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 2.2]}, layout="constrained",
    )

    draw_csf(ax_csf, **params)
    ax_csf.set_title("Contrast sensitivity function")

    mark_sweeps(ax_activity)
    ax_activity.plot(time_frames, activity, color=BLUE, linewidth=2)
    ax_activity.set_ylim(-0.05, 1.05)
    ax_activity.set_title("Predicted neuron population activity", loc="left", pad=14)
    style_axes(ax_activity, "Time frame (in TR)", "Predicted activity")

    plt.show()


show_interactive(plot_tuning, make_sliders("cs_peak", "sf_peak", "width_r", "slope_crf"))
```

> **Exercise 5:** Why does the predicted neuron population activity have this particular shape? How does it relate to the stimulus? Change `sf_peak` to 0.5 and to 12. How does the predicted activity change and why?

+++

> **Exercise 6:** The stimulus only contains gratings with six different SFs. Can you find two clearly different CSFs (i.e., combinations of `sf_peak` and `width_r`) that predict a similar neuron population activity? What does this mean for estimating the CSF from data?

+++

### Impulse response and scaling

The remaining two submodels are the same as in the canonical Gaussian 2D pRF model from the first practical: The predicted neuron population activity is convolved with an impulse response that accounts for the slow and delayed BOLD response (`prfmodel.impulse.DerivativeTwoGammaImpulse` with the parameter `weight_deriv`). Then, the convolved response is scaled with the parameters `baseline` and `amplitude` (`prfmodel.scaling.BaselineAmplitude`). See the first practical for interactive figures of both submodels.

The impulse response must be sampled at the same rate as the stimulus. The model takes the sampling rate from the `frame_duration` of the stimulus, so we do not need to create a custom impulse response model for the TR of 1.5 seconds.

+++

### The full canonical model

Now that we have seen every submodel, we can combine them into the full nCSF model. prfmodel implements it as the canonical model `prfmodel.models.csf.CSFModel`, which uses the submodels from above by default.

```{code-cell} ipython3
from prfmodel.models.csf import CSFModel

csf_model = CSFModel()

csf_model.parameter_names
```

The model has more parameters than the submodels that we have seen so far: `delay`, `dispersion`, `undershoot`, `u_dispersion`, and `ratio` determine the shape of the two gamma densities of the impulse response. These parameters are optional: If they are missing from the parameter DataFrame, the model uses default values, which correspond to the impulse response from the first practical. We make a prediction with all other parameters at once.

```{code-cell} ipython3
start_params = pd.DataFrame(
    {
        "cs_peak": [50.0],
        "sf_peak": [2.0],
        "width_r": [1.0],
        "slope_crf": [2.0],
        "weight_deriv": [-0.5],
        "baseline": [0.0],
        "amplitude": [1.0],
    },
)

# Make prediction with nCSF model
simulated_response = csf_model(stimulus, start_params)

simulated_response.shape  # One predicted timecourse with one value per time frame
```

The figure below shows the CSF (left) and the predicted BOLD response of the full model (right). The dashed gray line shows the prediction for the starting parameters above. Use the sliders to see how all parameters together shape the predicted BOLD response.

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Interactive figure: Full model { display-mode: "form" }
start_response = np.asarray(csf_model(stimulus, to_frame(**START_VALUES)))[0]


def plot_prediction(**params: float) -> None:
    response = np.asarray(csf_model(stimulus, to_frame(**params)))[0]

    fig, (ax_csf, ax_response) = plt.subplots(
        1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 2.2]}, layout="constrained",
    )

    draw_csf(ax_csf, **params)
    ax_csf.set_title("Contrast sensitivity function")

    mark_sweeps(ax_response)
    ax_response.plot(time_frames, start_response, color=GRAY, linestyle="--", linewidth=1.5, label="Starting parameters")
    ax_response.plot(time_frames, response, color=BLUE, linewidth=2, label="Current parameters")
    ax_response.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    ax_response.set_title("Predicted BOLD response", loc="left", pad=14)
    style_axes(ax_response, "Time frame (in TR)", "Predicted BOLD response")

    plt.show()


show_interactive(
    plot_prediction,
    make_sliders("cs_peak", "sf_peak", "width_r", "slope_crf", "weight_deriv", "baseline", "amplitude"),
)
```

> **Exercise 7:** Compare the predicted BOLD response to the predicted neuron population activity of the CSF tuning model for the same parameters. How does the impulse response change the response to each sweep? Why do you think the first sweep of every block goes from high to low contrast and the second sweep from low to high contrast?

+++

## Fitting the model to data

### Inspecting the data

Before we fit the nCSF model to empirical data, we inspect the recorded BOLD response timecourses, which we can access from the `dataset.response` attribute.

```{code-cell} ipython3
response = dataset.response

response.shape  # One timecourse for every vertex (left hemisphere first)
```

The dataset contains a timecourse with 214 time frames for each of the 118,584 vertices of the standardized surface (59,292 per hemisphere). The responses were already detrended, converted to percent signal change (PSC), and averaged over the runs of the experiment. Thus, every timecourse fluctuates around zero, and we do not need to convert it before fitting.

+++

To visualize the response on the inflated surface, we create a helper function with `make_surface_plot` from `plotting_helpers.py` like in the first practical. The helper shows three views that focus on the visual cortex in the occipital lobe: the inner (medial) side of each hemisphere, where the calcarine sulcus lies, and both hemispheres from behind (posterior). Like in the fourth practical, the response is already given for every vertex of the surface, so we do not need to project it from a volume onto the surface.

```{code-cell} ipython3
from plotting_helpers import make_surface_plot

plot_surf_stat_map_helper = make_surface_plot(surface.mesh)
```

We plot the standard deviation of each timecourse on the surface to get an overview of the variability in the BOLD timecourses across the surface.

```{code-cell} ipython3
# Calculate standard deviation of each timecourse
response_sd = response.std(axis=1)

plot_surf_stat_map_helper(response_sd, vmax=2.0, title="BOLD response standard deviation (in PSC)");
```

The dataset also contains a region of interest (ROI) label for every vertex. The labels come from the template atlas by Benson et al. (2014), which predicts the location of twelve visual areas from the anatomy of the subject's brain. Vertices outside these areas have the label 0.

```{code-cell} ipython3
roi_index = dataset.roi_index  # One ROI label per vertex

dataset.roi_mapping  # Which ROI each label refers to
```

We plot the ROIs on the surface. Note that the helper averages the values of neighboring vertices for plotting, so vertices at the borders between ROIs can get the color of a different ROI.

```{code-cell} ipython3
import matplotlib as mpl

roi_labels = [label for label in dataset.roi_mapping if label > 0]

# Only color vertices that belong to one of the ROIs
fig, _ = plot_surf_stat_map_helper(
    np.where(roi_index > 0, roi_index, np.nan),
    vmin=0.5,
    vmax=len(roi_labels) + 0.5,
    cmap=mpl.colors.ListedColormap(plt.get_cmap("Paired").colors[: len(roi_labels)]),
    title="Visual areas (Benson et al., 2014)",
)
# Label the colors of the colorbar (the last axes of the figure) with the names of the ROIs
fig.axes[-1].set_yticks(roi_labels, labels=[dataset.roi_mapping[label] for label in roi_labels]);
```

> **Exercise 8:** Compare the ROIs to the map of the response standard deviation. Do the vertices with the largest variability lie inside the ROIs? Do all parts of V1 have a similarly variable response?

+++

### Selecting valid vertices

The nCSF model only makes sense for vertices that respond to visual stimulation, and fitting all vertices of the surface would take a long time. Therefore, we only fit the model to vertices that belong to one of the ROIs. We also make sure that their responses vary over time: A vertex with a constant response contains no information about the stimulus.

```{code-cell} ipython3
# Vertices in one of the ROIs with a response that varies over time
response_is_valid = (roi_index > 0) & (response_sd > 0.0)

response_valid = response[response_is_valid]

# Name of the ROI of every valid vertex
roi_valid = pd.Series(roi_index[response_is_valid]).map(dataset.roi_mapping).to_numpy()


def fill_valid_vertices(values: np.ndarray) -> np.ndarray:
    """Fill values for the valid vertices into an array with one value per vertex, using NaN for all other vertices."""
    values_full = np.full((response_is_valid.shape[0],), fill_value=np.nan)
    values_full[response_is_valid] = values
    return values_full


response_valid.shape
```

> **Exercise 9:** Visualize the BOLD response of the valid vertices (hint: use `prfmodel.plotting.plot_response_heatmap`). Can you see the blocks of the stimulus? How does the response change within a block? Do all valid vertices respond to the stimulus?

```{code-cell} ipython3
# Your code goes here
```

Before we let prfmodel fit the model, you can try to fit it yourself. The figure below shows the observed BOLD response of a V1 vertex with a clear signal (gray) together with the prediction of the nCSF model (blue). The R-squared in the title shows how much of the variance in the observed response the prediction explains (1 is a perfect fit).

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Manual fit? { display-mode: "form" }
vertex_index = 4635  # A valid V1 vertex with a clear response to the stimulus
vertex_response = response_valid[vertex_index]


def plot_manual_fit(**params: float) -> None:
    prediction = np.asarray(csf_model(stimulus, to_frame(**params))[0])
    r_squared = 1.0 - np.sum((vertex_response - prediction) ** 2) / np.sum((vertex_response - vertex_response.mean()) ** 2)

    fig, (ax_csf, ax_response) = plt.subplots(
        1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 2.2]}, layout="constrained",
    )

    draw_csf(ax_csf, **params)
    ax_csf.set_title("Contrast sensitivity function")

    mark_sweeps(ax_response)
    ax_response.plot(time_frames, vertex_response, color=GRAY, linewidth=1, label="Observed")
    ax_response.plot(time_frames, prediction, color=BLUE, linewidth=2, label="Predicted")
    ax_response.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    ax_response.set_title(
        f"Vertex {vertex_index} ({roi_valid[vertex_index]}): R-squared = {r_squared:.2f}", loc="left", pad=14,
    )
    style_axes(ax_response, "Time frame (in TR)", "BOLD response (in PSC)")

    plt.show()


show_interactive(
    plot_manual_fit,
    make_sliders("cs_peak", "sf_peak", "width_r", "slope_crf", "weight_deriv", "baseline", "amplitude"),
)
```

> **Exercise 10:** Move the sliders until the prediction matches the observed response as well as you can. Which R-squared can you reach? Which parameters are easy to find and which are hard? Why?

+++

### Grid search

We fit the nCSF model in three stages like in the first practical. In the first stage, we perform a grid search to find good starting values for the parameters of the CSF tuning model (i.e., `cs_peak`, `sf_peak`, `width_r`, and `slope_crf`). prfmodel does not have a function that creates grid values for the nCSF model, so we create them ourselves. All four parameters are positive and act on a logarithmic scale, so we space their values logarithmically with `np.geomspace`:

- `cs_peak` between 2 and 400, which covers the contrast sensitivities in the stimulus (100 / 80 = 1.25 to 100 / 0.25 = 400).
- `sf_peak` between 0.25 and 10 cycles per degree. Values below the lowest SF in the stimulus (0.5 cycles per degree) allow CSFs that fall off across all SFs in the stimulus.
- `width_r` between 0.3 and 2.
- `slope_crf` between 0.5 and 8.

For `weight_deriv`, `baseline`, and `amplitude`, we only provide a single value so that they stay constant across the entire grid. The grid has 12 × 12 × 6 × 8 = 6,912 parameter combinations.

```{code-cell} ipython3
grid_param_ranges = {
    "cs_peak": np.geomspace(2.0, 400.0, 12),
    "sf_peak": np.geomspace(0.25, 10.0, 12),
    "width_r": np.geomspace(0.3, 2.0, 6),
    "slope_crf": np.geomspace(0.5, 8.0, 8),
    "weight_deriv": [-0.5],
    "baseline": [0.0],
    "amplitude": [1.0],
}
```

To run the grid search, we construct the `prfmodel.fitters.GridFitter`. We set `batch_size=100` to let the fitter evaluate 100 parameter combinations at the same time. By default, the `loss` is the negative correlation coefficient, which ignores differences in baseline and amplitude between model predictions and observed data.

```{code-cell} ipython3
from prfmodel.fitters import GridFitter

# Create grid fitter object
grid_fitter = GridFitter(
    model=csf_model,
    stimulus=stimulus,
    compile_step=True,  # Setting 'compile_step=True' speeds up the fitting
)

# Run grid search
grid_history, grid_params = grid_fitter.fit(
    data=response_valid,
    parameter_values=grid_param_ranges,
    batch_size=100,
)

grid_params
```

Because the grid search can only return values that are in the grid, we check how the estimates are distributed across the grid with `prfmodel.plotting.plot_grid_parameter_distribution`. Estimates at the smallest or largest value of the grid are highlighted in red.

```{code-cell} ipython3
from prfmodel.plotting import plot_grid_parameter_distribution

csf_param_names = ["cs_peak", "sf_peak", "width_r", "slope_crf"]

fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")

for ax, name in zip(axes.flat, csf_param_names):
    plot_grid_parameter_distribution(grid_params, grid_param_ranges, name, ax=ax)  # Detects log spacing
    # Label every other grid value instead of the powers of ten
    tick_values = grid_param_ranges[name][::2]
    ax.set_xticks(tick_values, labels=[f"{value:.3g}" for value in tick_values])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set(xlabel=name, title=name)
```

> **Exercise 11:** Which parameters often have estimates at the smallest or largest value of the grid? What could be the reason (hint: think about the vertices in the ROIs that do not respond to the stimulus)?

+++

### Least-squares

In the grid search, we ignored the `baseline` and `amplitude` parameters. Like in the first practical, we estimate them with least-squares by regressing the observed responses on the predicted responses for each vertex.

```{code-cell} ipython3
from prfmodel.fitters import LeastSquaresFitter

# Create least-squares fitter
ls_fitter = LeastSquaresFitter(
    model=csf_model,
    stimulus=stimulus,
)

# Run least squares fit
ls_history, ls_params = ls_fitter.fit(
    data=response_valid,
    parameters=grid_params,
    slope_name="amplitude",  # Names of parameters to be optimized with least squares
    intercept_name="baseline",
    batch_size=1000,
)

ls_params
```

### Stochastic gradient descent

Finally, we use SGD to finetune all parameters at once, starting from the least-squares estimates.

We need to be more careful with the nCSF parameters than with the pRF parameters in the first practical. All four CSF parameters must be positive, but that is not enough: The data only constrain the CSF within the range of SFs and contrasts in the stimulus. Beyond this range, SGD can move the parameters of some vertices to implausible values without changing the prediction much. Without upper bounds, some vertices end up with a peak SF of more than 100 cycles per degree, a `width_r` of more than 10, or a `slope_crf` of more than 100. Therefore, we keep each CSF parameter between a lower and an upper bound during SGD.

In the first practical, we used a `ParameterTransform` in an adapter to let SGD optimize `sigma` in log space. prfmodel also provides `prfmodel.fitters.adapter.ParameterConstraint`, which keeps a parameter above a lower *or* below an upper bound. To keep a parameter between a lower *and* an upper bound, we write our own `ParameterTransform`: SGD optimizes the parameter on an unbounded scale $y$, and the model sees the value $x = \text{lower} + (\text{upper} - \text{lower}) \cdot \text{sigmoid}(y)$, which always lies between the bounds. The transformation from the bounded to the unbounded scale is the inverse of this function: $y = \log\left((x - \text{lower}) / (\text{upper} - x)\right)$. Starting values must lie strictly between the bounds. The bounds are a bit wider than the grid, so that SGD can move the estimates beyond the grid values.

```{code-cell} ipython3
from keras import ops
from prfmodel.fitters.adapter import Adapter, ParameterTransform


def box_transform(parameter_name: str, lower: float, upper: float) -> ParameterTransform:
    """Keep a parameter strictly between `lower` and `upper` during SGD."""
    return ParameterTransform(
        [parameter_name],
        transform_fun=lambda x: ops.log((x - lower) / (upper - x)),  # From the bounded to the unbounded scale
        inverse_fun=lambda y: lower + (upper - lower) * ops.sigmoid(y),  # From the unbounded to the bounded scale
    )


# Lower and upper bound of each CSF parameter
parameter_bounds = {
    "cs_peak": (1.0, 1000.0),
    "sf_peak": (0.1, 20.0),
    "width_r": (0.1, 3.0),
    "slope_crf": (0.2, 10.0),
}

adapter = Adapter([box_transform(name, lower, upper) for name, (lower, upper) in parameter_bounds.items()])
```

The default optimizer of SGD (Adam) changes each parameter on the unbounded scale by at most about the learning rate in each step. With the default learning rate of 0.001, 500 steps can only move the parameters a little, so we increase the learning rate to 0.01. Like in the first practical, we fix `weight_deriv` to its starting value.

```{code-cell} ipython3
from keras.optimizers import Adam
from prfmodel.fitters import SGDFitter

sgd_fitter = SGDFitter(
    model=csf_model,
    stimulus=stimulus,
    adapter=adapter,
    optimizer=Adam(learning_rate=0.01),
    compile_step=True,  # Setting 'compile_step=True' speeds up the fitting
)

sgd_history, sgd_params = sgd_fitter.fit(
    data=response_valid,
    init_parameters=ls_params,
    fixed_parameters=["weight_deriv"],  # Fix weight_deriv to starting value
    num_steps=500,
)

sgd_params
```

## Evaluating the model fit

We make predictions with the SGD estimates and compute the R-squared of each vertex. Like in the first practical, we wrap our `csf_model` in `prfmodel.utils.batched` to make predictions for batches of vertices sequentially, and compute the R-squared of each vertex with `prfmodel.utils.calculate_r_squared`. We look at the distribution of the R-squared scores with `prfmodel.plotting.plot_r_squared_hist`.

```{code-cell} ipython3
from prfmodel.plotting import plot_r_squared_hist
from prfmodel.utils import batched, calculate_r_squared

predict_batched = batched(csf_model)

pred_response = np.asarray(predict_batched(stimulus, sgd_params, batch_size=1000))

r_squared = calculate_r_squared(response_valid, pred_response)  # One score per vertex

plot_r_squared_hist(r_squared);
```

> **Exercise 12:** Compare the R-squared between the least-squares and the SGD stage (hint: use `prfmodel.plotting.plot_r_squared_comparison`). Does SGD improve the fit and if so, by how much?

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 13:** Plot the R-squared on the inflated surface (hint: use `fill_valid_vertices`). In which parts of the visual cortex does the model fit the data well? Why does it fit poorly in large parts of V1 and V2? (hint: Plot the R-squared against the eccentricity that the template atlas predicts for each vertex, which is stored in `dataset.units["benson14_eccen"]`, and remember the size of the screen.)

```{code-cell} ipython3
# Your code goes here
```

We compare the R-squared between ROIs with `prfmodel.plotting.plot_parameter_by_roi`, which shows the mean and standard deviation across the vertices in each ROI. The function expects a DataFrame, so we wrap the R-squared in one.

```{code-cell} ipython3
from prfmodel.plotting import plot_parameter_by_roi

# Visual areas ordered roughly from early to higher visual cortex
roi_order = ["V1", "V2", "V3", "hV4", "VO1", "VO2", "V3a", "V3b", "LO1", "LO2", "TO1", "TO2"]

fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
plot_parameter_by_roi(pd.DataFrame({"R-squared": r_squared}), "R-squared", roi_valid, order=roi_order, ax=ax)
ax.set_ylim(-0.1, 1.0);
```

> **Exercise 14:** Plot the predicted vs. the observed response for vertices with a high, medium, and low R-squared (hint: use `prfmodel.plotting.plot_observed_predicted`). Can you see any patterns in the observed response that the nCSF model does not capture?

```{code-cell} ipython3
# Your code goes here
```

## Interpreting the model parameters

Like in the first practical, we only interpret the parameters of vertices for which we can trust the estimates. We select vertices that fulfill four criteria:

- The R-squared is above 0.3 (note that this threshold is somewhat arbitrary).
- The amplitude is positive.
- The peak SF lies within the range of SFs in the stimulus (0.5 to 18 cycles per degree).
- The peak contrast sensitivity lies within the range of contrast sensitivities in the stimulus (100 / 80 = 1.25 to 100 / 0.25 = 400).

The second criterion excludes vertices whose response *decreases* when the gratings are shown. The model can fit such a response with a negative amplitude, which flips the sign of the predicted response, but the CSF then describes which gratings decrease the response and not to which gratings the neuron population is sensitive. The last two criteria make sure that the peak of the CSF lies inside the stimulus in CSF space, similar to the criterion that the pRF center lies inside the stimulus in the first practical. A peak outside the stimulus is only weakly constrained by the data because we never measured the response at the peak.

```{code-cell} ipython3
cs_peak = sgd_params["cs_peak"].to_numpy()
sf_peak = sgd_params["sf_peak"].to_numpy()
width_r = sgd_params["width_r"].to_numpy()

r_squared_threshold = 0.3

# Range of contrast sensitivities in the stimulus
stimulus_sensitivity_range = 100.0 / stimulus.contrast[~is_blank].max(), 100.0 / stimulus.contrast[~is_blank].min()

selection_criteria = pd.DataFrame({
    f"r_squared > {r_squared_threshold}": r_squared > r_squared_threshold,
    "amplitude > 0": sgd_params["amplitude"].to_numpy() > 0.0,
    "sf_peak inside stimulus": (sf_peak >= sf_levels.min()) & (sf_peak <= sf_levels.max()),
    "cs_peak inside stimulus": (cs_peak >= stimulus_sensitivity_range[0]) & (cs_peak <= stimulus_sensitivity_range[1]),
})

is_selected = selection_criteria.all(axis=1).to_numpy()


def fill_selected_vertices(values: np.ndarray) -> np.ndarray:
    """Fill values for the selected vertices into an array with one value per vertex, using NaN for all others."""
    return fill_valid_vertices(np.where(is_selected, values, np.nan))


# Number of valid vertices that fulfill each criterion and all criteria together
selection_criteria.assign(all_criteria=is_selected).sum()
```

First, we look at the peak SF and plot it on the surface. Because the SFs are spaced logarithmically, we plot the logarithm of the peak SF.

```{code-cell} ipython3
plot_surf_stat_map_helper(
    fill_selected_vertices(np.log10(sf_peak)),
    vmin=np.log10(0.5),
    vmax=np.log10(5.0),
    cmap="viridis",
    title="Peak SF (log10 of cycles/degree)",
);
```

> **Exercise 15:** How does the peak SF change across the surface? Plot the peak contrast sensitivity (`cs_peak`) on the surface as well. How does it change across the surface?

```{code-cell} ipython3
# Your code goes here
```

To compare the CSFs between visual areas, we predict the CSF of every selected vertex and plot the median CSF of each ROI with at least 50 selected vertices. Note that the median CSF is not the same as the CSF of the median parameters. We group the ROIs into early visual cortex, ventral visual cortex, and dorsal and lateral visual cortex.

```{code-cell} ipython3
roi_groups = {
    "Early": ["V1", "V2", "V3"],
    "Ventral": ["hV4", "VO1", "VO2"],
    "Dorsal and lateral": ["V3a", "V3b", "LO1", "LO2", "TO1", "TO2"],
}
roi_colors = {name: color for names in roi_groups.values() for name, color in zip(names, plt.get_cmap("tab10").colors)}

# ROIs with at least 50 selected vertices
selected_rois = [name for name in roi_order if np.sum(is_selected & (roi_valid == name)) >= 50]

# Spatial frequencies to predict the contrast sensitivity for (in cycles per degree)
sf_range = np.geomspace(0.25, 40.0, 200)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, layout="constrained")

for ax, (group, names) in zip(axes, roi_groups.items()):
    for name in [name for name in names if name in selected_rois]:
        roi_params = sgd_params[is_selected & (roi_valid == name)]
        roi_sensitivity = np.asarray(
            predict_contrast_sensitivity(
                sf=sf_range,
                cs_peak=roi_params[["cs_peak"]],
                sf_peak=roi_params[["sf_peak"]],
                width_l=0.68,
                width_r=roi_params[["width_r"]],
            )
        )
        ax.plot(sf_range, np.median(roi_sensitivity, axis=0), color=roi_colors[name], linewidth=2, label=name)

    ax.set(xscale="log", yscale="log", ylim=(1.0, 300.0), title=group)
    ax.legend(frameon=False)
    style_axes(ax, "Spatial frequency (in cycles/degree)", "Median contrast sensitivity")
```

> **Exercise 16:** How do the CSFs differ between visual areas? Which areas are most sensitive to high SFs?

+++

The CSF is often summarized with two numbers that describe the whole curve:

- The *SF cutoff* (SFmax) is the highest SF that a neuron population can detect at all. It is the SF at which the sensitivity drops to 1, that is, where even a grating with 100% contrast lies at the threshold.
- The *area under the log CSF* (AULCSF) summarizes the sensitivity across all SFs. It is the area between the CSF and a sensitivity of 1 when both axes are logarithmic.

For the asymmetric log-parabola, both measures follow directly from the parameters (as long as `cs_peak` is above 1):

$$
\log_{10} \text{SF}_{max} = \log_{10} f_p + \frac{\sqrt{\log_{10} CS_p}}{w_R}
\qquad
\text{AULCSF} = \frac{2}{3} \left(\log_{10} CS_p\right)^{3/2} \left(\frac{1}{w_L} + \frac{1}{w_R}\right)
$$

We compute both measures and compare them, together with the peak SF, between ROIs. The SF cutoff of broad CSFs can lie far beyond the highest SF in the stimulus, where it is an extrapolation, so we compare the logarithm of the SFs.

```{code-cell} ipython3
width_l = 0.68

log_cs_peak = np.log10(cs_peak)

csf_measures = pd.DataFrame({
    "log10_sf_peak": np.log10(sf_peak),
    "log10_sf_max": np.log10(sf_peak) + np.sqrt(log_cs_peak) / width_r,
    "aulcsf": 2.0 / 3.0 * log_cs_peak**1.5 * (1.0 / width_l + 1.0 / width_r),
})

fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")

for ax, name, label in zip(
    axes,
    csf_measures.columns,
    ["Peak SF (log10 of cycles/degree)", "SF cutoff (log10 of cycles/degree)", "AULCSF"],
):
    plot_parameter_by_roi(csf_measures[is_selected], name, roi_valid[is_selected], order=selected_rois, ax=ax)
    style_axes(ax, "ROI", label)
```

Finally, we look at how the CSF changes across the visual field. In the visual system, acuity is highest at the center of the visual field and falls off towards the periphery. Do the CSFs of the neuron populations follow the same pattern? We do not know the pRFs of the vertices from the `csf` task, but we can use the eccentricity that the template atlas predicts for each vertex. The gratings covered the screen up to about 5 degrees from its center (the bars in the `prf` task cover about -5.1 to 5.1 degrees), so we only look at vertices with an eccentricity below 5 degrees. We plot the median peak SF in bins of 1 degree for V1, V2, and V3.

```{code-cell} ipython3
eccentricity = dataset.units["benson14_eccen"].to_numpy()[response_is_valid]

eccentricity_bins = np.arange(0.0, 6.0, 1.0)
bin_centers = (eccentricity_bins[:-1] + eccentricity_bins[1:]) / 2

fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")

for name in ["V1", "V2", "V3"]:
    is_roi = is_selected & (roi_valid == name) & (eccentricity < eccentricity_bins[-1])
    # Bin of every selected vertex in the ROI
    bin_index = np.digitize(eccentricity[is_roi], eccentricity_bins) - 1
    median_sf_peak = [
        np.median(sf_peak[is_roi][bin_index == i]) if np.sum(bin_index == i) >= 20 else np.nan
        for i in range(len(bin_centers))
    ]
    ax.plot(bin_centers, median_sf_peak, "o-", color=roi_colors[name], label=f"{name} (n = {is_roi.sum()})")

ax.legend(frameon=False)
ax.set_title("Peak SF across the visual field")
style_axes(ax, "Eccentricity of the template atlas (in degrees)", "Median peak SF (in cycles/degree)")
```

> **Exercise 17:** How does the peak SF change with eccentricity? Do the estimated CSFs agree with the results reported by Roelofzen et al. (2025)?

+++

> **Exercise 18 (optional, advanced):** In this practical, we fitted the model to the response averaged over all runs and evaluated it on the same data. The dataset also contains the average of the odd and of the even runs as separate splits (`load_dataset("7t-ncsf-visual", task="csf", split="odd")`). Following the [second practical](02-prf-visual-fmri-cv.md), fit the nCSF model to the odd runs and evaluate its predictions on the even runs. Then, fix `slope_crf` for all vertices at the median of its estimates (i.e., use a single value for `slope_crf` in the grid and add it to the `fixed_parameters` of SGD) and repeat. Which of the two models predicts the even runs better?

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 19 (optional, advanced):** Fit the canonical Gaussian 2D pRF model from the first practical to the `prf` task of the dataset (`load_dataset("7t-ncsf-visual", task="prf")`). How does the peak SF of the nCSF model relate to the eccentricity and size of the estimated pRFs?

```{code-cell} ipython3
# Your code goes here
```

## Summary

- The nCSF model describes to which SFs and contrasts a neuron population responds. Its tuning model combines a CSF (an asymmetric log-parabola with the parameters `cs_peak`, `sf_peak`, and `width_r`) with a contrast response function (a Naka-Rushton function with the exponent `slope_crf`).
- Unlike pRF models, the nCSF model does not need a stimulus grid or a stimulus encoding step: The stimulus only describes the SF and contrast of a full-screen grating at every time frame.
- We fit the nCSF model with the same stages as pRF models: grid search, least-squares, and SGD. During SGD, we keep the CSF parameters between bounds, because the data hardly constrain them beyond the range of SFs and contrasts in the stimulus.
- The model can only be fitted to vertices that respond to the stimulus. Large parts of V1 and V2 represent the visual field beyond the screen and are not driven by the gratings.
- The estimated CSFs differ between visual areas and across the visual field: The peak SF is highest in early visual cortex and decreases with eccentricity.

## References

Benson, N. C., Butt, O. H., Brainard, D. H., & Aguirre, G. K. (2014). Correction of distortion in flattened representations of the cortical surface allows prediction of V1-V3 functional organization from anatomy. *PLoS Computational Biology*, *10*(3), e1003538. https://doi.org/10.1371/journal.pcbi.1003538

Roelofzen, C., Daghlian, M., van Dijk, J. A., de Jong, M. C., & Dumoulin, S. O. (2025). Modeling neural contrast sensitivity functions in human visual cortex. *Imaging Neuroscience*, *3*, imag_a_00469. https://doi.org/10.1162/imag_a_00469

Van Essen, D. C., Ugurbil, K., Auerbach, E., Barch, D., Behrens, T. E. J., Bucholz, R., Chang, A., Chen, L., Corbetta, M., Curtiss, S. W., Della Penna, S., Feinberg, D., Glasser, M. F., Harel, N., Heath, A. C., Larson-Prior, L., Marcus, D., Michalareas, G., Moeller, S., ... WU-Minn HCP Consortium. (2012). The Human Connectome Project: A data acquisition perspective. *NeuroImage*, *62*(4), 2222-2231. https://doi.org/10.1016/j.neuroimage.2012.02.018

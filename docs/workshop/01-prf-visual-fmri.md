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

This is the notebook for the first practical of the workshop. This practical covers the canonical Gaussian 2D population receptive field (pRF) model, its different computationial parts, and how to fit it to empirical data from a visual functional magnetic resonance imaging (fMRI) experiment.

**We recommend going through this notebook first to get familiar with the prfmodel package before starting with the notebooks of the other practicals.**

+++

## Installation and Setup

First, we need to install prfmodel from GitHub. We install the package with the TensorFlow backend (for more information on the different backends, see the [Backends](https://popylar-org.github.io/prfmodel/development/backends.html) section in the package documentation).

**Please make sure to use a GPU instance when installing prfmodel and running this notebook in Colab.**

```{code-cell} ipython3
!pip install "prfmodel[tensorflow] @ git+https://github.com/popylar-org/prfmodel.git"
```

We set the backend to TensorFlow and a few defaults for printing and plotting.

```{code-cell} ipython3
import os

import pandas as pd
import plotly.io as pio
import plotly.offline as pyo

# Set keras backend to 'tensorflow' (this is normally the default)
os.environ["KERAS_BACKEND"] = "tensorflow"
# Print parameter DataFrames with three decimals
pd.set_option("display.precision", 3)
# Necessary for interactive plotly figures
pyo.init_notebook_mode()
pio.renderers.default = "colab"
```

> **Exercise 1:** Import the prfmodel package and check whether the TensorFlow backend has been installed with GPU support (hint: use `tensorflow.config.list_physical_devices('GPU')` and `tensorflow.test.is_built_with_cuda()`).

+++

## The canonical Gaussian 2D pRF model

A population receptive field (pRF) model maps the recorded activity of neuron populations to an experimental stimulus
that is presented during the recording. In this example, we use the canonical Gaussian 2D pRF model adapted from Dumoulin and Wandell (2008) to map the recorded blood oxygenation level-dependent (BOLD) response from an fMRI experiment to a series of bar apertures that move through the visual field of the recorded subject (i.e., the bar moves on a computer screen in front of the subject).

+++

### Input: Stimulus design matrix and grid

How the position of the bar on the screen changes during the experiment is recorded as a stimulus *design matrix*. The design matrix indicates which cell of the visual field is stimulated by the bar at which *time frame* (i.e., recording sample) of the experiment. A pRF model is defined in the *feature space* of the experimental stimulus. In this example, the stimulus is
defined in a two-dimensional visual feature space. The stimulus *grid* contains the visual field coordinates of every
cell in the stimulus design matrix. Together, the stimulus design matrix and grid define the feature space of the pRF model.

+++

We can look at the experimental stimulus that is attached to the dataset that we will use in this example. The dataset is
called the Arrow-of-Time (AOT) dataset and the BOLD response was recorded with a 7 Tesla MRI scanner. Therefore, the
dataset has the tag `7t-aot-visual`. We also specify that we want to load a flat surface of the subject that we will
use later for visualization.

```{code-cell} ipython3
from prfmodel.examples import load_dataset

# Downloads on first use and caches in a user data directory (see prfmodel.examples.get_data_dir)
dataset = load_dataset("7t-aot-visual", surface_type="flat")
```

The stimulus can be accessed through the `data.stimulus` attribute.

```{code-cell} ipython3
stimulus = dataset.stimulus
```

> **Exercise 2:** Inspect the stimulus design matrix and grid. What do the dimensions and values refer too? Hint: You can
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

> **Exercise 3:** What do you see in the video? Going back to Exercise 2, does the video confirm your assumptions about what the stimulus design and grid dimensions refer to? What do you expect the neuron population activity that responds to this stimulus to look like?

+++

### Output: Predicted neuron population activity

The pRF is the part of the stimulus feature space that stimulates activity of the neuron population. The pRF model takes
the stimulus design matrix and grid as input and predicts the neuron population activity given a set of parameters. By comparing the predicted against observed activity and optimizing the model parameters, the pRF model can estimate the
pRF properties for the neuron population of the observed activity. Typically, pRFs are defined through a location (e.g.,
a center x- and y-coordinate in visual space) and a size or tuning width.

+++

To see how input stimulus and model parameters lead to predicted neuron population acivity, we create an instance of the Gaussian 2D pRF model.

```{code-cell} ipython3
from prfmodel.models.prf import Gaussian2DPRFModel

prf_model = Gaussian2DPRFModel()
```

In prfmodel, we call this canonical model because it performs all necessary steps to get from input (i.e., stimulus design matrix and grid) to output (i.e., predicte neuron population activity). These steps are called *submodels* that each perform a dedicated computation. We will cover the different submodels soon, but first we define a set of parameters to make a prediction
with our model instance. Parameters in prfmodel live in `pandas.DataFrame` objects where each row corresponds to a different unit (e.g., a vertex or voxel in fMRI) and each column to a different model parameter.

```{code-cell} ipython3
import numpy as np
import plotly.express as px

start_params = pd.DataFrame(
    {
        "mu_x": [0.0],
        "mu_y": [0.0],
        "sigma": [1.0],
        "baseline": [0.0],
        "amplitude": [1.0],
        "weight_deriv": [-0.5],
    },
)

# Make prediction with pRF model
simulated_response = prf_model(stimulus, start_params)

fig = px.line(
    pd.DataFrame({
        "Time frame": np.arange(simulated_response.shape[1]),
        "Predicted BOLD response": simulated_response[0]
    }),
    x="Time frame",
    y="Predicted BOLD response",
)
fig.update_layout(height=450)
fig.show()
```

> **Exercise 4:** Why does the predicted BOLD response have this particular shape? How does the predicted BOLD response relate to the experimental stimulus (compare it to the video)?

> **Exercise 5:** Change the model parameters. How does the predicted BOLD response change?

+++

### Tuning profiles

In prfmodel, the core of every pRF model is called *tuning profile*. In the canonical Gaussian 2D pRF model, the tuning
profile is a isotropic multivariate Gaussian density that describes the stimulation of the neuron population by input from the visual field.

+++

To illustrate this, we create an instance of a Gaussian tuning profile and visualize its stimulation by our stimulus. The Gaussian tuning profile only has three parameters, namely, its center location x- and -y coordinate, and size defined as the standard deviation.

The predicted tuning profile assigns a value to every cell of the stimulus grid. We can visualize it as a heatmap in visual field coordinates: the brightest cells lie at the pRF center (`mu_x`, `mu_y`), and the spread of the profile is determined by its size (`sigma`).

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

# Visual field coordinates of the grid (dimensions are ordered as y, x)
y_coords = stimulus.grid[:, 0, 0]
x_coords = stimulus.grid[0, :, 1]

fig = px.imshow(
    simulated_tuning_profile[0],
    x=x_coords,
    y=y_coords,
    origin="lower",  # Positive y-coordinates point up
    color_continuous_scale=["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
    labels={"x": "x (degrees visual angle)", "y": "y (degrees visual angle)", "color": "Tuning"},
)
fig.update_traces(hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>Tuning: %{z:.3f}<extra></extra>")
fig.update_layout(height=450, width=500)
fig.show()
```

> **Exercise 6:** Change the tuning parameters. How does the predicted tuning profile change and why?

+++

### Stimulus encoding

In pRF models, the predicted tuning profile is typically encoded with the stimulus design matrix through a dot-product operation. This operation computes by how much the predicted tuning profile overlaps with the stimulus design matrix at every time frame and returns the stimulus-encoded pRF model response.

+++

We also illustrate this by creating an instance of the pRF stimulus encoder.

```{code-cell} ipython3
from prfmodel.models.prf import PRFStimulusEncoder

prf_encoder = PRFStimulusEncoder()

# We redefine the tuning parameters so that we can tweak them together with the encoding
tuning_params = pd.DataFrame(
    {
        "mu_x": [0.0],
        "mu_y": [0.0],
        "sigma": [1.0],
    },
)

# We first predict the tuning profile
simulated_tuning_profile = prf_tuning(stimulus, tuning_params)

# Then we stimulus-encode it
simulated_encoded_profile = prf_encoder(stimulus, simulated_tuning_profile, tuning_params)

fig = px.line(
    pd.DataFrame({
        "Time frame": np.arange(simulated_encoded_profile.shape[1]),
        "Predicted encoded response": simulated_encoded_profile[0]
    }),
    x="Time frame",
    y="Predicted encoded response",
)
fig.update_layout(height=450)
fig.show()
```

> **Exercise 7:** How does the encoded response differ from the predicted BOLD response of the full canonical model? Change the tuning parameters. How does the encoded response change and why?

+++

### Impulse response models for BOLD measurements

The encoded pRF model response still differs from the predicted canonical model response because it does not account for the characteristic shape of the BOLD recording. By default, the canonical Gaussian 2D pRF model in prfmodel approximates the shape of the BOLD response with the weighted difference of two gamma densities. From this difference, we then also subtract its weighted derivative. While the parameters for the difference weight and the gamma densities are typically fixed at default values, we estimate the weight of the derivative. This is the `weight_deriv` that we saw earlier.

+++

We create an instance of the default impulse response model, define a set of parameters, and predict the impulse response.

```{code-cell} ipython3
from prfmodel.impulse import DerivativeTwoGammaImpulse

impulse_model = DerivativeTwoGammaImpulse()

impulse_params = pd.DataFrame({"weight_deriv": [-0.5]})

simulated_impulse_response = impulse_model(impulse_params)

fig = px.line(
    pd.DataFrame({
        "Time frame": np.arange(simulated_impulse_response.shape[1]),
        "Predicted impulse response": simulated_impulse_response[0]
    }),
    x="Time frame",
    y="Predicted impulse response",
)
fig.update_layout(height=450)
fig.show()
```

To predict the response that accounts for the shape of the BOLD measurement, the canonical Gaussian 2D pRF model convolves the stimulus-encoded with the impulse response.

```{code-cell} ipython3
from prfmodel.impulse import convolve_prf_impulse_response

# We redefine the tuning parameters so that we can tweak them together with the encoding
start_params = pd.DataFrame(
    {
        "mu_x": [0.0],
        "mu_y": [0.0],
        "sigma": [1.0],
        "weight_deriv": [-0.5],
    },
)

# We first predict the tuning profile
simulated_tuning_profile = prf_tuning(stimulus, start_params)

# We stimulus-encode the tuning profile
simulated_encoded_profile = prf_encoder(stimulus, simulated_tuning_profile, start_params)

# We predict the impulse response
simulated_impulse_response = impulse_model(start_params)

# We convolve the encoded tuning profile with the impulse response
convolved_response = convolve_prf_impulse_response(simulated_encoded_profile, simulated_impulse_response)

time_frames = np.arange(convolved_response.shape[1])

fig = px.line(
    pd.concat([
        pd.DataFrame({
            "Time frame": time_frames,
            "Predicted response": simulated_encoded_profile[0],
            "Response": "Stimulus-encoded",
        }),
        pd.DataFrame({
            "Time frame": time_frames,
            "Predicted response": convolved_response[0],
            "Response": "Convolved",
        }),
    ]),
    x="Time frame",
    y="Predicted response",
    color="Response",
    color_discrete_map={"Stimulus-encoded": "#2a78d6", "Convolved": "#eb6834"},
)
fig.update_layout(height=450, hovermode="x unified")
fig.show()
```

> **Exercise 8:** Change the impulse model parameter `weight_deriv`. How does the convolved response change and why?

+++

### Scaling models

The predicted convolved pRF model response has a certain amplitude (i.e., scale) and a baseline of zero (i.e., no activity when the screen is blank). However, observed BOLD responses of different neuron populations often differ in their amplitude and baseline from the predictions of the canonical Gaussian 2D pRF model (e.g., due to noise). To account for these differences, a *scaling* submodel can modify the amplitude and baseline of the predicted response.

+++

We create an instance of a scaling model and modify the predicted convolved model response.

```{code-cell} ipython3
import numpy as np
import plotly.express as px

start_params = pd.DataFrame(
    {
        "mu_x": [0.0],
        "mu_y": [0.0],
        "sigma": [1.0],
        "baseline": [0.0],
        "amplitude": [1.0],
        "weight_deriv": [-0.5],
    },
)

# Make prediction with pRF model
simulated_response = prf_model(stimulus, start_params)

fig = px.line(
    pd.DataFrame({
        "Time frame": np.arange(simulated_response.shape[1]),
        "Predicted BOLD response": simulated_response[0]
    }),
    x="Time frame",
    y="Predicted BOLD response",
)
fig.update_layout(height=450)
fig.show()
```

> **Exercise 9:** Change the `baseline` and `amplitude` parameter. How does the predicted response change and why? Change both the `amplitude` and `sigma` parameter at the same time. What happens and why?

+++

## Fitting the model to data

### Inspecting the data

Before we fit the canonical Gaussian 2D pRF model to empirical data, we first load and inspect the recorded BOLD response timecourses. The BOLD response was recorded in a volume of voxels with a size of 2 mm. The dataset only contains the voxels that
fall inside a flattened (dilated) gray matter mask. We can access the timecourses from the `dataset.response` attribute.

```{code-cell} ipython3
response_raw = dataset.response
response_raw.shape 
```

> **Exercise 10:** What do the dimensions of `response_raw` refer to? Visualize one or multiple timecourses with matplotlib or plotly. What is the unit of the BOLD response? Do you see any patterns in the timecourses and if so, what do they mean?

+++

<!-- The `response_raw` object contains the BOLD response timecourse for each voxel inside the mask. It has shape `(num_voxels, num_frames)` where `num_voxels` is the number of voxels in the mask and `num_frames` the number of time frames of the recording. The BOLD response timecourses have 340 time frames. They have already been converted to percent signal change (PSC) relative to a baseline of 100, so we subtract 100 to center each timecourse around zero. -->

```{code-cell} ipython3
response_psc = response_raw - 100.0
```

<!-- Note that converting to PSC and centering are not strictly necessary for pRF model fitting, but they can facilitate it because baseline parameters will be very close to zero and amplitude parameters become smaller compared to fitting on raw BOLD signal. Fitting algorithms such as SGD are more efficient when parameters are small and in a similar range. -->

+++

To get a more complete overview of the BOLD timecourses, we can plot them for all voxels at once in a heatmap.

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt


aspect_ratio = response_psc.shape[1] / response_psc.shape[0]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# We use matplotlib because plotly cannot handle this many voxels
im = ax.imshow(
    response_psc,
    aspect=aspect_ratio,
    cmap="inferno",
    vmin=-2,
    vmax=5,
)

ax.set_xlabel("Time frame (in TR)")
ax.set_ylabel("Voxel index")
fig.colorbar(im, ax=ax, label="BOLD response");
```

> **Exercise 11:** What do you see in the heatmap? Does it confirm or contradict the impression you got from looking at single (or multiple) timecourses?

+++

To visualize the response on the cortical surface, we need to project it from the volume onto the surface. The volume and the surfaces of the subject share the same coordinate system, so we can use `nilearn.surface.vol_to_surf` for the projection. The function samples the volume at several depths between the pial surface and the white matter surface, so we load these surfaces as well. To map a value per voxel back into the volume, we use `nilearn.masking.unmask` with the gray matter mask from the dataset.

```{code-cell} ipython3
import numpy as np
from nilearn.masking import unmask
from nilearn.surface import PolyMesh, vol_to_surf

# The pial and white matter surfaces are always downloaded together with the dataset
pial_mesh = PolyMesh(left=dataset.files["pia_lh"], right=dataset.files["pia_rh"])
wm_mesh = PolyMesh(left=dataset.files["wm_lh"], right=dataset.files["wm_rh"])


def project_to_surface(voxel_values: np.ndarray) -> np.ndarray:
    """Project one value per voxel in the mask onto the vertices of both hemispheres (left first)."""
    volume = unmask(voxel_values, dataset.mask)
    return np.concatenate([
        vol_to_surf(
            volume,
            pial_mesh.parts[hemi],
            inner_mesh=wm_mesh.parts[hemi],
            mask_img=dataset.mask,  # Ignore samples outside the gray matter mask
        )
        for hemi in ("left", "right")
    ])
```

We also define a helper function to plot a per-voxel statistic on a flatmap of the surface.

```{code-cell} ipython3
from nilearn.plotting import plot_surf_stat_map


mesh = dataset.mesh

SURF_VIEW = (90, 270)


def plot_surf_stat_map_helper(
        stat_map: np.ndarray,
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = None,
        cmap: str = "inferno",
    ) -> tuple[plt.Figure, plt.Axes]:
    """Helper function to plot a surface with a stat map."""
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))

    plot_surf_stat_map(
        mesh,
        stat_map,
        vmin=vmin,
        vmax=vmax,
        hemi="both",
        view=SURF_VIEW,
        cmap=cmap,
        axes=ax,
        figure=fig,
        title=title,
    )

    # Expand the 3D axes to fill the space to the left of the colorbar
    surf_axes = [a for a in fig.axes if a.name == "3d"]
    cbar_axes = [a for a in fig.axes if a.name != "3d"]
    cbar_x0 = min(a.get_position().x0 for a in cbar_axes)
    for a in surf_axes:
        a.set_position([0.0, 0.0, cbar_x0 - 0.01, 0.97])

    return fig, ax
```

We can use the helper to plot, for example, the standard deviation of each timecourse on the surface to get an overview of the variability in the BOLD timecourses across the surface.

```{code-cell} ipython3
# Calculate standard deviation of each timecourse
response_sd = response_psc.std(axis=1)

plot_surf_stat_map_helper(project_to_surface(response_sd), vmax=5.0, title="BOLD response standard deviation");
```

> **Exercise 12:** What do you see on the surface flatmap? Can you explain the displayed pattern of response standard deviation? Plot a different statistic on the surface. Do you see any interesting patterns?

+++

### Splitting the data for cross-validation

A pRF model that fits the data it was estimated on well does not necessarily predict new data well, because it can also fit the noise in the data. That is, it can overfit the *training* data and generalize poorly to new *test* data. To evaluate how well our pRF model generalizes, we use cross-validation: We fit the model on the first half of the experiment (the training data) and evaluate its predictions on the second half (the test data). Because the second half repeats the bar sweeps of the first half in the opposite direction, both halves cover the same locations in the visual field but have independent noise.

This "split-half" cross-validation is only appropriate for symmetric or repeated designs, where both halfs repeat the same design matrix. This is the case for this dataset but might not be true for other datasets you encounter in the field and important to keep in mind when designing new pRF experiments.

We split both the stimulus design and the BOLD response at the midpoint of the time axis.

```{code-cell} ipython3
from prfmodel.stimuli import PRFStimulus

num_frames = response_psc.shape[1]

num_frames_train = num_frames // 2

# The training stimulus contains the first half of the design and the same grid
stimulus_train = PRFStimulus(
    design=stimulus.design[:num_frames_train],
    grid=stimulus.grid,
    dimension_labels=stimulus.dimension_labels,
)

response_train = response_psc[:, :num_frames_train]
response_test = response_psc[:, num_frames_train:]

print(stimulus_train)
```

Some voxels in the mask do not have valid timecourses, that is, their timecourses are constant and do not contain any signal. We filter out all voxels that have a constant timecourse in either half of the data.

```{code-cell} ipython3
response_is_valid = (response_train.std(axis=1) > 0.0) & (response_test.std(axis=1) > 0.0)

response_train_valid = response_train[response_is_valid]
response_test_valid = response_test[response_is_valid]

response_is_valid.mean()  # Fraction of valid voxels
```

### Defining the pRF model

Now that training stimulus and training BOLD response are in place, we can fit the canonical Gaussian 2D pRF model. However, we need to slightly customize the model: The stimulus design matrix and the BOLD response were recorded with a repetition time (TR) of 0.9 seconds. By default, pRF models in prfmodel assume a TR of 1 second. The TR is set through the resolution of the impulse response model, so we replace the default with a custom impulse response model that has a resolution of 0.9 seconds. Matching the TR of the impulse response model with the resoluton ot the impulse response model ensures that predicted pRF model response has the same sampling rate as the stimulus and the observed BOLD timecourses. For details, see the [Important Details](https://popylar-org.github.io/prfmodel/important_details.html) section in the online documentation of prfmodel.

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

We fit the canonical Gaussian 2D pRF model in two stages. In the first stage, we perform a "brute-force" grid search to find good estimates for the parameters of the pRF tuning profile (i.e., `mu_x`, `mu_y`, and `sigma`)

For the grid search, we define ranges of `mu_x`, `mu_y`, and `sigma` that we want to construct a grid of parameter values from. For `baseline`, `amplitude`, and `weight_deriv`, we only provide a single value so that they will stay constant across the entire grid.

```{code-cell} ipython3
grid_param_ranges = {
    "mu_x": np.linspace(-4, 4, 20),  # Range of visual field in experiment
    "mu_y": np.linspace(-4, 4, 20),
    "sigma": np.linspace(0.05, 4.0, 20),
    "weight_deriv": [-0.5],
    "baseline": [0.0],
    "amplitude": [1.0],
}
```

To run the grid search, we construct the `prfmodel.fitters.grid.GridFitter`. Note that we pass the training stimulus and set `batch_size=20` to let the fitter evaluate 20 parameter combinations at the same time which saves us some memory. By default, the `loss` (i.e., the metric the fitter minimizes between model predictions and data) is the negative correlation coefficient, which ignores differences in baseline and amplitude between model predictions and observed data.

```{code-cell} ipython3
from prfmodel.fitters import GridFitter

# Create grid fitter object
grid_fitter = GridFitter(
    model=prf_model,
    stimulus=stimulus_train,
    compile_step=True,  # Setting 'compile_step=True' speeds up the fitting
)

# Run grid search
grid_history, grid_params = grid_fitter.fit(
    data=response_train_valid,
    parameter_values=grid_param_ranges,
    batch_size=20,
)
```

We can print the parameters estimated by the grid search.

```{code-cell} ipython3
grid_params
```

> **Exercise 13:** Compare the parameter estimates to the ranges of the grid search. What do you observe? Change the grid ranges (you can also create a range for `weight_deriv`). Can you get to a lower (i.e., better) mean loss across voxels (shown in the progress bar under `loss`)? Change the `batch_size` in `GridFitter`. How does this affect the speed of the fit? You can also try setting `compile_step=False` and compare the fitting speed.

+++

### Least-squares optimization

In the grid search, we ignored `baseline` and `amplitude` parameters. In prfmodel, we estimate them with least-squares by regressing the predicted pRF model responses from the observed responses for each voxel. The intercept and slope of the least-squares regression become the `baseline` and `amplitude` parameter, respectively.

```{code-cell} ipython3
from prfmodel.fitters import LeastSquaresFitter

# Create least-squares fitter
ls_fitter = LeastSquaresFitter(
    model=prf_model,
    stimulus=stimulus_train,
)

# Run least squares fit
ls_history, ls_params = ls_fitter.fit(
    data=response_train_valid,
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

Now that the pRF tuning profile parameters and the auxiliary baseline and amplitude parameters are optimized, we can compare the model predictions against the observed responses. Because we want to make predictions for all voxels at once, we wrap our `prf_model` in the `prfmodel.utils.batched` modifier function. The modifier changes the behavior of the model to make predictions for batches of voxels sequentially. This saves us a lot of memory at the expense of minimal runtime overhead.

For the test data, we need to be careful: The hemodynamic response to the bar at the end of the first half carries over into the beginning of the second half. If we made predictions with the second half of the design alone, the model would miss this carry-over. Therefore, we make predictions for the *full* stimulus and split the predicted timecourses in the same way as the observed ones. The predictions for the first half are identical to the predictions for the training stimulus because nothing precedes the first time frame.

```{code-cell} ipython3
from prfmodel.utils import batched

predict_batched = batched(prf_model)

# Make predictions for the full stimulus with the parameters estimated on the training set
pred_response = np.asarray(predict_batched(stimulus, ls_params, batch_size=200))

pred_response_train = pred_response[:, :num_frames_train]
pred_response_test = pred_response[:, num_frames_train:]
```

We can quantify how well the predictions align with the observed timecourses using the R-squared metric. This metric indicates the proportion of variance in the observed data explained by our model predictions. We start by comparing the model predictions to the observed timecourses of the training data. Because we used the training data to fit our pRF model, we are assessing its in-sample fit.

```{code-cell} ipython3
from keras.metrics import R2Score

r2_metric = R2Score(class_aggregation=None)  # Don't aggregate score over voxels

r_squared_train = np.asarray(
    r2_metric(response_train_valid.T, pred_response_train.T)
)  # Transpose to compute score across time frames
r_squared_train.shape
```

> **Exercise 14:** Compute the R-squared score on the test data and assess the out-of-sample fit. How much do the test R-squared scores differ from the training R-squared scores? How well does the canonical Gaussian 2D pRF model generalize to the test data?

+++

> **Exercise 15:** Plot the test R-squared score on the flat surface. You can use the helper function below. For which voxels does the model generalize better or worse?

```{code-cell} ipython3
def fill_valid_voxels(values: np.ndarray) -> np.ndarray:
    """Fill values for the valid voxels into an array with one value per voxel in the mask."""
    values_full = np.full((response_psc.shape[0],), fill_value=np.nan)
    values_full[response_is_valid] = values
    return values_full
```

> **Exercise 16:** In addition to computing model fit with R-squared, we also recommend comparing model predictions against observed timecourses for individual voxels. Plot in- and out-of-sample predictions against observed response for a subset of voxels. Does the plot confirm the impression you got from the R-squared scores? Why does the model not generalize well for some voxels?

+++

## Interpreting the model parameters

To analyze and interpret the pRF parameters, we will zoom in on surface vertices with an out-of-sample R-squared above a certain threshold (e.g., 0.3; note that this threshold is often somewhat arbitrary). Because we select vertices based on the test set, which was not used for fitting, the selection is not biased towards vertices whose fits mostly capture noise.

For these vertices, we can look at different quantities to interpret the pRFs estimated by our model.

+++

> **Exercise 17:** Create a mask that selects voxels above a test R-squared threshold. Plot the pRF size `sigma` for voxels that pass the threshold on the flat surface. Which pattern do you see? Change the threshold and recreate the plot. Does the pattern change?

+++

> **Exercise 18:** Compute the polar angle of the pRF center using the helper function below. Plot the polar angle for voxels that pass the threshold on the flat surface (hint: use a cyclic colormap, e.g., `cmap="hsv"`). Which pattern do you see?

```{code-cell} ipython3
def calc_polar_angle(mu_x: float, mu_y: float) -> float:
    """Compute the polar angle of a pRF from the x- and y-coordinate of its center."""
    return np.angle(mu_x + mu_y * 1j)
```

> **Exercise 19:** Compute the eccentricity (i.e., the distance of the pRF center from the center of the screen) using the helper function below. Plot it for voxels that pass the threshold it on the flat surface. Which pattern do you see

```{code-cell} ipython3
def calc_eccentricity(mu_x: float, mu_y: float) -> float:
    """Compute the eccentricity of a pRF from the x- and y-coordinate of its center."""
    return np.abs(mu_x + mu_y * 1j)
```

> **Exercise 20 (optional):** Compare the polar angle and eccentricity patterns you observed to the ones reported in the paper by Dumouline and Wandell (2008). Do your results agree with theirs?

+++

> **Exercise 21 (optional, advanced):** Fit the model on the test data and evaluate it on the training data (i.e., two-fold cross-validation). How do you account for the carry-over from training stimulus when fitting the model? Do you see any difference in in- and out-of-sample fit? Average the test R-squared from both folds. Do your conlcusions about how well the model generalizes change?

+++

> **Exercise 22 (optional, advanced):** Fit the compressive spatial summation (CSS) pRF model (Kay et al., 2013) to the dataset using the same workflow. The CSS pRF model is implemented in the `prfmodel.models.prf.Gaussian2DCSSPRFModel` class. Look at the API documentation of the model class to get more information about its parameters. You can also take a look at the paper by Kay et al. (2013) for more background information.

+++

## References

Dumoulin, S. O., & Wandell, B. A. (2008). Population receptive field estimates in human visual cortex. *NeuroImage*, *39*(2), 647-660. https://doi.org/10.1016/j.neuroimage.2007.09.034

Kay, K. N., Winawer, J., Mezer, A., & Wandell, B. A. (2013). Compressive spatial summation in human visual cortex. *Journal of Neurophysiology*, *110*(2), 481-494. https://doi.org/10.1152/jn.00105.2013

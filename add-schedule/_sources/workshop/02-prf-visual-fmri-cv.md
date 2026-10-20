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

# Practical 2: Cross-validation

This is the notebook for the second practical of the workshop. This practical covers how to evaluate the out-of-sample model fit using cross-validation on empirical data from visual functional magnetic resonance imaging (fMRI) experiments. It uses the canonical Gaussian 2D population receptive field (pRF) model as an example and illustrates cross-validation across different runs and sessions from the same experiment.

**We recommend first going through the [first practical](01-prf-visual-fmri.md) because this practical builds on package-specific basics that are covered there.**

**We also recommend going through this notebook before starting with the notebooks of the other practicals which build on this one.**

In computational modelling, cross-validation is typically used to evaluate and compare the ability of different models to generalize their predictions to new observations. That is, from a set of candidate models, we want to select the model which can generalize best to unseen data. Here, we focus only on the cross-validation part and leave the model comparison and selection part to the [next practical](03-prf-visual-fmri-comparison.md).

As all computational and machine learning models, pRF models can overfit the data that they are "trained" on (i.e., the data they are fit to in order to estimate their parameters). Overfitting means that the pRF model fits noise in the training data that is not present or different in unseen (i.e., test) data. It leads to a degradation in prediction performance from the training to test data. The more free parameters a model has, the more flexibly it can fit to noise in the training data and the more likely it is to overfit. While some degree of overfitting is unavoidable, we want to at least know when a pRF model overfits and whether we can trust that it is modelling a robust pattern in the data and not just noise. When comparing different models, we also want to be sure that a model leads to better predictions because it captures the underlying pattern in the data better and not just because it fits better to noise in the data (e.g., because it has more parameters).

+++

## Contents

- [Installation and Setup](#installation-and-setup)
- [Loading the dataset](#loading-the-dataset)
- [Selecting vertices in visual cortex](#selecting-vertices-in-visual-cortex)
- [Fitting the model](#fitting-the-model)
- [Evaluating the model fit in-sample](#evaluating-the-model-fit-in-sample)
- [Same stimulus, new session](#same-stimulus-new-session)
- [New stimulus, same session](#new-stimulus-same-session)
- [Fitting and interpreting the final model](#fitting-and-interpreting-the-final-model)
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

## Loading the dataset

+++

In this example, we use the Center for Human Neuroscience (CHN) dataset (Chang et al., 2024, 2025) that includes BOLD responses recorded with a 3 Tesla MRI scanner while subjects were exposed to two different bar stimuli. We only use the standard (fixed) bar stimulus here.

We load the dataset which is included in the prfmodel package. The `load_dataset` function has a `split` argument that we can use to load a specific run of the experiment. This allows us to use different runs for fitting (i.e., training) and evaluating the out-of-sample fit (i.e., testing) the model. The dataset contains three runs from a single subject:

| Split | Session | Stimulus | Role in this practical |
|---|---|---|---|
| `ses-01_run-01` | 1 | Sequence of bar positions A | Training |
| `ses-02_run-01` | 2 | Sequence of bar positions A (identical to training) | Test: same stimulus, new session |
| `ses-01_run-02` | 1 | Sequence of bar positions B (different from training) | Test: new stimulus, same session |

The two test runs allow us to ask two different questions about how well the model generalizes: Can it predict a new measurement of the *same* experiment, recorded in a different session? And can it predict the response to a *new* stimulus that it was not fitted to?

We start with loading the training data which is the first run of the first session. We select the standard bar stimulus with the `task` argument. By default, `load_dataset` only loads the left hemisphere. On first use, every run downloads about 360 MB, so loading can take a moment.

```{code-cell} ipython3
from prfmodel.examples import load_dataset

# Downloads on first use and caches in a user data directory (see prfmodel.examples.get_data_dir)
dataset_train = load_dataset("3t-chn-visual", split="ses-01_run-01", task="fixedbar")

# Select the stimulus used for training
stimulus_train = dataset_train.stimulus
```

> **Exercise 1:** Using what you learned in the first practical, inspect the stimulus design matrix and grid. How many bar passes are in the design matrix and in which directions do they run? What is the size of the stimulus grid? (hint: use `prfmodel.plotting.animate_2d_prf_stimulus` to create a video of the stimulus).

```{code-cell} ipython3
# Your code goes here
```

Before fitting a pRF model to the data, we inspect the BOLD response timecourses. Note that this dataset contains more timecourses than the `7t-aot-visual` dataset from the first practical and each timecourse corresponds to a vertex on the surface (not to a voxel in the volume). This means that we don't need to project timecourse statistics and model parameters onto the surface for visualization.

```{code-cell} ipython3
response_train = dataset_train.response
response_train.shape
```

> **Exercise 2:** Inspect the BOLD timecourses. What is the unit of the BOLD response? Can you see the response to the bar sweeps when visualizing them?

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 3:** Using the helper function below, plot the standard deviation (SD) of each vertex on an inflated surface. Which areas of the brain have the largest variability in BOLD response? Can you find V1 on the surface?

```{code-cell} ipython3
---
cellView: form
jupyter:
  source_hidden: true
tags: [hide-input]
---
# @title Helper functions for plotting on the surface { display-mode: "form" }
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import InMemoryMesh, PolyMesh


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


# The surface is the same for all splits of the dataset, so we take it from the training data
mesh = separate_hemispheres(dataset_train.mesh)
hemispheres = [hemi for hemi in ("left", "right") if hemi in mesh.parts]  # Rows of the response are in this order
num_vertices = {hemi: mesh.parts[hemi].n_vertices for hemi in hemispheres}

# Downsample each hemisphere for plotting
downsampled = {hemi: downsample_mesh(mesh.parts[hemi], cell_size=2.0) for hemi in hemispheres}
plot_mesh = PolyMesh(**{hemi: plot_part for hemi, (plot_part, _) in downsampled.items()})
merged_indices = {hemi: merged_index for hemi, (_, merged_index) in downsampled.items()}

# Views as (hemisphere, elevation, azimuth); medial and lateral views are rotated slightly towards the back of the brain
# and the posterior view looks slightly upward from below to show more of the occipital cortex
if len(hemispheres) == 2:
    SURF_VIEWS = {
        "Left medial": ("left", -10, 330),
        "Posterior": ("both", -20, 270),
        "Right medial": ("right", -10, 210),
    }
else:
    hemi = hemispheres[0]
    SURF_VIEWS = {
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

    The stat map must have one value per row of the response (i.e., per vertex). Vertices with NaN values are not
    colored. Set `cyclic=True` for angles in radians (e.g., polar angle) to use the color range [-pi, pi] with the
    cyclic colormap "hsv" (like `prfmodel.plotting.plot_surface_stat_map`). The angles of merged vertices are then
    averaged as unit vectors.
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
        1, len(SURF_VIEWS), subplot_kw={"projection": "3d"}, figsize=(12, 4.5), layout="constrained",
    )

    for ax, (view_name, (hemi, elev, azim)) in zip(axes, SURF_VIEWS.items()):
        plot_surf_stat_map(
            # nilearn centers the coordinates of a single hemisphere in place, which would undo separate_hemispheres
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
```

```{code-cell} ipython3
# Your code goes here
```

## Selecting vertices in visual cortex

The response contains a timecourse for every vertex of the left hemisphere, but the pRF model only makes sense for vertices that respond to visual stimulation. Fitting the model to all vertices would also take a long time. The dataset includes a region of interest (ROI) label for each vertex, which tells us whether the vertex belongs to one of the visual areas V1, V2, or V3 (or to none of them).

```{code-cell} ipython3
roi_index = dataset_train.roi_index  # One ROI label per vertex

dataset_train.roi_mapping  # Which ROI each label refers to
```

```{code-cell} ipython3
# Only color vertices that belong to one of the ROIs; at the borders between ROIs, merged vertices get in-between colors
plot_surf_stat_map_helper(
    np.where(roi_index > 0, roi_index, np.nan),
    vmin=1,
    vmax=3,
    cmap="viridis",
    title="ROIs (1 = V1, 2 = V2, 3 = V3)",
);
```

> **Exercise 4:** Compare the ROIs to the map of the response SD that you plotted above. Do the vertices with the largest variability lie inside the ROIs?

+++

We select the vertices that belong to one of the ROIs. We also make sure that their responses vary over time: A vertex with a constant response contains no information about the stimulus.

```{code-cell} ipython3
# Vertices in V1, V2, or V3 with a response that varies over time
response_is_valid = (roi_index > 0) & (response_train.std(axis=1) > 0.0)

response_train_valid = response_train[response_is_valid]


def fill_valid_vertices(values: np.ndarray) -> np.ndarray:
    """Fill values for the valid vertices into an array with one value per vertex, using NaN for all other vertices."""
    values_full = np.full((response_is_valid.shape[0],), fill_value=np.nan)
    values_full[response_is_valid] = values
    return values_full


response_train_valid.shape
```

> **Exercise 5:** How many vertices of the whole hemisphere have a constant response (hint: `response_train.std(axis=1) == 0.0`) and where do they lie on the surface? Are any of them inside the ROIs?

```{code-cell} ipython3
# Your code goes here
```

## Fitting the model

We create a canonical Gaussian 2D pRF model with a custom impulse response submodel to fit to the training data. The repetition time (TR) for this dataset is 1.2 seconds.

```{code-cell} ipython3
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.prf import Gaussian2DPRFModel

# Define repetition time (TR)
tr = 1.2

# Create custom impulse model
impulse_model = DerivativeTwoGammaImpulse(resolution=tr)

# Define pRF model with custom impulse response submodel
prf_model = Gaussian2DPRFModel(
    impulse_model=impulse_model,
)
```

> **Exercise 6 (optional):** Why do we need to create a custom impulse response model with the TR of the dataset? Read the relevant [Important Details](https://popylar-org.github.io/prfmodel/important_details.html) section.

+++

To keep things simple, we only use grid search and least-squares in this example to fit the pRF model to the training data. We start by defining ranges of the grid search.

```{code-cell} ipython3
from prfmodel.fitters import grid_values_2d_prf

grid_param_ranges = grid_values_2d_prf(
    stimulus_train,  # We use the stimulus from the training data
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

Then, we create the `GridFitter` and run the grid search on the timecourses of the selected vertices.

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
    data=response_train_valid,  # Only the selected vertices
    parameter_values=grid_param_ranges,
    batch_size=20,
)
```

> **Exercise 7:** Inspect the results from the grid search. Are there any vertices for which the center of the pRF lies exactly on or outside the stimulus grid (hint: use `prfmodel.plotting.plot_2d_prf_centers`)? Are there any vertices for which the pRF size (`sigma`) lies on the floor or ceiling of the sigma range (hint: use `prfmodel.plotting.plot_grid_parameter_distribution`)? How should the parameters of these vertices be handled or interpreted?

```{code-cell} ipython3
# Your code goes here
```

After the grid search, we fit the `baseline` and `amplitude` parameters of the model with least-squares.

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

## Evaluating the model fit in-sample

Before we look at the out-of-sample fit of the pRF model, we look at its fit on the training data. For that, we make predictions with the training stimulus and the training parameters.

```{code-cell} ipython3
from prfmodel.utils import batched

predict_batched = batched(prf_model)

# Make predictions with the estimated parameters
pred_response_train = np.asarray(predict_batched(stimulus_train, ls_params, batch_size=200))
```

We compare the training data predictions against the observed BOLD timecourses using R-squared.

```{code-cell} ipython3
from prfmodel.utils import calculate_r_squared

r_squared_train = calculate_r_squared(response_train_valid, pred_response_train)  # One score per selected vertex
r_squared_train.shape
```

> **Exercise 8:** Visualize the distribution of the training R-squared values (hint: use `prfmodel.plotting.plot_r_squared_hist`). What fraction of the selected vertices has an R-squared above 0.3? Plot the R-squared on the inflated surface (hint: use `fill_valid_vertices`). Where do the well-fitting vertices lie?

```{code-cell} ipython3
# Your code goes here
```

## Same stimulus, new session

To assess the out-of-sample model fit, we first evaluate its predictions on the first run of the second session. The subject saw exactly the same stimulus as in the training run. What differs is the noise in the measurement and everything that changes between sessions, for example, the position of the head in the scanner, the alignment of the data to the anatomy, and the state of the subject. This tells us whether the model captures a pattern in the responses that we can reproduce in a new measurement, and not just the noise of the training run.

We load the dataset for this run using the `split` argument.

```{code-cell} ipython3
dataset_test_ses = load_dataset("3t-chn-visual", split="ses-02_run-01", task="fixedbar")

stimulus_test_ses = dataset_test_ses.stimulus

response_test_ses = dataset_test_ses.response

# Select the same vertices as for the training data
response_test_ses_valid = response_test_ses[response_is_valid]
```

> **Exercise 9:** Check that the stimulus of this test dataset is identical to the training stimulus (hint: use `np.array_equal` on the design matrices). What does this imply for the model predictions for training and test data? Where can differences in prediction performance between training and test data then come from?

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 10:** Visualize the BOLD response from this test dataset. Do you see any fundamental differences compared to the training data?

```{code-cell} ipython3
# Your code goes here
```

We make model predictions using the stimulus of the *test* data and the parameters of the *training* data. We can reuse the batched model object because nothing in it is specific to the training or test data. We compare the test predictions against the observed test timecourses.

```{code-cell} ipython3
pred_response_test_ses = np.asarray(predict_batched(stimulus_test_ses, ls_params, batch_size=200))

r_squared_test_ses = calculate_r_squared(response_test_ses_valid, pred_response_test_ses)
r_squared_test_ses.shape
```

> **Exercise 11:** Compare the training against the test R-squared (hint: use `prfmodel.plotting.plot_r_squared_comparison`). Also compare the predicted vs. observed timecourses for the best fitting vertices (hint: use `prfmodel.plotting.plot_observed_predicted`). Can you explain the differences in prediction performance between training and test data? Can you explain why some vertices have negative out-of-sample R-squared scores?

```{code-cell} ipython3
# Your code goes here
```

For full leave-one-out cross-validation, we would repeat the out-of-sample prediction but swap the training and the test data: The split `ses-02_run-01` would become the training and `ses-01_run-01` the test data. We would then average the out-of-sample R-squared scores across both directions (also called *folds*) for each vertex. In this way, our generalization assessment does not depend on the order of the two runs which we assume to be independent.

+++

> **Exercise 12 (optional):** Compute the out-of-sample R-squared in reverse order (using `ses-02_run-01` as training and `ses-01_run-01` as test data). How does out-of-sample R-squared differ between the two folds?

```{code-cell} ipython3
# Your code goes here
```

## New stimulus, same session

Next, we evaluate the model predictions on the second run of the first session. This run was recorded in the same session as the training run, but the bar moves through the visual field in a different sequence. This is a different test because the model must predict the response to a stimulus that it was not fitted to. A pRF model that captures how a vertex responds to stimulation in different parts of the visual field should be able to do that. In contrast, a model that only fits the timecourse of the training run (including its noise) would fail.

```{code-cell} ipython3
dataset_test_run = load_dataset("3t-chn-visual", split="ses-01_run-02", task="fixedbar")

stimulus_test_run = dataset_test_run.stimulus

response_test_run = dataset_test_run.response

# Select the same vertices as for the training data
response_test_run_valid = response_test_run[response_is_valid]
```

> **Exercise 13:** Compare the stimulus of this test dataset against the training stimulus (hint: create a video of both stimuli). How does the sequence of bar positions differ? Compute and evaluate the out-of-sample prediction performance of the pRF model for this dataset.

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 14:** Compare the out-of-sample prediction performance for this test dataset to the previous test dataset. For which test dataset is it better? Think about what differs between the training data and each of the test datasets (noise, session, stimulus) and how this can explain the difference.

```{code-cell} ipython3
# Your code goes here
```

## Fitting and interpreting the final model

Cross-validation is primarily used for comparing the out-of-sample performance between different candidate models and selecting the model that best generalizes to new observations. Model comparison and selection will be part of the [next practical](03-prf-visual-fmri-comparison.md).

Once we have selected a model, our goal changes: We no longer want to *evaluate* the model but to *estimate* its parameters as precisely as possible so that we can interpret them. For that, we want to use as much data with the least amount of noise that we can get, including the data that we held out for testing before. This means that there is no held-out data left to evaluate the final fit, and that its in-sample R-squared is optimistic for the same reasons as above. We accept this because cross-validation has already told us how well this model generalizes to new data.

For the CHN dataset, we can for example average the BOLD responses of the two runs that show the identical stimulus (the first runs of both sessions) and fit the pRF model on the averaged responses.

```{code-cell} ipython3
# Average the first runs of both sessions, which show the identical stimulus
response_avg = np.stack([
    response_train_valid,
    response_test_ses_valid,
], axis=-1).mean(axis=-1)

response_avg.shape
```

> **Exercise 15:** Compare the averaged responses against the responses from only the first session. Do you observe differences in the noise level?

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 16:** Why is averaging the responses across sessions a valid choice for this dataset?

+++

> **Exercise 17:** Why can we average the first run of the first session with the first run of the second session, but not with the second run of the first session?

+++

> **Exercise 18:** Fit the Gaussian 2D pRF model to the averaged responses (`response_avg`) using grid search and least-squares. Evaluate the model predictions on the averaged responses. How does the in-sample R-squared compare to the in-sample R-squared of the model fitted to only the first run of the first session?

```{code-cell} ipython3
# Your code goes here
```

> **Exercise 19:** Subset the model parameter estimates using an R-squared threshold. Plot the parameter subset on the inflated surface (hint: set the estimates of vertices below the threshold to `np.nan` and use `fill_valid_vertices` to get one value for every vertex on the surface). Do the results align with your findings in the first practical?

```{code-cell} ipython3
# Your code goes here
```

For thresholding vertices for which we can interpret the parameters, we face a similar problem as for model selection: We want to select those vertices for which the pRF model robustly estimates the parameters without overfitting to noise in the data. Thus, we should not rely on R-squared based on the training data to select vertices for interpretation. Instead, we can for example use out-of-sample R-squared from cross-validation.

+++

> **Exercise 20:** Use a threshold based on test R-squared on the `ses-01_run-02` split to select vertices to plot on the inflated surface. Do you see any differences from the results of the previous exercise?

```{code-cell} ipython3
# Your code goes here
```

## Summary

- The in-sample (training) R-squared is an optimistic estimate of how well a pRF model describes the data because the model is partly fitted to noise in the training data.
- Cross-validation evaluates the model on data that it was not fitted to. What the test data has in common with the training data determines which kind of generalization we test: A new session with the same stimulus tests whether the model captures a pattern that we can reproduce in a new measurement; a new stimulus tests whether the model captures how a vertex responds to stimulation in different parts of the visual field.
- The out-of-sample R-squared also depends on the noise in the test data. It can be negative when the model predicts the test data worse than the mean of the test data.
- We use cross-validation to compare models and to select vertices for which we can trust the parameter estimates. To interpret the parameters, we fit the selected model to all comparable data.

## References

Chang, K., Fine, I., & Boynton, G. M. (2024). CHN Retinotopic Mapping Dataset. *OpenNeuro*. https://doi.org/10.18112/openneuro.ds004698.v1.0.1

Chang, K., Fine, I., & Boynton, G. M. (2025). Improving the reliability and accuracy of population receptive field measures using a logarithmically warped stimulus. *Journal of Vision*, *25*(1), 5. https://doi.org/10.1167/jov.25.1.5

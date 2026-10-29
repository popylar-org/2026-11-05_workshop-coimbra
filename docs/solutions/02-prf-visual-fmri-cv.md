## Exercise 1

```python
from IPython.display import HTML
from prfmodel.plotting import animate_2d_prf_stimulus

print(stimulus_train)
print("Grid from", stimulus_train.grid.min(axis=(0, 1)), "to", stimulus_train.grid.max(axis=(0, 1)), "(y, x)")

# In this dataset, the first row of the grid is the upper edge of the visual field
ani = animate_2d_prf_stimulus(stimulus_train, origin="upper", interval=25)

HTML(ani.to_html5_video())
```

The design matrix has 305 time frames and a grid of 108 x 108 cells that spans -8 to 8 degrees of visual angle in both directions. The bar moves within a circular aperture with a radius of 8 degrees. Unlike in the first practical, the first row of the grid is the upper edge of the visual field, so we set `origin="upper"` to show the video the right way up.

The design contains eight bar passes of 37–38 time frames each, without blank periods in between (the bar is visible from time frame 2 to 301). Some of the passes are slightly tilted against the horizontal and vertical axes.

## Exercise 2

```python
print("Largest absolute mean of a timecourse:", np.abs(response_train.mean(axis=1)).max())
print("Median standard deviation of the timecourses:", np.median(response_train.std(axis=1)))
```

Every timecourse has a mean of zero, and the timecourses vary by several hundred units. Thus, the response is not in percent signal change like in the first practical, but in the arbitrary units of the scanner signal with the mean of each timecourse subtracted.

We plot the timecourses of all vertices as a heatmap, sorted by their standard deviation (SD). The heatmap has many more vertices than the figure has pixel rows, so it can only show a fraction of them. Therefore, we also zoom in on the 500 most variable vertices at the bottom of the heatmap.

```python
import matplotlib.pyplot as plt
from prfmodel.plotting import plot_response_heatmap

# All vertices, sorted by the standard deviation of their response (the most variable at the bottom)
sorted_vertices = np.argsort(response_train.std(axis=1))

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, layout="constrained")

for ax, vertices, title in zip(
    axes, [sorted_vertices, sorted_vertices[-500:]], ["All vertices", "500 most variable vertices"],
):
    plot_response_heatmap(
        response_train[vertices],
        vmin=-300,
        vmax=300,
        cmap="RdBu_r",
        xlabel="",
        ylabel="Vertex (sorted by SD)",
        colorbar_label="BOLD response",
        ax=ax,
    )
    ax.set_title(title)

axes[-1].set_xlabel("Time frame (in TR)");
```

Only for a small fraction of the vertices. In the heatmap of all vertices, the rows at the top have a constant response, and the other rows mostly show noise without a clear pattern: Most vertices of the hemisphere do not respond to the visual stimulus. Among the 500 most variable vertices, we can see vertical stripes where many vertices respond at the same time frames, roughly once for each of the eight bar passes. However, some of the most variable vertices (e.g., the rows at the very bottom) fluctuate quickly from one time frame to the next without following the bar passes. Their variability is caused by noise, so a large SD alone does not tell us that a vertex responds to the stimulus. Below, we will select the vertices in the visual cortex.

## Exercise 3

```python
plot_surf_stat_map_helper(
    response_train.std(axis=1), vmin=0.0, vmax=150.0, title="BOLD response standard deviation",
);
```

The largest variability lies in the occipital lobe at the back of the brain, especially on the medial side around the calcarine sulcus and at the occipital pole. This is where V1 lies. There are also patches with a large SD in other places, for example, around the medial wall (the black area without cortex) and on the lateral surface. Here, the variability is likely not caused by the visual stimulus but by noise.

## Exercise 4

Partly. The ROIs cover the occipital pole and the area around the calcarine sulcus, where many vertices have a large SD because they respond to the bar. However, not all vertices inside the ROIs are highly variable (e.g., the parts of V1–V3 further to the front, which represent the periphery of the visual field beyond the stimulus), and many highly variable vertices lie outside the ROIs. The SD does not distinguish between variability caused by the stimulus and variability caused by noise, so it is not a good criterion for selecting vertices on its own.

## Exercise 5

```python
is_constant = response_train.std(axis=1) == 0.0

print("Vertices with a constant response:", is_constant.sum(), "of", len(is_constant))
print("Of those inside the ROIs:", np.sum(is_constant & (roi_index > 0)))

# Only color the vertices with a constant response
plot_surf_stat_map_helper(
    np.where(is_constant, 1.0, np.nan), vmin=0.0, vmax=1.0, title="Vertices with a constant response",
);
```

About 15,600 of the 221,709 vertices (7%) have a constant response. Most of them lie in the medial wall, where the surface closes the hemisphere but there is no cortex. Others lie along the top edge of the hemisphere, which was probably outside the part of the brain that was scanned, and a few are scattered over the lateral surface. None of them lie inside the ROIs.

## Exercise 6

The impulse response describes how the BOLD signal unfolds over time in seconds: It peaks about 5 seconds after neural activity and returns to baseline after about 20–30 seconds. The model convolves the impulse response with the predicted neuron population activity, which has one value per time frame. For the convolution to be correct, the impulse response must be sampled with the same time step as the stimulus and the data, that is, once per TR. If we sampled it every second instead, every sample of the impulse response would be interpreted as one time frame of 1.2 seconds. The predicted BOLD response would then be stretched in time: For example, the peak 5 seconds after the activity would end up 5 time frames (6 seconds) later.

## Exercise 7

```python
import matplotlib.pyplot as plt
from prfmodel.plotting import plot_2d_prf_centers, plot_grid_parameter_distribution

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")

plot_2d_prf_centers(grid_params, grid_param_ranges, stimulus=stimulus_train, ax=axes[0])
axes[0].set(xlabel="mu_x (in degrees)", ylabel="mu_y (in degrees)", title="pRF centers")

plot_grid_parameter_distribution(grid_params, grid_param_ranges, "sigma", ax=axes[1])
axes[1].set(xlabel="sigma (in degrees)", title="pRF sizes");
```

```python
grid_min_xy, grid_max_xy = stimulus_train.grid.min(axis=(0, 1)), stimulus_train.grid.max(axis=(0, 1))
edge_tolerance = 1e-3  # in degrees

is_on_or_outside_edge = (
    (grid_params["mu_x"] < grid_min_xy[1] + edge_tolerance)
    | (grid_params["mu_x"] > grid_max_xy[1] - edge_tolerance)
    | (grid_params["mu_y"] < grid_min_xy[0] + edge_tolerance)
    | (grid_params["mu_y"] > grid_max_xy[0] - edge_tolerance)
)

print("Fraction with a center on or outside the edge of the stimulus:", is_on_or_outside_edge.mean())
print("Fraction with sigma at the floor:", np.isclose(grid_params["sigma"], np.min(grid_param_ranges["sigma"])).mean())
print("Fraction with sigma at the ceiling:", np.isclose(grid_params["sigma"], np.max(grid_param_ranges["sigma"])).mean())
```

Yes. Most pRF centers lie inside the stimulus and in the right half of the visual field, which the left hemisphere represents. However, about a third of the vertices have a center on the edge of or outside the stimulus. About 11% of the vertices have a `sigma` at the floor (0.05 degrees) and about 7% at the ceiling (4 degrees) of the grid.

We should not interpret the parameters of these vertices at face value:

- **Center on or outside the stimulus edge:** The stimulus never covers the pRF center, so the data only constrain the center weakly. Many of these vertices also have a response that is dominated by noise. We usually exclude them from the interpretation.
- **`sigma` at the ceiling:** The best fitting size may lie beyond the grid, so the estimate is only a lower bound. We can extend the grid or exclude these vertices.
- **`sigma` at the floor:** The size is smaller than what the stimulus can resolve (e.g., the width of the bar), so the estimate is only an upper bound. Very small pRFs are also often a sign of a noisy fit.

## Exercise 8

```python
from prfmodel.plotting import plot_r_squared_hist

plot_r_squared_hist(r_squared_train);

print("Fraction of vertices with R-squared > 0.3:", np.mean(r_squared_train > 0.3))
```

```python
plot_surf_stat_map_helper(
    fill_valid_vertices(r_squared_train), vmin=0.0, vmax=1.0, title="Variance explained (R-squared, training data)",
);
```

About 38% of the selected vertices have an R-squared above 0.3. The well-fitting vertices lie in the posterior parts of V1–V3 around the occipital pole, which represent the central part of the visual field covered by the stimulus. Towards the front of the ROIs on the medial surface, which represents the periphery of the visual field beyond the stimulus, the R-squared drops close to zero.

## Exercise 9

```python
print("Identical design matrices:", np.array_equal(stimulus_train.design, stimulus_test_ses.design))
print("Identical grids:", np.array_equal(stimulus_train.grid, stimulus_test_ses.grid))
```

Both are identical. Thus, the model with the training parameters makes exactly the same predictions for the training and the test data. Any difference in prediction performance must come from differences in the observed responses:

- **Noise:** Thermal and physiological noise (e.g., breathing and heartbeat) differ between the runs. The model is partly fitted to the noise of the training run, which does not repeat in the test run.
- **Session:** The position of the head in the scanner, the alignment of the data to the anatomy, and the scanner itself (e.g., drifts and calibration) differ between sessions.
- **Subject:** The state of the subject, for example, attention, fatigue, and how well they fixated, can differ between sessions.

## Exercise 10

We plot the responses of the selected vertices in both runs next to each other with the same color range.

```python
import matplotlib.pyplot as plt
from prfmodel.plotting import plot_response_heatmap

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")

for ax, response, title in zip(
    axes, [response_train_valid, response_test_ses_valid], ["Training (ses-01_run-01)", "Test (ses-02_run-01)"],
):
    plot_response_heatmap(
        response,
        vmin=-300,
        vmax=300,
        cmap="RdBu_r",
        xlabel="Time frame (in TR)",
        ylabel="Vertex index",
        colorbar_label="BOLD response",
        ax=ax,
    )
    ax.set_title(title)

print("Median SD (training):", np.median(response_train_valid.std(axis=1)))
print("Median SD (test):", np.median(response_test_ses_valid.std(axis=1)))
```

No. Both runs show the same pattern of responses to the bar passes at the same time frames, and the variability of the responses is similar (a median SD of about 77 vs. 74). The runs only differ in the details, that is, in the noise.

## Exercise 11

```python
import matplotlib.pyplot as plt
from prfmodel.plotting import plot_observed_predicted, plot_r_squared_comparison

plot_r_squared_comparison(
    r_squared_train,
    r_squared_test_ses,
    xlabel="R-squared (training: ses-01_run-01)",
    ylabel="R-squared (test: ses-02_run-01)",
)

print("Mean R-squared (training):", r_squared_train.mean())
print("Mean R-squared (test):", r_squared_test_ses.mean())
print("Fraction of vertices with a negative test R-squared:", np.mean(r_squared_test_ses < 0.0))
```

Neighboring vertices on the surface have almost the same response, so we do not plot the three best vertices but the best vertex, the vertex at the top 1%, and the vertex at the top 10% of the training R-squared. We also show the observed training response in gray. The prediction is the same for both runs.

```python
# Vertices sorted from the highest to the lowest training R-squared
order = np.argsort(r_squared_train)[::-1]
example_vertices = order[[0, len(order) // 100, len(order) // 10]]

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")

for ax, vertex in zip(axes, example_vertices):
    plot_observed_predicted(
        response_test_ses_valid[vertex], pred_response_test_ses[vertex], observed_label="Observed (test)", ax=ax,
    )
    ax.plot(response_train_valid[vertex], color="gray", linewidth=1, alpha=0.6, label="Observed (training)")
    ax.legend(frameon=False, loc="upper right")
    ax.set(
        xlabel="",
        ylabel="BOLD response",
        title=f"R-squared training = {r_squared_train[vertex]:.2f}, test = {r_squared_test_ses[vertex]:.2f}",
    )

axes[-1].set_xlabel("Time frame (in TR)");
```

For most vertices, the test R-squared is somewhat lower than the training R-squared (the points lie below the identity line), and the mean R-squared drops from about 0.22 to 0.19. The model was fitted to the training run, including part of its noise. The test run contains different noise, so this part of the fit does not carry over. For the best vertices, the predicted peaks match the peaks in both runs, but the noise between the peaks differs.

The R-squared compares the squared errors of the prediction with the variance of the observed response around its mean. When the prediction is worse than simply predicting the mean of the test response, the R-squared becomes negative. This cannot happen for the training data because least-squares can always set the amplitude to zero and the baseline to the mean. It can happen for the test data, where the parameters are fixed. About a third of the vertices have a slightly negative test R-squared. They almost all have a training R-squared close to zero: The model fitted noise in the training run, and this noise prediction only adds errors in the test run.

## Exercise 12

We fit the model to the first run of the second session in the same way as to the training data and evaluate it on the first run of the first session.

```python
from prfmodel.plotting import plot_r_squared_comparison

_, grid_params_rev = GridFitter(model=prf_model, stimulus=stimulus_test_ses, compile_step=True).fit(
    data=response_test_ses_valid,
    parameter_values=grid_param_ranges,
    batch_size=20,
)

_, ls_params_rev = LeastSquaresFitter(model=prf_model, stimulus=stimulus_test_ses).fit(
    data=response_test_ses_valid,
    parameters=grid_params_rev,
    slope_name="amplitude",
    intercept_name="baseline",
    batch_size=200,
)

pred_response_rev = np.asarray(predict_batched(stimulus_train, ls_params_rev, batch_size=200))
r_squared_test_rev = calculate_r_squared(response_train_valid, pred_response_rev)

plot_r_squared_comparison(
    r_squared_test_ses,
    r_squared_test_rev,
    xlabel="Out-of-sample R-squared (fold 1)",
    ylabel="Out-of-sample R-squared (fold 2)",
)

# Average across both folds for every vertex
r_squared_cv = (r_squared_test_ses + r_squared_test_rev) / 2.0

print("Mean out-of-sample R-squared (fold 1):", r_squared_test_ses.mean())
print("Mean out-of-sample R-squared (fold 2):", r_squared_test_rev.mean())
print("Mean cross-validated R-squared:", r_squared_cv.mean())
```

The two folds give very similar results: The mean out-of-sample R-squared is about 0.19 in both directions, and the scores of individual vertices are highly correlated between the folds. Small differences remain because each fold is fitted to and evaluated on different noise. For this dataset, the order of the runs hardly matters, but averaging both folds still gives a more reliable estimate than a single fold.

## Exercise 13

```python
from IPython.display import HTML, display
from prfmodel.plotting import animate_2d_prf_stimulus

for stimulus, title in [(stimulus_train, "Training (ses-01_run-01)"), (stimulus_test_run, "Test (ses-01_run-02)")]:
    ani = animate_2d_prf_stimulus(stimulus, title=title, origin="upper", interval=25)
    display(HTML(ani.to_html5_video()))
```

Both stimuli have the same timing: eight bar passes of 37–38 time frames each without blank periods. However, the passes run in different directions and in a different order. For example, the first pass of the test run moves from the lower left to the upper right instead of from the upper left to the lower right, and the second pass moves from left to right instead of from right to left. Hence, the bar is at a different position at almost every time frame.

```python
from prfmodel.plotting import plot_r_squared_comparison

pred_response_test_run = np.asarray(predict_batched(stimulus_test_run, ls_params, batch_size=200))
r_squared_test_run = calculate_r_squared(response_test_run_valid, pred_response_test_run)

plot_r_squared_comparison(
    r_squared_train,
    r_squared_test_run,
    xlabel="R-squared (training: ses-01_run-01)",
    ylabel="R-squared (test: ses-01_run-02)",
)

print("Mean R-squared (training):", r_squared_train.mean())
print("Mean R-squared (test, new stimulus):", r_squared_test_run.mean())
```

The model generalizes to the new stimulus: Vertices with a high training R-squared also have a high test R-squared. Thus, the model captures how the vertices respond to stimulation at different positions in the visual field and not only the timecourse of the training run. However, the test R-squared is lower than the training R-squared (a mean of about 0.17 vs. 0.22).

## Exercise 14

```python
from prfmodel.plotting import plot_r_squared_comparison

plot_r_squared_comparison(
    r_squared_test_ses,
    r_squared_test_run,
    xlabel="R-squared (same stimulus, new session)",
    ylabel="R-squared (new stimulus, same session)",
)

pd.DataFrame({
    "training": r_squared_train,
    "same stimulus, new session": r_squared_test_ses,
    "new stimulus, same session": r_squared_test_run,
}).agg(["mean", "median"])
```

The prediction performance is better for the test run with the same stimulus in a new session (a mean R-squared of about 0.19) than for the test run with a new stimulus in the same session (about 0.17).

- **Same stimulus, new session:** Only the noise and the session differ. The prediction is identical to the training prediction, so the model only has to reproduce the same stimulus-driven response in a new measurement. Any error in the pRF estimates that does not matter for the training stimulus does not matter for the test run either.
- **New stimulus, same session:** The noise and the stimulus differ, but the session is the same. The model has to predict the response to new bar positions. Errors in the pRF estimates and in the assumptions of the model (e.g., the Gaussian shape of the pRF or the impulse response) that were hidden by the specific training stimulus now show up.

For this dataset, generalizing to a new stimulus is harder than generalizing to a new session, even though the new session adds more differences in the measurement.

## Exercise 15

We plot both responses for the vertex with the highest training R-squared. To quantify the noise, we use the difference between the two sessions: It only contains noise because the stimulus-driven response is the same in both sessions.

```python
import matplotlib.pyplot as plt

vertex = np.argmax(r_squared_train)

fig, ax = plt.subplots(figsize=(10, 3), layout="constrained")
ax.plot(response_train_valid[vertex], color="gray", linewidth=1, label="First session (ses-01_run-01)")
ax.plot(response_avg[vertex], color="black", linewidth=1.5, label="Average of both sessions")
ax.legend(frameon=False, loc="upper right")
ax.set(xlabel="Time frame (in TR)", ylabel="BOLD response")

# For independent noise with the same SD in both sessions, the difference between the sessions has sqrt(2) times
# the noise SD of a single session and 2 times the noise SD of the average
session_difference_sd = (response_train_valid - response_test_ses_valid).std(axis=1)

print("Median noise SD (single session):", np.median(session_difference_sd / np.sqrt(2.0)))
print("Median noise SD (average):", np.median(session_difference_sd / 2.0))
```

Yes. The averaged response is smoother: The peaks for the bar passes stay, but the fluctuations between them become smaller. Averaging two sessions with independent noise reduces the noise SD by a factor of $\sqrt{2}$ (from about 62 to 44 units for the median vertex).

## Exercise 16

Both runs show the identical stimulus with the same timing, and the responses of both sessions are given for the same vertices of the same surface. Thus, the stimulus-driven part of the response is the same in both runs at every time frame, whereas the noise is independent between the sessions. Averaging keeps the stimulus-driven response and reduces the noise (for independent noise with equal variance, the noise variance is halved). This assumes that the pRFs of the subject do not change between the sessions and that the data of both sessions are well aligned to the same anatomy.

## Exercise 17

The second run of the first session shows a different sequence of bar positions. At the same time frame, the bar is at a different position than in the first run, so the stimulus-driven responses of the two runs differ. Averaging them would mix the responses to two different stimuli into a timecourse that does not belong to any stimulus, and the pRF model could not predict it with either design matrix. To use both runs, we can instead fit the model to both runs at the same time, for example, by concatenating their design matrices and responses along the time dimension.

## Exercise 18

Both runs have the same stimulus, so we use the training stimulus.

```python
from prfmodel.plotting import plot_r_squared_comparison

_, grid_params_avg = GridFitter(model=prf_model, stimulus=stimulus_train, compile_step=True).fit(
    data=response_avg,
    parameter_values=grid_param_ranges,
    batch_size=20,
)

_, ls_params_avg = LeastSquaresFitter(model=prf_model, stimulus=stimulus_train).fit(
    data=response_avg,
    parameters=grid_params_avg,
    slope_name="amplitude",
    intercept_name="baseline",
    batch_size=200,
)

pred_response_avg = np.asarray(predict_batched(stimulus_train, ls_params_avg, batch_size=200))
r_squared_avg = calculate_r_squared(response_avg, pred_response_avg)

plot_r_squared_comparison(
    r_squared_train,
    r_squared_avg,
    value_range=(0.0, 1.0),
    xlabel="In-sample R-squared (first session)",
    ylabel="In-sample R-squared (average)",
)

print("Mean in-sample R-squared (first session):", r_squared_train.mean())
print("Mean in-sample R-squared (average):", r_squared_avg.mean())
```

The in-sample R-squared is higher for the averaged responses (a mean of about 0.29 vs. 0.22), and it increases for most vertices. The stimulus-driven response is the same in both runs, but the noise is reduced, so the model can explain a larger fraction of the variance. Note that this in-sample R-squared is still optimistic because the model is partly fitted to the remaining noise.

## Exercise 19

We define a function that plots the polar angle, eccentricity, and size of the pRFs of a subset of the vertices, so that we can reuse it in the next exercise.

```python
from prfmodel.utils import calculate_eccentricity, calculate_polar_angle


def plot_prf_maps(params: pd.DataFrame, is_shown: np.ndarray, label: str) -> None:
    """Plot the polar angle, eccentricity, and size of the pRFs of the shown vertices on the surface."""
    mu_x = np.where(is_shown, params["mu_x"], np.nan)
    mu_y = np.where(is_shown, params["mu_y"], np.nan)
    sigma = np.where(is_shown, params["sigma"], np.nan)

    plot_surf_stat_map_helper(
        fill_valid_vertices(calculate_polar_angle(mu_x, mu_y)), cyclic=True, title=f"Polar angle ({label})",
    )
    plot_surf_stat_map_helper(
        fill_valid_vertices(calculate_eccentricity(mu_x, mu_y)), vmin=0.0, vmax=8.0, title=f"Eccentricity ({label})",
    )
    plot_surf_stat_map_helper(fill_valid_vertices(sigma), vmin=0.0, vmax=4.0, title=f"pRF size ({label})")


r_squared_threshold = 0.3
is_above_threshold = r_squared_avg > r_squared_threshold

print("Fraction of vertices above the threshold:", is_above_threshold.mean())

plot_prf_maps(ls_params_avg, is_above_threshold, f"in-sample R-squared > {r_squared_threshold}")
```

Yes, the maps show the same organization as in the first practical:

- **Polar angle:** The left hemisphere represents the right visual field, so the polar angles lie between $-\pi/2$ and $\pi/2$. The upper visual field (positive angles) is represented below the calcarine sulcus and the lower visual field (negative angles) above it.
- **Eccentricity:** The eccentricity increases from the occipital pole (center of the visual field) towards the front of the medial surface (periphery). Many vertices at the front of the ROIs have an eccentricity at or beyond the edge of the stimulus (8 degrees).
- **pRF size:** The pRF size increases with eccentricity.

## Exercise 20

We use the test R-squared of the model fitted to the training run (`r_squared_test_run`, see Exercise 13) to select the vertices, but still plot the parameters estimated from the averaged responses.

```python
is_above_test_threshold = r_squared_test_run > r_squared_threshold

print("Fraction of vertices above the threshold:", is_above_test_threshold.mean())
print("Selected by both thresholds:", np.mean(is_above_threshold & is_above_test_threshold))
print("Selected only by the in-sample threshold:", np.mean(is_above_threshold & ~is_above_test_threshold))
print("Selected only by the out-of-sample threshold:", np.mean(~is_above_threshold & is_above_test_threshold))

plot_prf_maps(ls_params_avg, is_above_test_threshold, f"test R-squared > {r_squared_threshold}")
```

The maps show the same organization, but fewer vertices pass the threshold (about 31% instead of 48%). Almost all vertices selected by the out-of-sample threshold are also selected by the in-sample threshold, but a third of the vertices selected by the in-sample threshold do not generalize well enough to the new stimulus. Many of them lie near the occipital pole and have small pRFs close to the center of the visual field, so there are more gaps in this region of the maps. Their in-sample R-squared is partly due to fitted noise, or their small pRFs are sensitive to small differences between runs, for example, in fixation. With the out-of-sample threshold, we only interpret the parameters of vertices for which the model makes reliable predictions for new data.

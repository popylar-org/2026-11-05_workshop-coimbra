## Exercise 1

```python
import prfmodel
import tensorflow as tf

print("prfmodel version:", prfmodel.__version__)
print("TensorFlow built with CUDA:", tf.test.is_built_with_cuda())
print("Available GPUs:", tf.config.list_physical_devices("GPU"))
```

On a Colab GPU instance, TensorFlow is built with CUDA and the list contains at least one GPU, for example, `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`. If the list is empty, TensorFlow runs on the CPU. The notebook still works, but fitting the models takes much longer. On Colab, you can switch to a GPU instance under *Runtime > Change runtime type*.

## Exercise 2

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

## Exercise 3

Every frame of the video shows one time frame of the design matrix (its first dimension), and the image shows the cells of the visual field (its second and third dimension: y and x). The axes of the video show the grid coordinates, which range from about -4 to 4 degrees of visual angle in both directions. Bright cells are covered by the bar. The bar sweeps across the visual field in four directions (top to bottom, left to right, bottom to top, right to left), with blank periods in between. After a long blank period in the middle of the run, the four sweeps are repeated in the reverse order.

## Exercise 4

A neuron population only responds when the bar covers the part of the visual field that it is sensitive to (its pRF). We therefore expect a short burst of activity every time the bar passes over the pRF, that is, one burst per sweep, and no activity during the blank periods. The timing of the bursts within the sweeps depends on the location of the pRF. For example, a pRF in the upper visual field is crossed early in the top-to-bottom sweep and late in the bottom-to-top sweep. The duration of the bursts depends on the size of the pRF: A large pRF overlaps with the bar for longer than a small one.

## Exercise 5

`mu_x` and `mu_y` move the peak of the tuning profile to the given location in the visual field. That is to the upper right for the first combination and to the lower left for the second. `sigma` changes the spread of the profile: With `sigma=2.0`, the profile is broad and parts of it extend beyond the area covered by the stimulus while it is narrow with `sigma=0.7`.

The height of the peak also changes. It is much lower for `sigma=2.0` than for `sigma=0.7` (note the colorbar). The tuning profile is a Gaussian density, which always integrates to one. A broader profile spreads the same total over a larger area, so its peak (which is proportional to $1 / \sigma^2$) must be lower.

## Exercise 6

The predicted activity at a time frame is the overlap between the bar and the tuning profile, that is, the sum of the tuning profile over all cells covered by the bar. The pRF in the center of the visual field is crossed by every sweep, so there is one peak per sweep (eight in total). The activity rises when the bar approaches the pRF, reaches its maximum when the bar covers the center of the pRF, and falls when the bar moves away. Thus, each peak has the shape of the Gaussian profile, smoothed by the width of the bar. During the blank periods (including the long one in the middle of the run), nothing overlaps with the pRF and the activity is zero.

## Exercise 7

Moving the pRF center shifts the peaks in time because the bar crosses the pRF earlier or later in each sweep. For the pRF in the upper right (`mu_x=2.1`, `mu_y=1.4`), the peaks come early in the top-to-bottom and right-to-left sweeps and late in the bottom-to-top and left-to-right sweeps. For the pRF in the lower left, it is the other way around.

The size changes the width and height of the peaks. With `sigma=2.0`, the bar overlaps with the pRF for a longer time, so the peaks are broader. They are also lower because the bar only covers a small part of the broad profile at any time frame and because part of the profile lies outside the stimulus area. Some peaks are cut off abruptly because the sweep ends while the bar still overlaps with the pRF. With `sigma=0.7`, the peaks are narrow and high because the bar covers almost the entire profile when it passes over the center.

## Exercise 8

- **Very small `sigma`:** The pRF is much narrower than the bar, so the bar either covers the entire pRF or nothing of it. The peaks become box-shaped with a flat top: The activity jumps up when the bar reaches the pRF center, stays constant while the bar covers it, and drops to zero when the bar has passed. The peak duration then only depends on the width and speed of the bar.
- **Center at the edge of the stimulus:** Only part of the profile lies inside the area that the bar can cover, so the peaks are lower. Sweeps that move towards the edge reach the pRF at their very end, and sweeps that move away from the edge reach it at their very beginning, so these peaks are cut off.
- **Large `sigma`:** The peaks become broad and low and overlap between consecutive sweeps.

In all cases, the reason is the same: The activity is the overlap between stimulus and tuning profile. Parts of the profile outside the stimulus area never contribute to the activity, and the profile cannot resolve details that are smaller than the bar.

## Exercise 9

First, note how the convolution changes the predicted activity in general: The convolved response is delayed by several seconds, smoother, and dips below zero after each peak (the undershoot), just like the impulse response.

Subtracting a weighted derivative from a function approximately shifts it in time: $g(t) - w \, g'(t) \approx g(t - w)$. Thus, a negative `weight_deriv` moves the peak of the impulse response to earlier time frames and a positive `weight_deriv` to later ones. The convolved response is shifted in the same direction, so `weight_deriv` adjusts the delay of the predicted BOLD response, which differs between brain regions and subjects. For large absolute values, the approximation breaks down, and the shape of the impulse response (e.g., the height of its peak and the depth of its undershoot) changes as well.

## Exercise 10

No, at least not with the impulse response of the BOLD signal. The impulse response models the slow hemodynamic response that follows neural activity. ECoG measures the electrical activity of the neuron population directly, and the broadband power follows the activity almost instantly. Thus, the predicted neuron population activity can be scaled and compared to the measurements directly. In prfmodel, we can create such a model with `Gaussian2DPRFModel(impulse_model=None)`, which is what we do in the practical on ECoG data.

## Exercise 11

The scaling model computes `amplitude * response + baseline`. `baseline` shifts the entire predicted response up or down, including the blank periods. `amplitude` stretches or compresses the response around zero: The peaks and the undershoots become larger or smaller, but the timing and the shape of the response stay the same. With `amplitude=0`, the prediction is a flat line at `baseline`. These two parameters allow the model to match the arbitrary scale and offset of the observed BOLD response of each voxel without changing what the model says about the pRF.

## Exercise 12

Both parameters change the scale of the predicted response. Increasing `sigma` lowers the peaks (see Exercise 7), and increasing `amplitude` raises them. Thus, we can increase `sigma` and compensate the lower peaks with a larger `amplitude` (or vice versa) and get a predicted response with a similar scale. Only the shape of the response, for example, the width of the peaks, still tells the two apart. This trade-off makes it harder to estimate `sigma` and `amplitude` independently from noisy data because different combinations of the two can fit the data almost equally well. See the [Important Details](https://popylar-org.github.io/prfmodel/important_details.html) section in the package documentation for details.

## Exercise 13

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

## Exercise 14

- Most rows only show noise around zero: Most voxels in the gray matter mask do not respond to the stimulus.
- Some rows are bright during the bar sweeps (frames 30–150 and 186–307) and dark at the start of the run and during the long blank period in the middle (frames 151–185). These are voxels in the visual cortex that respond to the bar. The rows are ordered by the position of the voxels in the volume, so neighboring rows often contain neighboring voxels. Neighboring voxels have similar pRFs and respond at similar times, so the responding voxels form bands of several rows.
- Some rows are bright or dark during the entire run. These voxels have an offset from zero that is unrelated to the stimulus.

## Exercise 15

- **Location of the pRF:** Visual cortex is organized retinotopically: Neighboring voxels represent neighboring parts of the visual field. The bar crosses the pRFs of different voxels at different times, so their responses peak at different times.
- **Size of the pRF:** Voxels with larger pRFs respond for longer and with broader peaks.
- **Responsiveness:** The mask contains the gray matter of the entire brain. Most voxels lie outside the visual cortex or represent parts of the visual field that the stimulus does not cover, so they do not respond to the bar at all and only contain noise.
- **Measurement:** The amplitude, delay, and noise level of the BOLD response differ between voxels, for example, because of differences in the vasculature (voxels close to large veins have large and delayed responses), the fraction of gray matter in the voxel, and the distance to the receiver coil.

## Exercise 16

The standard deviation measures how much a timecourse varies over time, whatever the cause. In voxels that respond to the stimulus, part of this variability is the stimulus-driven response, so the standard deviation is elevated in the occipital lobe. However, noise from breathing, the heartbeat, head motion, or large veins also contributes to the variability. Therefore, some regions far from the visual cortex have an even larger standard deviation, for example, the bright spots next to the front end of the medial wall (the gray area in the medial views) and at the lower edge of the surface. A large standard deviation alone does not tell us that the neurons in a region respond to the stimulus.

## Exercise 17

```python
# Average of each timecourse (after subtracting the baseline of 100)
response_mean = response_no_baseline.mean(axis=1)

plot_surf_stat_map_helper(
    project_to_surface(response_mean), vmin=-2.0, vmax=2.0, cmap="RdBu_r", title="Average BOLD response",
);
```

Because we subtracted the baseline, the average is close to zero for most of the surface. It is positive (red) in patches of the occipital lobe, especially on the posterior surface. The voxels there respond to the bar with peaks above baseline, which raises their average. A few patches have a negative average (blue), that is, their response lies below baseline for long parts of the run. This can be caused, for example, by slow drifts of the signal or by suppression of activity during the stimulation. The average is a less clear indicator of stimulus-driven activity than the standard deviation, because a response with peaks and undershoots partly averages out.

## Exercise 18

You can reach an R-squared of about 0.85, for example, with `mu_x` = 0.7, `mu_y` = 2.3, `sigma` = 0.9, `weight_deriv` = -0.5, `baseline` = 0.1, and `amplitude` = 0.11. With `weight_deriv` = 0, the R-squared increases to about 0.9. For comparison, the grid search and least-squares below reach an R-squared of 0.84 for this voxel.

- **Easy:** The pRF center determines *when* the peaks occur. The horizontal bar (top to bottom and bottom to top) crosses the pRF at a time that depends on `mu_y`, and the vertical bar at a time that depends on `mu_x`. Because each bar orientation moves in both directions, moving the center shifts the peaks of the two opposite sweeps in opposite directions in time. `baseline` and `amplitude` are also easy because they only shift and scale the prediction.
- **Hard:** `sigma` and `amplitude` trade off against each other (Exercise 12): A larger `sigma` makes the peaks broader and lower, which a larger `amplitude` can compensate. Only the width of the peaks distinguishes the two, and the noise makes the width hard to judge. `weight_deriv` shifts all peaks slightly in the same direction in time and changes their shape, which is easy to confuse with small changes of the center and size.

## Exercise 19

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

## Exercise 20

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

## Exercise 21

```python
r_squared_surf = project_to_surface(fill_valid_voxels(r_squared))

plot_surf_stat_map_helper(r_squared_surf, vmin=0.0, vmax=0.8, title="R-squared");
```

The model fits the data well only in the occipital lobe, i.e., around the occipital pole and the calcarine sulcus on the medial side (V1), and in the surrounding regions on the dorsal, ventral, and lateral surface of the occipital lobe (V2, V3, and higher visual areas). There, the R-squared reaches values of 0.6 to 0.9. A few small patches with a lower R-squared lie further up in the parietal lobe. In the rest of the brain, the R-squared is close to zero because these regions do not respond to the bar stimulus (or not in a way that the pRF model can describe).

## Exercise 22

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

## Exercise 23

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

## Exercise 24

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

## Exercise 25

```python
from prfmodel.utils import calculate_polar_angle

# Project mu_x and mu_y onto the surface separately and compute the polar angle afterwards
mu_x_surf = project_selected(mu_x)
mu_y_surf = project_selected(mu_y)

polar_angle_surf = calculate_polar_angle(mu_x_surf, mu_y_surf)

plot_surf_stat_map_helper(polar_angle_surf, cyclic=True, title="Polar angle");
```

Each hemisphere represents the opposite (contralateral) half of the visual field: The left hemisphere has polar angles between $-\pi/2$ and $\pi/2$ (right visual field; green, cyan, and blue), and the right hemisphere has angles beyond $\pm\pi/2$ (left visual field; yellow, red, and magenta). Within each hemisphere, the upper visual field (positive angles) is represented on the ventral side, below the calcarine sulcus, and the lower visual field (negative angles) on the dorsal side, above it. Moving away from the calcarine sulcus, the polar angle changes until it reaches the vertical meridian and then reverses its direction at the border between V1 and V2, and again at the border between V2 and V3. In the posterior view, these reversals show up as alternating bands of color.

## Exercise 26

```python
from prfmodel.utils import calculate_eccentricity

# Use the projections of mu_x and mu_y from Exercise 25
eccentricity_surf = calculate_eccentricity(mu_x_surf, mu_y_surf)

plot_surf_stat_map_helper(eccentricity_surf, vmin=0.0, vmax=5.0, title="Eccentricity");
```

The eccentricity is lowest (center of the visual field) at the occipital pole and on the posterior surface close to it. It increases with the distance from the pole: forward along the calcarine sulcus in the medial views, and outward on the posterior surface. The stimulus only covers the central 4 degrees (up to about 5.7 degrees in the corners), so we only see the central part of the eccentricity map: Voxels that represent the periphery of the visual field further forward in the calcarine sulcus are not stimulated and therefore not selected.

## Exercise 27

Dumoulin and Wandell (2008) found the classical retinotopic organization of the early visual cortex: Each hemisphere represents the contralateral visual field. In V1, the upper visual field is represented below and the lower visual field above the calcarine sulcus, and the polar angle reverses at the vertical and horizontal meridians, which form the borders between V1, V2, and V3. The eccentricity increases from the occipital pole forward.

Our polar angle and eccentricity maps agree with these patterns, although our stimulus only covers the central 4 degrees of the visual field, so we can only compare the central part of the maps.

Dumoulin and Wandell (2008) also found that the pRF size increases with eccentricity within each visual area and from V1 to the higher visual areas. Our size map shows larger pRFs in the higher visual areas (Exercise 24). However, among the selected voxels, the size hardly changes with eccentricity (the mean `sigma` is about 1.7 degrees in all eccentricity bins). This is a consequence of our fitting procedure rather than of the data: The grid only contains 10 sizes, least-squares does not change them, and we excluded the voxels with a size at the ceiling of the grid (4 degrees), which are almost half of the well-fitting voxels. With a grid that extends `sigma` to 10 degrees (Exercise 19), the size increases with eccentricity.

## Exercise 28

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

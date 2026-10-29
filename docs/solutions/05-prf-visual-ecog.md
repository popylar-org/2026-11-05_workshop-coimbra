## Exercise 1

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

## Exercise 2

An ECoG electrode only records the activity of the neuron population in the cortex directly below it. A channel only responds to the bar if this population responds to visual stimulation in the part of the visual field covered by the stimulus. Channels may not respond because:

- The electrode does not lie over visual cortex. Most of the standard grids and strips lie further to the front and top of the brain, and many channels are labeled `none` in the atlas.
- The pRF of the neuron population lies outside the stimulated part of the visual field (beyond about 8 degrees), or the population prefers other stimuli than a contrast pattern in a moving bar (e.g., objects, faces, or motion in higher visual areas).
- The recording is noisy, for example, because the electrode has poor contact with the cortex.

## Exercise 3

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

## Exercise 4

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

## Exercise 5

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

## Exercise 6

```python
plot_electrodes(r_squared_sgd, vmin=0.0, vmax=1.0, title="Variance explained (R-squared)");
```

The model fits best for the channels of the high-density grid at the back of the lateral occipital cortex, which lie in the visual areas V3, V3b, and LO1 (and V2). Here, the model explains up to 90% of the variance. The fit becomes worse for the more anterior channels of the high-density grid (e.g., in TO and IPS), and the model hardly explains any variance for the standard grids and strips outside of the visual cortex.

## Exercise 7

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

## Exercise 8

Yes, the results show the same organization of the visual cortex as in the first practical and in Dumoulin and Wandell (2008):

- **Polar angle:** Almost all selected pRFs lie in the left visual field, contralateral to the right hemisphere with the electrodes. Most of them lie in the lower left quadrant because the electrodes cover dorsal visual areas, which represent the lower visual field.
- **Eccentricity:** Most selected pRFs lie within 4 degrees of the center of the visual field. The high-density grid lies on the lateral occipital cortex close to the occipital pole, where the central visual field is represented (the foveal confluence of V1–V3).
- **pRF size:** The pRF size increases with eccentricity and from early to higher visual areas, from V2 over V3 to V3b, LO1, and IPS. Dumoulin and Wandell (2008) reported the same two trends, and we saw them on the surface in the first practical.

The main difference to fMRI is the coverage: ECoG only measures at a few positions on the cortex. Thus, we cannot see complete maps of polar angle and eccentricity, only the small part of the visual field that the pRFs of the channels cover.

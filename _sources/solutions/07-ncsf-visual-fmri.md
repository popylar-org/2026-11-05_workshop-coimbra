## Exercise 1

```python
print("SFs:", np.unique(stimulus.sf[~is_blank]))
print("Contrasts:", np.unique(stimulus.contrast[~is_blank]).round(2))

# Number of time frames and contrast levels per SF
for sf in sf_levels:
    is_sf = (stimulus.sf == sf) & ~is_blank
    print(f"{sf:g} cycles/degree: {is_sf.sum()} time frames, {len(np.unique(stimulus.contrast[is_sf]))} contrast levels")

# Number of blank time frames
print("Blank time frames:", is_blank.sum())
```

The experiment has 214 time frames (321 seconds). It contains six blocks with gratings of 24 time frames (36 seconds) each. The blocks are separated by blank periods of 10 time frames (15 seconds), and the experiment starts and ends with a blank period (70 blank time frames in total). Each block consists of two sweeps through the contrasts of 12 time frames each: In the first sweep, the contrast steps down from 80% to 0.25% at one SF. In the second sweep, the contrast steps back up from 0.25% to 80% at another SF. The blocks pair the SFs as follows: 3 and 18, 1 and 6, 0.5 and 12, 6 and 1, 12 and 3, and 18 and 0.5 cycles per degree.

Each of the six SFs (0.5, 1, 3, 6, 12, and 18 cycles per degree) is shown in two blocks, once with decreasing and once with increasing contrast, so 24 time frames per SF. There are 14 contrast levels in total, but each SF is shown at only 12 of them: The low SFs skip the high contrasts of 21% and 51% and instead include the low contrasts of 0.39% and 0.95%, and the high SFs do the opposite. This puts more contrast levels where the threshold is expected to lie: Sensitivity is usually higher for low SFs than for high SFs.

## Exercise 2

`cs_peak` moves the whole CSF up or down, `sf_peak` moves it left or right, and `width_r` determines how steeply the sensitivity falls off for SFs above the peak (larger values give a narrower CSF). The left branch does not change with `width_r` because `width_l` is fixed.

- `cs_peak=300`, `sf_peak=1.0`, `width_r=0.7`: A high and broad CSF with its peak at a low SF. The sensitivity at all SFs in the stimulus lies above 50 (i.e., the threshold lies below 2% contrast), so the population responds to 122 of the 144 time frames with a grating. Only the lowest contrasts are below its threshold.
- `cs_peak=20`, `sf_peak=6.0`, `width_r=1.5`: A low and narrow CSF with its peak at a high SF. Even at the peak, the threshold is 5% contrast, and at 0.5 and 18 cycles per degree it is above 15%. Only 52 time frames have a contrast above the threshold, mostly gratings with a high contrast at intermediate SFs (3 to 12 cycles per degree).

## Exercise 3

The left branch of the CSF is only constrained by the SFs in the stimulus that lie below the peak. The lowest SF in the stimulus is 0.5 cycles per degree. With `sf_peak` around 1 cycle per degree, only the gratings with 0.5 cycles per degree lie on the left branch, and for a peak at or below 0.5 cycles per degree, none of them do. Thus, the data contain little or no information about `width_l` for many neuron populations. If we estimated it anyway, it would trade off with `sf_peak` and `cs_peak` (e.g., a higher peak with a steeper left branch can predict the same sensitivity at 0.5 cycles per degree), and the estimates of all three parameters would become unstable. Fixing `width_l` removes this trade-off.

## Exercise 4

`slope_crf` determines how steeply the predicted activity rises with the contrast around the threshold. It does not change the thresholds: All curves still reach half of their maximum at the same contrast.

- With a small `slope_crf` (e.g., 0.5), the activity rises slowly over a wide range of contrasts. Even contrasts far below the threshold drive some activity, and contrasts far above it do not drive the maximum activity.
- With a large `slope_crf` (e.g., 10), the CRF becomes a step function: The activity is close to zero for all contrasts below the threshold and close to one for all contrasts above it.

With a large `slope_crf`, the neuron population acts as a threshold detector: It signals whether a grating is visible to it or not, but not how far its contrast lies above the threshold.

## Exercise 5

The predicted activity is zero during the blank periods because there is no contrast. In the first sweep of a block, the contrast decreases. As long as the contrast lies above the threshold at the SF of the sweep, the activity is high, and it falls when the contrast drops below the threshold. In the second sweep, the contrast increases at another SF and the activity rises again when the contrast passes the threshold at that SF. Thus, every block has a valley in the middle, where the contrast is lowest. The width of the valley depends on the sensitivity at the SFs of the block: For SFs close to the peak of the CSF, the threshold is low, so the activity stays high for more time frames. For SFs far from the peak, the threshold is high, so the activity drops early in the first sweep (or rises late in the second sweep), or the population does not respond at all.

- With `sf_peak=0.5`, the population is most sensitive to the lowest SFs. The sweeps at 0.5 and 1 cycles per degree drive the most activity, whereas the sweeps at 12 and 18 cycles per degree drive almost none, because the threshold lies above the highest contrast of 80%.
- With `sf_peak=12`, it is the other way around: The sweeps at high SFs drive the most activity, and the sweeps at 0.5 and 1 cycles per degree only drive activity at their highest contrasts.

## Exercise 6

```python
# The starting CSF and two CSFs with clearly different peak SFs (sf_peak, width_r)
example_csfs = {"Starting CSF": (2.0, 1.0), "Lower peak SF": (1.25, 0.8), "Higher peak SF": (2.8, 1.2)}

fig, (ax_csf, ax_activity) = plt.subplots(
    1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 2.2]}, layout="constrained",
)

example_activity = {}
for label, (example_sf_peak, example_width_r) in example_csfs.items():
    params = to_frame(cs_peak=50.0, sf_peak=example_sf_peak, width_r=example_width_r, slope_crf=2.0)
    example_activity[label] = np.asarray(csf_tuning(stimulus, params))[0]
    ax_csf.plot(sf_values, predict_sensitivity(sf_values, 50.0, example_sf_peak, example_width_r), label=label)
    ax_activity.plot(time_frames, example_activity[label], label=label)
    print(label, "- sensitivity at the SFs of the stimulus:", predict_sensitivity(sf_levels, 50.0, example_sf_peak, example_width_r).round(1))

for sf in sf_levels:
    ax_csf.axvline(sf, color=GRAY, linewidth=0.8, linestyle=":")  # SFs in the stimulus
ax_csf.set(xscale="log", yscale="log", ylim=(1.0, 100.0))
ax_csf.legend(frameon=False, fontsize=8)
style_axes(ax_csf, "Spatial frequency (in cycles/degree)", "Contrast sensitivity")
style_axes(ax_activity, "Time frame (in TR)", "Predicted activity")

for label in ["Lower peak SF", "Higher peak SF"]:
    correlation = np.corrcoef(example_activity["Starting CSF"], example_activity[label])[0, 1]
    print(f"Correlation between the activity of the starting CSF and the CSF with a {label.lower()}: {correlation:.3f}")
```

For example, the starting CSF (`sf_peak=2.0`, `width_r=1.0`), a CSF with a lower peak SF and a broader right branch (`sf_peak=1.25`, `width_r=0.8`), and a CSF with a higher peak SF and a narrower right branch (`sf_peak=2.8`, `width_r=1.2`) predict almost the same activity (correlations above 0.99), although their peak SFs differ by more than a factor of two. The predicted activity only depends on the sensitivity at the six SFs in the stimulus, and these CSFs have similar sensitivities at these SFs. Between and beyond these SFs, the CSFs can differ without changing the prediction. In addition, the contrast levels are spaced coarsely, and a smooth CRF changes the activity only a little when the threshold shifts slightly.

This means that `sf_peak` and `width_r` (and to a lesser degree `cs_peak`) trade off against each other: Different combinations can fit the data almost equally well, especially when the data are noisy. The estimates of the individual parameters are therefore less certain than the shape of the CSF at the SFs in the stimulus. Measures that summarize the whole CSF can be more robust than individual parameters, and a stimulus with more SFs would constrain the CSF better.

## Exercise 7

The impulse response delays the predicted response by several seconds and smooths it. The valley in the middle of each block becomes shallower, so the responses to the two sweeps of a block partly merge, and the response dips below zero after each block (the undershoot of the impulse response).

Because the BOLD response is slow, the response at a time frame also reflects the stimulus at previous time frames. The design deals with this by placing the highest contrasts, which drive the strongest activity, at the start and end of each block and the lowest contrasts in the middle. This way, the responses to the high contrasts of the two SFs in a block are as far apart in time as possible, and the response to the first SF has declined before the response to the second SF rises. If the contrast increased in the first sweep and decreased in the second, the strongest responses to both SFs would follow each other directly and could not be separated. In addition, every SF is shown once with decreasing and once with increasing contrast, which balances effects of the presentation order, such as adaptation.

## Exercise 8

Many of the vertices with the most variable responses lie in the occipital lobe inside the ROIs, especially at the occipital pole and on the lateral and dorsal side of the posterior view. These are the vertices that respond to the gratings. However, not all parts of the ROIs have a variable response: The front (anterior) part of V1 and V2 along the calcarine sulcus in the medial views varies about as little as the rest of the brain. Some vertices outside the ROIs also have a variable response, either because they belong to visual areas that the atlas does not include or because of noise (e.g., from large veins).

V1 is organized retinotopically: The occipital pole represents the center of the visual field, and the anterior part of the calcarine sulcus represents the periphery. The gratings only covered the central part of the visual field (up to about 5 degrees from the center), so the vertices that represent the periphery do not see the stimulus and do not respond to it (see also Exercise 13).

## Exercise 9

```python
from prfmodel.plotting import plot_response_heatmap

fig, ax = plot_response_heatmap(
    response_valid,
    vmin=-2,
    vmax=2,
    xlabel="Time frame (in TR)",
    ylabel="Valid vertex index",
    colorbar_label="BOLD response (in PSC)",
    figsize=(10, 6),
)

# Mark the start and the end of each block
for first_sweep, second_sweep in zip(sweeps[::2], sweeps[1::2]):
    ax.axvline(first_sweep[0] - 0.5, color="white", linestyle="--", linewidth=1)
    ax.axvline(second_sweep[1] + 0.5, color="white", linestyle="--", linewidth=1)

# Average response of the 1,000 valid vertices with the most variable responses
most_variable = np.argsort(response_valid.std(axis=1))[-1000:]

fig, ax = plt.subplots(figsize=(12, 3.5), layout="constrained")
mark_sweeps(ax)
ax.plot(time_frames, response_valid[most_variable].mean(axis=0), color=BLUE, linewidth=2)
ax.set_title("Average response of the 1,000 most variable valid vertices", loc="left", pad=14)
style_axes(ax, "Time frame (in TR)", "BOLD response (in PSC)");
```

Many of the valid vertices respond to the stimulus with the same pattern in every block, which shows up as vertical stripes in the heatmap. The average response of the most variable vertices shows this pattern more clearly:

- The response rises shortly after the start of each block, when the first sweep shows the highest contrasts, and falls in the middle of the block, when the contrast is lowest.
- The response rises again towards the end of each block, when the second sweep shows the highest contrasts. Because the BOLD response is delayed, this peak lies just after the end of the block.
- During the blank periods, the response falls below zero. Because the timecourses are centered around their mean, the baseline lies below zero.
- The height of the peaks depends on the SF of the sweep: The peaks for SFs of 1 and 3 cycles per degree are high, and the peaks for 18 cycles per degree are low (e.g., the first peak of the last block).

Not all valid vertices respond, though: Several groups of rows in the heatmap do not show the pattern. Many of these vertices represent parts of the visual field that the screen does not cover (see Exercises 8 and 13).

## Exercise 10

For vertex 4635 in V1, the fits of prfmodel reach an R-squared of about 0.59 with `cs_peak` ≈ 70, `sf_peak` ≈ 2.6 cycles per degree, `width_r` ≈ 1.9, `slope_crf` ≈ 1.1, `baseline` ≈ -0.45, and `amplitude` ≈ 1.6. You may reach a similar R-squared with quite different values for some of the parameters.

`baseline` and `amplitude` are easy to find because they shift and scale the entire prediction. `sf_peak` is also relatively easy to find because it determines which sweeps drive the largest responses. `cs_peak`, `width_r`, and `slope_crf` are harder to find: They change the prediction in similar ways, for example, both `cs_peak` and `slope_crf` determine at which contrast the response within a sweep drops, and changes in one parameter can be compensated by changes in the others (see Exercise 6). In addition, the observed response rises a bit earlier than the prediction, which none of the CSF parameters can fix and which limits the R-squared that we can reach.

## Exercise 11

```python
# The grid search loss is the negative correlation between the prediction and the response
grid_correlation = -np.asarray(grid_history.history["loss"])
is_responsive = grid_correlation > 0.5

# Fraction of vertices with an estimate at the smallest or largest grid value
edge_fraction = {}
for name in csf_param_names:
    estimates = grid_params[name].to_numpy()
    grid_values = grid_param_ranges[name]
    is_at_edge = np.isclose(estimates, np.min(grid_values)) | np.isclose(estimates, np.max(grid_values))
    edge_fraction[name] = {
        "correlation > 0.5": is_at_edge[is_responsive].mean(),
        "correlation <= 0.5": is_at_edge[~is_responsive].mean(),
    }

print(f"{is_responsive.sum()} of {len(is_responsive)} vertices have a correlation above 0.5")
pd.DataFrame(edge_fraction).round(2)
```

About 5,500 of the 12,388 valid vertices have a correlation above 0.5 between the best grid prediction and their response. For these vertices, the estimates rarely lie at the edge of the grid. The exception is `width_r`: About a fifth of them have the largest value (2.0), which suggests that their CSF falls off even more steeply at high SFs. SGD can move these estimates beyond the grid, up to the upper bound of 3.

For the other vertices, most estimates of `cs_peak`, `width_r`, and `slope_crf` lie at the edge of the grid: the smallest peak sensitivity, the narrowest CSF, and the steepest CRF. Many vertices in the ROIs do not respond to the stimulus, and their response only contains noise. The combination of a low and narrow CSF with a steep CRF predicts a response to only a few time frames with a high contrast at SFs close to the peak. Such a sparse prediction can match a few random peaks in the noise and therefore reaches the highest correlation with a noisy response. These estimates do not describe a CSF, so we do not interpret them.

## Exercise 12

```python
from prfmodel.plotting import plot_r_squared_comparison

pred_response_ls = np.asarray(predict_batched(stimulus, ls_params, batch_size=1000))
r_squared_ls = calculate_r_squared(response_valid, pred_response_ls)

fig, ax = plot_r_squared_comparison(
    r_squared_ls,
    r_squared,
    value_range=(0.0, 1.0),
    xlabel="R-squared (least-squares)",
    ylabel="R-squared (SGD)",
)

r_squared_gain = r_squared - r_squared_ls
print(f"Fraction of vertices with a higher R-squared after SGD: {np.mean(r_squared_gain > 0):.2f}")

# Vertices that least-squares already fits reasonably well
is_fitted_ls = r_squared_ls > 0.1
print(
    f"Median R-squared of {is_fitted_ls.sum()} vertices with a least-squares R-squared above 0.1: "
    f"{np.median(r_squared_ls[is_fitted_ls]):.3f} (least-squares), {np.median(r_squared[is_fitted_ls]):.3f} (SGD); "
    f"largest gain: {r_squared_gain[is_fitted_ls].max():.3f}"
)

# Vertices whose R-squared jumps from below 0.1 to above 0.3
is_jump = ~is_fitted_ls & (r_squared > 0.3)
print(
    f"{is_jump.sum()} vertices jump from an R-squared below 0.1 to above 0.3, "
    f"{np.mean(sgd_params['amplitude'].to_numpy()[is_jump] < 0.0):.0%} of them with a negative amplitude"
)
pd.Series(roi_valid[is_jump]).value_counts()
```

SGD improves the R-squared of almost all vertices, but in two different ways. For the vertices that least-squares already fits reasonably well (R-squared above 0.1), the improvement is small: Their median R-squared increases from about 0.52 to 0.53, and no vertex gains more than about 0.03. The grid is dense enough to find estimates close to the best fit, and the prediction changes smoothly with the CSF parameters. Thus, least-squares already gets close to the best fit, and SGD mostly finetunes the parameters. Note that some of these small changes in R-squared can still come with noticeable changes in the parameters (see Exercise 6).

Most of the other vertices stay close to an R-squared of zero because their response only contains noise. However, about 230 vertices, mostly in V1 and V2, jump from an R-squared below 0.1 to above 0.3 (up to about 0.75). All of them have a negative amplitude: Their response decreases when the gratings are shown. The grid search looks for the prediction with the highest correlation with the response. Every CSF predicts more activity for gratings with a higher contrast, so all predictions correlate negatively with such a response, and the grid search picks the prediction that correlates the least negatively. Least-squares then finds a negative amplitude for this prediction but cannot change the CSF. SGD changes all parameters at once and finds a CSF that matches the inverted response. Almost all of these vertices represent the visual field beyond the screen (see Exercise 13). We exclude them from the interpretation with the criterion that the amplitude is positive.

## Exercise 13

```python
from plotting_helpers import ORANGE

plot_surf_stat_map_helper(fill_valid_vertices(r_squared), vmin=0.0, vmax=0.8, title="Variance explained (R-squared)")

eccentricity_template = dataset.units["benson14_eccen"].to_numpy()[response_is_valid]
is_negative = sgd_params["amplitude"].to_numpy() < 0.0

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, layout="constrained")

for ax, name in zip(axes, ["V1", "V2"]):
    is_roi = roi_valid == name
    for is_shown, color, label in [(~is_negative, BLUE, "Positive amplitude"), (is_negative, ORANGE, "Negative amplitude")]:
        ax.scatter(
            eccentricity_template[is_roi & is_shown], r_squared[is_roi & is_shown], s=4, color=color, alpha=0.3, label=label,
        )
    ax.axvline(5.1, color=GRAY, linestyle="--", linewidth=1, label="Edge of the screen")
    ax.set(xscale="log", title=name)
    ax.legend(frameon=False)
    style_axes(ax, "Eccentricity of the template atlas (in degrees)", "R-squared")

for name in ["V1", "V2"]:
    is_roi = roi_valid == name
    is_inside = eccentricity_template[is_roi] < 5.1
    print(
        f"{name}: {is_inside.mean():.0%} of the vertices lie inside the screen; median R-squared "
        f"{np.median(r_squared[is_roi][is_inside]):.2f} inside and {np.median(r_squared[is_roi][~is_inside]):.2f} outside"
    )
```

The model fits the data well at the occipital pole and in the dorsal and lateral visual areas (R-squared up to about 0.8). In large parts of V1 and V2, especially their anterior parts along the calcarine sulcus, the R-squared is close to zero.

The template atlas predicts that only about a quarter of the V1 vertices and about a third of the V2 vertices represent the part of the visual field that the screen covers (up to about 5 degrees from the center). Inside the screen, the median R-squared is about 0.54 in both areas, and outside the screen it is about zero: The vertices that represent the periphery never see a grating, so their response does not contain any information about the CSF. The higher visual areas have larger pRFs and represent the periphery less, so a larger part of them responds to the stimulus. The R-squared is also lower for some vertices with an eccentricity below about 0.5 degrees. Close to the center of the visual field, the visual areas are small and lie close together, so the template atlas is less accurate there.

A few vertices beyond the screen still reach a high R-squared. Many of them have a negative amplitude (see Exercise 12): Their response decreases when the gratings are shown. Such negative responses often occur in the parts of the visual cortex that represent the visual field around a stimulus. The template atlas is not exact either, so some of the vertices that it places beyond the screen may represent a part of the visual field on the screen.

## Exercise 14

```python
from prfmodel.plotting import plot_observed_predicted

r_squared_order = np.argsort(r_squared)

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")

# Vertices at the 99th, 80th, and 60th percentile of the R-squared scores
for ax, percentile, label in zip(axes, [99, 80, 60], ["High", "Medium", "Low"]):
    vertex = r_squared_order[int(percentile / 100 * (len(r_squared_order) - 1))]
    plot_observed_predicted(response_valid[vertex], pred_response[vertex], ax=ax)
    mark_sweeps(ax)
    ax.set_title(
        f"{label}: vertex {vertex} ({roi_valid[vertex]}), R-squared = {r_squared[vertex]:.2f}", loc="left", pad=14,
    )
    ax.set(xlabel="", ylabel="BOLD response (in PSC)")

axes[-1].set_xlabel("Time frame (in TR)");
```

For well fitting vertices, the model captures the main pattern of the response: a peak after the start of each block (the high contrasts of the first sweep), a dip in the middle, a peak after the end of each block (the high contrasts of the second sweep), and different peak heights for different SFs. However, there are a few patterns that the model does not capture:

- **Timing:** The observed response often rises and peaks a bit earlier than the prediction. The impulse response has a fixed shape, and only `weight_deriv` can shift it in time, which we fixed during SGD. Estimating `weight_deriv` or the other parameters of the impulse response (e.g., `delay`) can improve the fit.
- **Responses at the end of a block:** Some vertices, especially in higher visual areas, respond after the end of a block even if the model predicts no response to the SF of the second sweep (e.g., after a sweep at 18 cycles per degree). This could be a response to the disappearance of the grating (an offset response), which the nCSF model does not include.
- **Responses below baseline:** Some vertices respond with a decrease below baseline to some sweeps (e.g., in the block with 0.5 and 12 cycles per degree). The nCSF model can only predict activity that increases with contrast, so it cannot capture such suppression.
- **Noise:** For vertices with a lower R-squared, a large part of the response is noise, which no model should capture.

## Exercise 15

```python
plot_surf_stat_map_helper(
    fill_selected_vertices(np.log10(cs_peak)),
    vmin=1.0,
    vmax=2.6,
    cmap="plasma",
    title="Peak contrast sensitivity (log10)",
);
```

The peak SF is highest (above 3 cycles per degree) at the occipital pole, where V1 to V3 represent the center of the visual field. It decreases towards the periphery within these areas and is lower (around 1 cycle per degree) in the dorsal and lateral visual areas.

The peak contrast sensitivity varies less systematically. Half of the selected vertices have a peak sensitivity between about 65 and 170 (log10 between 1.8 and 2.2), which corresponds to a contrast threshold between about 0.6% and 1.5%. Lower values occur in patches, mostly at the borders of the area that responds to the stimulus and in the ventral visual areas, where the fits are often worse.

## Exercise 16

```python
# Median parameters of the selected vertices in each ROI
sgd_params[is_selected].groupby(roi_valid[is_selected])[csf_param_names].median().loc[selected_rois]
```

The CSFs of the early visual areas peak at higher SFs than those of the higher visual areas: The median peak SF is about 2.7 cycles per degree in V1, about 2 in V2 and V3, and about 1 cycle per degree in the dorsal and lateral areas (V3a, V3b, LO1, LO2, TO1, and TO2). Between about 3 and 10 cycles per degree, V1 has the highest sensitivity of the early visual areas, whereas V2 and V3 are more sensitive to low SFs. Most dorsal and lateral areas have a higher peak sensitivity than the early areas and a larger area under the log CSF, but their sensitivity is highest at low SFs. VO2 has a much lower sensitivity than all other areas, but the model also fits it worse than most other areas.

In this dataset, V3b has the highest sensitivity to SFs above 10 cycles per degree because its CSFs are broad (low `width_r`). However, only 75 vertices of V3b are selected, and its SF cutoff lies beyond the highest SF in the stimulus (18 cycles per degree), so this part of its CSF is an extrapolation of the model and not a measurement.

## Exercise 17

In V1, V2, and V3, the median peak SF is highest close to the center of the visual field (about 2.7 to 3.7 cycles per degree below 1 degree eccentricity) and decreases with eccentricity, especially within the central 2 degrees. Beyond about 2 degrees, the peak SF levels off in our data. The peak SF is also higher in V1 than in V2 and V3 at all eccentricities.

Roelofzen et al. (2025) also report that the peak SF decreases with eccentricity, in line with the decrease of visual acuity from the center to the periphery of the visual field. Our analysis has several limitations, though: The eccentricity comes from a template atlas instead of a measurement in this subject, the stimulus only covers the central 5 degrees, and the selection only includes vertices that fit well. In Exercise 19, we replace the template eccentricity by the eccentricity of pRFs that we estimate from the `prf` task.

## Exercise 18

We load both splits and select the same vertices as for the full dataset. Both splits were measured with the same stimulus as the full dataset.

```python
dataset_odd = load_dataset("7t-ncsf-visual", task="csf", split="odd")
dataset_even = load_dataset("7t-ncsf-visual", task="csf", split="even")

response_odd = dataset_odd.response[response_is_valid]
response_even = dataset_even.response[response_is_valid]

# Mean of each timecourse
print("Odd runs:", np.percentile(response_odd.mean(axis=1), [5, 50, 95]).round(2))
print("Even runs:", np.percentile(response_even.mean(axis=1), [5, 50, 95]).round(2))
```

The odd runs and the even runs were processed differently: The mean of every timecourse is zero for the even runs (like for the average of all runs), but differs between vertices for the odd runs. This offset has nothing to do with the CSF, but it reduces the R-squared on held-out data when the training and test data have a different mean. We therefore remove the mean from the observed and the predicted timecourses before we compute the R-squared on the test data.

We wrap all three fitting stages in a function, so that we can fit both models in the same way.

```python
def fit_ncsf(
    response_train: np.ndarray, grid_values: dict, fixed_parameters: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit the nCSF model with grid search, least-squares, and SGD; return the estimates and the in-sample R-squared."""
    _, params_grid = GridFitter(model=csf_model, stimulus=stimulus, compile_step=True).fit(
        data=response_train, parameter_values=grid_values, batch_size=100,
    )
    _, params_ls = LeastSquaresFitter(model=csf_model, stimulus=stimulus).fit(
        data=response_train, parameters=params_grid, slope_name="amplitude", intercept_name="baseline", batch_size=1000,
    )

    # Keep the bounded parameters that are not fixed between their bounds
    fit_adapter = Adapter([
        box_transform(name, lower, upper)
        for name, (lower, upper) in parameter_bounds.items()
        if name not in fixed_parameters
    ])
    _, params_sgd = SGDFitter(
        model=csf_model, stimulus=stimulus, adapter=fit_adapter, optimizer=Adam(learning_rate=0.01), compile_step=True,
    ).fit(
        data=response_train, init_parameters=params_ls, fixed_parameters=fixed_parameters, num_steps=500,
    )

    r_squared_train = calculate_r_squared(response_train, np.asarray(predict_batched(stimulus, params_sgd, batch_size=1000)))

    return params_sgd, r_squared_train


def calculate_r_squared_without_mean(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Calculate the R-squared after removing the mean of each observed and predicted timecourse."""
    return calculate_r_squared(
        observed - observed.mean(axis=1, keepdims=True), predicted - predicted.mean(axis=1, keepdims=True),
    )
```

First, we fit the model with a free `slope_crf` to the odd runs and evaluate it on the even runs.

```python
params_free, r_squared_train_free = fit_ncsf(response_odd, grid_param_ranges, ["weight_deriv"])

r_squared_test_free = calculate_r_squared_without_mean(
    response_even, np.asarray(predict_batched(stimulus, params_free, batch_size=1000)),
)
```

Then, we fix `slope_crf` at the median of its estimates for the vertices that the first model fits well on the odd runs. We do not use the even runs to choose this value, so that they stay independent from the training data.

```python
slope_crf_fixed = np.median(params_free.loc[r_squared_train_free > 0.3, "slope_crf"])
print(f"slope_crf fixed at {slope_crf_fixed:.2f}")

params_fixed, r_squared_train_fixed = fit_ncsf(
    response_odd, grid_param_ranges | {"slope_crf": [slope_crf_fixed]}, ["weight_deriv", "slope_crf"],
)

r_squared_test_fixed = calculate_r_squared_without_mean(
    response_even, np.asarray(predict_batched(stimulus, params_fixed, batch_size=1000)),
)
```

We compare both models on the vertices that at least one of the models fits well on the training data.

```python
is_compared = (r_squared_train_free > 0.3) | (r_squared_train_fixed > 0.3)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

plot_r_squared_comparison(
    r_squared_train_free[is_compared], r_squared_test_free[is_compared], value_range=(-0.5, 1.0),
    xlabel="R-squared (odd runs)", ylabel="R-squared (even runs)", ax=axes[0],
)
axes[0].set_title("Free slope_crf: training vs. test")

plot_r_squared_comparison(
    r_squared_test_free[is_compared], r_squared_test_fixed[is_compared], value_range=(-0.5, 1.0),
    xlabel="R-squared on even runs (free slope_crf)", ylabel="R-squared on even runs (fixed slope_crf)", ax=axes[1],
)
axes[1].set_title("Free vs. fixed slope_crf on the test data")

pd.DataFrame({
    "free slope_crf": pd.Series(r_squared_test_free[is_compared]).groupby(roi_valid[is_compared]).median(),
    "fixed slope_crf": pd.Series(r_squared_test_fixed[is_compared]).groupby(roi_valid[is_compared]).median(),
    "fraction better with fixed slope_crf": pd.Series(
        r_squared_test_fixed[is_compared] > r_squared_test_free[is_compared]
    ).groupby(roi_valid[is_compared]).mean(),
}).loc[roi_order].round(3)
```

With a free `slope_crf`, the R-squared on the even runs is a bit lower than on the odd runs for most vertices, but the two are closely related: The nCSF model captures a pattern in the response that we can reproduce in independent runs, and not just the noise of the training runs.

Fixing `slope_crf` (here at about 1.2) hardly changes the R-squared on the even runs: In all ROIs, the median R-squared of both models differs by less than 0.01, and the fixed model is better for about half of the vertices in most ROIs (for about two thirds in V1 and TO2, and for about 40% in V3a and V3b). Thus, a single `slope_crf` for all vertices predicts held-out data as well as a separate `slope_crf` for every vertex. Following the [third practical](03-prf-visual-fmri-comparison.md), we would prefer the simpler model with a fixed `slope_crf`: It has one parameter less per vertex, and its other estimates do not trade off against `slope_crf`.

## Exercise 19

```python
from prfmodel.fitters import grid_values_2d_prf
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.utils import calculate_eccentricity

dataset_prf = load_dataset("7t-ncsf-visual", task="prf")
stimulus_prf = dataset_prf.stimulus
response_prf = dataset_prf.response[response_is_valid]  # Same vertices as for the nCSF model

print(stimulus_prf)
```

We fit the canonical Gaussian 2D pRF model with a grid search and least-squares like in the first practical.

```python
prf_model = Gaussian2DPRFModel()

grid_param_ranges_prf = grid_values_2d_prf(
    stimulus_prf,
    num_mu=21,
    mu_extent=2.0,
    num_sigma=10,
    sigma_range=(0.1, 5.0),
) | {
    "weight_deriv": [-0.5],
    "baseline": [0.0],
    "amplitude": [1.0],
}

_, grid_params_prf = GridFitter(model=prf_model, stimulus=stimulus_prf, compile_step=True).fit(
    data=response_prf, parameter_values=grid_param_ranges_prf, batch_size=20,
)
_, ls_params_prf = LeastSquaresFitter(model=prf_model, stimulus=stimulus_prf).fit(
    data=response_prf, parameters=grid_params_prf, slope_name="amplitude", intercept_name="baseline", batch_size=1000,
)

r_squared_prf = calculate_r_squared(
    response_prf, np.asarray(batched(prf_model)(stimulus_prf, ls_params_prf, batch_size=1000)),
)
```

We select vertices with a good fit of both models whose pRF center lies inside the stimulus and relate the peak SF to the eccentricity and size of the pRF.

```python
prf_mu_x = ls_params_prf["mu_x"].to_numpy()
prf_mu_y = ls_params_prf["mu_y"].to_numpy()
prf_sigma = ls_params_prf["sigma"].to_numpy()
prf_eccentricity = calculate_eccentricity(prf_mu_x, prf_mu_y)

grid_min_xy, grid_max_xy = stimulus_prf.grid.min(axis=(0, 1)), stimulus_prf.grid.max(axis=(0, 1))
is_prf_selected = (
    (r_squared_prf > 0.3)
    # Center inside the stimulus (the grid stores y in grid[..., 0] and x in grid[..., 1])
    & (prf_mu_x >= grid_min_xy[1]) & (prf_mu_x <= grid_max_xy[1])
    & (prf_mu_y >= grid_min_xy[0]) & (prf_mu_y <= grid_max_xy[0])
    # Least squares does not change sigma, so it is always one of the grid values
    & ~np.isclose(prf_sigma, np.max(grid_param_ranges_prf["sigma"]))
)
is_both_selected = is_selected & is_prf_selected
print(f"{is_both_selected.sum()} vertices are selected for both models")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout="constrained")

for name in ["V1", "V2", "V3"]:
    is_roi = is_both_selected & (roi_valid == name)
    axes[0].scatter(prf_eccentricity[is_roi], sf_peak[is_roi], s=6, alpha=0.5, color=roi_colors[name], label=name)
    axes[1].scatter(prf_sigma[is_roi], sf_peak[is_roi], s=6, alpha=0.5, color=roi_colors[name], label=name)

axes[0].set_yscale("log")
axes[0].legend(frameon=False)
style_axes(axes[0], "pRF eccentricity (in degrees)", "Peak SF (in cycles/degree)")
axes[1].set_xscale("log")
style_axes(axes[1], "pRF size (sigma, in degrees)", "Peak SF (in cycles/degree)")

# Rank correlation of the peak SF with the pRF eccentricity and size within each ROI
correlations = {}
for name in ["V1", "V2", "V3"]:
    is_roi = is_both_selected & (roi_valid == name)
    correlations[name] = {
        "number of vertices": is_roi.sum(),
        "eccentricity": pd.Series(prf_eccentricity[is_roi]).corr(pd.Series(sf_peak[is_roi]), method="spearman"),
        "size": pd.Series(prf_sigma[is_roi]).corr(pd.Series(sf_peak[is_roi]), method="spearman"),
    }

pd.DataFrame(correlations).round(2)
```

The `prf` task has the same vertices as the `csf` task, but some valid vertices have a constant response in the `prf` task. The fitters exclude them with a warning and return `NaN` parameters for them, so they are not selected.

In V1, V2, and V3, the peak SF clearly decreases with the eccentricity of the pRF: Vertices with pRFs close to the center of the visual field prefer high SFs, and vertices with pRFs in the periphery prefer low SFs. The rank correlation between peak SF and eccentricity is about -0.4 in V1 and about -0.7 to -0.8 in V2 and V3. The relationship is clearer than with the eccentricity of the template atlas in Exercise 17 because the pRF estimates come from measurements in this subject.

The peak SF also decreases with the size of the pRF: Neuron populations with small pRFs prefer fine patterns, and populations with large pRFs prefer coarse patterns. Note that pRF size increases with eccentricity, so the two relationships are not independent of each other. Also note that the pRF estimates only take the values of the grid because we did not finetune them with SGD.

## Exercise 1

```python
from IPython.display import HTML
from prfmodel.plotting import animate_2d_prf_stimulus

ani = animate_2d_prf_stimulus(stimulus, interval=25)  # Pause 25 ms between time frames

HTML(ani.to_html5_video())
```

Yes. The video shows the four sweeps of the first half (top to bottom, left to right, bottom to top, right to left), then the long blank period, and then the same four sweeps in the reverse order (right to left, bottom to top, left to right, top to bottom). To check the time frames of a sweep, you can also plot single frames of the design matrix, for example, `plt.imshow(stimulus.design[30], origin="lower")`.

## Exercise 2

The BOLD response to a stimulus peaks about 5 seconds later and, including the undershoot, lasts for 20 seconds or longer. If we split the run at frame 151, the response to the last sweep of the first half would continue into the second half. The model would not be able to predict this part of the second half because the stimulus of the second half does not contain the sweep that caused it: The model assumes that there is no response before the first frame of a stimulus. At the same time, the end of the response would be missing from the first half. Splitting in the middle of the blank period gives the response 17 frames (about 15 seconds) to return close to baseline before the second half starts, and the second half starts with another 18 blank frames before its first sweep.

## Exercise 3

```python
response_sd = response_valid.std(axis=1)

plot_surf_stat_map_helper(
    project_to_surface(fill_valid_voxels(response_sd)), vmin=0.0, vmax=5.0, title="BOLD response standard deviation",
)

fig, axes = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 2]}, layout="constrained")

# Noise ceiling against standard deviation for all valid voxels
axes[0].hexbin(response_sd, noise_ceiling, xscale="log", bins="log", gridsize=50, cmap="inferno")
axes[0].set(xlabel="Standard deviation", ylabel="Noise ceiling")

# Example of a voxel with a large standard deviation (top 1%) but a low noise ceiling
is_variable_unreliable = (response_sd > np.quantile(response_sd, 0.99)) & (noise_ceiling < 0.2)
voxel_index = np.flatnonzero(is_variable_unreliable)[0]
axes[1].plot(response_valid[voxel_index])
axes[1].axvline(split, color="black", linestyle="--")
axes[1].set(
    xlabel="Time frame",
    ylabel="BOLD response",
    title=f"Voxel {voxel_index}: noise ceiling = {noise_ceiling[voxel_index]:.2f}, standard deviation = {response_sd[voxel_index]:.1f}",
)
plt.show()

print(f"Voxels with a large standard deviation but a low noise ceiling: {is_variable_unreliable.sum()}")

# Where do these voxels lie?
plot_surf_stat_map_helper(
    project_to_surface(fill_valid_voxels(is_variable_unreliable.astype(float))),
    vmin=0.0,
    vmax=0.3,
    title="Large standard deviation, low noise ceiling",
);
```

Voxels with a high noise ceiling lie in the occipital lobe: around the occipital pole and the calcarine sulcus on the medial side, and on the posterior surface, that is, in the early and higher visual areas. This is roughly where the standard deviation is elevated, too. However, the two maps differ: The standard deviation is also large in regions far from the visual cortex, for example, next to the front end of the medial wall and along the lower edge of the surface, where the noise ceiling is close to zero. Of the 1% of voxels with the largest standard deviation, most (859 of 1,055) have a noise ceiling below 0.2.

The timecourses of these voxels fluctuate strongly, but the fluctuations are unrelated to the stimulus and do not repeat between the two halves. They lie in regions that are known for strong noise in fMRI: close to large veins and sinuses, close to air-filled cavities (which distort the magnetic field), or close to cerebrospinal fluid. Their variability comes from breathing, the heartbeat, head motion, and these artifacts, not from neuronal responses to the stimulus. This is why the standard deviation alone is not a good criterion for selecting voxels, and the noise ceiling, which measures how well the response repeats, is better.

## Exercise 4

The noise ceiling assumes that the two responses we correlate contain the same stimulus-driven signal and only differ in noise. This is not quite true here:

- The BOLD response is slow, so the response during a sweep still contains the tail of the response to the preceding sweep or blank period. This tail differs between the halves because the order of the sweeps differs.
- When a sweep lasts one frame longer in one half, the bar moves at a slightly different speed or starts at a slightly different time. The responses of the two halves are then slightly shifted against each other.
- We use the same fixed delay of 5 frames for all voxels and sweeps, which is only an approximation of the delay of the BOLD response.

These differences are stimulus-driven, but in the correlation they look like noise and reduce it. Thus, our noise ceiling is a conservative (too low) estimate of how well a model can predict the data. Two runs with the identical stimulus would only differ in the noise and would give a higher correlation.

## Exercise 5

```python
for threshold in [0.5, 0.7, 0.9]:
    print(f"Noise ceiling above {threshold}: {(noise_ceiling > threshold).sum()} voxels")
```

With a threshold of 0.5, we would select 5,961 voxels, with 0.7 (our choice) 3,103 voxels, and with 0.9 only 406 voxels.

- **Lower threshold:** We compare the models on more voxels, which makes the comparison more representative of the visual cortex as a whole (e.g., it includes more voxels from higher visual areas, where the responses are weaker). However, the additional voxels are noisier: Noise makes the differences between models smaller and their ranking less reliable, and each fit takes longer.
- **Higher threshold:** The remaining voxels have very reliable responses, so differences between the models are more likely to be real. However, they are few and mostly come from regions with the strongest responses (e.g., early visual cortex), so the result may not generalize to other regions. The threshold also depends on our noise ceiling estimate, which is conservative (Exercise 4).

## Exercise 6

```python
from prfmodel.models.prf import DivNormGaussian2DPRFModel

dn_check_model = DivNormGaussian2DPRFModel(impulse_model=impulse_model)

# Parameters for a single voxel with the starting values of the interactive figure
dn_check_params = pd.DataFrame({
    "mu_x": [0.0],
    "mu_y": [0.0],
    "sigma_activation": [1.0],
    "sigma_normalization": [2.0],
    "amplitude_activation": [1.0],
    "baseline_activation": [1.0],
    "amplitude_normalization": [1.0],
    "baseline_normalization": [1.0],
    "weight_deriv": [-0.5],
    "baseline": [0.0],
})

constant = 10.0


def predict_scaled(names: list[str]) -> np.ndarray:
    """Predict the response to the first half after multiplying the parameters `names` by the constant."""
    params = dn_check_params.copy()
    params[names] *= constant
    return np.asarray(dn_check_model(stimulus_first, params))[0]


prediction = predict_scaled([])

scaled_predictions = {
    "a, b, c, and d": predict_scaled(
        ["amplitude_activation", "baseline_activation", "amplitude_normalization", "baseline_normalization"],
    ),
    "a and b": predict_scaled(["amplitude_activation", "baseline_activation"]),
    "c and d": predict_scaled(["amplitude_normalization", "baseline_normalization"]),
    "only a": predict_scaled(["amplitude_activation"]),
}

for name, scaled_prediction in scaled_predictions.items():
    print(
        f"Multiplying {name}: identical = {np.allclose(scaled_prediction, prediction, atol=1e-6)}, "
        f"ratio of the maxima = {scaled_prediction.max() / prediction.max():.2f}, "
        f"correlation = {np.corrcoef(scaled_prediction, prediction)[0, 1]:.2f}"
    )
```

Multiplying all four parameters by the same constant does not change the prediction at all:

$$
\frac{k a A(t) + k b}{k c N(t) + k d} - \frac{k b}{k d} = \frac{a A(t) + b}{c N(t) + d} - \frac{b}{d}
$$

Thus, the data cannot distinguish between these parameter combinations. When we only multiply some of the parameters, the prediction changes:

- Multiplying only $a$ and $b$ (the numerator) multiplies the response by the constant, and multiplying only $c$ and $d$ (the denominator) divides it by the constant. The shape of the response stays the same (correlation of 1), only its scale changes.
- Multiplying a single parameter (e.g., only $a$) changes the balance between activation and normalization, which changes the shape of the response (the correlation drops below 1).

Because the response only depends on the ratios between the four parameters, fixing one of them (here $c = 1$) makes the model identifiable without losing any flexibility.

## Exercise 7

```python
for name, model in zip(["Gaussian", "CSS", "DoG", "DN"], [gaussian_model, css_model, dog_model, dn_model]):
    print(f"{name}: {model.parameter_names}")

dog_sigma_ratio = dog_params["sigma_surround"] / dog_params["sigma_center"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

axes[0].hist(css_params["n"], bins=np.linspace(0.0, 2.0, 41))
axes[0].axvline(1.0, color="black", linestyle="--")  # No compression
axes[0].set(xlabel="n", ylabel="Number of voxels", title="CSS exponent")

axes[1].hist(dog_sigma_ratio, bins=np.geomspace(0.5, 100.0, 41))
axes[1].set(xscale="log", xlabel="sigma_surround / sigma_center", ylabel="Number of voxels", title="DoG size ratio")
plt.show()

print(css_params["n"].describe())
print(dog_sigma_ratio.describe())
# Total response of the surround relative to the center (the Gaussians are normalized)
print((dog_params["amplitude_surround"] / dog_params["amplitude_center"]).describe())
```

The parameter names contain more parameters than the table: the parameters of the impulse response model (`delay`, `dispersion`, `undershoot`, `u_dispersion`, `ratio`, and `weight_deriv`). The first five are not part of our parameter DataFrames, so the model uses their default values, and we fix `weight_deriv`. Without them, the names match the free parameters in the table. For the DN model, the names also contain `amplitude_normalization` ($c$), which we fix to 1.

- **Compression:** The CSS exponent is below 1 for 96% of the selected voxels and below 0.5 for 69% (median of 0.30). Thus, the responses of most voxels are strongly compressive: Covering a larger part of the pRF increases the response much less than linearly. A few voxels have an exponent above 1, which is probably not well constrained by the data.
- **Suppression:** The DoG surround is about 2 to 3.5 times as large as the center for most voxels (median of 2.6). Many ratios lie close to the starting value of 3, which suggests that the data do not constrain the size of the surround well (see Exercise 18). The strength of the surround varies a lot: For a fifth of the voxels, the total response of the surround is below 1% of the center, so these voxels show practically no suppression, whereas the surround of a quarter of the voxels has more than 65% of the total response of the center.

## Exercise 8

```python
# Improvement of the in-sample R-squared over the Gaussian model
improvement_train = r_squared_train[model_names[1:]].sub(r_squared_train["Gaussian"], axis=0)

pd.DataFrame({
    "mean improvement": improvement_train.mean(),
    "fraction better": (improvement_train > 0.0).mean(),
    "fraction clearly worse (< -0.01)": (improvement_train < -0.01).mean(),
})
```

The DN model fits the training data best (mean R-squared of 0.690), followed by the CSS (0.679), DoG (0.667), and Gaussian model (0.654).

The CSS, DoG, and DN models can mimic the Gaussian model (e.g., the CSS model with `n` equal to 1), and we start their fits from the Gaussian estimates. Thus, they can always reach at least the fit of the Gaussian model, and with their additional parameters, they can fit the data more closely. The CSS model fits better than the Gaussian model for 91% of the voxels, the DoG model for 77%, and the DN model for 84%. Almost none of the voxels (at most 1% for the DN model) fit clearly worse.

If a more complex model fits a voxel clearly worse than the Gaussian model, its fit failed: For example, SGD got stuck in a poor local minimum, or the starting values or the constraints of the parameters (e.g., keeping `amplitude_surround` positive) prevented it from reaching the solution of the Gaussian model. We should check these voxels and, for example, refit them with different starting values.

The in-sample fit does not tell us which model describes the data best. A model with more parameters can also fit the noise in the training data better (overfitting), so a better in-sample fit is expected even if the additional parameters do not describe anything real. To compare the models, we need their out-of-sample fit.

## Exercise 9

```python
improvement_test = r_squared_test[model_names[1:]].sub(r_squared_test["Gaussian"], axis=0)

pd.DataFrame({
    "mean improvement (training set)": improvement_train.mean(),
    "mean improvement (test set)": improvement_test.mean(),
    "fraction better (training set)": (improvement_train > 0.0).mean(),
    "fraction better (test set)": (improvement_test > 0.0).mean(),
})
```

The ranking is the same as for the in-sample fit: The DN model predicts the test data best on average (mean R-squared of 0.656), closely followed by the CSS model (0.652), the DoG model (0.635), and the Gaussian model (0.625). All models predict the test data somewhat worse than they fit the training data.

The advantage of the more complex models shrinks slightly: The mean improvement over the Gaussian model drops from 0.036 to 0.030 for the DN model and from 0.013 to 0.010 for the DoG model, whereas it stays at about 0.026 for the CSS model. The CSS model also improves the prediction for the most voxels (84%, compared to 78% for the DN model and 54% for the DoG model). Thus, the DN model generalizes best on average, but only by a small margin over the CSS model, which has two parameters less and improves the prediction more consistently.

## Exercise 10

```python
from prfmodel.plotting import plot_observed_predicted

# Ratio between out-of-sample R-squared and noise ceiling, and fraction of voxels above the noise ceiling
print(pd.DataFrame({
    "median ratio to noise ceiling": r_squared_test.div(noise_ceiling_selected, axis=0).median(),
    "fraction above noise ceiling": r_squared_test.gt(noise_ceiling_selected, axis=0).mean(),
}))

# Voxels with a high noise ceiling that no model predicts well
is_poorly_predicted = (noise_ceiling_selected > 0.8) & (r_squared_test.max(axis=1) < 0.3)
print(f"Voxels with a noise ceiling above 0.8 and an out-of-sample R-squared below 0.3: {is_poorly_predicted.sum()}")

fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True, layout="constrained")

for ax, voxel_index in zip(axes, np.flatnonzero(is_poorly_predicted)):
    plot_observed_predicted(
        response_second[voxel_index],
        {name: pred_response_second[name][voxel_index] for name in model_names},
        ylabel="BOLD response",
        ax=ax,
    )
    ax.set_title(f"Voxel {voxel_index}: noise ceiling = {noise_ceiling_selected[voxel_index]:.2f}", loc="left")
```

For a typical voxel, the models reach about 80% (Gaussian and DoG) to 85% (CSS and DN) of the noise ceiling. The out-of-sample R-squared lies above the noise ceiling for 3% of the voxels with the Gaussian and DoG model and for about 10% with the CSS and DN model. This is possible because the noise ceiling is only an estimate: It is conservative (Exercise 4), so differences between the halves that a model can predict (e.g., the different order of the sweeps) count as noise. It also contains noise itself because we compute it from only 94 time frames per half.

There are 22 voxels with a noise ceiling above 0.8 for which no model reaches an R-squared of 0.3. Most of them have a clear response with large peaks (some above 10%), but the peaks occur 3 to 5 time frames (about 3 to 4.5 seconds) later than predicted, and they are often broader. When we shift the predictions by this delay, most of them correlate highly with the response (about 0.8 to 0.9). Such large and delayed responses are typical for voxels close to large veins, which drain blood from a larger region and react more slowly. Many of these voxels lie next to each other (e.g., voxels 2624 to 2631). Our models use the same impulse response with a fixed shape for all voxels, so they cannot shift their predictions in time. A few other voxels have strong negative responses (e.g., voxel 345) that none of the models describe. To describe the delayed voxels, we would also need to estimate the parameters of the impulse response (e.g., `delay`) for each voxel.

## Exercise 11

```python
# Only consider voxels for which the Gaussian fit did not fail (in-sample R-squared above 0.3)
has_gaussian_fit = r_squared_train["Gaussian"] > 0.3

fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True, layout="constrained")

# For each complex model, the voxel with the largest improvement over the Gaussian model on the test set
plotted_voxels = []
for ax, name in zip(axes, model_names[1:]):
    voxel_index = improvement_test.loc[has_gaussian_fit, name].drop(plotted_voxels).idxmax()
    plotted_voxels.append(voxel_index)
    plot_observed_predicted(
        response_second[voxel_index],
        {model_name: pred_response_second[model_name][voxel_index] for model_name in model_names},
        ylabel="BOLD response",
        ax=ax,
    )
    ax.set_title(
        f"Voxel {voxel_index}: R-squared Gaussian = {r_squared_test['Gaussian'][voxel_index]:.2f}, "
        f"{name} = {r_squared_test[name][voxel_index]:.2f}",
        loc="left",
    )
```

- **CSS:** The voxel responds to a sweep with a broad, plateau-like response: The response is almost as large when the bar only covers the edge of the pRF as when it covers the center. The Gaussian model predicts narrower peaks with a dip in between. Compression flattens the response, so the CSS model describes the plateau better.
- **DoG and DN:** The response drops clearly below baseline when the bar is close to but not on the pRF center, and it lies above this level during the blank periods. The Gaussian model can only predict an increase of the response (followed by the short undershoot of the impulse response) and misses these negative responses. The suppressive surround of the DoG model and the normalization of the DN model predict them.

We only consider voxels for which the Gaussian model fits the training data with an R-squared above 0.3. Without this restriction, the largest improvements of the CSS model come from voxels for which the Gaussian fit failed: Its prediction is flat (R-squared close to 0), although the voxel responds clearly. There, the improvement reflects a failed fit rather than compression. Because the DoG and DN models start from the Gaussian estimates, their fits fail for these voxels, too.

## Exercise 12

```python
plot_surf_stat_map_helper(
    project_to_surface(fill_selected_voxels(noise_ceiling_selected)),
    vmin=0.7,
    vmax=1.0,
    title="Noise ceiling of the selected voxels",
)
plt.show()

# Fraction of the selected models within bins of the noise ceiling
pd.crosstab(pd.cut(noise_ceiling_selected, [0.7, 0.8, 0.9, 1.0]), best_model, normalize="index")
```

The Gaussian model is mostly selected around the occipital pole on the posterior surface and along the calcarine sulcus on the medial side, where the center of the visual field is represented in early visual cortex. The CSS model is mostly selected in the outer parts of the selected region, on the lateral, dorsal, and ventral side of the posterior surface, where the higher visual areas and the more peripheral parts of the visual field are represented. This agrees with Kay et al. (2013), who found stronger compression in higher visual areas. The DN model is selected in scattered patches, mostly in the outer parts as well, and the DoG model is rarely selected.

The noise ceiling is highest along the calcarine sulcus and around the occipital pole and lower towards the edges of the selected region. However, the fractions of the selected models hardly change with the noise ceiling: For example, the CSS model is selected for 45% to 47% of the voxels in each bin. Thus, the regional differences are not just a consequence of differences in noise.

## Exercise 13

```python
pd.DataFrame({
    f"min_gain = {gain}": pd.Series(select_models(r_squared_test[model_names], min_gain=gain))
    .value_counts(normalize=True)
    .reindex(model_names, fill_value=0.0)
    for gain in [0.0, 0.01, 0.05]
})
```

| Model | `min_gain` = 0 | `min_gain` = 0.01 | `min_gain` = 0.05 |
|---|---|---|---|
| Gaussian | 6% | 37% | 77% |
| CSS | 47% | 46% | 20% |
| DoG | 5% | 4% | 3% |
| DN | 41% | 13% | 1% |

With a minimum gain of 0, any improvement is enough, however small. The complex models fit at least as well as the Gaussian model and slightly better for most voxels, often only by chance, so the Gaussian model is only selected for 6% of the voxels, and the most flexible DN model for 41%. With a minimum gain of 0.05, a model must clearly beat all simpler models, so the Gaussian model is selected for most voxels. The DN model almost disappears because it must also beat the CSS model by 0.05, which already captures most of the compression. The CSS model is still selected for a fifth of the voxels, which shows that compression clearly improves the prediction for these voxels. The minimum gain controls the trade-off between missing real improvements (high gain) and selecting complex models because of noise (low gain).

## Exercise 14

```python
from prfmodel.utils import calculate_eccentricity

gaussian_eccentricity = calculate_eccentricity(gaussian_params["mu_x"], gaussian_params["mu_y"])
eccentricity_bins = pd.cut(gaussian_eccentricity, [0, 1, 2, 3, 4, 6, 10, np.inf])
sigma_bins = pd.cut(gaussian_params["sigma"], [0, 0.5, 1, 2, 3, 4, 6, np.inf])

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), layout="constrained")

# Average improvement within bins of eccentricity and size
improvement_test.groupby(eccentricity_bins, observed=True).mean().plot(
    ax=axes[0], marker="o", xlabel="Eccentricity (degrees)", ylabel="Improvement over Gaussian (test set)",
)
improvement_test.groupby(sigma_bins, observed=True).mean().plot(
    ax=axes[1], marker="o", xlabel="sigma (degrees)", ylabel="Improvement over Gaussian (test set)",
)
for ax in axes:
    ax.axhline(0.0, color="black", linestyle="--")
    ax.tick_params(axis="x", labelrotation=30)
plt.show()

# Number of voxels per bin
print(pd.Series(eccentricity_bins).value_counts(sort=False))
print(pd.Series(sigma_bins).value_counts(sort=False))

for name in model_names[1:]:
    plot_surf_stat_map_helper(
        project_to_surface(fill_selected_voxels(improvement_test[name].to_numpy())),
        vmin=-0.1,
        vmax=0.1,
        cmap="RdBu_r",
        title=f"Improvement of {name} over Gaussian (test set)",
    )
```

The improvement grows with eccentricity: For pRFs within 1 to 2 degrees of the center of the visual field, the more complex models hardly improve the prediction (the DoG model not at all up to 3 degrees). For pRFs with centers between 6 and 10 degrees, the CSS and DN models improve the R-squared by about 0.06. The CSS and DN models also improve most for large pRFs (`sigma` above 2 degrees), whereas the DoG model improves most for small pRFs (below 1 degree). On the surface, the improvement is close to zero around the occipital pole and largest in the outer parts of the selected region.

Note that more than a third of the selected voxels (36%) have a Gaussian pRF center outside the stimulus. For these voxels, the mean improvement of the CSS and DN models is about twice as large as for the voxels with a center inside the stimulus, and the Gaussian model is only selected for 15% of them (compared to 50% inside). For these pRFs, the stimulus only covers their edge, so the response only tells us how the edge of the pRF responds. Different combinations of center, size, and the additional parameters predict almost the same response, for example, a center further outside the stimulus together with a larger size or amplitude. Their parameters are therefore poorly constrained, and the additional flexibility of the complex models helps them describe the response to the edge of the pRF. Thus, a better prediction for these voxels does not necessarily mean that their neuron populations show more compression or suppression, and we should not interpret their parameters. To describe these pRFs, we would need a stimulus that covers a larger part of the visual field.

## Exercise 15

We put the fitting steps from above into a function so that we can reuse it (also in the next exercise). Note that fitting all four models takes about as long as the fits above.

```python
def fit_models(stimulus: PRFStimulus, response: np.ndarray) -> dict[str, pd.DataFrame]:
    """Fit all four models to the response with the same steps as above and return their parameter estimates."""
    log_transform = {"transform_fun": ops.log, "inverse_fun": ops.exp}

    def fit_least_squares(model, init_params, slope_name):
        _, params = LeastSquaresFitter(model=model, stimulus=stimulus).fit(
            data=response, parameters=init_params, slope_name=slope_name, intercept_name="baseline", batch_size=200,
        )
        return params

    def fit_sgd(model, init_params, log_names, fixed_names):
        sgd_fitter = SGDFitter(
            model=model,
            stimulus=stimulus,
            adapter=Adapter([ParameterTransform(log_names, **log_transform)]),
            optimizer=Adam(learning_rate=learning_rate),
            compile_step=True,
        )
        _, params = sgd_fitter.fit(
            data=response, init_parameters=init_params, fixed_parameters=fixed_names, num_steps=num_steps,
        )
        return params

    # Gaussian model: grid search, least-squares, and SGD
    grid_values = grid_values_2d_prf(stimulus, num_mu=11, mu_extent=1.0, num_sigma=10, sigma_range=(0.05, 4.0)) | {
        "weight_deriv": [-0.5],
        "baseline": [0.0],
        "amplitude": [1.0],
    }
    _, params = GridFitter(model=gaussian_model, stimulus=stimulus, compile_step=True).fit(
        data=response, parameter_values=grid_values, batch_size=20,
    )
    params = fit_least_squares(gaussian_model, params, "amplitude")
    gaussian = fit_sgd(gaussian_model, params, ["sigma"], ["weight_deriv"])

    # CSS model: start from the Gaussian estimates
    params = fit_least_squares(css_model, init_css_from_gaussian(gaussian), "amplitude")
    css = fit_sgd(css_model, params, ["sigma", "n"], ["weight_deriv"])

    # DoG model: start from the Gaussian estimates and only keep suppressive least-squares estimates
    params = init_dog_from_gaussian(gaussian, sigma_ratio=3.0)
    dog_ls_params = fit_least_squares(dog_model, params, ["amplitude_center", "amplitude_surround"])
    is_suppressive = dog_ls_params["amplitude_surround"] > 0.0
    params.loc[is_suppressive] = dog_ls_params.loc[is_suppressive]
    dog = fit_sgd(dog_model, params, ["sigma_center", "sigma_surround", "amplitude_surround"], ["weight_deriv"])

    # DN model: start from the DoG and CSS estimates
    params = init_div_norm_from_dog_css(dog, css_n=css["n"].clip(upper=0.9), stimulus=stimulus)
    params = fit_least_squares(dn_model, params, ["amplitude_activation", "baseline_activation"])
    params[["amplitude_activation", "baseline_activation"]] = (
        params[["amplitude_activation", "baseline_activation"]].clip(lower=1e-3)
    )
    dn = fit_sgd(
        dn_model,
        params,
        ["sigma_activation", "sigma_normalization", "amplitude_activation", "baseline_activation", "baseline_normalization"],
        ["weight_deriv", "amplitude_normalization"],
    )

    return {"Gaussian": gaussian, "CSS": css, "DoG": dog, "DN": dn}


# Fit the models to the second half
params_second = fit_models(stimulus_second, response_second)

# Predict the first half with the parameters estimated on the second half
r_squared_test_swapped = pd.DataFrame({
    name: calculate_r_squared(
        response_first, np.asarray(predict_batched[name](stimulus_first, params_second[name], batch_size=200)),
    )
    for name in model_names
})

best_model_swapped = select_models(r_squared_test_swapped[model_names], min_gain=min_gain)

print(f"Same model selected in both folds: {(best_model == best_model_swapped).mean():.2f}")
print(pd.crosstab(pd.Series(best_model, name="First fold"), pd.Series(best_model_swapped, name="Second fold")))

# Out-of-sample R-squared averaged across both folds
pd.DataFrame({
    "first fold": r_squared_test.mean(),
    "second fold": r_squared_test_swapped.mean(),
    "average": ((r_squared_test + r_squared_test_swapped) / 2).mean(),
})
```

The same model is selected in both folds for 69% of the voxels. This is clearly more than expected by chance (36% for selections with the same fractions of models but no relation between the folds), but for almost a third of the voxels, the selection changes when we swap the halves:

- The selection of the Gaussian model is the most reliable: 85% of the voxels for which it is selected in the first fold also select it in the second fold.
- The CSS model is selected again for 66% of its voxels. Most of the other voxels switch to the Gaussian model.
- The DoG and DN models are selected again for fewer than half of their voxels (41% and 44%).

Thus, the selection for an individual voxel is not very reliable, especially for the DoG and DN models, whose advantage over the simpler models is small and noisy. The selection becomes more reliable when we look at many voxels together, for example, at the fraction of voxels for which a model is selected within a region.

Averaged across both folds, the out-of-sample R-squared is 0.623 for the Gaussian, 0.645 for the CSS, 0.631 for the DoG, and 0.646 for the DN model. In the second fold, the CSS model even predicts better than the DN model (0.639 compared to 0.635). Thus, the CSS and DN models are equally good on average, and the CSS model achieves this with two parameters less. The first fold alone made the DN model look slightly better than it is.

## Exercise 16

We reuse the function `fit_models` from the solution of Exercise 15. Fitting the full run takes about twice as long as fitting one half.

```python
from prfmodel.utils import calculate_eccentricity, calculate_polar_angle

# Fit all four models to the full run with the function from the solution of Exercise 15
params_full = fit_models(stimulus, response_valid[is_selected])

# One condition per model: the voxels for which the model is selected
conditions = [best_model == name for name in model_names]

mu_x_selected = np.select(conditions, [params_full[name]["mu_x"] for name in model_names])
mu_y_selected = np.select(conditions, [params_full[name]["mu_y"] for name in model_names])

centers = {
    "selected models": (mu_x_selected, mu_y_selected),
    "Gaussian model": (params_full["Gaussian"]["mu_x"].to_numpy(), params_full["Gaussian"]["mu_y"].to_numpy()),
}

for label, (mu_x, mu_y) in centers.items():
    # Project mu_x and mu_y onto the surface separately and compute polar angle and eccentricity afterwards
    mu_x_surf = project_to_surface(fill_selected_voxels(mu_x))
    mu_y_surf = project_to_surface(fill_selected_voxels(mu_y))
    plot_surf_stat_map_helper(calculate_polar_angle(mu_x_surf, mu_y_surf), cyclic=True, title=f"Polar angle ({label})")
    plot_surf_stat_map_helper(
        calculate_eccentricity(mu_x_surf, mu_y_surf), vmin=0.0, vmax=5.0, title=f"Eccentricity ({label})",
    )

# Distance between the pRF centers of the selected models and the Gaussian model
center_distance = np.hypot(mu_x_selected - centers["Gaussian model"][0], mu_y_selected - centers["Gaussian model"][1])
pd.Series(center_distance).groupby(best_model).describe()
```

The maps of the selected models and the Gaussian model look very similar: Both show the same retinotopic organization, with the contralateral visual field in each hemisphere and the polar angle and eccentricity changing in the same way across the surface (see the first practical). The pRF centers of the selected models differ only little from the Gaussian centers: The polar angle differs by a median of only 1 to 3 degrees. The center of the CSS model lies a median of 0.65 degrees away from the Gaussian center, mostly closer to the center of the visual field (its eccentricity is 0.5 degrees smaller), and the center of the DN model lies 0.92 degrees away. Where the CSS model is selected, the eccentricity map therefore shows somewhat smaller values. For the voxels for which the Gaussian model is selected, the maps are identical.

Thus, the choice of the model hardly changes the large-scale maps of the pRF centers, but it can shift the centers of individual voxels by up to a few degrees. The shifts are largest for pRFs near or outside the edge of the stimulus: For voxels with a Gaussian eccentricity above 4 degrees, the selected model moves the center by a median of 1 to 1.4 degrees, compared to 0.16 degrees within 2 degrees of the center of the visual field.

## Exercise 17

We use the pRF centers of the selected models from Exercise 16 and plot the additional parameters against eccentricity, together with their median within bins of eccentricity (black). For the DN model, we use the ratio between the sizes of the normalization and activation Gaussians.

```python
eccentricity_selected = calculate_eccentricity(mu_x_selected, mu_y_selected)
eccentricity_bins = pd.cut(eccentricity_selected, [0, 1, 2, 3, 4, 6, 12])

# Additional parameters of each complex model for the voxels for which it is selected
additional_parameters = {
    "CSS": ("n", params_full["CSS"]["n"]),
    "DoG": ("sigma_surround / sigma_center", params_full["DoG"]["sigma_surround"] / params_full["DoG"]["sigma_center"]),
    "DN": (
        "sigma_normalization / sigma_activation",
        params_full["DN"]["sigma_normalization"] / params_full["DN"]["sigma_activation"],
    ),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")

for ax, (name, (label, values)) in zip(axes, additional_parameters.items()):
    is_model = best_model == name
    ax.scatter(eccentricity_selected[is_model], values[is_model], s=5, alpha=0.5)
    # Median within bins of eccentricity
    binned_median = values[is_model].groupby(eccentricity_bins[is_model], observed=True).median()
    ax.plot([interval.mid for interval in binned_median.index], binned_median, color="black", marker="o")
    ax.set(
        yscale="log",
        xlabel="Eccentricity (degrees)",
        ylabel=label,
        title=f"{name} ({is_model.sum()} voxels)",
    )
```

- **Compression:** For the voxels for which the CSS model is selected, compression is strong: The median exponent `n` is about 0.15 and does not change with eccentricity.
- **Suppression (DoG):** The DoG model is only selected for about a hundred voxels. The surround is mostly 2 to 3 times as large as the center, without a clear relation to eccentricity. Some ratios are extremely large (above 10), which means that the surround is hardly constrained for these voxels (see Exercise 18).
- **Normalization (DN):** The normalization Gaussian is relatively larger for pRFs close to the center of the visual field (median ratio of about 2.3 within 2 degrees) than for pRFs further out (about 1.5 beyond 6 degrees).

These results should be interpreted with caution: pRFs with an eccentricity above about 4 degrees have their center outside the stimulus, so their parameters are poorly constrained (Exercise 14). Moreover, the selection of the DoG and DN models is not reliable for individual voxels (Exercise 15), and the eccentricity bins mix voxels from different visual areas, which can differ in compression and suppression (e.g., compression is stronger in higher visual areas; Kay et al., 2013).

## Exercise 18

We use the estimates from the first half because we selected the models with them. For each Gaussian of the DoG and DN model, we compute the fraction of its volume that lies inside the visual field covered by the stimulus, and we plot the Gaussians of one voxel for which the DoG model is selected with `prfmodel.models.prf.Gaussian2DPRFTuning` (see the first practical).

```python
from prfmodel.models.prf import Gaussian2DPRFTuning
from scipy.stats import norm

# Edges of the visual field covered by the stimulus (the grid stores y in grid[..., 0] and x in grid[..., 1])
(y_min, x_min), (y_max, x_max) = stimulus.grid.min(axis=(0, 1)), stimulus.grid.max(axis=(0, 1))


def fraction_inside_stimulus(mu_x: pd.Series, mu_y: pd.Series, sigma: pd.Series) -> np.ndarray:
    """Fraction of the volume of a 2D Gaussian that lies inside the visual field covered by the stimulus."""
    fraction_x = norm.cdf(x_max, mu_x, sigma) - norm.cdf(x_min, mu_x, sigma)
    fraction_y = norm.cdf(y_max, mu_y, sigma) - norm.cdf(y_min, mu_y, sigma)
    return np.asarray(fraction_x * fraction_y)


# The Gaussians of the DoG and DN model and the names of their size parameters
gaussians = {
    "DoG center": (dog_params, "sigma_center"),
    "DoG surround": (dog_params, "sigma_surround"),
    "DN activation": (dn_params, "sigma_activation"),
    "DN normalization": (dn_params, "sigma_normalization"),
}

coverage = pd.DataFrame({
    name: fraction_inside_stimulus(params["mu_x"], params["mu_y"], params[sigma_name])
    for name, (params, sigma_name) in gaussians.items()
})

# Median fraction inside the stimulus, separately for the voxels for which each model is selected
print(coverage.groupby(best_model).median().reindex(model_names))

# The voxel with the highest out-of-sample R-squared among the voxels for which the DoG model is selected
voxel_index = r_squared_test["DoG"].where(best_model == "DoG").idxmax()
tuning = Gaussian2DPRFTuning()

fig, axes = plt.subplots(1, 4, figsize=(18, 4), layout="constrained")

for ax, (name, (params, sigma_name)) in zip(axes, gaussians.items()):
    profile = np.asarray(tuning(stimulus, params.iloc[[voxel_index]].rename(columns={sigma_name: "sigma"})))[0]
    image = ax.imshow(profile, origin="lower", extent=(x_min, x_max, y_min, y_max), cmap="inferno")
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set(xlabel="x (degrees)", ylabel="y (degrees)", title=f"{name}: {coverage[name][voxel_index]:.0%} inside")
```

The surround and normalization Gaussians are much larger than the stimulus: The median `sigma_surround` is 5.6 degrees, and some estimates of `sigma_surround` and `sigma_normalization` reach tens of degrees, although the stimulus only spans 8 × 8 degrees. For the median voxel, only 21% of the DoG surround and 35% of the DN normalization Gaussian lie inside the stimulus. Even for voxels with a pRF center inside the stimulus, only about a third of the DoG surround lies inside. The plotted voxel is one of the more favorable cases: Its pRF lies well inside the stimulus, and still about a third of its surround lies outside.

Thus, the stimulus design does not provide enough information to estimate these Gaussians well. The data only constrain the part of a Gaussian that lies inside the stimulus. A larger, stronger surround can predict almost the same response as a smaller, weaker one, so the size and strength of the surround trade off against each other.

This also determines when the DoG and DN models can fit better than the Gaussian and CSS models. Suppression only becomes visible when the bar stimulates the surround but not the center, which produces a negative response (Exercise 11). For a pRF in the middle of the stimulus, the bar always covers parts of the center and the surround at the same time, and the effect of the surround is hard to distinguish from a smaller or compressed center. The voxels for which the DoG and DN models are selected have pRFs that lie closer to the edge of the stimulus: Their center or activation Gaussian lies only 56% (DoG) or 37% (DN) inside the stimulus, compared to more than 95% for the voxels for which the Gaussian model is selected.

## Exercise 1

```python
import matplotlib.pyplot as plt
from scipy.io import loadmat

design_raw = loadmat(dataset.files["design"])["stim"]
print(design_raw.shape, design_raw.min(), design_raw.max())

# Plot every fifth time frame
fig, axes = plt.subplots(4, 6, figsize=(14, 6), layout="constrained")

for ax, frame in zip(axes.flat, range(0, design_raw.shape[-1], 5)):
    ax.imshow(design_raw[..., frame], cmap="gray", vmin=0, vmax=255)
    ax.set(title=f"Time frame {frame}", xticks=[], yticks=[])
```

The design matrix has the shape `(281, 500, 120)`: The first two dimensions are the rows (y) and columns (x) of pixels on the screen (with an aspect ratio of 16:9), and the last dimension contains the 120 time frames (one per TR). Note that this order differs from the design matrices in prfmodel, where time frames are the first dimension. The values are pixel intensities from 0 (background) to 255 (bar).

The stimulus contains four bar passes, separated by blank periods:

- Time frames 16–28: horizontal bar from top to bottom
- Time frames 31–50: vertical bar from right to left
- Time frames 68–87: vertical bar from left to right
- Time frames 90–102: horizontal bar from bottom to top

## Exercise 2

It depends on how the target response is measured:

- **Target also measured with ECoG:** Source and target have the same temporal dynamics, so the model needs no adjustment (apart from the scaling, which already adapts the units).
- **Target measured with BOLD fMRI:** The source response reflects neural activity almost instantly, but the target response is the slow and delayed BOLD response to that activity. The model would need an impulse response submodel, like a pRF model for BOLD data: The CF-weighted source response (i.e., the predicted target neural activity) is convolved with the hemodynamic impulse response before scaling. The source and target recordings would also need to be aligned in time and sampled at the same rate.

The opposite case (source measured with BOLD, target with ECoG) is much harder because we would have to undo the slow hemodynamic response of the source (deconvolution), which amplifies the noise.

## Exercise 3

```python
from prfmodel.plotting import plot_response_heatmap

plot_response_heatmap(
    source_response,
    vmin=-3,
    vmax=3,
    cmap="RdBu_r",  # Diverging colormap: positive responses are red, negative responses blue
    xlabel="Time frame (in TR)",
    ylabel="V1 vertex index",
    colorbar_label="BOLD response (in PSC)",
    figsize=(6, 6),
);
```

The first 1,509 rows are the V1 vertices of the left hemisphere, the remaining rows those of the right hemisphere. In each hemisphere, a group of vertices responds strongly to each of the four bar passes. The responses form slanted bands: As the bar moves through the visual field, it activates one V1 vertex after the other because neighboring vertices in V1 represent neighboring positions in the visual field (retinotopy). Each response peaks a few time frames after the bar passed because of the delay of the BOLD response. Many other V1 vertices hardly respond or show a weak negative response during the bar passes. They likely represent parts of the visual field that the bar does not cover, for example, the periphery beyond the edges of the screen.

## Exercise 4

```python
from prfmodel.plotting import plot_r_squared_hist

# A log scale also shows the bins with few vertices
plot_r_squared_hist(r_squared, value_range=(-0.1, 1.0), log=True);
```

```python
is_v1_valid = is_v1[response_is_valid]

print("Fraction of V1 vertices with R-squared > 0.99:", np.mean(r_squared[is_v1_valid] > 0.99))
print("Fraction of other vertices with R-squared > 0.99:", np.mean(r_squared[~is_v1_valid] > 0.99))
print("Fraction of V1 vertices with the smallest CF size:", np.mean(np.isclose(ls_params["sigma"][is_v1_valid], 0.5)))
```

The spike consists of the V1 vertices. We fitted the model to all vertices, including those in V1, which are also the source vertices. For a V1 vertex, the best CF is centered on the vertex itself and has the smallest size in the grid, so that it practically only includes the vertex itself. The prediction is then the response of the vertex, and the fit is (almost) perfect. These R-squared scores are trivial and tell us nothing about the relationship between V1 and other areas, which is why we exclude V1 vertices when interpreting the parameters below.

## Exercise 5

V1 vertices that represent the stimulated part of the visual field respond with a clear peak to each bar pass (see Exercise 3). A vertex in V2–V4 that represents the same part of the visual field should respond in the same way, and the CF model can predict it by averaging the responses of the corresponding V1 vertices. This is what we see for the vertices in V3 and V4: The observed response has four clear peaks, and the prediction follows them.

The observed response of the vertex in V2 does not show any peaks for the bar passes. Instead, it slowly drifts upwards over the run and fluctuates from time frame to time frame. This vertex likely represents a part of the visual field that the bar does not cover, so its response is not driven by the stimulus. Nevertheless, the CF model reaches an R-squared of 0.65: The grid search finds a CF in V1 whose response contains a similar drift and similar fluctuations, for example, because of noise that is shared between brain regions (e.g., from breathing or head motion). The CF model can map any signal that source and target have in common, not only the response to the stimulus. A high R-squared therefore does not guarantee that the CF reflects how the stimulus-driven response is relayed from V1 to the target vertex.

## Exercise 6

On the surface, the CF size increases with the distance from V1: Vertices in V2 next to V1 have small CFs, and the CFs become larger in V3, V4, and the areas beyond. The ROI comparison shows the same: The average CF size increases from about 2 mm in V2 over about 3 mm in V3 to about 6 mm in V4.

A larger CF means that the response of a target vertex is best predicted by averaging the responses of a larger patch of V1. Neighboring V1 vertices represent neighboring positions in the visual field, so a larger patch of V1 covers a larger part of the visual field. Thus, neuron populations in higher-order visual areas integrate information from a larger part of V1, which is in line with their larger pRFs. This convergence along the visual hierarchy is what Haak et al. (2013) also reported.

## Exercise 7

We create a stimulus with the z-scored source response and repeat the grid search and least-squares. To save time, we only fit the vertices in V2, V3, and V4. The grid search uses the correlation as loss, which does not depend on the scale of the target response, so we can keep the target response in PSC.

```python
stimulus_z = CFStimulus(distance_matrix=distance_matrix, source_response=response_z[is_v1])

is_target = np.isin(visual_area_valid, ["V2", "V3", "V4"])

_, grid_params_z = GridFitter(model=cf_model, stimulus=stimulus_z, compile_step=True).fit(
    data=response_valid[is_target],
    parameter_values=grid_param_ranges,
    batch_size=20,
)

_, ls_params_z = LeastSquaresFitter(model=cf_model, stimulus=stimulus_z).fit(
    data=response_valid[is_target],
    parameters=grid_params_z,
    slope_name="amplitude",
    intercept_name="baseline",
    batch_size=200,
)

r_squared_z = calculate_r_squared(
    response_valid[is_target], np.asarray(predict_batched(stimulus_z, ls_params_z, batch_size=200)),
)
```

We compare the estimates for the vertices that we selected above.

```python
is_compared = is_selected[is_target]
estimates_psc = ls_params[is_target][is_compared].reset_index(drop=True)
estimates_z = ls_params_z[is_compared].reset_index(drop=True)

# Distance between the CF centers estimated with both source responses (in mm)
center_distance = distance_matrix[
    estimates_psc["center_index"].astype(int), estimates_z["center_index"].astype(int)
]

print("Fraction with the same CF center:", np.mean(center_distance == 0.0))
print("Fraction with CF centers less than 3 mm apart:", np.mean(center_distance < 3.0))
print("Fraction with the same CF size:", np.mean(np.isclose(estimates_psc["sigma"], estimates_z["sigma"])))
print("Mean R-squared (PSC):", r_squared[is_target][is_compared].mean())
print("Mean R-squared (z-scores):", r_squared_z[is_compared].mean())

# Mean CF size per visual area
pd.DataFrame({
    "PSC": estimates_psc["sigma"],
    "z-scores": estimates_z["sigma"],
    "area": visual_area_valid[is_target][is_compared],
}).groupby("area").mean()
```

For this dataset, z-scoring hardly changes the estimates: About two thirds of the vertices get the same CF center and more than 80% get a CF center less than 3 mm away, most vertices get the same CF size, and the mean CF size per area and the R-squared are almost identical.

In general, z-scoring changes how the source vertices are weighted. In PSC, vertices with large responses (e.g., close to large veins) contribute more to the weighted sum than vertices with small responses. In z-scores, every source vertex contributes with the same variance, which also gives more weight to vertices that mostly contain noise. The CF center and size keep their meaning as the location and extent of the CF on the surface of V1. However, the predicted response is no longer a weighted average of the actual responses, and `amplitude` and `baseline` change their units (from PSC per PSC to PSC per standard deviation).

## Exercise 8

No. The CF model predicts the target response as a weighted sum (i.e., a spatial average) of source responses. A V2 population has a larger pRF than a V1 population, and averaging the responses of neighboring V1 populations with small, overlapping pRFs mimics a larger pRF. Thus, V1 $\rightarrow$ V2 works well, and the CF size tells us how much of V1 a V2 population integrates over.

In the other direction, averaging V2 responses can only make the effective pRF larger, never smaller. The best prediction of a V1 population is then a CF that is as small as possible (a single V2 vertex at the corresponding position), and the fit is worse because the V2 response is blurrier than the V1 response. The CF centers would still roughly correspond to each other because both areas are retinotopically organized, but the CF sizes would differ and mean something different. Note also that the CF is a statistical mapping between responses, which does not imply that information flows only in the modeled direction: There are both feedforward and feedback connections between V1 and V2.

## Exercise 9

We compute the distance matrix and the source response for the V2 vertices in the same way as for V1. All V2 vertices have valid responses.

```python
label_v2 = visual_area_labels["V2"]
is_v2 = atlas_labels == label_v2

v2_indices_left = np.where(surface.atlas["left"] == label_v2)[0]
v2_indices_right = np.where(surface.atlas["right"] == label_v2)[0]

distance_matrix_v2_left = calculate_distance_matrix(surface.files["wm_lh"], v2_indices_left)
distance_matrix_v2_right = calculate_distance_matrix(surface.files["wm_rh"], v2_indices_right)

num_left_v2, num_right_v2 = len(v2_indices_left), len(v2_indices_right)
distance_matrix_v2 = np.block([
    [distance_matrix_v2_left, np.full((num_left_v2, num_right_v2), np.inf)],
    [np.full((num_right_v2, num_left_v2), np.inf), distance_matrix_v2_right],
])

stimulus_v2 = CFStimulus(distance_matrix=distance_matrix_v2, source_response=response_psc[is_v2])
```

We fit the model to the valid vertices in V3 and V4 with a grid that contains every V2 vertex as a CF center.

```python
is_target_v2 = np.isin(visual_area_valid, ["V3", "V4"])

grid_param_ranges_v2 = grid_param_ranges | {"center_index": np.arange(len(distance_matrix_v2))}

_, grid_params_v2 = GridFitter(model=cf_model, stimulus=stimulus_v2, compile_step=True).fit(
    data=response_valid[is_target_v2],
    parameter_values=grid_param_ranges_v2,
    batch_size=20,
)

_, ls_params_v2 = LeastSquaresFitter(model=cf_model, stimulus=stimulus_v2).fit(
    data=response_valid[is_target_v2],
    parameters=grid_params_v2,
    slope_name="amplitude",
    intercept_name="baseline",
    batch_size=200,
)

r_squared_v2 = calculate_r_squared(
    response_valid[is_target_v2], np.asarray(predict_batched(stimulus_v2, ls_params_v2, batch_size=200)),
)

# Same selection criteria as with V1 as the source ROI
is_selected_v2 = (r_squared_v2 > r_squared_threshold) & ~np.isclose(
    ls_params_v2["sigma"], np.max(grid_param_ranges_v2["sigma"]),
)

plot_parameter_by_roi(
    ls_params_v2[is_selected_v2],
    "sigma",
    visual_area_valid[is_target_v2][is_selected_v2],
    order=["V3", "V4"],
    error="sem",
    xlabel="Visual area",
    ylabel="CF size in V2 (sigma, in mm)",
    figsize=(4, 4),
);
```

Yes, the CFs of V4 vertices in V2 (about 2.7 mm) are larger than those of V3 vertices (about 2.2 mm). However, the difference between V3 and V4 is much smaller than with V1 as the source ROI (about 3 mm vs. 6 mm). V2 populations already integrate over a larger part of the visual field than V1 populations, so fewer source vertices are needed to build the larger pRFs of V3 and V4. Note that CF sizes in V1 and V2 cannot be compared directly because the same distance on the surface of V1 and V2 corresponds to different distances in the visual field (cortical magnification differs between areas).

## Exercise 10

First, we create a pRF stimulus from the raw design matrix. We binarize it, downsample it to a grid of 56 x 100 cells to make the fit faster, and move the time frames to the first dimension. The dataset does not tell us how large the screen was in degrees of visual angle, so we assume a height of 10 degrees. Scaling the visual field by a constant factor scales all pRF centers by the same factor, so the correlations below do not depend on this assumption.

```python
from prfmodel.fitters import grid_values_2d_prf
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.stimuli import PRFStimulus
from prfmodel.utils import calculate_eccentricity, calculate_polar_angle
from scipy.io import loadmat

design_raw = loadmat(dataset.files["design"])["stim"]  # Shape (y, x, time frames), values from 0 to 255

# Binarize, crop to 280 x 500 pixels, and average blocks of 5 x 5 pixels
design_binary = (design_raw[:280] > 127).astype("float32")
design_small = design_binary.reshape(56, 5, 100, 5, -1).mean(axis=(1, 3))

# Move the time frames to the first dimension and flip the rows, so that the first row is the bottom of the screen
design_prf = np.ascontiguousarray(np.moveaxis(design_small, -1, 0)[:, ::-1, :])

# Assumed height of the screen in degrees of visual angle
screen_height = 10.0
cell_size = screen_height / design_prf.shape[1]
y_coords = (np.arange(design_prf.shape[1]) - (design_prf.shape[1] - 1) / 2) * cell_size
x_coords = (np.arange(design_prf.shape[2]) - (design_prf.shape[2] - 1) / 2) * cell_size
grid = np.stack(np.meshgrid(y_coords, x_coords, indexing="ij"), axis=-1)  # Shape (y, x, 2)

tr = 1.5

prf_stimulus = PRFStimulus(design=design_prf, grid=grid, dimension_labels=["y", "x"], frame_duration=tr)
```

Second, we fit the canonical Gaussian 2D pRF model to the valid vertices in V1 to V4 with grid search and least-squares, like in the first practical.

```python
is_v1_to_v4 = visual_area_valid != ""

prf_model = Gaussian2DPRFModel(impulse_model=DerivativeTwoGammaImpulse(resolution=tr))

prf_grid_values = grid_values_2d_prf(prf_stimulus, num_mu=21, mu_extent=1.0, num_sigma=10) | {
    "weight_deriv": [-0.5],
    "baseline": [0.0],
    "amplitude": [1.0],
}

_, prf_grid_params = GridFitter(model=prf_model, stimulus=prf_stimulus, compile_step=True).fit(
    data=response_valid[is_v1_to_v4],
    parameter_values=prf_grid_values,
    batch_size=20,
)

_, prf_params = LeastSquaresFitter(model=prf_model, stimulus=prf_stimulus).fit(
    data=response_valid[is_v1_to_v4],
    parameters=prf_grid_params,
    slope_name="amplitude",
    intercept_name="baseline",
    batch_size=200,
)

prf_r_squared = calculate_r_squared(
    response_valid[is_v1_to_v4], np.asarray(batched(prf_model)(prf_stimulus, prf_params, batch_size=200)),
)
```

Third, we look up the pRF of the V1 vertex at the CF center of every target vertex. Each hemisphere represents the opposite half of the visual field, so the polar angles of the right hemisphere lie around $\pm \pi$, where they wrap around. To avoid this, we mirror the left half of the visual field onto the right half before computing the polar angle. It then runs from $-\pi/2$ (lower vertical meridian) to $\pi/2$ (upper vertical meridian). We only compare target vertices with a good CF fit for which the pRF model also fits well to the vertex and its CF center.

```python
def fill_v1_to_v4(values: np.ndarray) -> np.ndarray:
    """Fill values for the vertices in V1 to V4 into an array with one value per valid vertex; NaN for all others."""
    values_valid = np.full(len(response_valid), np.nan)
    values_valid[is_v1_to_v4] = values
    return values_valid


mu_x_prf = prf_params["mu_x"].to_numpy()
mu_y_prf = prf_params["mu_y"].to_numpy()

prf_eccentricity = fill_v1_to_v4(calculate_eccentricity(mu_x_prf, mu_y_prf))
prf_polar_angle = fill_v1_to_v4(calculate_polar_angle(np.abs(mu_x_prf), mu_y_prf))  # Mirrored visual field
prf_r_squared_valid = fill_v1_to_v4(prf_r_squared)

# Row of response_valid of the V1 vertex at the CF center of every valid vertex (all V1 vertices are valid)
v1_rows = np.flatnonzero(is_v1[response_is_valid])
cf_center_rows = v1_rows[center_index]

prf_r_squared_threshold = 0.3

is_compared = (
    is_selected
    & np.isin(visual_area_valid, ["V2", "V3", "V4"])
    & (prf_r_squared_valid > prf_r_squared_threshold)
    & (prf_r_squared_valid[cf_center_rows] > prf_r_squared_threshold)
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")

for ax, values, label in zip(
    axes, [prf_eccentricity, prf_polar_angle], ["Eccentricity (in degrees)", "Polar angle (in radians)"],
):
    for area in ["V2", "V3", "V4"]:
        is_area = is_compared & (visual_area_valid == area)
        correlation = np.corrcoef(values[cf_center_rows][is_area], values[is_area])[0, 1]
        ax.scatter(
            values[cf_center_rows][is_area], values[is_area], s=3, alpha=0.3, label=f"{area} (r = {correlation:.2f})",
        )
    ax.legend(frameon=False, markerscale=4)
    ax.set(xlabel=f"{label} of the CF center in V1", ylabel=f"{label} of the target vertex")
```

Yes. The pRF position associated with the CF center agrees well with the pRF position of the target vertex itself. The correlations are very high for eccentricity (about 0.98 in V2 and V3 and 0.91 in V4) and somewhat lower for polar angle (about 0.9 in V2, 0.8 in V3, and 0.7 in V4). Like Haak et al. (2013), we find that the CF model recovers the retinotopic organization of the target areas from the V1 responses alone, without any knowledge of the stimulus. The correlations decrease along the visual hierarchy because the larger CFs and pRFs of higher-order areas are less precisely localized.

## Exercise 1

A neuron population that prefers numerosity 2 responds most strongly when two dots are shown. This happens twice per cycle: in the ascending sequence (time frames 2 and 3) and in the descending sequence (time frames 32 and 33). Depending on how broadly it is tuned, it also responds to the neighboring numerosities 1 and 3, but hardly to the numerosities 5 to 7 and 20. The BOLD response follows the neuron activity with a delay of a few seconds and smooths it. Thus, we expect two broad peaks about two time frames (4 seconds) after these frames, one early and one late in the cycle. In between, during the higher numerosities and the numerosity-20 block in the middle of the cycle, the response stays low.

A population that prefers numerosity 6 responds most strongly to the time frames 10 and 11 and the time frames 24 and 25. Its two peaks lie close to the middle of the cycle, on both sides of the numerosity-20 block (time frames 14 to 21), with a dip in between. Its response is low at the beginning and the end of the cycle, where a population that prefers numerosity 2 responds. Thus, the preferred numerosity determines *when* the response peaks within the cycle.

The baseline should be a stimulus that numerosity-tuned populations hardly respond to. A blank screen would fulfill this, but it also differs from the dot displays in its other visual properties: Switching between a blank screen and dots changes, for example, the contrast and luminance of the display. This strongly drives the visual cortex (and possibly attention), and these responses would mix with the responses to numerosity. With numerosity 20, the screen always shows dots, so the change between the sequences and the baseline is mostly a change in numerosity. On the log scale, 20 lies far from the numerosities 1 to 7 ($\log(20) \approx 3.0$ vs. $\log(7) \approx 1.9$), so populations that prefer one of these numerosities respond only weakly to it. In addition, the eight time frames of the baseline give the BOLD response time to return towards baseline between the ascending and the descending sequence, which makes their responses easier to distinguish.

## Exercise 2

The response of a population that is tuned to the total area of the dots depends on how the total area changes with numerosity in each condition:

- **Area:** The total area is the same for all numerosities, so the response does not change with numerosity. The timecourse is flat (apart from noise), and the pRF model cannot fit it.
- **Size:** Every dot has the same size, so the total area increases in proportion to numerosity. The response changes with numerosity, and the population appears to be tuned to the numerosity at which the dots cover its preferred area.
- **Dens:** The dots are the same as in the Area condition, so the total area is again constant and the response is flat. Only the density of the dots differs from the Area condition.
- **Circ:** The total circumference of $n$ dots with radius $r$ is $n \cdot 2 \pi r$. To keep it constant, the radius must decrease in proportion to $1/n$, so the total area $n \pi r^2$ also decreases in proportion to $1/n$. The total area thus changes in the opposite direction compared to the Size condition: A population that prefers a large area appears to prefer high numerosities in the Size condition but low numerosities in the Circ condition.

Thus, an area-tuned population only appears to be tuned to numerosity in some conditions, and its preferred numerosity differs between them. In contrast, a population that is tuned to numerosity has the same preferred numerosity in all four conditions. The same reasoning applies to the other visual properties: The conditions change each property in a different way, so a population that is tuned to one of them has a different preferred numerosity in different conditions.

## Exercise 3

```python
N_DCT_NEW = 1.0

# Cosine basis with a constant, half a cycle, and a full cycle
frequencies_new = np.arange(0, N_DCT_NEW + 0.25, 0.5)
trends_new = np.cos(np.outer(time, frequencies_new))

# Fit the trends to each timecourse and subtract them
beta_new = np.linalg.pinv(trends_new) @ average_response_psc.T
average_response_detrended_new = average_response_psc - (trends_new @ beta_new).T

fig, ax = plt.subplots(figsize=(12, 3.8), layout="constrained")

ax.axhline(0.0, color="#bdbdbd", linewidth=1)
ax.plot(time_frames, average_response_detrended[voxel_index], color=BLUE, linewidth=2, label="N_DCT = 0.5")
ax.plot(time_frames, average_response_detrended_new[voxel_index], color=ORANGE, linewidth=2, label="N_DCT = 1.0")
add_legend(ax)
ax.set_title(f"Voxel {voxel_index}: BOLD response after detrending", loc="left")
style_axes(ax, "Time frame (in TR)", "BOLD response (in PSC)")

plt.show()

# Fraction of the variance that the full cosine removes in addition
1 - average_response_detrended_new[voxel_index].var() / average_response_detrended[voxel_index].var()
```

The responses are averaged across the four cycles. A slow drift across the run (e.g., a steady decrease of the signal) therefore appears as a trend that rises or falls steadily within the averaged cycle. The constant and the half cosine, which falls steadily from the beginning to the end of the timecourse, are enough to remove such a trend.

With `N_DCT = 1.0`, the basis also contains a full cosine, which is high at the beginning and end of the timecourse and low in the middle. For voxel 100, it removes another 11% of the variance: The response at the beginning and end of the cycle is lowered and the response in the middle is raised.

The problem is that the response to the stimulus has a similar shape. Each cycle contains an ascending and a descending sequence, so the response has two peaks per cycle (a frequency of about two cycles per timecourse). Populations that prefer low numerosities respond at the beginning and end of the cycle, and populations that prefer high numerosities in the middle (Exercise 1). The difference between the two looks like a full cosine. Thus, the full cosine removes part of the response to the stimulus, specifically the part that distinguishes low from high preferred numerosities. In model predictions with `sigma = 0.5`, it removes about 10% of the variance for the preferred numerosities 1 and 7 (but almost nothing for 3). A basis that contains the frequency of the response itself (`N_DCT = 2.0`) removes up to two thirds of the predicted response.

Because we detrend the predictions in the same way, the model is still compared against the data in a consistent way. However, the removed part of the response is lost: The remaining response is smaller compared to the noise, and the responses of populations with different preferred numerosities become more similar, so the estimates become less precise. Therefore, the detrending basis should only contain frequencies that are clearly lower than the frequency of the response to the stimulus.

## Exercise 4

The timecourses in the left NumTO show several patterns:

- **Two peaks at the beginning and end of the cycle** (e.g., vertex 1248): The response rises after the low numerosities of the ascending sequence, drops in the middle of the cycle, and rises again after the low numerosities of the descending sequence. These vertices prefer low numerosities (vertex 1248: about 2).
- **Two peaks close to the middle of the cycle** (e.g., vertex 832): The response peaks after the higher numerosities, on both sides of the numerosity-20 block. These vertices prefer intermediate to high numerosities (vertex 832: about 5).
- **A response that increases with numerosity** (e.g., vertices 0 and 166): The response is lowest shortly after numerosity 1 is shown (around time frames 3 and 36) and highest around the numerosity-20 block. These vertices respond more strongly the more dots are shown, so the model estimates a preferred numerosity above 7 for them (vertex 0: about 11, vertex 166: about 13). About a quarter of the well-fitting vertices in the left NumTO show this pattern. NumTO lies close to visual areas, so such responses could also reflect visual properties of the dots that change with numerosity.
- **Weak or flat responses** that mostly contain noise.

The first two patterns match our expectations from Exercise 1: The response has two peaks per cycle, one for the ascending and one for the descending sequence, and their timing depends on the preferred numerosity. Responses that increase across the whole range of numerosities are not tuned to one of the numerosities 1 to 7.

Neighboring vertices on the slider often have similar timecourses because they lie close to each other on the surface and can be assigned to the same voxels.

## Exercise 5

The timecourses in the left NumF are more similar to each other than in the left NumTO: The median correlation between the timecourse of a vertex and the average timecourse of its map is 0.81 in the left NumF, compared to 0.54 in the left NumTO. Most timecourses show two peaks at the beginning and end of the cycle and a dip in the middle around the numerosity-20 block (e.g., vertices 354 and 450). Thus, most vertices in the left NumF prefer low numerosities (about 2.5 on average, see the table in the section on interpreting the model parameters). Almost none of them have a response that increases with numerosity like vertices 0 and 166 in the left NumTO. The other maps also contain many vertices with clear peaks, but NumTO in both hemispheres stands out with its many vertices whose response increases with numerosity.

## Exercise 6

The preferred numerosity determines *when* the predicted response peaks. Every peak follows the time frames that show the preferred numerosity by a few seconds because of the delay of the BOLD response. As `mu` increases, the two peaks move from the beginning and end of the cycle towards the middle:

- With `mu = 0` (numerosity 1), the peaks lie at about time frames 3 and 37, shortly after numerosity 1 is shown in the ascending (time frames 0 and 1) and descending sequence (time frames 34 and 35).
- With `mu = 1.95` (numerosity 7), they lie at about time frames 13 and 26, on both sides of the numerosity-20 block in the middle of the cycle.

The two peaks move in opposite directions because the descending sequence mirrors the ascending one. Each numerosity is shown for two time frames, so the peaks move by two time frames from one numerosity to the next. However, the numerosities are not evenly spaced on the log scale: $\log(2) - \log(1) \approx 0.69$, but $\log(7) - \log(6) \approx 0.15$. Thus, the peaks move slowly when you change `mu` at low numerosities and quickly at high numerosities.

Beyond numerosity 7 (`mu` above about 1.95), the population responds more and more to numerosity 20. The two peaks merge in the middle of the cycle, and with `mu = 3.0` (numerosity 20), the response peaks after the numerosity-20 blocks and drops during the sequences, which is the opposite of a population that prefers low numerosities.

## Exercise 7

- **Small `sigma` (0.2):** The pRF only covers a single numerosity of the stimulus. With `mu = 1.1` (numerosity 3), the tuning is 1 at numerosity 3, 0.36 at 4, and 0.13 at 2, and close to zero for all other numerosities. Thus, the predicted response has two narrow peaks after the time frames that show numerosity 3, and it is flat in between. The peaks are also higher than for `sigma = 0.5` because the tuning profile is a Gaussian density, which integrates to one: A narrower profile has a higher peak (as in the first practical). Once `sigma` is smaller than the spacing between the numerosities, making it even smaller only scales the response, but no longer changes its shape. The `amplitude` parameter absorbs this scaling, so the data cannot tell such small values of `sigma` apart.
- **Large `sigma` (2.0):** The tuning is almost flat across the numerosities 1 to 7 (between 0.86 and 1 relative to its peak) and only somewhat lower for numerosity 20 (0.64). The population responds almost equally to all numerosities in the sequences, so the response mainly reflects the switch between the sequences and the numerosity-20 blocks: It is high during the sequences and drops after the switch to numerosity 20. The response is also much smaller (the difference between its highest and lowest value is about 0.13 compared to 1.2 for `sigma = 0.5`). Because the shape of the response hardly depends on `mu` anymore, the preferred numerosity of such broad pRFs cannot be identified. This is why we exclude vertices with a `sigma` above 2.8 when we interpret the parameters.

## Exercise 8

- **Preferred numerosity:** The maps in the occipital cortex (NumTO and NumPO) prefer higher numerosities on average (about 3.5 to 4.6) than the maps in the parietal cortex, NumPC1 and NumPC2 (about 2.5 to 3.5), and the frontal map NumF (about 2.5 to 3.3). NumPC3 is an exception, especially in the right hemisphere, where its average (about 4.7) is as high as in NumTO. In every map, the average preferred numerosity is higher in the right than in the left hemisphere (by about 0.6 to 1.3).
- **pRF size:** The average `sigma` is similar in all maps (about 0.8 to 1.1 in log numerosity), with the smallest averages in the left NumF and the right NumPC3. A `sigma` of 1 is quite broad: A population that prefers numerosity 3 responds at half its maximum to numerosities of about 1 and 10.

The standard deviations within the maps (about 0.9 to 1.8 for the preferred numerosity and 0.3 to 0.6 for `sigma`) are about as large as the differences between the maps. This is expected because each map is topographic and covers a range of preferred numerosities. Thus, the average of a map also depends on which part of the map contains well-fitting vertices. In addition, the averages of small maps, such as the right NumPO with only about 100 selected vertices, are less reliable than those of the large maps, such as the right NumPC1 with more than 1,000 vertices. The differences between the maps should therefore be seen as tendencies of a single subject.

## Exercise 9

```python
plot_numerosity_flatmaps(
    sgd_params=sgd_params,
    r_squared=r_squared,
    valid_vertex_indices=valid_vertex_indices,
    roi_ids=dataset.vertex_roi_index,
    roi_mapping=roi_mapping,
    surf_dir=surf_dir,
    lh_roi_id=1,  # lh_NumTO
    rh_roi_id=7,  # rh_NumTO
    r2_threshold=0.2,
)
```

NumTO lies in the lower part of both flatmaps (in the temporal-occipital cortex) and is small compared to the entire cortex, so you have to look closely. Within the outline in both hemispheres, the preferred numerosity changes gradually from low numerosities (yellow and green, about 2 to 3) on one side of the map to high numerosities (blue and purple, about 5 to 7) on the other side, rather than varying randomly from vertex to vertex. Thus, neighboring vertices prefer similar numerosities, which is what makes NumTO a topographic map. The map as a whole prefers higher numerosities than NumPC1, which mostly shows red to yellow (about 1 to 3), especially in the left hemisphere (see also Exercise 8).

Note that many vertices outside the outlined maps also pass the selection criteria. Some of them form other numerosity maps, for example, the large patches in the parietal cortex, but others are scattered and change their preferred numerosity from vertex to vertex. A gradient is only convincing when it covers a contiguous part of the cortex. We quantify the gradient within NumTO with the geodesic distance in the next section.

## Exercise 10

In the Average condition, the preferred numerosity increases from about 2.7 to 3 at the low end of the right NumTO (0 to 8 mm) to about 5 to 6 at the high end (from about 16 mm on), with a correlation of 0.67 between distance and preferred numerosity. The logarithmic function describes this increase well: It rises more steeply at the low end of the map than at the high end. Between about 8 and 16 mm, only a few vertices are selected (2 to 12 per bin), so the means of these bins are noisy.

The four stimulus conditions show the same increase: The preferred numerosity is low at the low end of the map and high at the high end in every condition. For individual vertices, the preferred numerosities in the four conditions correlate strongly with those in the Average condition (between 0.74 and 0.87 on the log scale). There are differences in the details: In the first half of the map, the preferred numerosity is higher in the Area condition (about 4 to 6 between 4 and 14 mm) and lower in the Dens condition (about 2 to 3 up to about 24 mm) than in the other conditions. Note that the Average condition is the average of the responses to the four conditions, so it is not independent of them.

In Exercise 2, we saw that a population that is tuned to the total area of the dots would not be tuned at all in the Area and Dens conditions and would have opposite preferred numerosities in the Size and Circ conditions. Instead, the responses are tuned in all four conditions, and the preferred numerosities follow the same gradient along the map. Because the conditions keep different visual properties constant, this gradient cannot be explained by dot size, total area, density, or circumference alone. Thus, the results support that the right NumTO represents numerosity rather than these visual properties. The differences between the conditions (e.g., in the Dens condition) suggest that the visual properties of the dots still have a small effect on the estimates.

## Exercise 11

```python
roi_index_new = 3  # lh_NumPC1

condition_results_new = {}

for condition in conditions:
    condition_results_new[condition] = fit_condition_roi(
        dataset=dataset,
        condition=condition,
        prf_model=prf_model,
        stimulus=stimulus,
        param_ranges=grid_param_ranges,
        convert_psc=convert_psc,
        roi_index=roi_index_new,
    )

plot_data_new = prepare_distance_data(
    dataset=dataset,
    sgd_params=sgd_params,
    r_squared=r_squared,
    vertex_valid=vertex_valid,
    roi_index_valid=roi_index_valid,
    condition_results=condition_results_new,
    roi_index=roi_index_new,
    r2_threshold=0.3,
    num_range=(1, 7),
    sigma_max=2.8,
)

distance_results_new = fit_distance_numerosity(
    plot_data=plot_data_new,
    bin_width=2.0,
    n_bootstrap=1000,
    random_state=22,
)

plot_distance_numerosity(
    binned_data=distance_results_new["binned_data"],
    fit_params=distance_results_new["fit_params"],
    x_fit=distance_results_new["x_fit"],
    lower=distance_results_new["lower"],
    upper=distance_results_new["upper"],
    roi_name=roi_mapping[roi_index_new],
);
```

In the left NumPC1, the preferred numerosity also increases along the map, but much less than in the right NumTO: In the Average condition, it rises from about 1.6 at the low end to about 2 to 2.7 in most of the map and to about 3 at the high end (correlation of 0.27 between distance and preferred numerosity). Most vertices prefer low numerosities, so the map covers only a small part of the range of the stimulus. The logarithmic function does not describe this pattern well: It rises steeply within the first few millimeters, driven by the first bin, and is almost flat afterwards.

The stimulus conditions agree less than in the right NumTO. The Size and Dens conditions show a similar slow increase as the Average condition, and the Area condition is similar except for a few bins with higher values close to the low end. However, in the Circ condition, the preferred numerosity is higher throughout the map (about 3 to 5) and correlates only weakly with the Average condition for individual vertices (0.28 compared to about 0.75 for the other conditions). Thus, the evidence for a numerosity gradient that is independent of the visual properties of the dots is weaker in this map. With a single subject, we cannot tell whether this reflects a real difference between the maps or noise in the estimates.

# Workshop materials

This section contains the materials for the hands-on sessions of the workshop.
Pages marked with an **Open in Colab** badge are Jupyter notebooks: click the
badge to run them in your browser on [Google Colab](https://colab.research.google.com),
or download the notebook to run it locally.

The diagram below shows the order in which the practicals can be followed.
Start with the first practical, which introduces the prfmodel package. An arrow
points from a practical to the practicals that build on it. The dashed arrow
means that only the last part of the fifth practical needs the third practical
(and thus also the second practical). Click on a practical to open it.

```{mermaid}
flowchart TD
    p1["1: Gaussian 2D pRF model"]
    p2["2: Cross-validation"]
    p3["3: Model comparison and selection"]
    p4["4: Gaussian connective field model"]
    p5["5: pRF models for ECoG data"]

    p1 --> p2 --> p3 -. last part .-> p5
    p1 --> p5
    p1 --> p4

    click p1 href "01-prf-visual-fmri.html"
    click p2 href "02-prf-visual-fmri-cv.html"
    click p3 href "03-prf-visual-fmri-comparison.html"
    click p4 href "04-cf-visual-fmri.html"
    click p5 href "05-prf-visual-ecog.html"
```

```{toctree}
:maxdepth: 1

01-prf-visual-fmri
02-prf-visual-fmri-cv
03-prf-visual-fmri-comparison
04-cf-visual-fmri
05-prf-visual-ecog
```

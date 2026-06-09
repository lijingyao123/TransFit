# Model Citation Guide

If you use TransFit in a paper, cite the TransFit software paper and the
model-specific method paper(s) relevant to the model family you used.

This file gives the default recommendation for TransFit's public model names.
It is not intended to replace a complete bibliography for a scientific paper.

## Software Citation

Always cite the TransFit paper:

```bibtex
@ARTICLE{2025ApJ...992...20L,
       author = {{Liu}, Liang-Duan and {Zhang}, Yu-Hao and {Yu}, Yun-Wei and {Du}, Ze-Xin and {Li}, Jing-Yao and {Wu}, Guang-Lei and {Dai}, Zi-Gao},
        title = "{TransFit: An Efficient Framework for Transient Light-curve Fitting with Time-dependent Radiative Diffusion}",
      journal = {\apj},
         year = 2025,
       volume = {992},
       number = {1},
          eid = {20},
        pages = {20},
          doi = {10.3847/1538-4357/adfed6},
archivePrefix = {arXiv},
       eprint = {2505.13825}
}
```

## Model-specific Citations

| TransFit model | Also cite |
|---|---|
| `nickel` | [Arnett 1982, ApJ, 253, 785](https://doi.org/10.1086/159681) |
| `magnetar` | [Kasen & Bildsten 2010, ApJ, 717, 245](https://doi.org/10.1088/0004-637X/717/1/245) |
| `magnetar_ni` | [Arnett 1982](https://doi.org/10.1086/159681) and [Kasen & Bildsten 2010](https://doi.org/10.1088/0004-637X/717/1/245) |
| `csm` | [Chatzopoulos, Wheeler & Vinko 2012, ApJ, 746, 121](https://doi.org/10.1088/0004-637X/746/2/121) |

## BibTeX for Model Papers

```bibtex
@ARTICLE{1982ApJ...253..785A,
       author = {{Arnett}, W. David},
        title = "{Type I supernovae. I - Analytic solutions for the early part of the light curve}",
      journal = {\apj},
         year = 1982,
       volume = {253},
        pages = {785-797},
          doi = {10.1086/159681}
}

@ARTICLE{2010ApJ...717..245K,
       author = {{Kasen}, Daniel and {Bildsten}, Lars},
        title = "{Supernova Light Curves Powered by Young Magnetars}",
      journal = {\apj},
         year = 2010,
       volume = {717},
        pages = {245-249},
          doi = {10.1088/0004-637X/717/1/245}
}

@ARTICLE{2012ApJ...746..121C,
       author = {{Chatzopoulos}, Emmanouil and {Wheeler}, J. Craig and {Vinko}, Jozsef},
        title = "{Generalized Semi-analytical Models of Supernova Light Curves}",
      journal = {\apj},
         year = 2012,
       volume = {746},
          eid = {121},
        pages = {121},
          doi = {10.1088/0004-637X/746/2/121}
}
```

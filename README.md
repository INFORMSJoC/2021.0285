[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# PyNumero and Parapint

This archive is distributed in association with the [INFORMS Journal
on Computing](https://pubsonline.informs.org/journal/ijoc) under the
[BSD 3-Clause License](LICENSE.md).

The software and data in this repository are a snapshot of the
software and data that were used in the research reported on in the
paper [Scalable Parallel Nonlinear Optimization with PyNumero and
Parapint](TBD) by J. Rodriguez, R. Parker, C. Laird, B. Nicholson,
J. Siirola, and M. Bynum.  Note that PyNumero is a module within
Pyomo. The snapshot is based on [this
SHA](https://github.com/Pyomo/pyomo/commit/a415dbfe3e1dfe343e7f829b6219a0e0b7fa8f0f)
for PyNumero and [this
SHA](https://github.com/Parapint/parapint/commit/6fcce1642a72faab54ad81cfe74da0cd57256c7b)
for Parapint in their respective development repositories.

**Important: This code is being developed on an on-going basis at
https://github.com/Pyomo/pyomo/tree/main/pyomo/contrib/pynumero and at
https://github.com/Parapint/parapint. Please go there if you would like to
get a more recent version or would like support.**

## Cite

To cite this software, please cite the [paper](TBD) using its DOI and the software itself, using one of the following DOIs.

PyNumero: https://doi.org/10.11578/dc.20201001.29

Parapint: https://doi.org/10.11578/dc.20201109.2

Below is the BibTex for citing PyNumero.

```
@misc{pyomo,
title = {Pyomo v6.0},
author = {Woodruff, David and Hackebeil, Gabe and Laird, Carl D. and Nicholson, Bethany L. and Hart, William E. and Siirola, John D. and Watson, Jean-Paul},
doi = {10.11578/dc.20201001.29},
url = {https://doi.org/10.11578/dc.20201001.29},
howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20201001.29}},
year = {2017},
month = {may}
}
```

Below is the BibTex for citing Parapint.

```
@misc{parapint,
title = {Parapint, Version 0.4.0},
author = {Bynum, Michael and Laird, Carl and Nicholson, Bethany and Rizdal, Denis},
doi = {10.11578/dc.20201109.2},
url = {https://doi.org/10.11578/dc.20201109.2},
howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20201109.2}},
year = {2020},
month = {sep}
}
```

## Description

PyNumero is a package for developing parallel algorithms for nonlinear
programs (NLPs). Documentation can be found at
https://pyomo.readthedocs.io/en/stable/contributed_packages/pynumero/index.html.

Parapint is a Python package for parallel solution of structured
nonlinear programs. Documentation can be
found at https://parapint.readthedocs.io/en/latest/.


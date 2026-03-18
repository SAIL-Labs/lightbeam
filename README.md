# lightbeam
Simulate light through weakly guiding waveguides using the finite-differences beam propagation method on an adaptive grid.

This repository is derived from:

https://github.com/jw-lin/lightbeam.git

The original implementation was developed by Jonathan Lin at MIT.

Modifications are:
* Code generally neatened
* 19-core photonic lantern tutorial `Lightbeam_19cpl_tutorial.ipynb` added to `tutorial`
* 19-core photonic latnern example script `lightbeam_19cpl_example.py` added to `tutorial`
* Output field `u` weights returned from `prop2end`

## installation
Use pip: 

```
pip install git+https://github.com/SAIL-Labs/lightbeam.git
```

Update:

```
pip install --force-reinstall git+https://github.com/SAIL-Labs/lightbeam.git
```

Python dependencies: `numpy`,`scipy`,`matplotlib`,`numba`,`numexpr`,`jupyter`

## getting started and further resources
Check out the Python notebook in the `tutorial` folder for a quickstart guide. <a href="tutorial/Lightbeam.ipynb">Direct link.</a>

A second Python notebook in the `tutorial` folder describes how to propagate 17 excited LP modes through a 19-core photonic latnern. <a href="tutorial/Lightbeam_pl19c_tutorial.ipynb">Direct link.</a>

Further, a report is included in the root of the repository, which validates lightbeam against RSoft, and provides a meshing convergence and a computational temporal analysis, using a 19-core photonic lantern as the base optical object. <a href="lightbeam_report_pl19c.pdf">Direct link.</a>

## references
J. Shibayama, K. Matsubara, M. Sekiguchi, J. Yamauchi and H. Nakano, "Efficient nonuniform schemes for paraxial and wide-angle finite-difference beam propagation methods," in Journal of Lightwave Technology, vol. 17, no. 4, pp. 677-683, April 1999, <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=754799">doi: 10.1109/50.754799.</a> 

The Beam-Propagation Method. (2015). In Beam Propagation Method for Design of Optical Waveguide Devices (pp. 22–70). <a href="https://onlinelibrary.wiley.com/doi/book/10.1002/9781119083405">  doi:10.1002/9781119083405.ch2</a>

## J. Lin acknowledgments
NSF grants 2109231, 2109232, 2308360, 2308361

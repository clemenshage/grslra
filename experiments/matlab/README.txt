In order to compare GRPCA to related methods, their implementations need to be downloaded from the respective authors sites and copied into this directory

LMaFit & IALM from http://lmafit.blogs.rice.edu/, folder named 'LMaFit-SMS.v1'
GoDec from https://sites.google.com/site/godecomposition/code, folder named 'GoDec'
GRASTA from https://sites.google.com/site/hejunzz/grasta, folder named 'grasta.1.2.0'
RMC from https://people.stanford.edu/lcambier/rmc, folder named 'RMC_1.1', MANOPT toolbox is available at http://www.manopt.org, folder named 'manopt'


Whenever possible, the parameterization provided by the authors has been used. Sometimes the parameters have been adjusted to better fit the respective task.

GoDec: iter_max=2e+2; error_bound=1e-8;
IALM: The code for IALM is provided with the LMaFit code
GRASTA: The wrapper 'grasta.m' has been written , which is in the style of the grasta_RobustMC_demo.m and uses the same parameters. As the mex file could not successfully be compiled, OPTIONS.USE_MEX is set to 0.
RMC: The wrapper 'rmc_wrapper.m' has been written, which generates the required index set of coordinates and provides it to the actual function.

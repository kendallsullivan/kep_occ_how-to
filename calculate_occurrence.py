# import statements
import numpy as np
from glob import glob
import os, sys, getopt
import multiprocessing as mp
import subprocess


# initialize the various keyword arguments the different pieces of the code require

# for all scripts
plots = True
verbose = True
savepath = os.getcwd()
spt = 'GK'

# For create_stellar_catalogs:
young_age = 0.1 #Gyr
old_age = 19.5 #Gyr

# for the vetting completeness
vettingModel = 'logisticX0xRotatedLogisticY02'
pmin = 1 # days
pmax = 400 # days 
rpmin = 1 # Rearth
rpmax = 15 # Rearth
vetNwalkers = 200
vetNsteps = 5000

# for the composite completeness
ncores = os.cpu_count()

# for FP effectiveness
fpEffModel = 'rotatedLogisticX0'
FPNwalkers = 100
FPNsteps = 10000
FPNfits = 100

# calculate FP rate
fpRateModel = 'rotatedLogisticX0'

# run the MCMC with KDE
prefix = 'sboot_'
catName = os.getcwd() + "/GKbaseline/koiCatalogs/dr25_{}_PCs.csv".format(spt) # should be the full path to the file
period_gridsize = 101
rp_gridsize = 101
niter = 1000


#create the stellar catalog
# stellar_cat_options = 'i:v:s:t:y:o:' #plots y/n, verbose y/n, savepath for files, spectral types to include, lower age limit, upper age limit
#os.system('python createStellarCatalogs.py -i {} -v {} -s {} -t {} -y {} -o {}'.format(plots, verbose, savepath, spt, young_age, old_age))


# calculate the vetting completeness
# vetting_completeness_options = 'm:i:v:s:t:p:P:r:R:n:N:' #model name, plots y/n, verbose y/n, savepath for files, spectral types to include, period min, period max, rp min, rp max, nwalkers, nsteps
#os.system('cd GKbaseline')
#os.system('python GKbaseline/binomialVettingCompleteness.py -m {} -i {} -v {} -s {} -t {} -p {} -P {} -r {} -R {} -n {} -N {}'.format(vettingModel, plots, verbose, savepath, spt, pmin, pmax, rpmin, rpmax, vetNwalkers, vetNsteps))
#os.system('cd ..')


#calculate the composite completeness
#completeness_options = 'm:s:t:p:P:r:R:c:'# model:savepath:spt:pmin:pmax:rpmin:rpmax:ncores
# os.system('python calc_composite_completeness.py - {} -s {} -t {} -p {} -P {} -r {} -R {} -c {}'.format(vettingModel, savepath, spt, pmin, pmax, rpmin, rpmax, ncores))


# calculate the FP effectiveness
#FP_eff_options = 'm:i:v:s:t:p:P:r:R:n:N:f:' #model name, plots y/n, verbose y/n, savepath for files, spectral types to include, period min, period max, rp min, rp max, nwalkers, nsteps, nfits
# os.system('cd GKbaseline')
# os.system('python GKbaseline/binomialFPEffectiveness.py -m {} -i {} -v {} -s {} -t {} -p {} -P {} -r {} -R {} -n {} -N {} -f {}'.format(fpEffModel, plots, verbose, savepath, spt, pmin,pmax, rpmin, rpmax, FPNwalkers, FPNsteps, FPNfits))


# calculate the FP rate
#FP_rate_options = 'm:i:v:s:t:p:P:r:R:n:N:f:' #model name, plots y/n, verbose y/n, savepath for files, spectral types to include, period min, period max, rp min, rp max, nwalkers, nsteps, nfits
#os.system('python GKbaseline/binomialObsFPRate.py -m {} -i {} -v {} -s {} -t {} -p {} -P {} -r {} -R {} -n {} -N {} -f {}'.format(fpRateModel, plots, verbose, savepath, spt, pmin, pmax, rpmin, rpmax, FPNwalkers, FPNsteps, FPNfits))


# make the planet catalog
# planet_options = 'E:F:i:v:t:p:P:r:R:' #model name for FP Eff, model name for FP Rate, plots y/n, verbose y/n, spt range, pmin, pmax, rpmin, rpmax
os.system('python GKbaseline/makePlanetInput.py -E {} -F {} -i {} -v {} -s {} -p {} -P {} -r {} -R {}'.format(fpEffModel, fpRateModel, plots, verbose, spt, pmin, pmax, rpmin, rpmax))

# os.system('cd ..')
# Do the KDE MCMC
'''
command-line order for sboot
output_prefix = sys.argv[1]
pcCatalog = sys.argv[2] #'koiCatalogs/dr25_FGK_PCs_B20_ruwe.csv'
pmin, pmax, period_grid = sys.argv[3], sys.argv[4], sys.argv[5]
rpmin, rpmax, rp_grid = sys.argv[6], sys.argv[7], sys.argv[8]
niter = sys.argv[9]
'''

os.system('python par_sboot_shuffle.py {} {} {} {} {} {} {} {} {}'.format(prefix, catName, pmin, pmax, period_gridsize, rpmin, rpmax, rp_gridsize, niter))




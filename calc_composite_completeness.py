import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join
from astropy.io import ascii, fits
from glob import glob
import os, sys, getopt
import multiprocessing as mp
import subprocess
sys.path.insert(0, os.getcwd() + '/completenessContours')
import compute_num_completeness_mproc as mproc
from setuptools._distutils.util import strtobool


''' 
Instructions in Steve's readme for a single run:
None of it is necessary to run this script, it's just here for reference.

1) Edit "./make_contours_multiproc.txt" and change "/Volumes/yoru/dr25/dr25CompletenessFits" to the location of your Window Function and One-sigma Depth Functions downloaded above. 
You may want to set the number of threads as appropriate to your machine via the variable nworkers in this script.  Currently nworkers = 8.  
Then run "./make_contours_multiproc.txt" . This takes about 4 hours on a 12-processor machine.

2) wait until all the out*.fits files are converted to out*.fits.gz

3) run "python accumulate.py", which will create a file "out__.fits.gz".  Wait until the file "out__.fits" is deleted by the program.

4) rename "out__.fits.gz" to "out_sc0_GK_baseline.fits.gz"

5) run "python examgrid.py" to generate diagnostic plots. [I won't do this step]

You're done!  Now you can calculate reliability for planet candidates and compute the occurrence rate

'''


def caclulate_composite_completeness(vetpath, starpath, path, stellarType = 'GK', ncores = 64, pmin = 1, pmax = 400, pgrid = 2001, rpmin = 0.5, rpmax = 15, rpgrid = 3001,\
		model = 'logisticX0xRotatedLogisticY02'):

	# change to the correct directory (i.e., the completenessContours directory) from the CWD
	# subprocess.run(['cd', vetpath])

	# instantiate the parallelization
	with mp.Pool() as pool:
		#then run on however many cores using the symtax in compute_num_completeness_mproc.py
		out = [pool.apply_async(mproc.nas_multi_grid_dr25, \
			args = (k, ncores, pmin, pmax, pgrid, rpmin, rpmax, rpgrid, '{}/dr25_stellar_supp_gaia_clean_{}.txt'.format(starpath, stellarType), \
			path + 'completenessContours/' , path + '/GKbaseline/vetCompletenessTable.pkl', model, '{}/out'.format(vetpath))) for k in range(ncores)]

		# get the results
		a = [o.get() for o in out]

	# go back to the initial directory
	# subprocess.run(['cd', path])

	# return 
	return

def cleanup(vetpath, stellarType = 'gk'):

	# this is purely to clean up files (accumulate.py) and then rename them so they work with the next step.
	subprocess.run(['python', vetpath + '/accumulate.py'])
	subprocess.run(['mv', vetpath + '/out__.fits.gz', vetpath + '/out_sc0_{}_baseline.fits.gz'.format(stellarType.upper())])

	return

def main(argv):

	try:
		argument_list = argv[1:]
		short_options = 'm:s:t:p:P:r:R:c:'
		long_options = 'model:savepath:spt:pmin:pmax:rpmin:rpmax:ncores:' 
		arguments, values = getopt.getopt(argument_list, short_options, long_options)

		model = arguments[0][1]
		path = arguments[1][1] + '/'
		spt = arguments[2][1].upper()
		pmin = float(arguments[3][1])
		pmax = float(arguments[4][1])
		rpmin = float(arguments[5][1])
		rpmax = float(arguments[6][1])
		ncores = int(arguments[7][1])

	except:
		print('No inputs given, running with default settings:\
		 model = logisticX0xRotatedLogisticY02, savepath = current working directory,\
		 spt = \'GK\', min. period = 1 d, max period = 400 d, \
		 min Rp = 0.5 Re, max Rp = 15 Re, ncores = all available cores. You can check how many cores you have available by importing python os, then typing\
		 os.cpu_count(). \
		 The code always uses the same gridsize: 2001 (period) x 3001 (radius).')

		model = 'logisticX0xRotatedLogisticY02'
		path = os.getcwd() + '/'
		spt = "GK"
		pmin = 1
		pmax = 400
		rpmin = 0.5
		rpmax = 15
		ncores = os.cpu_count()

	# check the spectral type inputs are valid
	test_spt = [s not in 'FGKM' for s in spt]

	# if they are not, throw an error and exit
	if any([s == True for s in test_spt]):
		print('Invalid spectral type entered! Please enter some combination of FGKM (case-insensitive). Terminating program.')
		sys.exit(1)

	if pmin > pmax:
		print('Minimum period must be less than maximum period. Terminating program.')
		sys.exit(1)

	if rpmin > rpmax:
		print('Minimum planet radius must be less than maximum planet radius. Terminating program.')
		sys.exit(1)

	pgrid = 2001
	rpgrid = 3001
 
	starpath = path + "stellarCatalogs" 
	vetpath = path + "completenessContours"


	print('Calculating composite completeness using the following parameters: model = {}, path = {}, spt = {}, pmin = {}, pmax = {}, rpmin = {}, rpmax = {}, ncores = {}.'.format\
		(model, path, spt, pmin, pmax, rpmin, rpmax, ncores))
	#now we have the required files, calculate the completeness contours
	caclulate_composite_completeness(vetpath, starpath, path, spt, ncores, pmin, pmax, pgrid, rpmin, rpmax, rpgrid, model)

	print('Cleaning up files. This might take a while.')
	cleanup(vetpath, spt)

if __name__ == '__main__':
	main(sys.argv)

import numpy as np
from astropy.table import Table
import os


keptable = Table.read('../../stellarCatalogs/dr25_stellar_supp_gaia_clean_FGK.txt', format = 'ascii.csv')

kicnumbers = keptable['kepid']

for n, kic in enumerate(kicnumbers):
	kic = str(kic).zfill(9)
	kepstring1 = str(kic)[:3]
	kepstring2 = str(kic)[:6]

	os.system('wget -O kplr{}_dr25_onesigdepth.fits http://exoplanetarchive.ipac.caltech.edu:80/data/KeplerData/{}/{}/{}/tps/kplr{}_dr25_onesigdepth.fits -a PDM.log'.format(kic, kepstring1, kepstring2, kic, kic))
	os.system('wget -O kplr{}_dr25_window.fits http://exoplanetarchive.ipac.caltech.edu:80/data/KeplerData/{}/{}/{}/tps/kplr{}_dr25_window.fits -a PDM.log'.format(kic, kepstring1, kepstring2, kic, kic))

	print(n, 'of', len(kicnumbers))
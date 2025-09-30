
'''
Make a stellar catalog for computing occurrence rates. Modified from Steve Bryson's Kepler DR25 tutorial jupyter notebooks (https://github.com/stevepur/DR25-occurrence-tutorial)
by Kendall Sullivan (find my contact info at kendallsullivan.github.io).
'''

import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys, getopt, os
from astropy.io import ascii

def makeCatalog(savepath, plots = True, verbose = True, spt = 'GK', age_lower = 0.1, age_upper = 19.5):

    dr25Stellar = pd.read_csv("stellarCatalogs/dr25_stellar_archive.txt", dtype={"st_quarters":str})
    dr25StellarSup = pd.read_csv("stellarCatalogs/dr25_stellar_supp_archive.txt", dtype={"st_quarters":str})


    # Merge the two catalogs.  The supplemental catalog has fewer entries, so we have to be careful about the merge. 
    #The merged catalog has the same number of entries as the supplemental catalog.
    # merging with inner gives the intersection of kepids
    mergedDr25Stellar = pd.merge(dr25Stellar, dr25StellarSup, on="kepid", how="inner")
    # get the original column names
    sColumns = list(dr25Stellar)


    # Because both catalogs share column names, mergedStellar has a column for each catalog, 
    # with the column names given an "\_x" for the stellar catalog and "\_y" for the supplemental.
    # We have to reconcile this by 
    #     a) restoring the original names, 
    #     b) copying the values in the supplemental catalog to the stellar columns for those columns specified in the supplemental documentation, and 
    #     c) removing the supplemental colums.


    suppColNames = [ "kepid","teff","teff_err1","teff_err2","logg","logg_err1","logg_err2","feh","feh_err1","feh_err2",\
    "radius","radius_err1","radius_err2","mass","mass_err1","mass_err2","dens","dens_err1","dens_err2","dist","dist_err1","dist_err2","av","av_err1","av_err2","prov_sec" ]

    # first copy all the stellar parameters to their original names
    for i in range(1,len(sColumns)):
        mergedDr25Stellar[sColumns[i]] = mergedDr25Stellar[sColumns[i]+"_x"]
    # now overwrite using the new columns (from Savita's erratum file)
    for i in range(1,len(suppColNames)):
        mergedDr25Stellar[suppColNames[i]] = mergedDr25Stellar[suppColNames[i]+"_y"]
    # drop the doubled columns
    for i in range(1,len(sColumns)):
        mergedDr25Stellar = mergedDr25Stellar.drop([sColumns[i]+"_x", sColumns[i]+"_y"], axis=1)

    if verbose == True:
        print("the merged catalog has " + str(len(mergedDr25Stellar)) + " entries")


    mergedDr25Stellar.to_csv("stellarCatalogs/dr25_stellar_updated_feh.txt", index=False)


    # Now restore the entries in the original stellar catalog that are not in the supplemental catalog.  
    #First we identify the original-only entries by repeating the merge, then entries in the merge from the stellar table.  
    #Then we concatenate the remaining entries with the merged table.  Finally we save the final full table.
    # now get the stars not in the stellar_supp by dropping stars in the merge from stellar (using reset index)
    merge2 = pd.merge(dr25Stellar.reset_index(), dr25StellarSup, on="kepid", how="inner")
    dr25StellarNoSup = dr25Stellar.drop(merge2["index"])

    # now concatenate the merged set and the non-supp stars
    dr25StellarFullMerged = pd.concat([mergedDr25Stellar, dr25StellarNoSup])

    if verbose == True:
        # print(list(stellar_full_merged))
        print("the final catalog has " + str(len(dr25StellarFullMerged)) + " entries")

    dr25StellarFullMerged.to_csv("stellarCatalogs/dr25_stellar_updated_feh_all.txt", index=False)


    # Next we bring in the Gaia-derived properties from Berger et. al. (http://adsabs.harvard.edu/abs/2018ApJ...866...99B).  
    #This provides improved radii and distances.  
    #It also includes effective temperatures that differ from the DR25 catalog, but we believe the DR25 catalog's effective temperatures are more reliable. 
    #The Berger table has 177,911 entries, and we will restrict our analysis to the stellar population in the Berger table.

    # read the published table from Berger et al. 2018
    gaiaUpdates = ascii.read("stellarCatalogs/apj_table1_published.txt")
    berger2020 = ascii.read("stellarCatalogs/berger2020.txt")
    berger2020 = berger2020['KIC', 'Age', 'f_Age', 'e_Age', 'E_Age']

    gaiaUpdatesPd = gaiaUpdates.to_pandas();
    berger2020Pd = berger2020.to_pandas();

    dr25GaiaStellar_merge1 = pd.merge(dr25StellarFullMerged, gaiaUpdatesPd, left_on="kepid", right_on="KIC", how="inner")

    dr25GaiaStellar = pd.merge(dr25GaiaStellar_merge1, berger2020Pd, left_on="kepid", right_on="KIC", how="inner")

    # copy the dr25 distance and radius to renamed columns in case anyone wants to compare
    dr25GaiaStellar["dist_DR25"] = dr25GaiaStellar["dist"]
    dr25GaiaStellar["dist_DR25_err1"] = dr25GaiaStellar["dist_err1"]
    dr25GaiaStellar["dist_DR25_err2"] = dr25GaiaStellar["dist_err2"]

    dr25GaiaStellar["radius_DR25"] = dr25GaiaStellar["radius"]
    dr25GaiaStellar["radius_DR25_err1"] = dr25GaiaStellar["radius_err1"]
    dr25GaiaStellar["radius_DR25_err2"] = dr25GaiaStellar["radius_err2"]

    # copy Gaia distance to the dr25 distance column
    dr25GaiaStellar["dist"] = dr25GaiaStellar["D"]
    # for DR25, err1 is the upper uncertainty, err2 is the lower uncertainty and is < 0
    # for Gaia, E_D is the upper uncertainty, E_d is the lower uncertainty and is > 0
    dr25GaiaStellar["dist_err1"] = dr25GaiaStellar["E_D"]
    dr25GaiaStellar["dist_err2"] = -dr25GaiaStellar["e_D"]
    # copy Gaia radius to the dr25 radius column
    dr25GaiaStellar["radius"] = dr25GaiaStellar["R*"]
    # for DR25, err1 is the upper uncertainty, err2 is the lower uncertainty and is < 0
    # for Gaia, E_R* is the upper uncertainty, e_R* is the lower uncertainty and is > 0
    dr25GaiaStellar["radius_err1"] = dr25GaiaStellar["E_R*"]
    dr25GaiaStellar["radius_err2"] = -dr25GaiaStellar["e_R*"]

    if verbose == True:
        # we'll leave the Gaia columns in place because they have different names
        print("the Gaia/Berger catalog has " + str(len(dr25GaiaStellar)) + " entries")

    dr25GaiaStellar.to_csv("stellarCatalogs/dr25_stellar_supp_gaia.txt", index=False)


    #pick out the desired age range as specified by the function call:
    dr25GaiaStellar = dr25GaiaStellar[(dr25GaiaStellar['Age'] > age_lower) & (dr25GaiaStellar['Age'] < age_upper) & (dr25GaiaStellar['f_Age'] != '*')]
    if verbose == True:
        print('In the desired age range of {}-{} Gyr, there are {} entries after the initial cross-match between Berger+2018, Berger+2020, and the KIC.'\
            .format(age_lower, age_upper, len(dr25GaiaStellar)))

    # dr25_stellar_supp_gaia.txt is our base catalog.  
    #We now apply several cuts to restrict our occurrence rate analysis to targets for which we have reliable completeness information, 
    #following the guidelines in Section 3.6 of https://exoplanetarchive.ipac.caltech.edu/docs/KSCI-19111-002.pdf.

    cleanDr25GaiaStellar = dr25GaiaStellar


    # First we remove the stars marked binary due to Gaia radius (Bin = 1 or 3) or evolved (Evol > 0). 
    #We do not remove targets with AO companions (Bin = 2) because the target star population was not uniformly surveyed with AO.  
    #We expect few evolved stars because most should have been removed by the R <= 1.25 cut.

    binaryDr25GaiaStellar = cleanDr25GaiaStellar[(cleanDr25GaiaStellar.Bin == 1)|(cleanDr25GaiaStellar.Bin == 3)]
    if verbose == True:
        print("There are " + str(len(binaryDr25GaiaStellar)) + " targets with binary flag == 1 or 3")

    evolvedDr25GaiaStellar = cleanDr25GaiaStellar[cleanDr25GaiaStellar.Evol > 0]
    if verbose == True:
        print("There are " + str(len(evolvedDr25GaiaStellar)) + " targets with evolved flag > 0")

    cleanDr25GaiaStellar = cleanDr25GaiaStellar[(cleanDr25GaiaStellar.Bin == 0)|(cleanDr25GaiaStellar.Bin == 2)]
    if verbose == True:
        print(str(len(cleanDr25GaiaStellar)) + " entries after removing marked binaries")

    cleanDr25GaiaStellar = cleanDr25GaiaStellar[cleanDr25GaiaStellar.Evol == 0]
    if verbose == True: 
        print(str(len(cleanDr25GaiaStellar)) + " entries after removing marked evolved stars")


    if plots == True:
        plt.figure(figsize=(10,10));
        plt.semilogy(dr25GaiaStellar.teff, dr25GaiaStellar["R*"], ".k", ms=3, alpha=0.5)
        plt.semilogy(cleanDr25GaiaStellar.teff, cleanDr25GaiaStellar["R*"], ".r", ms=3, alpha=0.5)
        plt.semilogy(binaryDr25GaiaStellar.teff, binaryDr25GaiaStellar["R*"], ".b", ms=3, alpha=0.5)
        plt.semilogy(evolvedDr25GaiaStellar.teff, evolvedDr25GaiaStellar["R*"], ".g", ms=3, alpha=0.5)
        plt.semilogy([9000,3000], [1.35,1.35], linestyle='--', linewidth=1, alpha=0.5)
        plt.xlim(9500, 2500)
        plt.legend(("all DR25/Gaia stars", "cleaned DR25/Gaia stars", "binary DR25/Gaia stars", "evolved DR25/Gaia stars"));
        plt.ylabel("radius");
        plt.xlabel("teff");
        plt.savefig(savepath + 'DR25andGaiaHRD.pdf')

    # Next we remove stars that are on the list https://github.com/nasa/KeplerPORTs/blob/master/DR25_DEModel_NoisyTargetList.txt.
    noisyTargets = pd.read_csv("data/DR25_DEModel_NoisyTargetList.txt", header=9)
    cleanDr25GaiaStellar = cleanDr25GaiaStellar[~dr25GaiaStellar.kepid.isin(noisyTargets['# 1- Kepler ID'])]
    if verbose == True:
        print(str(len(cleanDr25GaiaStellar)) + " entries after removing noisy targets")

    cleanDr25GaiaStellar = cleanDr25GaiaStellar[dr25GaiaStellar.radius <= 1.25]
    if verbose == True:
        print(str(len(cleanDr25GaiaStellar)) + " entries after removing R > 1.25 targets")


    # Next we remove stars with nan limb darkening coefficients.
    cleanDr25GaiaStellar = cleanDr25GaiaStellar[~np.isnan(dr25GaiaStellar.limbdark_coeff1)]
    cleanDr25GaiaStellar = cleanDr25GaiaStellar[~np.isnan(dr25GaiaStellar.limbdark_coeff2)]
    cleanDr25GaiaStellar = cleanDr25GaiaStellar[~np.isnan(dr25GaiaStellar.limbdark_coeff3)]
    cleanDr25GaiaStellar = cleanDr25GaiaStellar[~np.isnan(dr25GaiaStellar.limbdark_coeff4)]
    if verbose == True:
        print(str(len(cleanDr25GaiaStellar)) + " entries after removing Nan limb darkening targets")


    # Next we remove stars with duty cycle = nan.
    cleanDr25GaiaStellar = cleanDr25GaiaStellar[~np.isnan(dr25GaiaStellar.dutycycle)]
    if verbose == True:
       print(str(len(cleanDr25GaiaStellar)) + " entries after removing Nan dutycycle targets")


    # Next we remove stars that have had a drop in duty cycle > 30% due to the removal of transits. (10% in KSCI)
    dutyCycleChange = cleanDr25GaiaStellar.dutycycle - cleanDr25GaiaStellar.dutycycle_post
    cleanDr25GaiaStellar = cleanDr25GaiaStellar[dutyCycleChange/cleanDr25GaiaStellar.dutycycle <= 0.3]
    if verbose == True:
        print(str(len(cleanDr25GaiaStellar)) + " entries after removing duty cycle drops > 0.3")


    # Next we remove stars that have duty cycles < 0.6.
    cleanDr25GaiaStellar = cleanDr25GaiaStellar[cleanDr25GaiaStellar.dutycycle >= 0.6]
    if verbose == True:
        print(str(len(cleanDr25GaiaStellar)) + " entries after removing after removing stars with duty cycle < 0.6")


    # Next we remove stars with data span < 1000
    cleanDr25GaiaStellar = cleanDr25GaiaStellar[cleanDr25GaiaStellar.dataspan >= 1000]
    if verbose == True:
        print(str(len(cleanDr25GaiaStellar)) + " entries after removing after removing stars with data span < 1000")


    # Next we remove stars with the timeoutsumry flag != 1.
    cleanDr25GaiaStellar = cleanDr25GaiaStellar[cleanDr25GaiaStellar.timeoutsumry == 1]
    if verbose == True:
        print(str(len(cleanDr25GaiaStellar)) + " entries after removing after removing stars with timeoutsumry != 1")

    catalogHeader = "dr25_stellar_supp_gaia_clean"

    cleanDr25GaiaStellar.to_csv("stellarCatalogs/" + catalogHeader + "_all.txt", index=False)


    # This produces the clean master list dr25_stellar_supp_gaia_clean.txt from which we will extract specific spectral types.
    # We extract desired spectral types using the teff boundaries from Pecaut and Mamajek 2013 
    # http://iopscience.iop.org/article/10.1088/0067-0049/208/1/9/meta;jsessionid=698F3A9F5272B070DC62876C1764BFDB.c1#apjs480616s3:
    # M: 2400 <= T < 3900  <br>
    # K: 3900 <= T < 5300<br>
    # G: 5300 <= T < 6000<br>
    # F: 6000 <= T < 7300<br>

    spt = spt.lower()
    spt_tables = {}

    if 'f' in spt:
        cleanDr25GaiaStellarF = cleanDr25GaiaStellar[(cleanDr25GaiaStellar.teff >= 6000)&(cleanDr25GaiaStellar.teff < 7300)]
        spt_tables['f'] = cleanDr25GaiaStellarF
        if verbose == True:
            print(str(len(cleanDr25GaiaStellarF)) + " F targets")
    if 'g' in spt:
        cleanDr25GaiaStellarG = cleanDr25GaiaStellar[(cleanDr25GaiaStellar.teff >= 5300)&(cleanDr25GaiaStellar.teff < 6000)]
        spt_tables['g'] = cleanDr25GaiaStellarG
        if verbose == True:
            print(str(len(cleanDr25GaiaStellarG)) + " G targets")

    if 'k' in spt:
        cleanDr25GaiaStellarK = cleanDr25GaiaStellar[(cleanDr25GaiaStellar.teff >= 3900)&(cleanDr25GaiaStellar.teff < 5300)]
        spt_tables['k'] = cleanDr25GaiaStellarK
        if verbose == True:
            print(str(len(cleanDr25GaiaStellarK)) + " K targets")

    if 'm' in spt:
        cleanDr25GaiaStellarM = cleanDr25GaiaStellar[(cleanDr25GaiaStellar.teff >= 2400)&(cleanDr25GaiaStellar.teff < 3900)]
        spt_tables['m'] = cleanDr25GaiaStellarM
        if verbose == True:
            print(str(len(cleanDr25GaiaStellarM)) + " M targets")

    if len(spt) > 1:
        tab_to_write = [spt_tables[spt[n]] for n in range(len(spt))]
        writeTable = pd.concat(tab_to_write, ignore_index = True)

    else:
        writeTable = spt_tables[spt]


    # always save the GK catalog
    writeTable.to_csv("stellarCatalogs/" + catalogHeader + "_{}.txt".format(spt.upper()), index=False)

    return

def main(argv):

    try:
        argument_list = argv[1:]
        short_options = 'p:v:s:t:y:o:' #plots, verbose, savepath for files, spectral types to include, lower age limit, upper age limit
        long_options = 'plots:verbose:savepath:spt:age_lower:age_upper:' 
        arguments, values = getopt.getopt(argument_list, short_options, long_options)

        plots = arguments[0][1]
        verbose = arguments[1][1]
        savepath = arguments[2][1] + '/'
        spt = arguments[3][1]
        lower_age_limit = float(arguments[4][1])
        upper_age_limit = float(arguments[5][1])

        if 'f' in plots.lower():
            plots = False
        else:
            plots = True

        if 'f' in verbose.lower():
            verbose = False
        else:
            verbose = True

        if not ('f' in spt.lower() or 'g' in spt.lower() or 'k' in spt.lower() or 'm' in spt.lower()):
            print('Invalid spectral type entered! Please enter some combination of FGKM (case-insensitive). Terminating program.')
            sys.exit(1)

        if lower_age_limit < 0.1 or upper_age_limit > 19.5:
            print('Invalid age range: ages must be between 0.1 Gyr and 19.5 Gyr. Terminating program.')
            sys.exit(1)

        try:
            os.mkdir('stellarCatalogs')
        except:
            pass;

        print('Making a catalog with the following settings: plots = {}, verbose = {}, savepath = {}, spectral types = {}, \
            lower age limit = {} Gyr, upper age limit = {} Gyr'.format(plots, verbose, savepath, spt, lower_age_limit, upper_age_limit))

        makeCatalog(savepath, plots, verbose, spt, lower_age_limit, upper_age_limit)

    except:

        try: 
            os.mkdir('stellarCatalogs')
        except:
            pass;

        print('No inputs given, running with default settings: plots = True, verbose = True, stellar spectral types of GK only, all ages.\n \
            Savepath defaults to current working directory for plots. Files are saved in the stellarCatalogs subdirectory, \n \
            which must be contained in your current working directory. If stellarCatalogs did not exist, it has been created.')
        makeCatalog(os.getcwd() + '/')

    return

if __name__ == "__main__":
    main(sys.argv)


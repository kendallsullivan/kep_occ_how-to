import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spec
import pandas as pd
from astropy.io import ascii
from astropy.table import Table, vstack
import pickle
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
import sys
sys.path.insert(0, '..')
import dr25Models as funcModels
import requests
from cStringIO import StringIO

def computeReliabiltyPosterior(xp, yp, eSamples, oSamples):
    r = np.zeros(np.shape(eSamples)[0])
    for i in range(np.shape(eSamples)[0]):
        e = funcModels.evaluateModel(xp, yp, eSamples[i,:], fpEffXRange, fpEffYRange, fpEffModel)
        o = funcModels.evaluateModel(xp, yp, oSamples[i,:], obsXRange, obsYRange, obsModel)
        r[i] = 1 - (o/(1-o))*((1-e)/e)

    e = funcModels.evaluateModel(xp, yp, fpEffTheta, fpEffXRange, fpEffYRange, fpEffModel)
    o = funcModels.evaluateModel(xp, yp, obsTheta, obsXRange, obsYRange, obsModel)
    f = 1 - (o/(1-o))*((1-e)/e)

    return r, f

def makePlanetInput(fpEffModel = 'rotatedLogisticX0', obsModel = 'rotatedLogisticX0', plots = True, verbose = True, pmin = 1, pmax = 400, rpmin = 0.5, rpmax = 15, spt = 'GK'):

    # read in the model parameters
    tt = pd.read_pickle("fpEffectivenessTable.pkl")
    tm = tt[tt.Model == fpEffModel]
    fpEffXRange = tm.periodRange.values[0]
    fpEffYRange = tm.mesRange.values[0]
    fpEffTheta = tm.medianMCMCTheta.values[0] 

    tt = pd.read_pickle("obsFpTable.pkl")
    tm = tt[tt.Model == obsModel]
    obsXRange = tm.periodRange.values[0]
    obsYRange = tm.mesRange.values[0]
    obsTheta = tm.medianMCMCTheta.values[0] 

    cellPeriod, cellMes = np.meshgrid(np.array(np.linspace(fpEffXRange[0], fpEffXRange[1], 200)), 
                              np.array(np.linspace(fpEffYRange[0], fpEffYRange[1], 200)))

    effFit = funcModels.evaluateModel(cellPeriod, cellMes, fpEffTheta, fpEffXRange, fpEffYRange, fpEffModel)
    obsFit = funcModels.evaluateModel(cellPeriod, cellMes, obsTheta, obsXRange, obsYRange, obsModel)

    fig = plt.figure(figsize=plt.figaspect(0.3));
    R = 1 - (obsFit/(1-obsFit))*((1-effFit)/effFit)

    pR = R;
    pR[pR<0] = 0;

    if plots == True:
        fig, ax = plt.subplots(figsize=(10,10));
        CS = ax.contour(cellPeriod, cellMes, pR, colors='k', levels = [.45, .5, .55, .6, .7, .75, .8, .85, .9, .95, .99]);
        ax.clabel(CS, inline=1, fontsize=18);
        ax.tick_params(axis = "both", labelsize = 18)
        plt.xlabel("period (days)", fontsize = 24);
        plt.ylabel("MES", fontsize = 24);
        plt.title("Reliability Against False Alarms", fontsize = 24);
        plt.savefig("reliabilityContours.pdf",bbox_inches='tight')

    # fig = plt.figure(figsize=plt.figaspect(0.3));
    R = (1-effFit)/effFit

    pR = R;
    pR[pR<0] = 0;

    # ax = fig.add_subplot(1, 3, 1, projection='3d')
    # surf = ax.plot_surface(cellPeriod, cellMes, pR, alpha = 0.5);
    # plt.xlabel("period");
    # plt.ylabel("MES");
    # ax.view_init(0,0)

    # ax = fig.add_subplot(1, 3, 2, projection='3d')
    # surf = ax.plot_surface(cellPeriod, cellMes, pR, alpha = 0.5);
    # plt.xlabel("period");
    # plt.ylabel("MES");
    # ax.view_init(0,-90)
    # plt.title("1-E/E");

    # ax = fig.add_subplot(1, 3, 3, projection='3d')
    # surf = ax.plot_surface(cellPeriod, cellMes, pR, alpha = 0.5);
    # plt.xlabel("period");
    # plt.ylabel("MES");


    # fig, ax = plt.subplots(figsize=(5,5));
    # CS = ax.contour(cellPeriod, cellMes, pR);
    # ax.clabel(CS, inline=1, fontsize=10);
    # plt.xlabel("period");
    # plt.ylabel("MES");


    # fig = plt.figure(figsize=plt.figaspect(0.3));
    R = obsFit/(1-obsFit)

    pR = R;
    pR[pR<0] = 0;

    # ax = fig.add_subplot(1, 3, 1, projection='3d')
    # surf = ax.plot_surface(cellPeriod, cellMes, pR, alpha = 0.5);
    # plt.xlabel("period");
    # plt.ylabel("MES");
    # ax.view_init(0,0)

    # ax = fig.add_subplot(1, 3, 2, projection='3d')
    # surf = ax.plot_surface(cellPeriod, cellMes, pR, alpha = 0.5);
    # plt.xlabel("period");
    # plt.ylabel("MES");
    # ax.view_init(0,-90)
    # plt.title("obs/(1-obs)");

    # ax = fig.add_subplot(1, 3, 3, projection='3d')
    # surf = ax.plot_surface(cellPeriod, cellMes, pR, alpha = 0.5);
    # plt.xlabel("period");
    # plt.ylabel("MES");


    # fig, ax = plt.subplots(figsize=(5,5));
    # CS = ax.contour(cellPeriod, cellMes, pR);
    # ax.clabel(CS, inline=1, fontsize=10);
    # plt.xlabel("period");
    # plt.ylabel("MES");


    R = 1 - (obsFit/(1-obsFit))*((1-effFit)/effFit)

    pR = R;
    pR[pR<0] = 0;


    eSamples = np.load("binEffPosteriors_" + str(fpEffModel) + ".npy");
    oSamples = np.load("binObsPosteriors_" + str(obsModel) + ".npy");

    r1, f1 = computeReliabiltyPosterior(200., 25., eSamples, oSamples)
    r2, f2 = computeReliabiltyPosterior(365., 10., eSamples, oSamples)
    r3, f3 = computeReliabiltyPosterior(365., 8., eSamples, oSamples)

    ymax = 10000
    if plots == True:
        plt.figure(figsize=(15,10))
        plt.hist(r1, 20);
        plt.plot([f1, f1], [0, ymax], color='k', linestyle='--', linewidth=1, alpha = 0.2)

        plt.hist(r2, 100, alpha = 0.5);
        plt.plot([f2, f2], [0, ymax], color='k', linestyle='--', linewidth=1, alpha = 1)

        plt.hist(r3, 100, alpha = 0.5);
        plt.plot([f3, f3], [0, ymax], color='k', linestyle='--', linewidth=1, alpha = 1)
        plt.ylim(0, ymax)
        plt.xlim(0, 1.2)
        plt.tick_params(labelsize = 18)
        plt.xlabel(r"$R_\mathrm{FA}$", fontSize = 24);

        plt.savefig("reliabilityExamples.pdf",bbox_inches='tight')

    if verbose == True:
        print("f1:" + str(f1))
        print("f2:" + str(f2))
        print("f3:" + str(f3))


    if False:
        selectStr = "kepid,kepoi_name,koi_tce_plnt_num,koi_pdisposition,koi_score,koi_period,koi_max_mult_ev,koi_prad,koi_prad_err1,koi_prad_err2,koi_ror,koi_ror_err1,koi_ror_err2"
        urlDr25Koi = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=q1_q17_dr25_koi&select=" + selectStr

        r = requests.get(urlDr25Koi)
        if r.status_code != requests.codes.ok:
            r.raise_for_status()
        fh = StringIO(r.content)
        dr25Koi = pd.read_csv(fh, dtype={"kepoi_name":str})
        dr25Koi.to_csv("koiCatalogs/dr25_kois_archive.txt", index=False)
    else:
        dr25Koi = pd.read_csv("koiCatalogs/dr25_kois_archive.txt", dtype={"kepoi_name":str})

    print("Loaded " + str(len(dr25Koi)) + " KOIs")


    # restrict the population to stars in the Berger catalog

    dr25CleanStellarIso = pd.read_csv("../stellarCatalogs/dr25_stellar_supp_gaia_clean_{}.txt".format(spt))
    dr25Koi = dr25Koi[dr25Koi.kepid.isin(dr25CleanStellarIso.kepid)]
    dr25Koi = dr25Koi.reset_index(drop=True)
    if verbose == True:
        print("After removing planets not in the stellar catalog, we have " + str(len(dr25Koi)) + " KOIs")

    # merge in only radius with uncertainties and teff from the stellar table
    dr25Koi = pd.merge(dr25Koi, dr25CleanStellarIso[["kepid","radius","radius_err1","radius_err2","teff"]], on="kepid", how="inner")

    # correct the planet radii with the new catalog
    rEarth = 6356.8 # km
    rSun = 695700 # km

    dr25Koi['corrected_prad'] = dr25Koi['koi_ror']*dr25Koi['radius']*rSun/rEarth;
    dr25Koi['corrected_prad_err1'] = np.sqrt(dr25Koi['koi_ror_err1']**2*dr25Koi['radius']**2
                                            +dr25Koi['koi_ror']**2*dr25Koi['radius_err1']**2)*rSun/rEarth;
    dr25Koi['corrected_prad_err2'] = -np.sqrt(dr25Koi['koi_ror_err2']**2*dr25Koi['radius']**2
                                            +dr25Koi['koi_ror']**2*dr25Koi['radius_err2']**2)*rSun/rEarth;

    dr25Koi = dr25Koi[~np.isnan(dr25Koi.koi_prad)]

    v = dr25Koi.corrected_prad_err1/dr25Koi.koi_prad_err1
    # plt.hist(v[v<5], 100);

    # plt.hist(dr25Koi['corrected_prad'][dr25Koi['corrected_prad']<10], 100);

    # fig, ax = plt.subplots(figsize=(15,10));
    # ax.errorbar(dr25Koi.koi_period, dr25Koi.koi_prad, 
    #             yerr = [-dr25Koi.koi_prad_err2, dr25Koi.koi_prad_err1],
    #             fmt="k.", alpha = 0.5);
    # ax.errorbar(dr25Koi.koi_period, dr25Koi.corrected_prad, 
    #             yerr = [-dr25Koi.corrected_prad_err2, dr25Koi.corrected_prad_err1],
    #             fmt="r.", alpha = 0.5);

    # plt.xlabel("period");
    # plt.ylabel("planet radius");
    # plt.title("KOI Radius Change");
    # plt.ylim([0, 2.5])
    # plt.xlim([50, 400])


    dr25Fpp = ascii.read("../data/q1_q17_dr25_koifpp.txt")
    dr25FppPd = dr25Fpp.to_pandas()


    mergedDr25Koi = pd.merge(dr25Koi, dr25FppPd, on="kepoi_name", how="inner")


    mergedDr25Koi.loc[:,"fpEffectiveness"] = pd.Series(
                                funcModels.evaluateModel(mergedDr25Koi.koi_period,
                                 mergedDr25Koi.koi_max_mult_ev, fpEffTheta, 
                                 fpEffXRange, fpEffYRange, fpEffModel), index = mergedDr25Koi.index)
    mergedDr25Koi.loc[:,"obsFpRate"] = pd.Series(
                                funcModels.evaluateModel(mergedDr25Koi.koi_period,
                                 mergedDr25Koi.koi_max_mult_ev, obsTheta, 
                                 obsXRange, obsYRange, obsModel), index = mergedDr25Koi.index)

    mergedDr25Koi.loc[:,"reliability"] = pd.Series(
        1-(mergedDr25Koi.obsFpRate/(1-mergedDr25Koi.obsFpRate))
        *(1-mergedDr25Koi.fpEffectiveness)/mergedDr25Koi.fpEffectiveness, index = mergedDr25Koi.index)
        
    mergedDr25Koi.reliability[mergedDr25Koi.reliability < 0.] = 0.

    plt.hist(mergedDr25Koi.koi_score, 40);
    plt.yscale('log', nonposy='clip')

    np.sum(np.isnan(mergedDr25Koi.fpp_prob) & mergedDr25Koi.koi_period > 50)

    mergedDr25Koi[np.abs(mergedDr25Koi.koi_period - mergedDr25Koi.fpp_koi_period)>1e-2]

    mergedDr25Koi["fpp_prob_use"] = mergedDr25Koi["fpp_prob"]
    mergedDr25Koi.fpp_prob_use[np.isnan(mergedDr25Koi.fpp_prob)] = 1
    mergedDr25Koi.fpp_prob_use[np.abs(mergedDr25Koi.koi_period - mergedDr25Koi.fpp_koi_period)>1e-2] = 1

    mergedDr25Koi[np.abs(mergedDr25Koi.koi_period - mergedDr25Koi.fpp_koi_period)>1e-2]

    mergedDr25Koi["totalReliability"] = (1-mergedDr25Koi.fpp_prob_use)*mergedDr25Koi.reliability

    # fig, ax = plt.subplots(figsize=(15,10));
    # scf = ax.scatter(mergedDr25Koi.koi_period, mergedDr25Koi.koi_max_mult_ev, cmap="viridis", 
    #                  c=mergedDr25Koi.reliability, edgecolors='k', s=100*mergedDr25Koi.totalReliability, alpha = 0.5);
    # plt.xlabel("period");
    # plt.ylabel("MES");
    # plt.title("KOI Reliability, size = total reliability");
    # plt.ylim([7, 50])
    # plt.xlim([50, 400])

    # cbh = plt.colorbar(scf);
    # cbh.ax.set_ylabel("Reliability");

    # fig, ax = plt.subplots(figsize=(15,10));
    # scf = ax.scatter(mergedDr25Koi.koi_period, mergedDr25Koi.corrected_prad, cmap="viridis", 
    #                  c=mergedDr25Koi.reliability, edgecolors='k', s=100*mergedDr25Koi.totalReliability, alpha = 0.5);
    # plt.xlabel("period");
    # plt.ylabel("planet radius");
    # plt.title("KOI FA Reliability, size = total reliability");
    # plt.ylim([0, 2.5])
    # plt.xlim([50, 400])

    # cbh = plt.colorbar(scf);
    # cbh.ax.set_ylabel("FA Reliability");


    dr25PC = mergedDr25Koi[mergedDr25Koi.koi_pdisposition == "CANDIDATE"]
    dr25FP = mergedDr25Koi[mergedDr25Koi.koi_pdisposition == "FALSE POSITIVE"]
    # remove those with corrected_prad = NAN
    dr25PC = dr25PC[~np.isnan(dr25PC.corrected_prad)]
    dr25FP = dr25FP[~np.isnan(dr25FP.corrected_prad)]
    mergedDr25Koi = mergedDr25Koi[~np.isnan(mergedDr25Koi.corrected_prad)]


    if verbose == True:
        print("There are " + str(len(dr25PC)) + " PCs in " + str(len(dr25CleanStellarIso)) + " observed targets")
        print("There are " + str(len(dr25FP)) + " FPs in " + str(len(dr25CleanStellarIso)) + " observed targets")

    if plots == True:
        # fig, ax = plt.subplots(figsize=(15,10));
        # scf = ax.scatter(dr25PC.koi_period, dr25PC.koi_max_mult_ev, cmap="viridis", 
        #                  c=dr25PC.reliability, edgecolors='k', s=100*dr25PC.totalReliability, alpha = 0.5);
        # plt.xlabel("period");
        # plt.ylabel("MES");
        # plt.title("PC Reliability, size = total reliability");
        # #plt.ylim([7, 30])
        # #plt.xlim([50, 400])

        # cbh = plt.colorbar(scf);
        # cbh.ax.set_ylabel("Reliability");

        # fig, ax = plt.subplots(figsize=(15,10));
        # scf = ax.scatter(dr25PC.koi_period, dr25PC.corrected_prad, cmap="viridis", 
        #                  c=dr25PC.reliability, edgecolors='k', s=100*dr25PC.totalReliability, alpha = 0.8);
        # scf = ax.scatter(dr25PC.koi_period, dr25PC.corrected_prad, s=100*dr25PC.totalReliability, 
        #                  c=dr25PC.reliability, facecolors='none', edgecolors='k', alpha = 0.8);
        # plt.yscale('log', nonposy='clip')
        # plt.xscale('log', nonposx='clip')
        # plt.xlabel("period");
        # plt.ylabel("planet radius");
        # plt.title("PC Reliability, size = reliability");

        # cbh = plt.colorbar(scf);
        # cbh.ax.set_ylabel("Reliability");


        fig, ax = plt.subplots(figsize=(15,10));
        scf = ax.scatter(dr25PC.koi_period, dr25PC.corrected_prad, cmap="viridis", 
                         c=dr25PC.reliability, edgecolors='k', s=100*dr25PC.totalReliability, alpha = 0.5);
        plt.xlabel("period", fontsize = 24);
        plt.ylabel("corrected planet radius", fontsize = 24);
        plt.title("PC Reliability, size = total reliability", fontsize = 24);
        plt.ylim([rpmin, rpmax])
        plt.xlim([pmin, pmax])

        cbh = plt.colorbar(scf);
        cbh.ax.set_ylabel("Reliability");
        plt.savefig("pcReliability_correctedrp.pdf",bbox_inches='tight')

        # plt.plot([200, 200], [1, 2], color='k', linestyle='--', linewidth=1)
        # plt.plot([50, 200], [1, 1], color='k', linestyle='--', linewidth=1)
        # plt.plot([50, 200], [2, 2], color='k', linestyle='--', linewidth=1)

        fig, ax = plt.subplots(figsize=(15,10));
        scf = ax.scatter(dr25PC.koi_period, dr25PC.koi_prad, cmap="viridis", 
                         c=dr25PC.reliability, edgecolors='k', s=100*dr25PC.totalReliability, alpha = 0.5);
        plt.xlabel("period", fontsize = 24);
        plt.ylabel("DR25 planet radius", fontsize = 24);
        plt.title("PC Reliability, size = total reliability", fontsize = 24);
        plt.ylim([rpmin, rpmax])
        plt.xlim([pmin, pmax])

        cbh = plt.colorbar(scf);
        cbh.ax.set_ylabel("Reliability");
        plt.savefig("pcReliability.pdf",bbox_inches='tight')

        # plt.plot([200, 200], [1, 2], color='k', linestyle='--', linewidth=1)
        # plt.plot([50, 200], [1, 1], color='k', linestyle='--', linewidth=1)
        # plt.plot([50, 200], [2, 2], color='k', linestyle='--', linewidth=1)


    dr25PcInRange = dr25PC[(dr25PC.koi_period>50)&(dr25PC.koi_period<400)&(dr25PC.corrected_prad>0)&(dr25PC.corrected_prad<2.5)]


    # plt.hist(dr25PC.corrected_prad/dr25PC.koi_prad, 100);

    # plt.hist(dr25CleanStellarIso.radius[dr25CleanStellarIso.radius<2]/dr25CleanStellarIso.radius_DR25[dr25CleanStellarIso.radius<2], 100);


    dr25PC.to_csv("koiCatalogs/dr25_{}_PCs.csv".format(spt), index=False)
    mergedDr25Koi.to_csv("koiCatalogs/dr25_{}_KOIs.csv".format(spt), index=False)

    if plots == True:
        fig, ax = plt.subplots(figsize=(15,10));
        ax.errorbar(dr25PC.koi_period, dr25PC.koi_prad, 
                    yerr = [-dr25PC.koi_prad_err2, dr25PC.koi_prad_err1],
                    fmt="k.", alpha = 0.5);
        ax.errorbar(dr25PC.koi_period, dr25PC.corrected_prad, 
                    yerr = [-dr25PC.corrected_prad_err2, dr25PC.corrected_prad_err1],
                    fmt="r.", alpha = 0.5);

        plt.xlabel("period");
        plt.ylabel("planet radius");
        plt.title("KOI Radius Change");
        plt.ylim([0, 2.5])
        plt.xlim([50, 400])
        plt.savefig('radius_change.pdf')


        plt.hist(dr25PC.koi_score, 40);
        plt.yscale('log', nonposy='clip')
        plt.title("PC score distribution")
        plt.savefig('pc_score_dist.pdf')

        plt.hist(dr25FP.koi_score, 40, alpha=0.5);
        plt.yscale('log', nonposy='clip')
        plt.title("FP score distribution")
        plt.savefig('fp_score_dist.pdf')


    period_rng = (pmin, pmax)
    rp_rng = (rpmin, rpmax)

    occPcs = dr25PC[(dr25PC.koi_period>=period_rng[0])&(dr25PC.koi_period<=period_rng[1])&(dr25PC.corrected_prad>=rp_rng[0])&(dr25PC.corrected_prad<=rp_rng[1])]
    if verbose == True:
        print("After radius correction there are " + str(len(occPcs)) + " PCs in " + str(len(dr25CleanStellarIso)) + " observed targets")
    occPcs2 = dr25PC[(dr25PC.koi_period>=period_rng[0])&(dr25PC.koi_period<=period_rng[1])&(dr25PC.koi_prad>=rp_rng[0])&(dr25PC.koi_prad<=rp_rng[1])]
    if verbose == True:
        print("Before radius correction there are " + str(len(occPcs2)) + " PCs in " + str(len(dr25CleanStellarIso)) + " observed targets")

    return


def main(argv)    
    try:
        argument_list = argv[1:]
        short_options = 'E:F:i:v:s:p:P:r:R:' #model name for FP Eff, model name for FP Rate, plots y/n, verbose y/n, spt range, pmin, pmax, rpmin, rpmax
        long_options = 'fpeff:fprate:plots:verbose:spt:pmin:pmax:rpmin:rpmax:' 
        arguments, values = getopt.getopt(argument_list, short_options, long_options)

        # get out the various results of the keywords
        fpEffModel = str(arguments[0][1])
        fpRateModel = str(arguments[1][1])
        plots = strtobool(str(arguments[2][1]))
        verbose = strtobool(str(arguments[3][1]))
        spt = str(arguments[4][1]).upper()
        pmin = float(arguments[5][1])
        pmax = float(arguments[6][1])
        rpmin = float(arguments[7][1])
        rpmax = float(arguments[8][1])

    except:
        print('No inputs given, running with default settings: FP rate and FP effectiveness models = rotatedLogisticX0, plots = True, verbose = True, spectral types = \'GK\', min. period = 1 d, max period = 400 d, min Rp = 0.5 Re, max Rp = 15 Re')
       
        fpEffModel = 'rotatedLogisticX0'
        fpRateModel = 'rotatedLogisticX0'
        plots = True
        verbose = True
        spt = 'GK'
        pmin = 1
        pmax = 400
        rpmin = 0.5
        rpmax = 15

    # check the spectral type inputs are valid
    test_spt = [s.lower() not in 'fgkm' for s in spt]

    # if they are not, throw an error and exit
    if any([s == True for s in test_spt]):
        print('Invalid spectral type entered! Please enter some combination of FGKM (case-insensitive). Terminating program.')
        sys.exit(1)

    if pmin > pmax:
        print('Minimum period must be less than maximum period. Terminating program.')
        sys.exit(1)

    if rpmin > rpmax:
        print('Minimum planet radius must be less than maximum planet radius. Terminating program')
        sys.exit(1)


    # print out the running statement 
    print('Creating planet catalog with the following settings: FP effectiveness model = {}, FP rate model = {}, plots = {}, verbose = {}, spt range = {}, P min = {} d, P max = {} d, Rp min = {} Re, Rp max = {} Re.'.\
        format(fpEffModel, fpRateModel, plots, verbose, spt, pmin, pmax, rpmin, rpmax))

    makePlanetInput(fpEffModel = fpEffModel, obsModel = fpRateModel, plots = plots, verbose = verbose, spt = spt, pmin = pmin, pmax = pmax, rpmin = rpmin, rpmax = rpmax)
    return


if __name__ == '__main__':
    main(sys.argv)




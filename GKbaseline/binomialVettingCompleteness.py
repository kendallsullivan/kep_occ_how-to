

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spec
import pandas as pd
from astropy.io import ascii
import scipy.stats as stats
import sys
sys.path.insert(0, '..')
import dr25Models as funcModels
import os.path
import scipy.optimize as op
import pickle
import corner
import emcee
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm


def drawHeatMap(dataArray, imageSize, x, y, nData=[], colorBarLabel="", textOn=True, forceInt=True):
    dx = x[(1,0)] - x[(0,0)];
    dy = y[(0,1)] - y[(0,0)];
    extent = [x[(0,0)], x[(-1,0)]+dx,y[(0,0)],y[(0,-1)]+dy];

    plt.figure(figsize=imageSize);
    ax = plt.gca()

    da = np.transpose(dataArray);
    im = ax.imshow(da, extent = extent, origin='lower', cmap="Greys");
    ax.set_aspect(10);
    
    if len(nData) == 0:
        nData = np.ones(dataArray.shape)

    arrayShape = da.shape;
    minda = np.min(da)
    maxda = np.max(da)
    daRange = maxda - minda;
    for i in range(arrayShape[0]):
        for j in range(arrayShape[1]):
            if da[i, j] > minda + daRange*0.5:
                cstr = "w"
            else:
                cstr = "k"
            if np.abs(da[i,j]) < 100:
                fsz = 9
            else:
                fsz = 6
            
            if textOn:
                if nData[(j,i)] > 0:
                    if forceInt:
                        ax.text(x[(j,i)]+dx/2, y[(j,i)]+dy/2, da[i, j].astype("int"),
                               ha="center", va="center", color=cstr, fontsize=fsz)
                    else:
                        ax.text(x[(j,i)]+dx/2, y[(j,i)]+dy/2, da[i, j],
                               ha="center", va="center", color=cstr, fontsize=fsz)                        
                else:
                    ax.text(x[(j,i)]+dx/2, y[(j,i)]+dy/2, "-",
                           ha="center", va="center", color=cstr, fontsize=fsz)
    
    ax.tick_params(axis = "both", labelsize = 12)
    im_ratio = float(da.shape[0])/da.shape[1] 
    cbh = plt.colorbar(im,fraction=0.0477*im_ratio, pad=0.02)
    cbh.ax.set_ylabel(colorBarLabel, fontSize = 16);
    return

# we define the binomial probability distribution function.
def binPdf(n, r, c):
    return sp.comb(n,c)*(r**c)*((1-r)**(n-c));


def lnBinlike(theta, data, model):
    x, y, n, c = data;
    r = funcModels.rateModel(x,y,theta,model);
    clnr = c*np.log(r);
    clnr[c==0] = 0;
    lpl = np.sum(np.log(spec.comb(n,c)) + clnr + (n-c)*np.log(1-r));
    if np.isnan(lpl):
        return -np.inf
    else:
        return lpl


# For our prior $p(r)$, we'll set the prior = 1 on the interval $[0, 1]$ and zero otherwise.
def lnBinprior(theta, data, model):
    x, y, n, c = data
    if model == "linearX":
        if -5000.0 < theta[0] < 5000.0\
            and 0.0 < theta[1] < 1\
            and np.min(funcModels.rateModel(x, y, theta, model)) >= 0\
            and np.max(funcModels.rateModel(x, y, theta, model)) <= 1:
            return 1.0
    elif model == "linearXY":
        if -5000.0 < theta[0] < 5000.0\
            and -5000.0 < theta[1] < 5000.0\
            and 0.0 < theta[2] < 1         \
            and np.min(funcModels.rateModel(x, y, theta, model)) >= 0\
            and np.max(funcModels.rateModel(x, y, theta, model)) <= 1:
            return 1.0
    elif model == "gaussian":
        if -10.0 < theta[0] < 10.0\
            and -10.0 < theta[1] < 10.0\
            and 1e-4 < theta[2] < 1e4\
            and 1e-4 < theta[3] < 1e4\
            and -1 <= theta[4] <= 1\
            and 0 <= theta[5] <= 1\
            and np.min(funcModels.rateModel(x, y, theta, model)) >= 0\
            and np.max(funcModels.rateModel(x, y, theta, model)) <= 1:
            return 1.0
    elif model == "logisticX":
        if -10.0 < theta[0] < 10.0\
             and 1e-4 < theta[1] < 1e4\
             and 0 < theta[2] < 1e6 \
             and 0 < theta[3] < 1e6\
             and np.min(funcModels.rateModel(x, y, theta, model)) >= 0\
             and np.max(funcModels.rateModel(x, y, theta, model)) <= 1:
            return 1.0
    elif model == "logisticX0":
        if 0 <= theta[0] <= 1000\
            and 1e-4 < theta[1] < 1e4\
            and 0 < theta[2] < 1e6\
            and np.min(funcModels.rateModel(x, y, theta, model)) >= 0\
            and np.max(funcModels.rateModel(x, y, theta, model)) <= 1:
            return 1.0
    elif model == "logisticY0":
        if -1 <= theta[0] <= 2\
            and 1e-4 < theta[1] < 1e4\
            and 0 < theta[2] < 1\
            and np.min(funcModels.rateModel(x, y, theta, model)) >= 0\
            and np.max(funcModels.rateModel(x, y, theta, model)) <= 1:
            return 1.0
    elif model == "logisticX0xlogisticY0":
        if -1 <= theta[0] <= 2\
            and -1 <= theta[1] <= 2\        
            and 1e-4 < theta[2] < 1e4\
            and 1e-4 < theta[3] < 1e4\
            and 0 < theta[4] < 1\
            and np.min(funcModels.rateModel(x, y, theta, model)) >= 0\
            and np.max(funcModels.rateModel(x, y, theta, model)) <= 1:
            return 1.0
    elif model == "logisticX0xlogisticY02":
        if -1 <= theta[0] <= 2\
            and -1 <= theta[1] <= 2\
            and 1e-4 < theta[2] < 1e4\
            and 1e-4 < theta[3] < 1e4\
            and 0.01 < theta[4] < 10\
            and 0.01 < theta[5] < 10\
            and 0 < theta[6] < 1\
            and np.min(funcModels.rateModel(x, y, theta, model)) >= 0\
            and np.max(funcModels.rateModel(x, y, theta, model)) <= 1:
            return 1.0
    elif model == "logisticX0xRotatedLogisticY0":
        if -1 <= theta[0] <= 2\
            and -1 <= theta[1] <= 2\
            and 1e-4 < theta[2] < 1e4\
            and 1e-4 < theta[3] < 1e4\
            and -180 < theta[4] < 180\
            and 0 < theta[5] < 1\
            and np.min(funcModels.rateModel(x, y, theta, model)) >= 0\
            and np.max(funcModels.rateModel(x, y, theta, model)) <= 1:
            return 1.0
    elif model == "logisticX0xRotatedLogisticY02":
        if -1 <= theta[0] <= 2\
            and -1 <= theta[1] <= 2\
            and 1e-4 < theta[2] < 1e4\
            and 1e-4 < theta[3] < 1e4\
            and 0.01 < theta[4] < 10\
            and -180 < theta[5] < 180\
            and 0 < theta[6] < 1\
            and np.min(funcModels.rateModel(x, y, theta, model)) >= 0\
            and np.max(funcModels.rateModel(x, y, theta, model)) <= 1:
            return 1.0
    elif model == "rotatedLogisticX0xlogisticY0":
        if -1 <= theta[0] <= 2\
            and -1 <= theta[1] <= 2\
            and 1e-4 < theta[2] < 1e4\
            and 1e-4 < theta[3] < 1e4\
            and -3 < theta[4] < 3\
            and -180 < theta[5] < 180\
            and 0 < theta[6] < 1\
            and np.min(funcModels.rateModel(x, y, theta, model)) >= 0\
            and np.max(funcModels.rateModel(x, y, theta, model)) <= 1:
            return 1.0
    elif model == "rotatedLogisticX0xlogisticY02":
        if -1 <= theta[0] <= 2\
            and -1 <= theta[1] <= 2\
            and 1e-4 < theta[2] < 1e4\
            and 1e-4 < theta[3] < 1e4\
            and 0.01 < theta[4] < 10\
            and 0.01 < theta[5] < 10\
            and -180 < theta[6] < 180\
            and -180 < theta[7] < 180\
            and 0 < theta[8] < 1\
            and np.min(funcModels.rateModel(x, y, theta, model)) >= 0\
            and np.max(funcModels.rateModel(x, y, theta, model)) <= 1:
            return 1.0
    elif model == "rotatedLogisticX0":
        if -1 <= theta[0] <= 2\
            and 1e-4 < theta[1] < 100\
            and 0 < theta[2] <= 1\
            and -180 < theta[3] < 180\
            and np.min(funcModels.rateModel(x, y, theta, model)) >= 0\
            and np.max(funcModels.rateModel(x, y, theta, model)) <= 1:
            return 1.0
    elif model == "dualBrokenPowerLaw":
        if 0 <= theta[0] <= 1\
            and 0 <= theta[1] <= 1\
            and -2 < theta[2] < 0\
            and -2 < theta[3] < 0\
            and 0 < theta[4] < 1e4\
            and 0 < theta[5] < 1e4\
            and 0 < theta[6] < 1\
            and np.min(funcModels.rateModel(x, y, theta, model)) >= 0\      
            and np.max(funcModels.rateModel(x, y, theta, model)) <= 1:
            return 1.0
    else:
        raise ValueError('Bad model name');

    return -np.inf

# Then the log posterior probability is $\log p(\theta|c, n, p, m) = \log p(c|\theta, n, p, m) + \log p(\theta)$, which we code as 

def lnBinprob(theta, data, model):
    lp = lnBinprior(theta, data, model)
    # print("lnPoisprior = " + str(lp))
    if not np.isfinite(lp):
        return -np.inf
    # print(str(lnPoislike(theta, A, c)))
    return lp + lnBinlike(theta, data, model)

'''
In this notebook we measure DR25 vetting completeness, defined as the fraction of detections (TCEs) that are correctly vetted as planet candidates.  
We use the set of injected on-target planets that were detected at the correct ephemeris as the base set of TCEs.  All of these TCEs are "true planets" by definition.  
Then vetting completeness is the fraction of these TCEs that are vetted as PC by the robovetter.  We study how vetting completeness depends on period and MES.
 
We think of TCEs as consisting of two sets: those that are dispositioned as FP and those that are dispositioned as PC. 
Then we can think of the vetting process as drawing from the set of TCEs, with a probability $r$ of selecting PCs. We identify $r$ with vetting completeness.  
Then the probability distribution of selecting $c$ FPs from $n$ TCEs is given by the binomial distribution.
 
 
In this spirit, we define the vetting effectiveness $r$ as the probability of drawing PCs from inverted/scrambled TCEs, 
found via the Bayesian inference $p(r|n, c) \propto p(c|r, n) p(r)$, where $c$ is the number of TCEs vetted as PCs, $n$ is the total number of TCEs,
By putting the data on a grid indexed by $i,j$, we can fit effectiveness as a function parameterized by a vector $\theta$, 
$r(\theta,\mathrm{period},\mathrm{MES})$, 
as $p(\theta)|n_{i,j}, c_{i,j}, \mathrm{period}_{i,j},\mathrm{MES}_{i,j}) \propto p(c_{i,j}|\theta, n_{i,j}, \mathrm{period}_{i,j},\mathrm{MES}_{i,j}) p(\theta)$, 
where $p(\theta)$ is some prior distribution of the parameters.
''' 


def vettingCompleteness(plots = True, verbose = True, savepath = os.getcwd(), stellarType = 'gk', periodMin = 1, periodMax = 400, rpMin = 0.5, rpMax = 15, mesMin = 0, mesMax = 30):

    # Read in our data.
    dataLoc = "../data/"
    injTceList = dataLoc + "kplr_dr25_inj1_tces.txt"
    tcelist = dataLoc + "DR25-Injected-Recovered-OnTarget-Planet-TCEs-1-1-Prat.txt"
    # starlist = dataLoc + "dr25_stellar_updated_feh_" + stellarType + ".txt"
    starlist = "../stellarCatalogs/dr25_stellar_supp_gaia_clean_" + stellarType + ".txt"


    # Load the stellar population we want to use
    kic = pd.read_csv(starlist)
    # Load D2 table
    injTces = ascii.read(injTceList)
    tces = np.genfromtxt(tcelist, dtype='str')

    tceKepids = np.zeros(len(tces));
    for i in range(len(tces)):
        s = tces[i].split('-');
        tceKepids[i] = int(s[0]);
    print(tceKepids)

    print("num injected/recovered TCEs: " + str(np.size(tceKepids)))
    print("num injected TCEs: " + str(np.size(injTces)))

    injTces = injTces[np.in1d(injTces['TCE_ID'],tces)]
    print("num injected TCEs after trimming to injected/recovered: " + str(np.size(injTces)))


    # Select only those TCEs that are in this stellar population
    injTces = injTces[np.in1d(injTces['KIC'],kic.kepid)]
    print("after: " + str(np.size(injTces)))


    # Do some basic stats
    print(injTceList)
    print("# of injected TCEs: " + str(len(injTces)))
    print("# of injected PCs: " + str(len(injTces[injTces['Disp']=='PC'])))
    print("# of injected FPs: " + str(len(injTces[injTces['Disp']=='FP'])))
    print(' ')

    print("for " + str(rpMax) + " < Rp < " + str(rpMax) + ", " + str(periodMin) + " < period < " + str(periodMax) + ":");
    print("# of injected injected TCEs: " + str(len(injTces[np.all([injTces['Rp']>rpMin,injTces['Rp']<rpMax,injTces['period']>periodMin,injTces['period']<periodMax], axis=0)])))
    print("# of injected PCs: " + str(len(injTces[np.all([injTces['Disp']=='PC', injTces['Rp']>rpMin,injTces['Rp']<rpMax,injTces['period']>periodMin,injTces['period']<periodMax], axis=0)])))
    print("# of injected FPs: " + str(len(injTces[np.all([injTces['Disp']=='FP', injTces['Rp']>rpMin,injTces['Rp']<rpMax,injTces['period']>periodMin,injTces['period']<periodMax], axis=0)])))


    # Separate out the PCs and FPs
    pcIndex = np.where(injTces[injTces['Disp']=='PC'])
    fpIndex = np.where(injTces[injTces['Disp']=='FP'])
    pcs = injTces[pcIndex]
    fps = injTces[fpIndex]


    # Select the TCEs that are in our desired population and plot them.
    spIndex = np.where(np.all([
        injTces['Rp']>rpMin,injTces['Rp']<rpMax,injTces['period']>periodMin,injTces['period']<periodMax], axis=0))
    spInjTces = injTces[spIndex]
    spInjPcs = spInjTces[(spInjTces['Disp']=='PC') & (spInjTces['Score']>=scoreCut)]
    spInjFps = spInjTces[(spInjTces['Disp']=='FP') | (spInjTces['Score']<scoreCut)]


    plt.figure(figsize=(15,15));
    plt.subplot(2,2,1);
    plt.scatter(injTces['period'], injTces['Expected_MES'], marker=".");
    plt.ylim(0,mesMax);
    plt.title("all injected TCEs");
    plt.ylabel('Expected MES');
    plt.xlabel('Period');
    plt.subplot(2,2,2);
    plt.scatter(spInjTces['period'], spInjTces['Expected_MES'], marker=".");
    plt.ylim(0,mesMax);
    plt.title("injected TCEs in period/MES range");
    plt.xlabel('Period');
    plt.ylabel('Expected MES');
    plt.subplot(2,2,3);
    plt.scatter(spInjPcs['period'], spInjPcs['Expected_MES'], marker=".");
    plt.ylim(0,mesMax);
    plt.title("injected PCs in period/MES range");
    plt.ylabel('Expected MES');
    plt.xlabel('Period');
    plt.subplot(2,2,4);
    plt.scatter(spInjFps['period'], spInjFps['Expected_MES'], marker=".");
    plt.ylim(0,mesMax);
    plt.title("injected FPss in period/MES range");
    plt.ylabel('Expected MES');
    plt.xlabel('Period');


    # Bin the populations onto a grid.  The binned TCEs and PCs are the input to our MCMC analysis.
    dPeriod = 10; 
    dMes = 1;

    p0 = periodMin;
    pEnd = periodMax;
    m0 = mesMin;
    mEnd = mesMax;

    # make the period-mes grid
    NPeriod = int((pEnd - p0)/dPeriod);
    NMes = int((mEnd - m0)/dMes);
    tceGrid = np.zeros((NPeriod,NMes));
    cellPeriod = np.zeros((NPeriod,NMes));
    cellMes = np.zeros((NPeriod,NMes));
    pcGrid = np.zeros((NPeriod,NMes));
    fpGrid = np.zeros((NPeriod,NMes));

    # count how many points are in each cell
    for p in range(NPeriod):
        for m in range(NMes):
            cellPeriod[(p,m)] = p0 + p*dPeriod;
            cellMes[(p,m)] = m0 + m*dMes;
            pointsInCell = np.where(
                (spInjTces['period'] > cellPeriod[(p,m)]) 
                & (spInjTces['period'] <= cellPeriod[(p,m)]+dPeriod) 
                & (spInjTces['Expected_MES'] > cellMes[(p,m)]) 
                & (spInjTces['Expected_MES'] <= cellMes[(p,m)]+dMes));
            tceGrid[(p,m)] = len(pointsInCell[0]);

            pointsInCell = np.where(
                (spInjPcs['period'] > cellPeriod[(p,m)]) 
                & (spInjPcs['period'] <= cellPeriod[(p,m)]+dPeriod) 
                & (spInjPcs['Expected_MES'] > cellMes[(p,m)]) 
                & (spInjPcs['Expected_MES'] <= cellMes[(p,m)]+dMes));
            pcGrid[(p,m)] = len(pointsInCell[0]);


            pointsInCell = np.where(
                (spInjFps['period'] > cellPeriod[(p,m)]) 
                & (spInjFps['period'] <= cellPeriod[(p,m)]+dPeriod) 
                & (spInjFps['Expected_MES'] > cellMes[(p,m)]) 
                & (spInjFps['Expected_MES'] <= cellMes[(p,m)]+dMes));
            fpGrid[(p,m)] = len(pointsInCell[0]);


    drawHeatMap(tceGrid, (15,15), cellPeriod, cellMes, colorBarLabel="# of TCEs");  
    plt.ylabel('Expected MES', fontsize = 16);
    plt.xlabel('Period', fontsize = 16);
    plt.savefig("vetCompNTCEs.eps",bbox_inches='tight')
    plt.title("All Injected TCEs");


    drawHeatMap(pcGrid, (15,15), cellPeriod, cellMes, colorBarLabel="# of PCs");           
    plt.title("Injected TCEs Dispositioned as PCs");
    plt.ylabel('Expected MES', fontsize = 16);
    plt.xlabel('Period', fontsize = 16);
    drawHeatMap(fpGrid, (15,15), cellPeriod, cellMes, colorBarLabel="# of FPs");           
    plt.title("Injected TCEs Dispositioned as FPs");
    plt.ylabel('Expected MES', fontsize = 16);
    plt.xlabel('Period', fontsize = 16);



    # Compute the PC fraction in each cell.  This is not used in our analysis, but is a qualitative guide suggesting what functions may be the best fit.

    pcFrac = np.zeros(np.shape(tceGrid))
    minTcePerCell = 0;
    pcFrac[tceGrid>minTcePerCell] = pcGrid[tceGrid>minTcePerCell]/tceGrid[tceGrid>minTcePerCell];
    drawHeatMap(np.round(100*pcFrac), (15,15), cellPeriod, cellMes, colorBarLabel="% PCs", nData = tceGrid);           
    plt.ylabel('Expected MES', fontsize = 16);
    plt.xlabel('Period', fontsize = 16);
    plt.savefig("vetCompInjRate.eps",bbox_inches='tight')
    plt.title("% of Injected TCEs Dispositioned as PCs", fontsize = 16);


    pcFlat = pcGrid.flatten();
    tceFlat = tceGrid.flatten();

    # convert to homogeneous coordinates on unit square [0,1]
    cellX, cellY = funcModels.normalizeRange(cellPeriod, cellMes, [periodMin, periodMax], [mesMin, mesMax]);
    gridShape = np.shape(cellX);
    dx = 1./gridShape[0];
    dy = 1./gridShape[1];
    print("gridShape = " + str(gridShape) + ", dx = " + str(dx) + ", dy = " + str(dy))

    cellXFlat = cellX.flatten();
    cellYFlat = cellY.flatten();

    pcFlat[tceFlat<minTcePerCell] = 0;
    tceFlat[tceFlat<minTcePerCell] = 0;

    tceData = [cellXFlat, cellYFlat, tceFlat, pcFlat];


    # We're ready to compute a Bayesian inference of the success probability $r$:
    # 
    # $$p(r|c, n) \propto p(c|r, n) p(r).$$
    # 
    # But we're computing $r$ as a function of period $p$, MES $m$, and parameters $\theta$, $r(\theta, p, m)$. So our inference becomes 
    # 
    # $$p(\theta|c, n, p, m) \propto p(c|\theta, n, p, m) p(\theta).$$
    # 
    # Because each cell is independent, we linearize the array to a list of cells indexed by $k$.  Then the likelihood for each cell is
    # 
    # $$p(c_k|\theta, n_k, p_k, m_k) = \left( \begin{array}{c_k} n_k \\ c_k \end{array} \right) r(\theta, p_k , m_k )^{c_k} (1-r(\theta, p_k , m_k ))^{n_k-c_k}$$
    # 
    # Because the $N$ cells are independent, the likelihood for the collection of cells is
    # 
    # $$p(c|\theta, n, p, m) \equiv p(c_1, \ldots, c_N|\theta, n_1, \ldots, n_N, p_1, \ldots, p_N, m_1, \ldots, m_N) = \
    # \prod_k \left( \begin{array}{c_k} n_k \\ c_k \end{array} \right) r(\theta, p_k , m_k )^{c_k} (1-r(\theta, p_k , m_k ))^{n_k-c_k}.$$
    # 
    # The log-likelihood is then
    # 
    # $$\log p(c|\theta, n, p, m) = \sum_k \log \left(\left( \begin{array}{c_k} n_k \\ c_k \end{array} \right) r(\theta, p_k , m_k )^{c_k} (1-r(\theta, p_k , m_k ))^{n_k-c_k} \right)$$ 
    # $$= \sum_k \left[ \log \left(\begin{array}{c_k} n_k \\ c_k \end{array} \right) + c_k \log \left(r(\theta, p_k , m_k ) \right) + \left( n_k-c_k \right) \log(1-r(\theta, p_k , m_k )) \right] $$


    # We now perform the MCMC calculation of the log posterior. 
    #We use a manual three-part strategy to find the best seed point: first we use the initial values in the model definitions. 
    # This MCMC result is used to seed the second iterations, whose result is used to seed the third and final iteration.  
    #We then initialize the MCMC walkers with a small Gaussian distribution around these starting points.


    model = "logisticX0xRotatedLogisticY02"

    if model == "logisticY0":
        initialPos = [ 0.14828331, 17.33408434,  0.92967224];
    elif model == "logisticX0xlogisticY0":
        initialPos = [ 0.99560747,  0.14886312,  4.89150268, 22.22093359,  0.94770438];
    elif model == "logisticX0xlogisticY02":
        initialPos = [ 0.81952689,  0.23524868,  4.7697679,  19.05683796,  2.05838984,  4.60039486, 0.99113786];
    elif model == "rotatedLogisticX0xlogisticY0":
        initialPos = [ 1.08006382,  0.14448152,  4.07613918, 15.42298136,  0.75826285,  7.55065579,  0.98498245];
    elif model == "logisticX0xRotatedLogisticY0":
        initialPos = [  1.01211435,   0.14979708,   5.4723763,   13.37211524, 0,   0.98427409]
    elif model == "logisticX0xRotatedLogisticY02":
        initialPos = [  1.01211435,   0.14979708,   5.4723763,   13.37211524, 5, 0,   0.98427409]
    elif model == "rotatedLogisticX0xlogisticY02":
        initialPos = [ 1.21729902,  0.25877038,  5.03162786, 21.4166877,   0.51381796,  4.01130569,  3.30628676,  6.15551683,  0.97623034];
    elif model == "dualBrokenPowerLaw":
        initialPos = [ 0.44265552,  0.3179024,  -0.10705209, -0.98008117,  2.72776688,  0.06786803, 0.92982262];
    else:
        initialPos = funcModels.initRateModel(model);

    print(initialPos)


    nll = lambda *args: -lnBinlike(*args)
    result = op.minimize(nll, initialPos, args=(tceData, model))
    maxLikelihoodResult = result["x"];
    modelLabels = funcModels.getModelLabels(model)
    for i in range(0,len(maxLikelihoodResult)):
        print("maximum Likelihood " + modelLabels[i] + ":={:.3f}".format(maxLikelihoodResult[i]))
        
    if lnBinprior(maxLikelihoodResult, tceData, model) == -np.inf:
        maxLikelihoodResult = initialPos;
        print("violates prior, replacing maxLikelihoodResult result with initialPos")

    x, y, n, c = tceData
    r = funcModels.rateModel(x,y,maxLikelihoodResult,model);
    print("maximum Likelihood rate min = {:.3f}".format(np.min(np.min(r))) + ", max = {:.3f}".format(np.max(np.max(r))))



    # we'll use 20 walkers and 2000 steps so this doesn't take too long
    # 100 walkers and 5000 steps is better

    ndim, nwalkers = len(maxLikelihoodResult), 20
    pos = [maxLikelihoodResult + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnBinprob, args=(tceData, model))
     
    sampler.run_mcmc(pos, 2000);

    samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
    modelLabels = funcModels.getModelLabels(model)
    dataResult = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    dataResult = list(dataResult)
    for i in range(0,ndim):
        v = dataResult[i];    
        print("MCMC " + modelLabels[i] + ":={:.3f}".format(v[0]) + "+{:.3f}".format(v[1]) + "-{:.3f}".format(v[2]))
        # print("true " + modelLabels[i] + ":={:.3f}".format(trueTheta[i]))

    resultSize = np.shape(dataResult);
    fitTheta = np.zeros(resultSize[0]);
    for i in range(resultSize[0]):
        fitTheta[i] = dataResult[i][0]
    print("pcFitTheta:")
    print(fitTheta)



    reload(funcModels)
    plt.figure(figsize=(10,5))
    for i in range(0,ndim):
        plt.subplot(ndim,1,i+1)
        plt.plot(np.transpose(sampler.chain[:, :, i]), color="k", alpha=0.1);
        plt.ylabel(modelLabels[i]);
        

    modelLabels = funcModels.getModelLabels(model)

    fig = corner.corner(samples, labels = modelLabels, label_kwargs = {"fontsize": 32}, truths = fitTheta)
    plt.savefig("vetCompPost.eps",bbox_inches='tight')


    # We test the result by reconstructing the distribution of PCs and % of PCs using binomial-distributed random numbers, to see if they match the actual data.

    fitGrid = np.zeros(np.shape(tceGrid));

    resultSize = np.shape(dataResult);
    fitTheta = np.zeros(resultSize[0]);
    for i in range(resultSize[0]):
        fitTheta[i] = dataResult[i][0]
    print("fitTheta = " + str(fitTheta))

    for p in range(NPeriod):
        for m in range(NMes):
            fitGrid[(p,m)] = np.random.binomial(tceGrid[(p,m)], 
                funcModels.rateModel(cellX[(p,m)]+dx/2, cellY[(p,m)]+dy/2, fitTheta, model), 1);
            
    drawHeatMap(fitGrid, (15,15), cellPeriod, cellMes, nData = tceGrid);           
    plt.title("Simulated TCEs Dispositioned as PCs");
    plt.ylabel('Expected MES');
    plt.xlabel('Period');

    fitFrac = np.zeros(np.shape(tceGrid))
    fitFrac[tceGrid>minTcePerCell] = fitGrid[tceGrid>minTcePerCell]/tceGrid[tceGrid>minTcePerCell];
    drawHeatMap(np.round(100*fitFrac), (15,15), cellPeriod, cellMes, nData = tceGrid);           
    plt.title("Simulated % of TCEs Dispositioned as PCs");
    plt.ylabel('Expected MES');
    plt.xlabel('Period');


    # Now do many realizations, and subtract the average from the observed to look for systematic differences.

    nFits = 100;
    fitGrid = np.zeros([np.shape(tceGrid)[0],np.shape(tceGrid)[1],nFits]);
    sidx = [0]*nFits
    progress = FloatProgress(min=0, max=nFits)
    display(progress)

    for f in range(nFits):
        sidx[f] = int(np.random.uniform(high=samples.shape[0]-1));
        tTheta = samples[sidx[f],:]
        for p in range(NPeriod):
            for m in range(NMes):
                rm = funcModels.rateModel(cellX[(p,m)]+dx/2, cellY[(p,m)]+dy/2, tTheta, model)
                if rm > 1:
                    rm = 1;
                fitGrid[(p,m,f)] = np.random.binomial(tceGrid[(p,m)], rm, 1);
        progress.value += 1

        
    meanFit = np.mean(fitGrid, 2)
    stdFit = np.std(fitGrid, 2)



    drawHeatMap(meanFit, (15,15), cellPeriod, cellMes, nData = tceGrid);           
    plt.title("Mean Number of Simulated TCEs Dispositioned as PCs");
    plt.ylabel('Expected MES');
    plt.xlabel('Period');

    fitFracMean = np.zeros(np.shape(tceGrid))
    fitFracMean[tceGrid>minTcePerCell] = meanFit[tceGrid>minTcePerCell]/tceGrid[tceGrid>minTcePerCell];
    drawHeatMap(np.round(100*fitFracMean), (15,15), cellPeriod, cellMes, nData = tceGrid, colorBarLabel="Mean PC %");           
    plt.ylabel('Expected MES', fontsize = 16);
    plt.xlabel('Period', fontsize = 16);
    plt.savefig("vetCompMean.eps",bbox_inches='tight')
    plt.title("Mean Simulated % of TCEs Dispositioned as PCs");

    fitFracStd = np.zeros(np.shape(tceGrid))
    fitFracStd[tceGrid>minTcePerCell] = stdFit[tceGrid>minTcePerCell]/tceGrid[tceGrid>minTcePerCell];
    drawHeatMap(np.round(100*fitFracStd), (15,15), cellPeriod, cellMes, nData = tceGrid, colorBarLabel="Standard Deviation %");           
    plt.ylabel('Expected MES', fontsize = 16);
    plt.xlabel('Period', fontsize = 16);
    plt.savefig("vetCompStd.eps",bbox_inches='tight')
    plt.title("Standard Deviation of the Simulated % of TCEs Dispositioned as PCs");



    fitDiff = fitFracMean - pcFrac
    fitDiffNorm =np.zeros(fitDiff.shape)
    fitDiffNorm[tceGrid>minTcePerCell] = fitDiff[tceGrid>minTcePerCell]/stdFit[tceGrid>minTcePerCell];

    drawHeatMap(np.round(100*fitDiff), (15,15), cellPeriod, cellMes, nData = tceGrid);           
    plt.title("Residual from mean");
    plt.ylabel('Expected MES');
    plt.xlabel('Period');

    drawHeatMap(np.round(fitDiffNorm), (15,15), cellPeriod, cellMes, nData = tceGrid, 
                colorBarLabel=r"Mean Residual ($\sigma$)"); 
    plt.ylabel('Expected MES', fontsize = 16);
    plt.xlabel('Period', fontsize = 16);
    plt.savefig("vetCompMeanResid.eps",bbox_inches='tight')
    plt.title("Residual from mean (standard deviations)");



    # Plot the fitted functional form of the vetting efficiency.


    Z = funcModels.rateModel(cellX, cellY, fitTheta, model);

    fig = plt.figure(figsize=plt.figaspect(0.3));
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    surf = ax.plot_surface(cellPeriod, cellMes, Z, alpha = 0.5);
    scat = ax.scatter(cellPeriod[tceGrid>0], cellMes[tceGrid>0], pcFrac[tceGrid>0], c='r', marker = '.');
    plt.xlabel("period");
    plt.ylabel("Expected MES");
    ax.view_init(0,0)

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    surf = ax.plot_surface(cellPeriod, cellMes, Z, alpha = 0.5);
    scat = ax.scatter(cellPeriod[tceGrid>0], cellMes[tceGrid>0], pcFrac[tceGrid>0], c='r', marker = '.');
    plt.xlabel("period");
    plt.ylabel("Expected MES");
    ax.view_init(0,-90)
    plt.title("Vetting completeness");

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(cellPeriod, cellMes, Z, alpha = 0.5);
    scat = ax.scatter(cellPeriod[tceGrid>0], cellMes[tceGrid>0], pcFrac[tceGrid>0], c='r', marker = '.');
    plt.xlabel("period");
    plt.ylabel("Expected MES");

    fig, ax = plt.subplots(figsize=(15,10));
    CS = ax.contour(cellPeriod, cellMes, Z, colors='k', levels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .96, .97, .98, .99]);
    ax.clabel(CS, inline=1, fontsize=18);
    scf = ax.scatter(cellPeriod[tceGrid>0], cellMes[tceGrid>0], cmap="cividis", c=pcFrac[tceGrid>0], s=5*tceGrid[tceGrid>0], alpha = 0.5);
    plt.xlabel("period", fontSize = 24);
    plt.ylabel("Expected MES", fontSize = 24);
    cbh = plt.colorbar(scf);
    cbh.ax.set_ylabel("Measured Fraction");
    plt.tick_params(labelsize = 12)

    plt.savefig("vetCompContours.eps",bbox_inches='tight')
    plt.title("Vetting completeness.  Size of marker = # of TCEs in cell", fontSize = 24);



    r1 = np.zeros(np.shape(samples)[0])
    xp = 50.;
    yp = 25.;
    cx = (xp - periodMin)/(periodMax - periodMin);
    cy = (yp - mesMin)/(mesMax - mesMin);
    for i in range(np.shape(samples)[0]):
        r1[i] = funcModels.rateModel(cx, cy, samples[i,:], model)
    f1 = funcModels.rateModel(cx, cy, fitTheta, model)
        
    r2 = np.zeros(np.shape(samples)[0])
    xp = 365.;
    yp = 10.;
    cx = (xp - periodMin)/(periodMax - periodMin);
    cy = (yp - mesMin)/(mesMax - mesMin);
    for i in range(np.shape(samples)[0]):
        r2[i] = funcModels.rateModel(cx, cy, samples[i,:], model)
    f2 = funcModels.rateModel(cx, cy, fitTheta, model)



    greyLevel = "0.7"
    rr = np.percentile(r1, [5, 95]);
    plt.hist(r1[(r1 > 0.95*rr[0]) & (r1 < 1.05*rr[1])], 100, color=greyLevel);
    plt.plot([f1, f1], [0, 3000], color='k', linestyle='--', linewidth=1)

    rr = np.percentile(r2, [5, 95]);
    plt.hist(r2[(r2 > 0.95*rr[0]) & (r2 < 1.05*rr[1])], 100, color=greyLevel);
    plt.plot([f2, f2], [0, 3000], color='k', linestyle='--', linewidth=1)

    plt.xlabel("Vetting Completeness")
    plt.savefig("vetCompExamples.eps",bbox_inches='tight')


    return

def main(argv):
    '''

    '''

    try:
        argument_list = argv[1:]
        short_options = 'i:v:s:t:p:P:r:R:' #plots y/n, verbose y/n, savepath for files, spectral types to include, period min, period max, rp min, rp max
        long_options = 'images:verbose:savepath:spt:pmin:pmax:rpmin:rpmax:' 
        arguments, values = getopt.getopt(argument_list, short_options, long_options)

        # get out the various results of the keywords
        plots = strtobool(str(arguments[0][1]))
        verbose = strtobool(str(arguments[1][1]))
        savepath = arguments[2][1] + '/'
        spt = arguments[3][1].lower()
        pmin = float(arguments[4][1])
        pmax = float(arguments[5][1])
        rpmin = float(arguments[6][1])
        rpmax = float(arguments[7][1])
        mesMin = 0
        mesMax = 30
    except:
        print('No inputs given, running iwth default settings: plots = True, verbose = True, savepath = current working directory, spt = \'GK\', min. period = 1 d\
            max period = 400 d, min Rp = 0.5 Re, max Rp = 15 Re. The code always uses all MES values: a range of 0-30.')

        plots = True
        verbose = True
        savepath = os.getcwd() + '/'
        spt = 'GK'.lower()
        pmin = 1
        pmax = 400
        rpmin = 0.5
        rpmax = 15
        mesMin = 0
        mesMax = 30

    # check the spectral type inputs are valid
    test_spt = [s not in 'fgkm' for s in spt]

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
    print('Assessing vetting completeness with the following settings: plots = {}, verbose = {}, savepath = {}, spectral types = {}, \
        P min = {} d, P max = {} d, Rp min = {} Re, Rp max = {} Re, MES min = {}, MES max = {}'.format(plots, verbose, savepath, spt, pmin, pmax, rpmin, rpmax, mesMin, mesMax))

    vettingCompleteness(plots = plots, verbose = verbose, savepath = savepath, stellarType = spt,\
        periodMin = pmin, periodMax = pmax, rpMin = rpmin, rpMax = rpmax, mesMin = mesMin, mesMax = mesMax)

    return

if __name__ == "__main__":
    main(sys.argv)


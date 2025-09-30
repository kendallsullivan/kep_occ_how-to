import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import sys

output_prefix = sys.argv[1]
pcCatalog = sys.argv[2] #'koiCatalogs/dr25_FGK_PCs_B20_ruwe.csv'

x = np.linspace(np.log10(1), np.log10(500), num=100)
y = np.linspace(np.log10(1), np.log10(10), num=100)
xg, yg = np.meshgrid(x, y, indexing='ij')

period_rng = (1, 500)
n_period = 101
rp_rng = (1, 10)
n_rp = 101

whichRadii = "corrected"
def getRadii(catalog):
	if whichRadii == "corrected":
		return catalog.corrected_prad
	elif whichRadii == "kic":
		return catalog.koi_prad
	else:
		raise ValueError('Bad whichRadii string')

base_kois = pd.read_csv(pcCatalog)
m = (period_rng[0] <= base_kois.koi_period) & (base_kois.koi_period <= period_rng[1])
thisRadii = getRadii(base_kois)
m &= np.isfinite(thisRadii) & (rp_rng[0] <= thisRadii) & (thisRadii <= rp_rng[1])
	
kois = pd.DataFrame(base_kois[m])
#kois = kois[:10] # for testing
print(len(kois))

def gauss_2D(x, y, amp, mux, muy, sigx, sigy):
	return(amp * np.exp(-0.5 * (((x - mux) / sigx)**2 + ((y - muy) / sigy)**2)))

def make_kde(xg, yg, sigx, sigy, per, rad, r):
	Z = np.zeros(xg.shape)
	g = np.zeros(xg.shape)

	for i in range(len(per)):
		g = gauss_2D(xg, yg, r[i] /(sigx[i]*sigy[i]*2*np.pi),
					np.log10(per[i]), 
					np.log10(rad[i]),
					sigx[i],
					sigy[i])
		Z += g

	return(Z)


def sample(rel=True):
   # Re-seed the random number generator
	np.random.seed()

	if rel==True:
		s = np.random.choice(len(kois), len(kois), replace=True)
		r = kois.totalReliability.values

	else:
		s = np.random.choice(len(kois), len(kois), replace=True) 
		r = np.zeros(len(kois)) + 1.0
	
	sx = np.zeros(len(kois)) + 0.1
	sy = np.zeros(len(kois)) + 0.1
	
	# shake values based on uncertainties on period and radius
	ep_x = np.random.normal(scale=(0.01/(np.log(10)*kois.koi_period.iloc[s].values)),
							# size=len(kois))
							)
	ep_y = np.random.normal(scale=(kois.corrected_prad_err1.iloc[s].values/(np.log(10)*kois.corrected_prad.iloc[s].values)),
							# size=len(kois))
							)
	ep_y[np.abs(ep_y) > 1.] = 1. # for those crazy values idk

	periods = 10**(np.log10(kois.koi_period.iloc[s].values) + ep_x)
	radii = 10**(np.log10(kois.corrected_prad.iloc[s].values) + ep_y)

	return make_kde(xg, yg, sx, sy, periods, radii, r)

def _parallel_mc(iter=100, rel=True):
	pool = mp.Pool(4) # change this

	future_res = np.array([pool.apply_async(sample, (rel,)) for _ in range(iter)])
	print(future_res.shape)
	res = np.array([f.get() for f in future_res])
	print(res.shape)

	return res

def parallel_monte_carlo(iter=100, rel=True):
	samples = _parallel_mc(iter, rel)
	print(samples.shape)
	print(samples[0].shape)

	o = np.average(samples, axis=0)
	#o_std = np.std(samples, axis=0)
	print(o.shape)

	return samples

if __name__=="__main__":

	np.save(output_prefix + "_5000.npy", parallel_monte_carlo(iter=5000, rel=True))
	# parallel_monte_carlo(iter=5, rel=True)
import sys, os

from Payne.fitting import fitstar
from astropy.table import Table
from astropy.io import fits
import h5py
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

from chelle import get_chelle

inputdict = {}

# set parameter for sampler
inputdict['sampler'] = {}
inputdict['sampler']['samplertype'] = 'Static'
inputdict['sampler']['samplemethod'] = 'rwalk'
inputdict['sampler']['npoints'] = 125
inputdict['sampler']['samplerbounds'] = 'multi'
inputdict['sampler']['flushnum'] = 100
inputdict['sampler']['delta_logz_final'] = 1.0
inputdict['sampler']['bootstrap'] = 0
inputdict['sampler']['walks'] = 25
# inputdict['sampler']['slices'] = 500

# set some flat priors for defining the prior volume
inputdict['priordict'] = {}
inputdict['priordict']['Teff']   = {'pv_uniform':[4000.0,7000.0]}
inputdict['priordict']['log(g)'] = {'pv_uniform':[4.0,5.5]}
inputdict['priordict']['[Fe/H]'] = {'pv_uniform':[-1.5,0.5]}
inputdict['priordict']['[a/Fe]'] = {'pv_uniform':[-0.1,0.1]}
inputdict['priordict']['Vrad'] = {'pv_uniform':[-500.0,500.0]}
inputdict['priordict']['Vrot']   = {'pv_uniform':[0.0,300.0]}

# set an additional guassian prior on the instrument profile
inputdict['priordict']['Inst_R'] = (
	{'pv_uniform':[27000.0,38000.0],
	'pv_gaussian':[32000.0,1000.0]})
# Distance/redenning priors set for each object

numpoly = 4
# list of [value, sigma] pairs
coeffarr = [[1.0,0.5],[0.0,0.1]] + [[0.0,0.05] for ii in range(numpoly-2)]
inputdict['priordict']['blaze_coeff'] = coeffarr

##################

def run_one_chelle(wav, flu, err, obj_id, output_dir, inputdict=inputdict,
                  runspec=True, runphot=False):
    
    if runspec:
        inputdict['spec'] = {}
        inputdict['specANNpath'] = '/n/home13/stdouglas/code/python/thepayne/data/specANN/FALANN_RVS31_v7.1.h5'

        inputdict['spec']['obs_wave'] = wav
        inputdict['spec']['obs_flux'] = flu
        inputdict['spec']['obs_eflux'] = err
        inputdict['spec']['normspec'] = True #Means spectrum already normalized
        inputdict['spec']['convertair'] = True #models are vac, obs are air

    # if Gaia parallax, can change to using a dist prior (otherwise degen & breaks)
    # inputdict['priordict']['Dist']   = {'pv_uniform':[1.0,200.0]}
    # inputdict['priordict']['log(R)'] = {'pv_uniform':[-0.1,0.1]}
    inputdict['photscale'] = True
    inputdict['priordict']['log(A)'] = {'pv_uniform':[-3.0,7.0]}
    inputdict['priordict']['Av']     = {'pv_uniform':[0.0,1.0]}

    inputdict['output'] = os.path.join(output_dir,
                                       'chelle_test_{0}.dat'.format(obj_id))

    FS = fitstar.FitPayne()
    print('---------------')
    if 'phot' in inputdict.keys():
        print('    PHOT:')
        for kk in inputdict['phot'].keys():
            print('       {0} = {1} +/- {2}'.format(kk,inputdict['phot'][kk][0],inputdict['phot'][kk][1]))
    if 'spec' in inputdict.keys():
        print('    Median Spec Flux: ')
        print('       {0}'.format(np.median(inputdict['spec']['obs_flux'])))
        print('    Median Spec Err_Flux:')
        print('       {0}'.format(np.median(inputdict['spec']['obs_eflux'])))

    if 'priordict' in inputdict.keys():
        print('    PRIORS:')
        for kk in inputdict['priordict'].keys():
            if kk == 'blaze_coeff':
                pass
            else:
                for kk2 in inputdict['priordict'][kk].keys():
                    if kk2 == 'pv_uniform':
                        print('       {0}: min={1} max={2}'.format(kk,inputdict['priordict'][kk][kk2][0],inputdict['priordict'][kk][kk2][1]))
                    if kk2 == 'pv_gaussian':
                        print('       {0}: N({1},{2})'.format(kk,inputdict['priordict'][kk][kk2][0],inputdict['priordict'][kk][kk2][1]))

    print('--------------')

    sys.stdout.flush()
    result = FS.run(inputdict=inputdict)
    sys.stdout.flush()


def run_chelle(filename):

    wav, flu, err, realspec, obj_id  = get_chelle.read_chelle(chellefile)
    #i = np.where(obj_id=='9531080')[0][0]
    #i = np.where(obj_id=='9470739')[0][0]
    #i = np.where(obj_id=='9471796')[0][0]
    real_idx = np.where(realspec)[0]
    print(len(real_idx),"total spectra in this field")

    j=os.getenv("SLURM_ARRAY_TASK_ID",default=20)
    print(j)
    i = real_idx[int(j)]
    print(i,obj_id[i])

    print("w",wav[i])
    print("f",flu[i])
    print("e",err[i])
    good = np.isfinite(wav[i]) & np.isfinite(flu[i]) & np.isfinite(err[i]) & (wav[i]<5290) & (wav[i]>5160)
    print(len(np.where(good)[0]),"good points in the spectrum")

    run_one_chelle(wav[i][good], flu[i][good], err[i][good], obj_id[i],
                  output_dir="/n/scratchlfs/conroy_lab/stdouglas/chelle/outputs")

if __name__=="__main__":

    chellefile = "/n/home13/stdouglas/data/Hectochelle/2018.0527/RV31_1x1/hectochelle_NGC6811_2018a_1/spHect-hectochelle_NGC6811_2018a_1.2512-0100.fits"
    run_chelle(chellefile)

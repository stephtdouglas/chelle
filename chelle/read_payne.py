import astropy.io.ascii as at
from astropy.table import Table
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.

    from:
    https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def collect_all(file_list, table_file):

    nf = len(file_list)
    tab = Table([np.empty(nf,"U200"),np.ones(nf)*np.nan,np.ones(nf)*np.nan,
                 np.ones(nf)*np.nan,np.ones(nf)*np.nan,np.ones(nf)*np.nan,
                 np.ones(nf)*np.nan,np.ones(nf)*np.nan,np.ones(nf)*np.nan,
                 np.ones(nf)*np.nan,np.ones(nf)*np.nan,np.ones(nf)*np.nan,
                 np.ones(nf)*np.nan],
                names=["filename","Teff","e_Teff","log(g)","e_log(g)",
                       "[Fe/H]","e_[Fe/H]","[a/Fe]","e_[a/Fe]",
                       "Vrad","e_Vrad","Vrot","e_Vrot"], 
               masked=False)

    print(tab.dtype)
    print(tab.dtype.names)

    for i, fname in enumerate(file_list):
        dat = at.read(fname)
        weights = np.exp(dat["log(wt)"]-dat["log(z)"][-1])

        for val in ['filename', 'Teff', 'log(g)', '[Fe/H]', '[a/Fe]', 'Vrad', 'Vrot']:
            if val=="filename":
                tab[i][val] = fname
            else:
                e_val = "e_"+val
                w_median = weighted_quantile(dat[val],0.5,sample_weight=weights)
                tab[i][val] = w_median
                w_u = weighted_quantile(dat[val],[0.16,0.84],
                                        sample_weight=weights)
                tab[i][e_val] = np.sum(abs(w_u-w_median))/2
                

    at.write(tab,table_file,delimiter=",",overwrite=True)


def plot_test():
    dat = at.read("/n/scratchfls/conroy_lab/stdouglas/outputs/chelle_testout.dat")

    weights = np.exp(dat["log(wt)"]-dat["log(z)"][-1])

    plt.figure(figsize=(8,10))
    ax = plt.subplot(311)
    _ = ax.hist(dat["Teff"],weights=weights,bins=30)
    ax.set_xlabel("Teff")
    ax = plt.subplot(312)
    _ = ax.hist(dat["Vrad"],weights=weights,bins=30)
    ax.set_xlabel("RV")
    ax = plt.subplot(313)
    _ = ax.hist(dat["Vrot"],weights=weights,bins=30)
    ax.set_xlabel("Vrot")
    plt.savefig("/n/scratchfls/conroy_lab/stdouglas/outputs/hecto_testout.png")
    plt.close()

if __name__=="__main__":

    files = at.read("/n/scratchfls/conroy_lab/stdouglas/outputs/test_outs.lst")
    collect_all(files["filename"],
                "/n/scratchfls/conroy_lab/stdouglas/outputs/test_collect.csv")

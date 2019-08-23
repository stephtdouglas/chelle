import os, sys
from astropy.io import fits
import astropy.io.ascii as at
import numpy as np
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

# blaze functions are different on each chip
# 1-120 on one, 121-240 on the other (except 0 indexed instead)
def read_blaze(filename):
    with fits.open(filename) as hdu:
        wave = hdu[1].data["WAVE"]*10
        blaze = hdu[1].data["BLAZE"]
    return wave, blaze

def make_blaze(filename):
    ww, bb = read_blaze(filename)
    blaze = interp1d(ww,bb,kind="linear",bounds_error=False,fill_value=np.nan,
                     assume_sorted=False)
    print(min(ww),max(ww))
    return blaze

blaze1 = make_blaze("/n/home13/stdouglas/data/Hectochelle/master_blaze_det1_smooth.fits")
blaze2 = make_blaze("/n/home13/stdouglas/data/Hectochelle/master_blaze_det2_smooth.fits")

def read_chelle(filename,wave_ext=0,flux_ext=1,err_ext=2,idl=True):
    if os.path.exists(filename):
        wave = np.array(fits.getdata(filename,0))
        flux_raw = np.array(fits.getdata(filename,1))
        inv_err = np.array(fits.getdata(filename,2))
        err_raw = 1/np.sqrt(inv_err)
        mask = np.array(fits.getdata(filename,3))
        obj_data = fits.getdata(filename,5)
    else:
        print(filename,"NOT FOUND")
        return None, None, None

    # remove masked pixels (AND mask)
    flux_raw[mask] = np.nan
    
    # sigma clip outliers
    # TODO: vectorize!
    std = np.std(flux_raw,axis=1)
    fmed_raw = np.nanmedian(flux_raw, axis=1)
    outlier_mask = np.zeros_like(flux_raw,dtype=bool)
    for i,fs in enumerate(flux_raw):
        outlier_mask[i] = abs(fs-fmed_raw[i])>(std[i]*5)
        #print(len(np.where(outlier_mask[i])[0]))
    flux_raw[outlier_mask] = np.nan

    # TODO: divide by nanmedian before dividing by blaze
    fmed = np.nanmedian(flux_raw,axis=1)
    #print(fmed)
    #print(fmed.reshape(240,-1))
    flux0 = np.divide(flux_raw,fmed.reshape(240,-1))
    err0 = np.divide(err_raw,fmed.reshape(240,-1))

    obj_id = obj_data["OBJTYPE"]
    realspec = np.ones(240)
    realspec[obj_data["OBJTYPE"]=="UNUSED"] = False
    realspec[obj_data["OBJTYPE"]=="SKY"] = False
    realspec[np.all(np.isnan(flux_raw),axis=1)] = False
    for i in np.where(realspec)[0]:
        if "REJECT" in obj_data["OBJTYPE"][i]:
            realspec[i] = False

    fblaze1 = blaze1(wave[:120])
    fblaze2 = blaze2(wave[120:])
    
    flux = np.ones_like(flux0)
    err = np.ones_like(err0)
    
    flux[:120] = flux0[:120]/fblaze1
    err[:120] = err0[:120]/fblaze1

    flux[120:] = flux0[120:]/fblaze2
    err[120:] = err0[120:]/fblaze2

    return wave, flux, err, realspec, obj_id

def write_chelle(filename,outfile_base,output_dir):

    if os.path.exists(filename):
        wav, flu, err, realspec, obj_id = read_chelle(filename)
    else:
        print(filename,"does not exist")
        return

    for i in np.where(realspec)[0]:
        outfile = os.path.join(output_dir,
                               "{0}_{1}.csv".format(outfile_base,obj_id[i]))
        at.write({"wavelength":wav[i],"flux":flu[i],"error":err[i]},outfile,
                 names=["wavelength","flux","error"],overwrite=True)        

if __name__=="__main__":
    hectofile = "/n/home13/stdouglas/data/Hectochelle/2018.0527/RV31_1x1/hectochelle_NGC6811_2018a_1/spHect-hectochelle_NGC6811_2018a_1.2512-0100.fits"

#    write_chelle(hectofile,"test",
#                "/n/scratchlfs/conroy_lab/stdouglas/payne_demo/spectra/")


    wav, flu, err, realspec, obj_id  = read_chelle(hectofile)
    print(wav[np.where(realspec)[0][0]])

    plt.figure(figsize=(10,5))
    plt.xlim(5160,5200)
    plt.ylim(-10,10)
    for i in np.where(realspec)[0]:
        plt.plot(wav[i], flu[i], alpha=0.25)

    plt.savefig("/n/scratchlfs/conroy_lab/stdouglas/chelle/plots/test_hecto.png")
    plt.close()

    i = np.where(obj_id=='9531080')[0][0]
    plt.figure(figsize=(10,5))
    print(i,wav[i],flu[i])
    plt.plot(wav[i], flu[i])
    #plt.ylim(np.percentile(flu[i][np.isfinite(flu[i])],1),
    #         np.percentile(flu[i][np.isfinite(flu[i])],99))
    plt.savefig("/n/scratchlfs/conroy_lab/stdouglas/chelle/plots/test_hecto_9531080.png")
    plt.close()

import emcee
import time
from multiprocessing import Pool
import numpy as np
from IPython import embed
from fitting_script_helpers import log_posterior
import os
from disksurf import observation

def main():
    t0 = time.time()
    ndim = 4
    nthreads = 6
    nsteps = 10000
    ninits = 500
    nwalk = 64
    if nthreads > 1: os.environ["OMP_NUM_THREADS"] = "1"

    cube_CO = observation('CO/HD_163296_CO_220GHz.0.3arcsec.JvMcorr.image.pbcor.fits', FOV=10.0)
    cube_13CO =  observation('13CO/FinalSingleProject/HD163296_13CO_beam0.20_100ms_3sigma.clean.image.fits', FOV=10.0)
    cube_C18O = observation('C18O/HD163296_C18O_robust1.00_40ms_3sigma.clean.image.fits', FOV=10.0)

    # Channels
    chans_CO = [(50, 61), (65, 76)]
    chans_13CO = [(30, 55), (59, 84)]
    chans_C18O =  [(70, 134), (150, 214)]

    # Disk Params
    x0 = 0 # arcsec
    y0 = 0  # arcsec
    inc = 46.7   # deg
    PA = 312.0   # deg
    vlsr = 5.7e3 # m/s

    # Get the emission surface
    surface_CO = cube_CO.get_emission_surface(x0=x0, y0=y0, inc=inc, PA=PA,
                                        vlsr=vlsr, chans=chans_CO, smooth=1.0)
    surface_13CO = cube_13CO.get_emission_surface(x0=x0, y0=y0, inc=inc, PA=PA,
                                        vlsr=vlsr, chans=chans_13CO, smooth=1.0)
    surface_C18O = cube_C18O.get_emission_surface(x0=x0, y0=y0, inc=inc, PA=PA,
                                                vlsr=vlsr, chans=chans_C18O, smooth=1.0)

    # Mask the surface  
    surface_CO.mask_surface(side='both', min_zr=0.0, max_zr=1.0, reflect=True, min_SNR=10.0)
    surface_13CO.mask_surface(side='both', min_zr=0.0, max_zr=1.0, reflect=True, min_SNR=10.0)
    surface_C18O.mask_surface(side='both', min_zr=0.0, max_zr=1.0, reflect=True, min_SNR=10.0)

    r_CO, z_CO, dz_CO = surface_CO.rolling_surface()

    p0 = np.array([0.1, 1.2, 1.3, 4]) * \
            (1 + 0.1 * np.random.randn(nwalk, ndim))
        
    # Initialization sampling
    with Pool(processes=nthreads) as pool:
        isampler = emcee.EnsembleSampler(nwalk, ndim, log_posterior, pool=pool,
                                        args=(r_CO, z_CO, dz_CO))
        isampler.run_mcmc(p0, ninits)

    # Prune stray walkers
    isamples = isampler.get_chain()
    lop0 = np.quantile(isamples[-1,:,:], 0.25, axis=0)
    hip0 = np.quantile(isamples[-1,:,:], 0.75, axis=0)
    p00 = [np.random.uniform(lop0, hip0, ndim) for iw in range(nwalk)]

    # Run MCMC
    with Pool(processes=nthreads) as pool:
        sampler = emcee.EnsembleSampler(nwalk, ndim, log_posterior, pool=pool,
                                        args=(r_CO, z_CO, dz_CO))
        sampler.run_mcmc(p00, nsteps, progress=True)

    # flattened chain after burn-in
    chain = sampler.get_chain()
    print(chain.shape)

    # get log-posteriors
    lnprob = sampler.get_log_prob()

    # Save the posterior samples
    ofile = 'initial.npz'
    np.savez(ofile, chain=chain, lnprob=lnprob)
    t1 = time.time()
    np.savez('data_for_sean.npz', r=r_CO, z=z_CO, dz=dz_CO)

    # progress monitoring
    print('Surface in %.1f seconds\n' % (t1 - t0))

 

    # embed()

if __name__ == '__main__':
    main()
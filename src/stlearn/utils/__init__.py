"""
Useful utility functions.
"""
import numpy as np
import bottleneck as bn
import astropy.units as a_units

from scipy.stats import binned_statistic
from stlearn.conventions import MAD_TO_SIGNMA


def rms_timescale(lc, timescale=3600/86400):
	"""
	Compute robust RMS on specified timescale. Using MAD scaled to RMS.
	
    Parameters
    ----------
    lc : lightkurve.TessLightCurve
        Timeseries to calculate RMS for.
    timescale : float, optioanl
        Timescale to bin timeseries before calculating RMS. Default=1 hour.
	
    Returns
    -------
	float
        Robust RMS on specified timescale.
	
    .. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	time = np.asarray(lc.time)
	flux = np.asarray(lc.flux)
	if len(flux) == 0 or bn.allnan(flux):
		return np.nan
	if len(time) == 0 or bn.allnan(time):
		raise ValueError("Invalid time-vector specified. No valid timestamps.")

	time_min = np.nanmin(time)
	time_max = np.nanmax(time)
	if not np.isfinite(time_min) or not np.isfinite(time_max) or time_max - time_min <= 0:
		raise ValueError("Invalid time-vector specified")

	# Construct the bin edges seperated by the timescale:
	bins = np.arange(time_min, time_max, timescale)
	bins = np.append(bins, time_max)

	# Bin the timeseries to one hour:
	indx = np.isfinite(flux)
	flux_bin, _, _ = binned_statistic(
        time[indx], flux[indx], bn.nanmean, bins=bins
    )

	# Compute robust RMS value (MAD scaled to RMS)
	return MAD_TO_SIGNMA * bn.nanmedian(np.abs(flux_bin - bn.nanmedian(flux_bin)))


def ptp(lc):
	"""
	Compute robust Point-To-Point scatter.

	Parameters
    ----------
	lc : lightkurve.TessLightCurve
        Lightcurve to calculate PTP for.
	
    Returns
    -------
	float
        Robust PTP.
	
    .. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	if len(lc.flux) == 0 or bn.allnan(lc.flux):
		return np.nan
	if len(lc.time) == 0 or bn.allnan(lc.time):
		raise ValueError("Invalid time-vector specified. No valid timestamps.")
	return bn.nanmedian(np.abs(np.diff(lc.flux)))


def get_periods(featdict, nfreqs, time, in_days=True, ignore_harmonics=False):
	"""
	Cuts frequency data down to desired number of frequencies (in uHz) and 
    optionally transforms them into periods in days.
	
    Parameters
    ----------
    featdict : dict
    nfreq : int
        Number of frequencies/periods to extract.
    time : numpy.array
    in_days : bool, optional
        Return periods in days instead of frequencies in uHz.
    ignore_harmonics : bool, optional
        Sort frequency table by amplitude (i.e. ignore into harmonic 
        structure).
	
    Returns
    -------
    periods :
    n_usedfreqs : int
        Number of true periods/frequencies that are used.
    usedfreqs : 
        Indices of the used periods/frequencies in the astropy table.
	
    .. codeauthor:: Jeroen Audenaert <jeroen.audenaert@kuleuven.be>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	tab = featdict['frequencies']
	tab = tab[~np.isnan(tab['amplitude'])]
	if ignore_harmonics:
		tab.sort('amplitude', reverse=True)
		selection = tab[:min(len(tab), nfreqs)]
	else:
		selection = tab[tab['harmonic'] == 0][:nfreqs]

	periods = selection['frequency'].quantity
	usedfreqs = selection[['num', 'harmonic']]

	if in_days:
		periods = (1/periods).to(a_units.day)

	per = (np.max(time) - np.min(time)) * a_units.day
	gap = nfreqs - len(periods)
	if gap > 0:
		if in_days:
			for i in range(gap):
				periods = periods.insert(len(periods), per)
		else:
			for i in range(gap):
				periods = periods.insert(len(periods), (1/per).to(a_units.uHz))

	n_usedfreqs = len(usedfreqs)

	return periods.value, n_usedfreqs, usedfreqs
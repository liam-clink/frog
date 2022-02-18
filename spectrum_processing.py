import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.interpolate

folder = './First Stage/'
filename = 'SpectraAfterFirstStage_2021_08_03.txt'


frequencies = np.loadtxt(folder+'processed_data_freqs.tsv')
# for SHG, the spectrum of the trace is double the frequency of the original pulse
# Shift the spectrum so the center is at half the center, but the width is the same
frequencies -= 0.25*(frequencies[0] + frequencies[-1])
max_wavelength = 2.99e8/frequencies[0]*1.e9 # nm
min_wavelength = 2.99e8/frequencies[-1]*1.e9 # nm
centered_frequencies = frequencies - (frequencies[-1]+frequencies[0])/2
current_bandwidth = frequencies[-1]-frequencies[0]
desired_timestep = 1.e-15 # s, set by user
desired_bandwidth = 1/desired_timestep
padded_frequencies = centered_frequencies*desired_bandwidth/current_bandwidth

spectrum = np.loadtxt(folder+filename,skiprows=14)
plt.plot(spectrum[:,0],spectrum[:,1])
plt.title('Raw Spectrum')
plt.xlabel('Wavelength (nm)')
plt.show()

# Fit line to points away from data
def constant(x, a):
    return a

index_filter = spectrum[:,0] >= max_wavelength + 50.
popt, pcov = scipy.optimize.curve_fit(constant, spectrum[index_filter,0], spectrum[index_filter,1])
plt.plot(spectrum[:,0],spectrum[:,1])
plt.plot([spectrum[0,0], spectrum[-1,0]],[popt[0],popt[0]])
plt.title('Spectrum with Background Fit')
plt.xlabel('Wavelength (nm)')
plt.show()

# Shift fit height down to zero
spectrum[:,1] -= popt[0]

# Change negative values to zero
index_filter = spectrum[:,1] < 0.
spectrum[index_filter,1] = 0.
plt.plot(spectrum[:,0],spectrum[:,1])
plt.title('Background Subtracted Spectrum')
plt.xlabel('Wavelength (nm)')
plt.show()

'''
# Crop Spectrum
index_filter = np.logical_and(spectrum[:,0] > min_wavelength, spectrum[:,0] < max_wavelength)
spectrum = spectrum[index_filter,:]
plt.plot(spectrum[:,0],spectrum[:,1])
plt.title('Cropped Spectrum')
plt.xlabel('Wavelength (nm)')
plt.show()
'''

# Transform to frequency
spectrometer_frequencies = 2.99e8/(spectrum[::-1,0]*1.e-9)
spectral_intensity = spectrum[::-1,1]
'''
plt.plot(frequencies, spectral_intensity)
plt.title('Frequencies with Original Data')
plt.xlabel('Frequencies (Hz)')
plt.show()
'''

# Interpolate to evenly spaced power of 2 samples
sample_count = len(spectrometer_frequencies)
# Round up to the nearest power of 2
interpolator = scipy.interpolate.interp1d(spectrometer_frequencies, spectral_intensity, bounds_error=False, fill_value=0.)
interpolated_spectral_intensity = interpolator(frequencies)

interpolator = scipy.interpolate.interp1d(centered_frequencies, interpolated_spectral_intensity, bounds_error=False, fill_value=0.)
interpolated_spectral_intensity = interpolator(padded_frequencies)

plt.plot(padded_frequencies, interpolated_spectral_intensity)
plt.title('Final Data')
plt.xlabel('Frequencies (Hz)')
plt.savefig(folder+'spectrum.svg')
plt.show()

processed_spectrum = np.array([*zip(frequencies, interpolated_spectral_intensity)])
np.savetxt(folder+'processed_spectrum.tsv', processed_spectrum, delimiter='\t')

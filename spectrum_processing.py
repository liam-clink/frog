import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.interpolate

folder = './Second Stage/'
filename = 'AfterChirpmirr_2ndStage.txt'

max_wavelength = 1150. # nm
min_wavelength = 900. # nm

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

# Transform to angular frequency
angular_frequencies = 2.*np.pi*2.99e8/(spectrum[::-1,0]*1.e-9)
spectral_intensity = spectrum[::-1,1]
plt.plot(angular_frequencies, spectral_intensity)
plt.title('Angular frequencies with Original Data')
plt.xlabel('Angular Frequencies (rad/s)')
plt.show()


# Interpolate to match image spectrum
sample_count = len(angular_frequencies)

interpolator = scipy.interpolate.interp1d(angular_frequencies, spectral_intensity, bounds_error=False, fill_value=0)
image_angular_frequencies = np.loadtxt(folder+'processed_data_ang_freqs.tsv', delimiter='\t')
# Halve the center of the band, but keep the same spectral width
start = image_angular_frequencies[0]
end = image_angular_frequencies[-1]
pulse_angular_frequencies = np.linspace(.75*start-.25*end, .75*end-.25*start, len(image_angular_frequencies), endpoint=True)
interpolated_spectral_intensity = interpolator(pulse_angular_frequencies)

plt.plot(pulse_angular_frequencies, interpolated_spectral_intensity)
plt.title('Final Data')
plt.xlabel('Angular Frequencies (rad/s)')
plt.savefig(folder+'spectrum.svg')
plt.show()

processed_spectrum = np.array([*zip(pulse_angular_frequencies, interpolated_spectral_intensity)])
np.savetxt(folder+'processed_spectrum.tsv', processed_spectrum, delimiter='\t')

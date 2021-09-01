import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.interpolate

folder = './Raw PHAROS/'
filename = 'Raw PHAROS Spectrum.txt'

spectrum = np.loadtxt(folder+filename,skiprows=14)
plt.plot(spectrum[:,0],spectrum[:,1])
plt.show()

# Fit line to points away from data
def constant(x, a):
    return a

index_filter = spectrum[:,0] >= 1100.
popt, pcov = scipy.optimize.curve_fit(constant, spectrum[index_filter,0], spectrum[index_filter,1])
plt.plot(spectrum[:,0],spectrum[:,1])
plt.plot([spectrum[0,0], spectrum[-1,0]],[popt[0],popt[0]])
plt.show()

# Shift fit height down to zero
spectrum[:,1] -= popt[0]

# Change negative values to zero
index_filter = spectrum[:,1] < 0.
spectrum[index_filter,1] = 0.
plt.plot(spectrum[:,0],spectrum[:,1])
plt.show()

# Crop Spectrum
index_filter = np.logical_and(spectrum[:,0] > 950., spectrum[:,0] < 1100.)
spectrum = spectrum[index_filter,:]
plt.plot(spectrum[:,0],spectrum[:,1])
plt.show()

# Transform to angular frequency
angular_frequencies = 2.*np.pi*2.99e8/(spectrum[::-1,0]*1.e-9)
spectral_intensity = spectrum[::-1,1]
plt.plot(angular_frequencies, spectral_intensity)
plt.show()


# Interpolate to evenly spaced power of 2 samples
sample_count = len(angular_frequencies)
# Round up to the nearest power of 2
interpolate_count = 2**(int(np.ceil(np.log2(sample_count))))

interpolator = scipy.interpolate.interp1d(angular_frequencies,spectral_intensity)
angular_frequencies = np.linspace(angular_frequencies[0],angular_frequencies[-1],interpolate_count,endpoint=True)
interpolated_spectral_intensity = interpolator(angular_frequencies)

plt.plot(angular_frequencies,interpolated_spectral_intensity)
plt.show()

processed_spectrum = np.array([*zip(angular_frequencies,interpolated_spectral_intensity)])
np.savetxt('processed_spectrum.tsv', processed_spectrum, delimiter='\t')

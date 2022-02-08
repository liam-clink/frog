import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from numpy.random import default_rng
rng = default_rng()

folder = 'Second Stage/'

trace = np.loadtxt(folder+'processed_data.tsv', delimiter='\t')
trace /= np.max(trace)
delays = np.loadtxt(folder+'processed_data_delays.tsv', delimiter='\t')
frequencies = np.loadtxt(folder+'processed_data_freqs.tsv', delimiter='\t')

# Spectrum of the original pulse (not necessary, but helpful)
original_spectrum = np.loadtxt(folder+'processed_spectrum.tsv')
original_spectrum[:,1] /= np.max(original_spectrum[:,1])
original_frequencies = original_spectrum[:,0]
original_spectrum[:,1] -= 0.0075
# Change negative values to zero
index_filter = original_spectrum[:,1] < 0.
original_spectrum[index_filter,1] = 0.
original_spectral_amplitude = original_spectrum[:,1]**0.5

# Calculate pulse from spectrum
# FFT produces a spectrum centered at zero, IFFT expects the same
# This means that FFT calculates spectrum of the envelope, unless the spectrum
# is padded so that the negative range of frequencies equals the positive range
timestep = 1./(original_frequencies[-1]-original_frequencies[0])
grid_size = len(original_frequencies) #TODO: throw error if odd grid size, or grids don't match
duration = grid_size*timestep 
times = np.fft.ifftshift(np.fft.fftfreq(grid_size, 1./duration))
shifted_original_frequencies = np.fft.ifftshift(np.fft.fftfreq(grid_size, timestep))
initial_guess = np.fft.ifft(original_spectral_amplitude)

fig, axes = plt.subplots(1, 2)
axes[0].plot(shifted_original_frequencies, original_spectral_amplitude)
axes[0].set_title('Original Spectrum')
axes[1].plot(times, np.fft.ifftshift(initial_guess.real))
axes[1].plot(times, np.fft.ifftshift(initial_guess.imag))
axes[1].set_title('Flat Spectral Phase Pulse')
plt.show()


# I'm using frequency instead of angular frequency because that is
# numpy's convention, and it's thus simpler this way

# FFT produces a spectrum centered at zero, IFFT expects the same
#TODO: add error if shape is odd (assert ValueError)
shifted_frequencies = frequencies - frequencies[int(frequencies.shape[0]/2)]

# Uncomment to assess soft threshold level visually
'''
for i in range(trace.shape[-1]):
    plt.plot(shifted_frequencies, trace[i,:])
plt.show()
'''
threshold = 1.5e-3

iterations = 5

indices = np.arange(len(delays))
for i in range(iterations):
    rng.shuffle(indices)
    # Iterate through lines
    print(indices)
    for j in indices:
        print(i,j)
        # Calculate SHG
        # FFT(SHG)
        # Determine indices above threshold
        # Update the part above threshold
        # Zero the part below the threshold
        # IFFT(SHG)
        # Update E

#TODO: add error if shifted_frequencies != shifted_original frequencies
#plt.pcolormesh(shifted_frequencies, delays, trace)
#plt.show()
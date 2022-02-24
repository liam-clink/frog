from pypret import pypret
import numpy as np

import scipy.interpolate

folder = 'Second Stage/'

# Parameters
guess_fwhm = 25.e-15 # FWHM in s
spectrum_provided = True

# Measured data
# axis 0 of data is delay, axis 1 is signal angular frequency
data = np.loadtxt(folder+'processed_data.tsv', delimiter='\t')
# Data has shape (delay pixels, spectrum pixels)
data = data[:,::-1]
time_pixels = data.shape[0]
spectrum_pixels = data.shape[1]

delays = np.loadtxt(folder+'processed_data_delays.tsv')
image_angular_frequencies = 2.*np.pi*np.loadtxt(folder+'processed_data_freqs.tsv')

if time_pixels != len(delays) or spectrum_pixels != len(image_angular_frequencies):
    raise ValueError('Dimensions of input image must match axis dimensions')

if spectrum_provided:
    spectral_intensity = np.loadtxt(folder+'processed_spectrum.tsv')
    central_angular_frequency = spectral_intensity[np.argmax(spectral_intensity[:,1]),0]
else:
    central_wavelength = 1030.e-9 # wavelength in m
    wavelengths = np.linspace(-25.e-9, 25.e-9, spectrum_pixels) + central_wavelength
    angular_frequencies = 2.*np.pi*2.99e8/wavelengths[::-1]

measured_data = pypret.MeshData(data, delays, image_angular_frequencies)
measured_data.interpolate()
image_angular_frequencies = measured_data.axes[1]

####################################################################################################
# Angular frequencies need to be interpolated to match the "measured_data.axes[1] = pnps.process_w"
# condition, and of course have the same length in the first place
# In other words: "image_angular_frequencies = 2*pulse.w0 + pulse.ft.w"
# This means that the measured spectral range is the same for both the original pulse and the nonlinear one
# pulse.w0 is calculated from wl0 argument of pypret.Pulse(), *not* pulse.ft.w0
# n = np.arange(pulse.ft.N)
# pulse.ft.w = pulse.ft.w0 + n * pulse.ft.dw

# The range of signal angular frequencies is fixed, as is the center wavelength for the pulse,
# so this means that the pulse.ft.w frequencies need to be resampled appropriately to satisfy this condition

angular_frequencies = image_angular_frequencies - 2.*central_angular_frequency

##################################################################
# Create Fourier Transform grid
# This FT grid is for the *original* pulse, not the nonlinear process result
# The FT grid is meant to be centered at zero in w space?
#TODO: The w0 should not need to be set. Something is wrong... the frequencies I get are off center
ft = pypret.FourierTransform(len(angular_frequencies), dw=angular_frequencies[1]-angular_frequencies[0], w0=angular_frequencies[0])
# instantiate a pulse object, angular frequency in rad/s
#TODO: Consider leaving input in wavelength
pulse = pypret.Pulse(ft, central_angular_frequency, unit='om')

# Spectrum specified from spectrometer data
if spectrum_provided:
    # Spectrum must be provided in angular frequency
    # Since documentation says "spectral envelope", the peak may need to be centered at w=0
    interpolator = scipy.interpolate.interp1d(spectral_intensity[:,0],spectral_intensity[:,1]**(0.5), bounds_error=False, fill_value=0.)
    interpolated_spectral_amplitude = interpolator(angular_frequencies+central_angular_frequency)
    pulse.spectrum = interpolated_spectral_amplitude
else:
    # create a random pulse with time-bandwidth product of 2.
    pypret.random_pulse(pulse, 2.0)
    # start with a Gaussian spectrum with random phase as initial guess
    pypret.random_gaussian(pulse, guess_fwhm, phase_max=0.0)


# plot the pulse
pypret.PulsePlot(pulse)

# simulate a frog measurement
pnps = pypret.PNPS(pulse, "frog", "shg")
# calculate the measurement trace
pnps.calculate(pulse.spectrum, delays)
# and plot it
pypret.MeshDataPlot(pnps.trace)

##################################################################
## Finally doing the actual retrieval

# and do the retrieval
ret = pypret.Retriever(pnps, "copra", verbose=True, maxiter=300)

# Retrieve from the measured data
ret.retrieve(measured_data, pulse.spectrum)

# and print the retrieval results
result = ret.result(pulse.spectrum)
np.savetxt(folder+'retrieved_trace.tsv', result.trace_retrieved, delimiter='\t')
np.savetxt(folder+'retrieved_pulse.tsv', result.pulse_retrieved, delimiter='\t')
np.savetxt(folder+'retrieved_wavelengths.tsv', pulse.wl, delimiter='\t')
np.savetxt(folder+'retrieved_spectrum.tsv', pulse.ft.forward(result.pulse_retrieved), delimiter='\t')
np.savetxt(folder+'pulse_time.tsv', pulse.t, delimiter='\t')
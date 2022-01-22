import sys
sys.path.append('pypret')
import pypret
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

folder = 'Second Stage/'

# Measured data
# axis 0 of data is delay, axis 1 is signal angular frequency
data = np.loadtxt(folder+'processed_data.tsv', delimiter='\t')
# Data has shape (delay pixels, spectrum pixels)
data = data[:,::-1]
time_pixels = data.shape[0]
spectrum_pixels = data.shape[1]

spectrum_provided = True
if spectrum_provided:
    spectral_intensity = np.loadtxt(folder+'processed_spectrum.tsv')


# Parameters

guess_fwhm = 25.e-15 # FWHM in s
timestep = 0.1e-15 # seconds of delay per vertical pixel
time_range = timestep*time_pixels
delays = np.linspace(-time_range/2., time_range/2., time_pixels,endpoint=True)  # delays in s

if spectrum_provided:
    central_angular_frequency = spectral_intensity[np.argmax(spectral_intensity[:,1]),0]
else:
    central_wavelength = 1030.e-9 # wavelength in m
    wavelengths = np.linspace(-25.e-9, 25.e-9, spectrum_pixels) + central_wavelength
    angular_frequencies = 2.*np.pi*2.99e8/wavelengths[::-1]

signal_wavelengths = np.loadtxt(folder+'image_wavelengths.tsv')
signal_angular_frequencies = 2.*np.pi*2.99e8/signal_wavelengths
print(signal_angular_frequencies[0], signal_angular_frequencies[-1])

measured_data = pypret.MeshData(data, delays, signal_angular_frequencies)
measured_data.interpolate()
signal_angular_frequencies = measured_data.axes[1]

####################################################################################################
# Angular frequencies need to be interpolated to match the "measured_data.axes[1] = pnps.process_w"
# condition, and of course have the same length in the first place
# In other words: "signal_angular_frequencies = 2*pulse.w0 + pulse.ft.w"
# This means that the measured spectral range is the same for both the original pulse and the nonlinear one
# pulse.w0 is calculated from wl0 argument of pypret.Pulse()
# n = np.arange(pulse.ft.N)
# pulse.ft.w = pulse.ft.w0 + n * pulse.ft.dw

# The range of signal angular frequencies is fixed, as is the center wavelength for the pulse,
# so this means that the pulse.ft.w frequencies need to be resampled appropriately to satisfy this condition

angular_frequencies = signal_angular_frequencies - 2.*central_angular_frequency

##################################################################
# Create Fourier Transform grid
# This FT grid is for the *original* pulse, not the nonlinear process result
# The FT grid is meant to be centered at zero in w space?
ft = pypret.FourierTransform(len(angular_frequencies), dw=angular_frequencies[1]-angular_frequencies[0], w0=angular_frequencies[0])
# instantiate a pulse object, angular frequency in rad/s
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

## Plot the measured and retrieved traces
signal_wavelengths = 2.*np.pi*2.99e8/signal_angular_frequencies*1.e9 # wavelengths in nm
pulse_wavelengths = 2.*np.pi*2.99e8/(pulse.w+pulse.w0)*1.e9
delays *= 1.e15 # convert delays to fs

fig = plt.figure()
axes = fig.subplots(2,2)

axes[0,0].pcolormesh(signal_wavelengths, delays, measured_data.data, shading='auto')
axes[0,0].set_xlabel('wavelength (nm)')
axes[0,0].set_ylabel('delays (fs)')

axes[0,1].pcolormesh(signal_wavelengths, delays, result.trace_retrieved, shading='auto')
axes[0,1].set_xlabel('wavelength (nm)')
axes[0,1].set_ylabel('delays (fs)')

def fwhm(intensity):
    max_index = np.argmax(intensity)
    for i in range(max_index,0,-1):
        if intensity[i]<intensity[max_index]/2.:
            low_index = i
            break
    for i in range(max_index,len(intensity)):
        if intensity[i]<intensity[max_index]/2.:
            high_index = i
            break
    return low_index, high_index

field = result.pulse_retrieved
field_intensity = np.abs(field)**2
field_intensity /= np.max(field_intensity)
low, high = fwhm(field_intensity)
fwhm_time = pulse.t[high]-pulse.t[low]
axes[1,0].plot([pulse.t[low]*1.e15,pulse.t[high]*1.e15], [0.5, 0.5], 'g-')
axes[1,0].plot(pulse.t*1.e15, field_intensity,'r-')
axes10phase = axes[1,0].twinx()
masked_field_phases = np.ma.masked_where(field_intensity<5.e-2, np.unwrap(np.angle(field)))
masked_field_phases -= np.mean(masked_field_phases)
axes10phase.plot(pulse.t*1.e15, masked_field_phases,'b-')
axes[1,0].set_xlabel('time (fs)')
axes[1,0].set_ylabel('intensity (arb.)')
axes10phase.set_ylabel('phase (rad)')
axes[1,0].set_title('Duration: 0:.4g'.format(fwhm_time*1.e15))

# TODO: This doesn't convert from w to wavelength...
spectrum = pulse.ft.forward(field)
spectral_intensity = np.abs(spectrum)**2
spectral_intensity /= np.max(spectral_intensity)
axes[1,1].plot(pulse_wavelengths, spectral_intensity,'r-')
axes11phase = axes[1,1].twinx()
masked_spectral_phases = np.ma.masked_where(spectral_intensity<5.e-2, np.unwrap(np.angle(result.pulse_retrieved)))
masked_spectral_phases -= np.mean(masked_spectral_phases)
axes11phase.plot(pulse_wavelengths, masked_spectral_phases,'b-')
axes[1,1].set_xlabel('wavelength (nm)')
axes[1,1].set_ylabel('intensity (arb.)')
axes11phase.set_ylabel('phase (rad)')

fig.tight_layout()
plt.savefig(folder+'frog_result.png', dpi=600)

plt.show()

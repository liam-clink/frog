"""
Retrieves the pulse from the FROG data using the Pypret library and saves the result

To run, it requires
processed_data.tsv: 
processed_data_delays.tsv
processed_data_freqs.tsv
processed_spectrum.tsv
"""

import numpy as np
import scipy.interpolate
from pypret import pypret

C = 2.99e8

FOLDER = "Raw PHAROS/"

# Parameters
GUESS_FWHM = 25.0e-15  # FWHM in s
SPECTRUM_PROVIDED = False

# Measured data
# axis 0 of data is delay, axis 1 is signal angular frequency
data = np.loadtxt(FOLDER + "processed_data.tsv", delimiter="\t")
# Data has shape (delay pixels, spectrum pixels)
data = data[:, ::-1]
time_pixels = data.shape[0]
spectrum_pixels = data.shape[1]


delays = np.loadtxt(FOLDER + "processed_data_delays.tsv")
image_angular_frequencies = (
    2.0 * np.pi * np.loadtxt(FOLDER + "processed_data_freqs.tsv")
)

if time_pixels != len(delays) or spectrum_pixels != len(image_angular_frequencies):
    raise ValueError("Dimensions of input image must match axis dimensions")


if SPECTRUM_PROVIDED:
    spectral_intensity = np.loadtxt(FOLDER + "processed_spectrum.tsv")
    central_angular_frequency = spectral_intensity[
        np.argmax(spectral_intensity[:, 1]), 0
    ]
else:
    CENTRAL_WAVELENGTH = 1030.0e-9  # wavelength in m
    central_angular_frequency = 2.0 * np.pi * C / CENTRAL_WAVELENGTH
    BANDWIDTH = 50.0e-9  # m
    angular_frequencies = (
        2.0
        * np.pi
        * C
        * np.linspace(
            1 / (CENTRAL_WAVELENGTH + BANDWIDTH / 2),
            1 / (CENTRAL_WAVELENGTH - BANDWIDTH / 2),
            spectrum_pixels,
        )
    )


measured_data = pypret.MeshData(data, delays, image_angular_frequencies)
measured_data.interpolate()
image_angular_frequencies = measured_data.axes[1]


####################################################################################################
# Angular frequencies need to be interpolated to match the "measured_data.axes[1] = pnps.process_w"
# condition, and of course have the same length in the first place
# In other words: "image_angular_frequencies = 2*pulse.w0 + pulse.ft.w"
# This means that the measured spectral range is the same
# for both the original pulse and the nonlinear one
# pulse.w0 is calculated from wl0 argument of pypret.Pulse(), *not* pulse.ft.w0
# n = np.arange(pulse.ft.N)
# pulse.ft.w = pulse.ft.w0 + n * pulse.ft.dw

# The range of signal angular frequencies is fixed, as is the center wavelength for the pulse,
# so this means that the pulse.ft.w frequencies need
# to be resampled appropriately to satisfy this condition


##################################################################
# Create Fourier Transform grid
# This FT grid is for the *original* pulse, not the nonlinear process result
# The FT grid is meant to be centered at zero in w space?
# w0 is supposed to be 0 because the pulse is supposed to be just the envelope

ft = pypret.FourierTransform(
    128,
    dt=2.86e-15,
)

# instantiate a pulse object, angular frequency in rad/s
# The central angular frequency just tracks where the center should be
# The actual pulse object only represents the envelope, which is centered at 0 rad/s
pulse = pypret.Pulse(ft, central_angular_frequency, unit="om")

# Spectrum specified from spectrometer data

# Spectrum must be provided in angular frequency
# Since documentation says "spectral envelope", the peak may need to be centered at w=0
interpolator = scipy.interpolate.interp1d(
    spectral_intensity[:, 0],
    spectral_intensity[:, 1] ** (0.5),
    bounds_error=False,
    fill_value=0.0,
)
interpolated_spectral_amplitude = interpolator(
    angular_frequencies + central_angular_frequency
)
pulse.spectrum = interpolated_spectral_amplitude


##################################################################
## Finally doing the actual retrieval

# and do the retrieval
ret = pypret.Retriever(pnps, "copra", verbose=True, maxiter=300)

# Retrieve from the measured data
ret.retrieve(measured_data, pulse.spectrum)

# and print the retrieval results
result = ret.result(pulse.spectrum)
np.savetxt(FOLDER + "retrieved_trace.tsv", result.trace_retrieved, delimiter="\t")
np.savetxt(FOLDER + "retrieved_pulse.tsv", result.pulse_retrieved, delimiter="\t")
np.savetxt(FOLDER + "retrieved_wavelengths.tsv", pulse.wl, delimiter="\t")
np.savetxt(
    FOLDER + "retrieved_spectrum.tsv",
    pulse.ft.forward(result.pulse_retrieved),
    delimiter="\t",
)
np.savetxt(FOLDER + "pulse_time.tsv", pulse.t, delimiter="\t")

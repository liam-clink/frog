"""
A template for how to use pypret for pulse retrieval.
"""

import numpy as np
from pypret import pypret


C = 2.99e8
SAMPLE_N = 128
EXAMPLE_FWHM = 250.0e-15

central_angular_frequency = 2.0 * np.pi * C / 1032.0e-9

delays = np.linspace(-2000.0e-15, 2000.0e-15, SAMPLE_N)
ft = pypret.FourierTransform(
    SAMPLE_N,
    dt=delays[1] - delays[0],
)
pulse = pypret.Pulse(ft, central_angular_frequency, unit="om")


# Create a random pulse with time-bandwidth product of 2.
pypret.random_pulse(pulse, 1.5)
# Or Start with a Gaussian spectrum with random phase as initial guess
# pypret.random_gaussian(pulse, EXAMPLE_FWHM, phase_max=0.0)

# plot the pulse
pypret.PulsePlot(pulse)

# simulate a frog measurement
pnps = pypret.PNPS(pulse, "frog", "shg")
# calculate the measurement trace

pnps.calculate(pulse.spectrum, delays)
# and plot it
pypret.MeshDataPlot(pnps.trace)

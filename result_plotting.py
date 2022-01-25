import matplotlib.pyplot as plt
import numpy as np

folder = './Second Stage/'
measured_data = np.loadtxt(folder+'processed_data.tsv', delimiter='\t')
delays = np.loadtxt(folder+'processed_data_delays.tsv', delimiter='\t')
signal_wavelengths = np.loadtxt(folder+'image_wavelengths.tsv', delimiter='\t')
retrieved_pulse = np.loadtxt(folder+'retrieved_pulse.tsv', delimiter='\t')
pulse_time = np.loadtxt(folder+'pulse_time.tsv', delimiter='\t')
retrieved_trace = np.loadtxt(folder+'retrieved_trace.tsv', delimiter='\t')
pulse_wavelengths = np.loadtxt(folder+'retrieved_wavelengths.tsv', delimiter='\t')


delays *= 1.e15 # convert delays to fs

fig = plt.figure()
axes = fig.subplots(2,2)

axes[0,0].pcolormesh(signal_wavelengths*1e9, delays, measured_data, shading='auto')
axes[0,0].set_xlabel('wavelength (nm)')
axes[0,0].set_ylabel('delays (fs)')

axes[0,1].pcolormesh(signal_wavelengths, delays, retrieved_trace, shading='auto')
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

field = retrieved_pulse
field_intensity = np.abs(field)**2
field_intensity /= np.max(field_intensity)
low, high = fwhm(field_intensity)
fwhm_time = pulse_time[high]-pulse_time[low]
axes[1,0].plot([pulse_time[low]*1.e15,pulse_time[high]*1.e15], [0.5, 0.5], 'g-')
axes[1,0].plot(pulse_time*1.e15, field_intensity,'r-')
axes10phase = axes[1,0].twinx()
masked_field_phases = np.ma.masked_where(field_intensity<5.e-2, np.unwrap(np.angle(field)))
masked_field_phases -= np.mean(masked_field_phases)
axes10phase.plot(pulse_time*1.e15, masked_field_phases,'b-')
axes[1,0].set_xlabel('time (fs)')
axes[1,0].set_ylabel('intensity (arb.)')
axes10phase.set_ylabel('phase (rad)')
axes[1,0].set_title('Duration: {0:.4g}'.format(fwhm_time*1.e15))

# TODO: This doesn't convert from w to wavelength...
spectrum = np.loadtxt(folder+'retrieved_spectrum.tsv', delimiter='\t')
spectral_intensity = np.abs(spectrum)**2
spectral_intensity /= np.max(spectral_intensity)
axes[1,1].plot(pulse_wavelengths, spectral_intensity,'r-')
axes11phase = axes[1,1].twinx()
masked_spectral_phases = np.ma.masked_where(spectral_intensity<5.e-2, np.unwrap(np.angle(retrieved_pulse)))
masked_spectral_phases -= np.mean(masked_spectral_phases)
axes11phase.plot(pulse_wavelengths, masked_spectral_phases,'b-')
axes[1,1].set_xlabel('wavelength (nm)')
axes[1,1].set_ylabel('intensity (arb.)')
axes11phase.set_ylabel('phase (rad)')

fig.tight_layout()
plt.savefig(folder+'frog_result.png', dpi=600)

plt.show()

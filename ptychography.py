from operator import index
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
initial_scale = 8. # Change to make initial SHG have similar scale to trace
initial_guess = initial_scale*np.fft.ifftshift(np.fft.ifft(original_spectral_amplitude))

fig, axes = plt.subplots(1, 2)
axes[0].plot(shifted_original_frequencies, original_spectral_amplitude)
axes[0].set_title('Original Spectrum')
axes[1].plot(times, initial_guess.real)
axes[1].plot(times, initial_guess.imag)
axes[1].set_title('Flat Spectral Phase Pulse')
plt.show()


# Define functions for error calculation
def calc_trace(pulse, delays):
    pulse_interpolator = scipy.interpolate.interp1d(times, pulse, bounds_error=False, fill_value=0.)
    trace = np.zeros((len(delays), len(pulse)))
    for i in range(len(delays)):
        gate_pulse = pulse_interpolator(times-delays[i])
        shg = pulse*gate_pulse
        shg_fft = np.fft.fftshift(np.fft.fft(shg))
        trace[i,:] = abs(shg_fft*np.conj(shg_fft))
    return trace

def scale_factor(trace_measured, trace_calculated):
    return sum(trace_measured*trace_calculated)/sum(trace_calculated*trace_calculated)

# Calculate norm for error function, since it need only be calculated once
norm = trace.shape[0]*trace.shape[1]*np.max(trace)**2
def calc_error(pulse, trace_measured, delays):
    # Calculate trace
    trace_calculated = calc_trace(pulse, delays)
    # Calculate scale factor
    mu = scale_factor(trace_measured, trace_calculated)
    # Sum squares of residuals
    r = np.sum((trace_measured-mu*trace_calculated)**2)
    # Return normalized error
    return np.sqrt(r/norm)

# I'm using frequency instead of angular frequency because that is
# numpy's convention, and it's thus simpler this way

# FFT produces a spectrum centered at zero, IFFT expects the same
#TODO: add error if shape is odd (assert ValueError)
#TODO: perhaps shifting not being perfect is the issue...
shifted_frequencies = frequencies - frequencies[int(frequencies.shape[0]/2)]

# Uncomment to assess soft threshold level visually
'''
for i in range(trace.shape[-1]):
    plt.plot(shifted_frequencies, trace[i,:])
plt.show()
'''
threshold = 1.5e-3    

iterations = 300
pulse = initial_guess
indices = np.arange(len(delays))
frog_errors = []
for i in range(iterations):
    #fig, axes = plt.subplots(1, 2)
    #axes[0].plot(times, pulse.real)
    #axes[0].plot(times, pulse.imag)
    rng.shuffle(indices)
    
    # Iterate through lines
    #print(indices)
    alpha = rng.uniform(0.1, 0.5)
    for j in indices:
        # Calculate SHG
        pulse_interpolator = scipy.interpolate.interp1d(times, pulse, bounds_error=False, fill_value=0.)
        gate_pulse = pulse_interpolator(times-delays[j])
        shg = pulse*gate_pulse
        
        # FFT(SHG)
        shg_fft = np.fft.fftshift(np.fft.fft(shg))
        #axes[0].plot(shifted_frequencies, trace[j,:])
        #axes[1].plot(shifted_frequencies, shg_fft*np.conj(shg_fft))
        
        # Update the part above threshold
        index_filter = trace[j,:] >= threshold
        shg_fft[index_filter] = abs(trace[j,index_filter])**0.5*np.exp(1.0j*np.angle(shg_fft[index_filter]))
        #axes[0].plot(shifted_frequencies[index_filter], trace[j,index_filter])
        #axes[1].plot(shifted_frequencies[index_filter], abs(shg_fft[index_filter])**2)
        
        # Soft threshold the weak part
        index_filter = trace[j,:] < threshold
        shg_fft[index_filter].real = np.where(abs(shg_fft[index_filter].real)<1.e-3, 0., shg_fft[index_filter].real)
        shg_fft[index_filter].imag = np.where(abs(shg_fft[index_filter].imag)<1.e-3, 0., shg_fft[index_filter].imag)
        #axes[0].plot(shifted_frequencies, trace[j,:])
        #axes[1].plot(shifted_frequencies, shg_fft.real)
        #axes[1].plot(shifted_frequencies, shg_fft.imag)
        
        # IFFT(SHG)
        shg_new = np.fft.ifft(np.fft.ifftshift(shg_fft))
        #axes[0].plot(times, abs(shg)**2)
        #axes[1].plot(times, abs(shg_new)**2)

        # Update E
        
        scale = alpha*np.conj(gate_pulse)/(np.max(abs(gate_pulse*np.conj(gate_pulse))) + 1.e-3)
        pulse += scale*(shg_new - shg)

        # Correct Spectrum of E TODO: probably need to normalize to be the same energy before and after
        pulse_fft = np.fft.fftshift(np.fft.fft(pulse))
        energy = np.sum(abs(pulse_fft*np.conj(pulse_fft)))
        pulse_fft = original_spectral_amplitude*np.exp(1.0j*np.angle(pulse_fft))
        new_energy = np.sum(abs(pulse_fft*np.conj(pulse_fft)))
        pulse_fft *= np.sqrt(energy/new_energy)
        pulse = np.fft.ifft(np.fft.ifftshift(pulse_fft))

        # Remove time ambiguity
        peak_index = np.argmax(abs(pulse*np.conj(pulse)))
        pulse = np.roll(pulse, int(len(pulse)/2 - peak_index))

    #plt.imshow(calc_trace(pulse, delays))
    #plt.show()    

    # Print frog error
    frog_error = calc_error(pulse, trace, delays)
    print("Iteration: ", i, "FROG Error: ", frog_error)
    frog_errors.append(frog_error)

    #axes[1].plot(times, pulse.real)
    #axes[1].plot(times, pulse.imag)
    #plt.show()

np.savetxt(folder+'frog_errors.tsv', np.array(frog_errors), delimiter='\t')

#plt.plot(times, pulse.real)
#plt.plot(times, pulse.imag)
fig, ax = plt.subplots(1, 1)
ax.pcolormesh(shifted_frequencies, delays, calc_trace(pulse, delays))
ax.set_title('Trace of Retrieved Pulse')
ax.set_xlabel('frequencies (Hz)')
ax.set_ylabel('delays (s)')
ax.set_aspect(1.e25)
plt.savefig(folder+'final_trace.png', dpi=600)
plt.show()

fig, axes = plt.subplots(1, 2)
axes[0].plot(times, abs(pulse)**2)
axes[0].set_xlabel('time (s)')
axes[0].set_ylabel('amplitude (a.u.)')
index_filter = abs(pulse)**2 > np.max(abs(pulse)**2)/10.
ax_phase = axes[0].twinx()
ax_phase.plot(times[index_filter], np.unwrap(np.angle(pulse[index_filter])), color='red')
ax_phase.set_ylabel('phase (rad)')
axes[0].title('Retrieved Pulse')
original = abs(original_spectral_amplitude)**2
axes[1].plot(shifted_frequencies, original/np.max(original))
retrieved = abs(np.fft.fftshift(np.fft.fft(pulse)))**2
axes[1].plot(shifted_frequencies, retrieved/np.max(retrieved))
axes[1].set_xlabel('frequencies (Hz)')
plt.savefig(folder+'final_pulse.png', dpi=600)
plt.show()

#TODO: add error if shifted_frequencies != shifted_original frequencies
#plt.pcolormesh(shifted_frequencies, delays, trace)
#plt.show()
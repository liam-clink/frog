import numpy as np
import cv2 as cv #opencv
import matplotlib.pyplot as plt
import scipy.interpolate

folder = './First Stage/'
filenames = ['First Stage Processed Smoothed.png']

# Read the 16 bit hdr image
raw_data = cv.imread(folder+filenames[0], cv.IMREAD_UNCHANGED)
# Returns a numpy array with dtype uint16
# First index is vertical axis going downwards, second index is horizontal axis

calibration_wavelength = 609.e-9 # m
calibration_pixel = 2029 # px
wavelength_per_pixel = 0.0903e-9 # m/px
delay_per_pixel = 0.17e-15 # s/px
crop_left_pixel = 822

delays = delay_per_pixel*np.linspace(-raw_data.shape[0]/2., raw_data.shape[0]/2., raw_data.shape[0])
left_wavelength = calibration_wavelength + wavelength_per_pixel*(crop_left_pixel-calibration_pixel)
right_wavelength = left_wavelength + wavelength_per_pixel*raw_data.shape[1]
wavelengths = np.arange(left_wavelength, right_wavelength, wavelength_per_pixel)


# Shift minimum to zero
shifted_data = raw_data - np.min(raw_data)
plt.pcolormesh(wavelengths, delays, raw_data)
plt.show()

#TODO: Pad spectrum so time resolution is higher
# FROG image should be square with power of 2 side length if using General Projection.
# Here the image is interpolated in case this isn't already the case.
# Also, the dt and dw need to match, so dw = 2*pi/(duration) and dt = 2*pi/(ang. freq. bandwidth)
old_size = max(shifted_data.shape[0], shifted_data.shape[1])
exponent = (int(np.ceil(np.log2(shifted_data.shape[1]))))
grid_size = 1024 #2**exponent

old_frequencies = 2.99e8/wavelengths
frequencies = np.linspace(2.99e8/right_wavelength, 2.99e8/left_wavelength, grid_size, endpoint=True)
new_delays = np.linspace(delays[0], delays[-1], grid_size, endpoint=True)

interpolant = scipy.interpolate.interp2d(old_frequencies, delays, shifted_data)
interpolated_data = interpolant(frequencies, new_delays)

centered_frequencies = frequencies - (frequencies[0]+frequencies[-1])/2.
current_bandwidth = frequencies[-1]-frequencies[0]
desired_timestep = 1.e-15 # s, set by user
desired_bandwidth = 1/desired_timestep
padded_frequencies = centered_frequencies*desired_bandwidth/current_bandwidth

interpolant = scipy.interpolate.interp2d(centered_frequencies, new_delays, interpolated_data, bounds_error=False, fill_value=0.)
interpolated_data = interpolant(padded_frequencies, new_delays)

plt.contour(frequencies, new_delays, interpolated_data, levels=100)
plt.xlabel('frequency (THz)')
plt.ylabel('delay (fs)')
plt.title('Contour Plot of Interpolated Data')
print(interpolated_data.shape)
plt.show()

print('output datatype: ', interpolated_data.dtype)
np.savetxt(folder+'processed_data.tsv', interpolated_data, delimiter='\t')
np.savetxt(folder+'processed_data_delays.tsv', new_delays, delimiter='\t')
np.savetxt(folder+'processed_data_freqs.tsv', frequencies, delimiter='\t')

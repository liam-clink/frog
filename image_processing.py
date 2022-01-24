import wave
import numpy as np
import cv2 as cv #opencv
import matplotlib.pyplot as plt
import scipy.interpolate

folder = './Second Stage/'
filenames = ['final.png']

# Read the 16 bit hdr image
raw_data = cv.imread(folder+filenames[0], cv.IMREAD_UNCHANGED)
# Returns a numpy array with dtype uint16
# First index is vertical axis going downwards, second index is horizontal axis

calibration_wavelength = 609.e-9 # m
calibration_pixel = 2029 # px
wavelength_per_pixel = 0.0903e-9 # m/px
delay_per_pixel = 0.1e-15 # s/px
crop_left_pixel = 356

delays = delay_per_pixel*np.linspace(-raw_data.shape[0]/2., raw_data.shape[0]/2., raw_data.shape[0])
left_wavelength = calibration_wavelength + wavelength_per_pixel*(crop_left_pixel-calibration_pixel)
right_wavelength = left_wavelength + wavelength_per_pixel*raw_data.shape[1]
wavelengths = np.arange(left_wavelength, right_wavelength, wavelength_per_pixel)


# Shift minimum to zero
shifted_data = raw_data - np.min(raw_data)
plt.imshow(raw_data)
plt.show()


# Interpolating the spectra to power of 2 for better FFT results
def is_pow_2(n):
    return (n & (n-1) == 0) and (n != 0)

if not is_pow_2(shifted_data.shape[1]):
    exponent = (int(np.ceil(np.log2(shifted_data.shape[1]))))
    new_width = 2**exponent

    interpolant = scipy.interpolate.interp2d(wavelengths, delays, shifted_data)
    new_wavelengths = np.linspace(left_wavelength, right_wavelength, new_width, endpoint=True)
    interpolated_data = interpolant(new_wavelengths, delays)
else:
    new_width = shifted_data.shape[1]


np.savetxt(folder+'image_wavelengths.tsv', new_wavelengths,delimiter='\t')

plt.contour(new_wavelengths*1e9, delays*1e15, interpolated_data, levels=100)
plt.xlabel('wavelength (nm)')
plt.ylabel('delay (fs)')
plt.title('Contour Plot of Interpolated Data')
print(interpolated_data.shape)
plt.show()

print('output datatype: ', interpolated_data.dtype)
np.savetxt(folder+'processed_data.tsv', shifted_data, delimiter='\t')

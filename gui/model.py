import numpy as np
from astropy.io import fits
import cv2 as cv
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats

class Model:
    def __init__(self):
        self.fits_path = None
        self.kernel_size = 21
        self.threshold = 0.5
        self.blur_sigma = 5
        self.mask_dilate_size = 5
        self.attenuation_factor = 0.4

    def load_fits(self, path):
        self.fits_path = path
        hdul = fits.open(path)
        data = hdul[0].data
        hdul.close()
        return data

    def process_image(self, data, kernel_size, threshold):
        # Handle both monochrome and color images
        if data.ndim == 3:
            if data.shape[0] == 3:
                data = np.transpose(data, (1, 2, 0))
            data_gray = np.mean(data, axis=2)
            image = np.zeros_like(data, dtype='uint8')
            for i in range(data.shape[2]):
                channel = data[:, :, i]
                image[:, :, i] = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype('uint8')
        else:
            data_gray = data
            image = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')

        # Star detection
        mean, median, std = sigma_clipped_stats(data_gray, sigma=3.0)
        daofind = DAOStarFinder(fwhm=5.0, threshold=median + threshold * std)
        sources = daofind(data_gray)

        # Create mask
        mask = np.zeros_like(data_gray, dtype=np.uint8)
        if sources is not None:
            positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
            for pos in positions:
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < data_gray.shape[1] and 0 <= y < data_gray.shape[0]:
                    mask[y, x] = 255

        thresh_value = median + 3 * std
        bright_mask = (data_gray > thresh_value).astype(np.uint8) * 255
        mask = cv.bitwise_or(mask, bright_mask)

        kernel_dilate = np.ones((mask_dilate_size, mask_dilate_size), np.uint8)
        mask = cv.dilate(mask, kernel_dilate, iterations=2)

        # Erosion
        kernel_erosion = np.ones((5, 5), np.uint8)
        Ierode = cv.erode(image, kernel_erosion, iterations=2)

        # Blurred mask
        M = mask.astype(np.float32) / 255.0
        M = cv.GaussianBlur(M, (kernel_size, kernel_size), blur_sigma)

        if image.ndim == 3:
            M = np.stack([M] * image.shape[2], axis=2)

        Ifinal = (M * Ierode.astype(np.float32) + (1 - M) * image.astype(np.float32)).astype(np.uint8)

        # Reintegrate bright stars
        final = Ifinal.copy()
        if sources is not None:
            threshold_flux = np.percentile(sources['flux'], 90)
            for source in sources:
                if source['flux'] > threshold_flux:
                    x, y = int(source['xcentroid']), int(source['ycentroid'])
                    half_size = 15
                    x1, x2 = max(0, x - half_size), min(image.shape[1], x + half_size)
                    y1, y2 = max(0, y - half_size), min(image.shape[0], y + half_size)
                    star_patch = image[y1:y2, x1:x2].astype(np.float32)
                    attenuated_star = star_patch * attenuation_factor
                    final[y1:y2, x1:x2] = np.maximum(final[y1:y2, x1:x2], attenuated_star.astype(np.uint8))

        return final
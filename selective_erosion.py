from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import scipy.ndimage as ndi

# Open and read the FITS file
fits_file = './examples/HorseHead.fits'
hdul = fits.open(fits_file)

# Display information about the file
hdul.info()

# Access the data from the primary HDU
data = hdul[0].data

# Access header information
header = hdul[0].header

# Handle both monochrome and color images
if data.ndim == 3:
    # Color image - need to transpose to (height, width, channels)
    if data.shape[0] == 3:  # If channels are first: (3, height, width)
        data = np.transpose(data, (1, 2, 0))
    # If already (height, width, 3), no change needed

    # Normalize the entire image to [0, 1] for matplotlib
    data_normalized = (data - data.min()) / (data.max() - data.min())

    # Save the data as a png image (no cmap for color images)
    plt.imsave('./results/original.png', data_normalized)

    # Normalize each channel separately to [0, 255] for OpenCV
    image = np.zeros_like(data, dtype='uint8')
    for i in range(data.shape[2]):
        channel = data[:, :, i]
        image[:, :, i] = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype('uint8')
else:
    # Monochrome image
    plt.imsave('./results/original.png', data, cmap='gray')

    # Convert to uint8 for OpenCV
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')

# Étape A : Création du masque d'étoiles
# Utiliser DAOStarFinder pour détecter les étoiles
mean, median, std = sigma_clipped_stats(data, sigma=3.0)

daofind = DAOStarFinder(fwhm=5.0, threshold=median + 0.5*std)  # Baisser le seuil
sources = daofind(data)

# Créer un masque binaire pour les étoiles
mask = np.zeros_like(data, dtype=np.uint8)
if sources is not None:
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    for pos in positions:
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
            mask[y, x] = 255  # Marquer les étoiles

# Ajouter un seuillage pour les grosses étoiles (valeurs élevées)
thresh_value = median + 3 * std
bright_mask = (data > thresh_value).astype(np.uint8) * 255

# Combiner les masques : DAO + seuillage sur valeurs élevées
mask = cv.bitwise_or(mask, bright_mask)

# Dilater plus largement le masque pour couvrir les étoiles
kernel_dilate = np.ones((7,7), np.uint8)  # Plus grand kernel
mask = cv.dilate(mask, kernel_dilate, iterations=2)

# Étape B : Réduction localisée
# 1. Créer une version érodée de l'image originale (Ierode)
kernel_erosion = np.ones((5,5), np.uint8)
Ierode = cv.erode(image, kernel_erosion, iterations=2)

# 2. Créer un masque d'étoiles (M) avec bords adoucis par flou gaussien
M = mask.astype(np.float32) / 255.0  # Normaliser à [0, 1]
M = cv.GaussianBlur(M, (5, 5), 0)  # Flou gaussien pour adoucir les bords

# 3. Calculer l'image finale Ifinal = (M * Ierode) + ((1 - M) * image)
Ifinal = (M * Ierode.astype(np.float32) + (1 - M) * image.astype(np.float32)).astype(np.uint8)

# Sauvegarder les résultats
cv.imwrite('./results/eroded_selective.png', Ifinal)
cv.imwrite('./results/mask.png', mask)

# Afficher les informations
print(f"Nombre d'étoiles détectées : {len(sources) if sources is not None else 0}")

# Close the file
hdul.close()
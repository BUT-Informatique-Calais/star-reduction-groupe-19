import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QSpinBox,
    QVBoxLayout, QFileDialog
)
import numpy as np
import cv2 as cv
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

class StarReductionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Réduction d'étoiles - Interface simplifiée")
        self.fits_path = None

        # Widgets
        self.label_file = QLabel("Aucun fichier sélectionné")
        self.bouton_select = QPushButton("Choisir un fichier FITS")
        self.bouton_select.clicked.connect(self.select_file)

        self.kernel_size = QSpinBox(); self.kernel_size.setRange(1, 25); self.kernel_size.setValue(5)
        self.iterations = QSpinBox(); self.iterations.setRange(1, 10); self.iterations.setValue(2)

        self.bouton_run = QPushButton("Lancer le traitement")
        self.bouton_run.clicked.connect(self.run_pipeline)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label_file)
        layout.addWidget(self.bouton_select)
        layout.addWidget(QLabel("Taille du kernel d'érosion :")); layout.addWidget(self.kernel_size)
        layout.addWidget(QLabel("Itérations :")); layout.addWidget(self.iterations)
        layout.addWidget(self.bouton_run)
        self.setLayout(layout)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choisir un fichier FITS", "", "FITS (*.fits)")
        if path:
            self.fits_path = path
            self.label_file.setText(f"Fichier sélectionné : {path}")

    def run_pipeline(self):
        if not self.fits_path:
            self.label_file.setText("Aucun fichier sélectionné")
            return

        # Lecture FITS
        hdul = fits.open(self.fits_path)
        data = hdul[0].data
        hdul.close()

        # Conversion fu fits en png
        if data.ndim == 3:
            if data.shape[0] == 3:
                data = np.transpose(data, (1, 2, 0))
            image = np.zeros_like(data, dtype='uint8')
            for i in range(data.shape[2]):
                channel = data[:, :, i]
                image[:, :, i] = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype('uint8')
            data = np.mean(data, axis=2)
        else:
            image = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')

        cv.imwrite('./results/original.png', image)

        # Masque DAO + seuil
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        daofind = DAOStarFinder(fwhm=5.0, threshold=median + 0.5*std)
        sources = daofind(data)

        mask = np.zeros_like(data, dtype=np.uint8)
        if sources is not None:
            for x, y in zip(sources['xcentroid'], sources['ycentroid']):
                mask[int(y), int(x)] = 255

        bright_mask = (data > median + 3 * std).astype(np.uint8) * 255
        mask = cv.bitwise_or(mask, bright_mask)

        # Dilatation
        kernel_dilate = np.ones((5, 5), np.uint8)
        mask = cv.dilate(mask, kernel_dilate, iterations=2)
        dilated_bright = cv.dilate(bright_mask, kernel_dilate, iterations=1)

        # Érosion
        kernel_erosion = np.ones((self.kernel_size.value(), self.kernel_size.value()), np.uint8)
        Ierode = cv.erode(image, kernel_erosion, iterations=self.iterations.value())
        cv.imwrite('./results/eroded.png', Ierode)

        # Masque flou
        M = mask.astype(np.float32) / 255.0
        M = cv.GaussianBlur(M, (21, 21), 5)
        cv.imwrite('./results/blurred_mask.png', (M * 255).astype(np.uint8))

        if image.ndim == 3:
            M = np.stack([M] * image.shape[2], axis=2)
            dilated_bright = np.stack([dilated_bright] * image.shape[2], axis=2)

        # Fusion
        Ifinal = (M * Ierode.astype(np.float32) + (1 - M) * image.astype(np.float32)).astype(np.uint8)
        final = Ifinal.copy()

        # Réintégration atténuée (valeur fixe)
        if sources is not None:
            threshold_flux = np.percentile(sources['flux'], 90)
            for source in sources:
                if source['flux'] > threshold_flux:
                    x, y = int(source['xcentroid']), int(source['ycentroid'])
                    half_size = 15
                    x1, x2 = max(0, x - half_size), min(image.shape[1], x + half_size)
                    y1, y2 = max(0, y - half_size), min(image.shape[0], y + half_size)
                    star_patch = image[y1:y2, x1:x2].astype(np.float32)
                    attenuated_star = star_patch * 0.4
                    final[y1:y2, x1:x2] = np.maximum(final[y1:y2, x1:x2], attenuated_star.astype(np.uint8))

        cv.imwrite('./results/final.png', final)
        cv.imwrite('./results/mask.png', mask)

        self.label_file.setText("Traitement terminé. Résultats dans ./results/")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = StarReductionGUI()
    gui.show()
    sys.exit(app.exec_())

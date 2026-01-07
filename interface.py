import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QSlider,
    QVBoxLayout, QHBoxLayout, QFileDialog, QSpinBox
)
from PyQt5.QtCore import Qt
import cv2 as cv
import numpy as np
from astropy.io import fits

class ErosionGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("erosion d'image")
        self.fits_path = None

        # --- Widgets ---
        self.label_file = QLabel("Aucun fichier sélectionné")
        self.bouton_select = QPushButton("Choisir un fichier FITS")
        self.bouton_select.clicked.connect(self.select_file)

        # Kernel size
        self.kernel_label = QLabel("Taille du kernel :")
        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(1, 25)
        self.kernel_spin.setValue(5)

        # Iterations
        self.iter_label = QLabel("Itérations :")
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 10)
        self.iter_spin.setValue(2)

        # Launch button
        self.bouton_run = QPushButton("Lancer l'érosion")
        self.bouton_run.clicked.connect(self.run_erosion)

        # --- Layout ---
        layout = QVBoxLayout()
        layout.addWidget(self.label_file)
        layout.addWidget(self.bouton_select)

        layout.addWidget(self.kernel_label)
        layout.addWidget(self.kernel_spin)

        layout.addWidget(self.iter_label)
        layout.addWidget(self.iter_spin)

        layout.addWidget(self.bouton_run)

        self.setLayout(layout)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choisir un fichier FITS", "", "FITS (*.fits)")
        if path:
            self.fits_path = path
            self.label_file.setText(f"Fichier sélectionné : {path}")

    def run_erosion(self):
        if not self.fits_path:
            self.label_file.setText("Aucun fichier sélectionné")
            return

        # chargement du fichier
        hdul = fits.open(self.fits_path)
        data = hdul[0].data
        hdul.close()

        # convertit l'image fits en image png
        image = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')

        # Kernel
        k = self.kernel_spin.value()
        kernel = np.ones((k, k), np.uint8)

        # iterations
        it = self.iter_spin.value()

        # erosion
        eroded = cv.erode(image, kernel, iterations=it)

        # resultat d el'erosion
        cv.imwrite("eroded.png", eroded)
        self.label_file.setText(" Érosion terminée (eroded.png)")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ErosionGUI()
    gui.show()
    sys.exit(app.exec_())

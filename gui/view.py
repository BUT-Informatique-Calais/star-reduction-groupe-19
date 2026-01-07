from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QGroupBox, QGridLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np

class View(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Selective Erosion GUI')
        self.setGeometry(200, 150, 1000, 700)

        # style
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #dddddd;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3a3f44;
                padding: 10px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #50565c;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #444;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00aaff;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QGroupBox {
                border: 1px solid #444;
                margin-top: 12px;
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
            }
        """)

        main_layout = QVBoxLayout()

        #  BUTTON 
        self.load_button = QPushButton(' Charger un fichier FITS')
        self.load_button.setFixedHeight(40)
        main_layout.addWidget(self.load_button)

        #  SLIDERS GROUP 
        sliders_group = QGroupBox("Paramètres du traitement")
        sliders_layout = QGridLayout()

        # Kernel size
        self.kernel_label = QLabel('Taille du noyau : 21')
        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setMinimum(5)
        self.kernel_slider.setMaximum(51)
        self.kernel_slider.setValue(21)
        self.kernel_slider.setTickInterval(2)
        sliders_layout.addWidget(self.kernel_label, 0, 0)
        sliders_layout.addWidget(self.kernel_slider, 0, 1)

        # Threshold
        self.threshold_label = QLabel('Seuil du masque : 0.5')
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(20)
        self.threshold_slider.setValue(5)
        sliders_layout.addWidget(self.threshold_label, 1, 0)
        sliders_layout.addWidget(self.threshold_slider, 1, 1)

        # Blur sigma
        self.blur_sigma_label = QLabel('Sigma du flou : 5')
        self.blur_sigma_slider = QSlider(Qt.Horizontal)
        self.blur_sigma_slider.setMinimum(1)
        self.blur_sigma_slider.setMaximum(20)
        self.blur_sigma_slider.setValue(5)
        sliders_layout.addWidget(self.blur_sigma_label, 2, 0)
        sliders_layout.addWidget(self.blur_sigma_slider, 2, 1)

        # Mask dilate
        self.mask_dilate_label = QLabel('Taille dilatation : 5')
        self.mask_dilate_slider = QSlider(Qt.Horizontal)
        self.mask_dilate_slider.setMinimum(1)
        self.mask_dilate_slider.setMaximum(15)
        self.mask_dilate_slider.setValue(5)
        sliders_layout.addWidget(self.mask_dilate_label, 3, 0)
        sliders_layout.addWidget(self.mask_dilate_slider, 3, 1)

        # Attenuation
        self.attenuation_label = QLabel('Atténuation : 0.4')
        self.attenuation_slider = QSlider(Qt.Horizontal)
        self.attenuation_slider.setMinimum(0)
        self.attenuation_slider.setMaximum(10)
        self.attenuation_slider.setValue(4)
        sliders_layout.addWidget(self.attenuation_label, 4, 0)
        sliders_layout.addWidget(self.attenuation_slider, 4, 1)

        sliders_group.setLayout(sliders_layout)
        main_layout.addWidget(sliders_group)

        #  IMAGE DISPLAY 
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #111; border: 1px solid #333;")
        self.image_label.setFixedHeight(400)
        main_layout.addWidget(self.image_label)

        self.setLayout(main_layout)

    def update_image(self, image):
        image = np.ascontiguousarray(image)
        if image.ndim == 3:
            h, w, c = image.shape
            q_img = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            h, w = image.shape
            q_img = QImage(image.data, w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def update_labels(self, kernel_size, threshold, blur_sigma, mask_dilate_size, attenuation_factor):
        self.kernel_label.setText(f'Taille du noyau : {kernel_size}')
        self.threshold_label.setText(f'Seuil du masque : {threshold:.1f}')
        self.blur_sigma_label.setText(f'Sigma du flou : {blur_sigma}')
        self.mask_dilate_label.setText(f'Taille dilatation : {mask_dilate_size}')
        self.attenuation_label.setText(f'Atténuation : {attenuation_factor:.1f}')

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Langkah 1: Pilih file gambar menggunakan file dialog
def pilih_file():
    root = tk.Tk()
    root.withdraw()  # Sembunyikan jendela utama
    print("Pilih file gambar batik:")
    file_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg;.png;*.jpeg")])
    return file_path

# Langkah 2: Muat gambar dan ubah menjadi grayscale
def muat_gambar(file_path):
    image = cv2.imread(file_path)
    if image is None:
        print("Gambar tidak ditemukan atau format tidak didukung!")
        exit()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, image_gray

# Langkah 3: Hitung GLCM
def hitung_glcm(image_gray):
    distances = [1]  # Jarak piksel
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Sudut dalam radian
    glcm = graycomatrix(image_gray, distances=distances, angles=angles, symmetric=True, normed=True)
    return glcm

# Langkah 4: Ekstraksi fitur tekstur
def ekstraksi_fitur(glcm):
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    ASM = graycoprops(glcm, 'ASM')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    return contrast, dissimilarity, homogeneity, ASM, energy

# Main Program
if _name_ == "_main_":
    file_path = pilih_file()
    if not file_path:
        print("Tidak ada file yang dipilih. Program selesai.")
        exit()

    # Muat gambar
    image, image_gray = muat_gambar(file_path)

    # Tampilkan gambar grayscale
    plt.imshow(image_gray, cmap='gray')
    plt.title("Gambar Batik dalam Grayscale")
    plt.axis('off')
    plt.show()

    # Hitung GLCM dan ekstraksi fitur
    glcm = hitung_glcm(image_gray)
    contrast, dissimilarity, homogeneity, ASM, energy = ekstraksi_fitur(glcm)

    # Tampilkan hasil fitur tekstur
    print("\nHasil Ekstraksi Fitur Tekstur:")
    print(f"Contrast: {contrast}")
    print(f"Dissimilarity: {dissimilarity}")
    print(f"Homogeneity: {homogeneity}")
    print(f"ASM: {ASM}")
    print(f"Energy: {energy}")
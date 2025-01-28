import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import t

# import sympy as sym
from sklearn.metrics import r2_score
from scipy.special import erf

import os
import argparse

this_file_path = os.path.abspath(__file__)

plots_path = os.path.join(os.path.dirname(this_file_path), "plots")
os.makedirs(plots_path, exist_ok=True)

plt.rcParams["font.family"] = "serif"


def PI2(name):
    P = np.zeros(256)
    for i in range(len(name)):
        P = P + name[i, :]
    mean = P / len(name)
    return mean


def PI(name, inicial, final, h, y):
    P = np.zeros(y - h)
    for i in range(inicial, final):
        P = P + name[i, h:y]
    mean = P / (final - inicial)
    return mean


def func(x, A, n, m, b):
    k = A * erf((x - n) / (np.sqrt(2) * b)) + m
    return k


def derfunc(x, A, n, b):
    k = A * np.sqrt(2 / np.pi) * np.exp(-((x - n) ** 2) / (2 * b**2)) / b
    return k


def func2(x, A, B):
    y = A / x + B
    return y


def exponential(x, a, b):
    y = a * np.exp(-x * b)
    return y


def exponential2(x, a, b, c):
    y = a * np.exp(x * b) + c
    return y


def Lineal(x, a, b):
    y = a * x + b
    return y


def Cuadratic(x, a, b, c):
    y = a * x**b + c
    return y


def ImagenesR(name, Npixeles):
    I = []
    for k in range(0, len(name), Npixeles * Npixeles):
        Im = np.zeros((Npixeles, Npixeles))
        for j in range(Npixeles):
            for i in range(Npixeles):
                Im[j, i] = name[i + Npixeles * j + k]
        I.append(Im)
    return I


def Imagenes(name, Npixeles, NI):
    D = []
    for i in range(1, NI + 1):
        Raw = np.fromfile(name + "{}.raw".format(i), dtype="float32")
        K = ImagenesR(Raw, Npixeles)
        D.append(K)
    return D


def ImagenesA(name, Npixeles, NI, NThr):
    A = []
    for i in range(NThr):
        Mean = np.zeros((Npixeles, Npixeles))
        for j in range(NI):
            Mean = Mean + name[j][i]
        A.append(Mean)
    return A


def CNR(meanS, meanB, stdB):
    CNR = (meanB - meanS) / stdB
    return CNR






def tiba(
    cardName: str,
    nameSAM: str,
    nameREF: str,
    nameFF: str,
    Npixeles: int = 256,
    NI: int = 1,
    NF: int = 1,
    NThr: int = 1,
) -> None:

    SAMP = Imagenes(nameSAM, Npixeles, NI)
    SAMPA = ImagenesA(SAMP, Npixeles, NI, NThr)
    REF = Imagenes(nameREF, Npixeles, NI)
    REFA = ImagenesA(REF, Npixeles, NI, NThr)
    FF = Imagenes(nameFF, Npixeles, NF)
    FFA = ImagenesA(FF, Npixeles, NF, NThr)

    Sampc = []
    Refc = []
    for i in range(NThr):
        SAMPC = SAMPA[i] / FFA[i]
        REFC = REFA[i] / FFA[i]
        Sampc.append(SAMPC)
        Refc.append(REFC)

    k = 0
    ImG = plt.hist(np.ravel(Sampc[k]), bins=50, color="blue")
    images_path = os.path.join(plots_path, cardName)
    os.makedirs(images_path, exist_ok=True)
    # save
    plt.savefig(os.path.join(images_path, "histogram.png"))

    plt.figure(figsize=(7, 7))
    plt.imshow(Sampc[k], cmap="bone")
    plt.colorbar()

    plt.savefig(os.path.join(images_path, "Sampc.png"))


def read_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    nameSAM = lines[0].strip()
    nameREF = lines[1].strip()
    nameFF = lines[2].strip()
    
    return nameSAM, nameREF, nameFF

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesar el nombre de un archivo.")
    parser.add_argument("file_path", type=str, help="El nombre del archivo a procesar")
    args = parser.parse_args()

    file_path=args.file_path

    file_name = os.path.basename(file_path)
    
    print(file_name)

    # remove extension
    cardName = os.path.splitext(file_name)[0]
    nameSAM, nameREF, nameFF = read_file(file_path)
    print(cardName)
    print("nameSAM: ", nameSAM)
    print("nameREF: ", nameREF)
    print("nameFF: ", nameFF)

    tiba(cardName, nameSAM, nameREF, nameFF)

#python3 CNR_C.py /home/c.tibambre/CNR_Speckle_Results/tarjetas/tarjeta1.txt

"""
#!/bin/bash

# Directorio que contiene las tarjetas
DIRECTORIO_TARJETAS="/home/c.tibambre/CNR_Speckle_Results/tarjetas"

# Iterar sobre cada archivo en el directorio
for TARJETA in "$DIRECTORIO_TARJETAS"/*.txt; do
    # Ejecutar el script Python con la tarjeta actual
    python3 CNR_C.py "$TARJETA"
done
"""
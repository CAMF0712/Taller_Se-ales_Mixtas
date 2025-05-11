import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import sosfilt, sosfreqz
import scipy.io.wavfile as wav
import os

# === UTILIDADES ===
def leer_fcf(path):
    sos = []
    scales = []
    leyendo_sos = False
    leyendo_scales = False
    with open(path, 'r') as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith('%'):
                continue
            if "SOS Matrix:" in linea:
                leyendo_sos = True
                leyendo_scales = False
                continue
            elif "Scale Values:" in linea:
                leyendo_scales = True
                leyendo_sos = False
                continue
            if leyendo_sos:
                valores = list(map(float, linea.split()))
                if len(valores) == 6:
                    sos.append(valores)
            elif leyendo_scales:
                try:
                    val = float(linea)
                    scales.append(val)
                except ValueError:
                    continue
    sos = np.array(sos)
    scales = np.array(scales)
    return sos, scales

def plot_fft(signal, fs, title):
    N = len(signal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    fft_mag = np.abs(np.fft.rfft(signal)) / N
    plt.plot(freqs, 20*np.log10(fft_mag + 1e-12), label=title)

"""
def imprimir_frecuencias_de_corte(sos_scaled, fs, tipo):
    w, h = sosfreqz(sos_scaled, worN=2000, fs=fs)
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    umbral = -3
    indices = np.where(mag_db <= umbral)[0]
    if tipo in ["Pasa Bajas", "Pasa Altas"]:
        if indices.size > 0:
            fc = w[indices[0]] if tipo == "Pasa Bajas" else w[indices[-1]]
            print(f"\n Frecuencia de corte real del filtro {tipo}: {fc:.2f} Hz")
        else:
            print(f"⚠ No se encontró una frecuencia de corte clara para el filtro {tipo}")
    elif tipo in ["Pasa Bandas", "Rechaza Banda"]:
        if indices.size > 1:
            f1 = w[indices[0]]
            f2 = w[indices[-1]]
            print(f"\n Frecuencias de corte reales del filtro {tipo}: {f1:.2f} Hz – {f2:.2f} Hz")
        else:
            print(f"⚠ No se identificaron dos puntos de corte claros para el filtro {tipo}")
"""

def procesar_audio(audio_path, filtro_path, tipo_filtro):
    fs, audio = wav.read(audio_path)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]  # solo canal izquierdo si es estéreo

    sos, scales = leer_fcf(filtro_path)
    sos_scaled = sos.copy()
    sos_scaled[:, 0:3] *= scales.reshape(-1, 1)

    #imprimir_frecuencias_de_corte(sos_scaled, fs, tipo_filtro)

    print("\nReproduciendo sonido original...")
    sd.play(audio / np.max(np.abs(audio)), fs)
    sd.wait()

    audio_filtrado = sosfilt(sos_scaled, audio)
    audio_filtrado_norm = audio_filtrado / np.max(np.abs(audio_filtrado))

    print("Reproduciendo sonido filtrado...")
    sd.play(audio_filtrado_norm, fs)
    sd.wait()

    plt.figure(figsize=(10, 6))
    plot_fft(audio, fs, 'Original')
    plot_fft(audio_filtrado, fs, 'Filtrado')
    plt.title(f'FFT antes y después del filtrado ({tipo_filtro})')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (dB)')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, fs // 2)
    plt.show()

# === MAPEO DE OPCIONES ===
filtros = {
    "1": ("FPasaBajas.fcf", "Pasa Bajas"),
    "2": ("FPasaAltas.fcf", "Pasa Altas"),
    "3": ("FPasaBandas.fcf", "Pasa Bandas"),
    "4": ("FSuprimeBandas.fcf", "Rechaza Banda"),
}

audios = {
    "1": "muelle_san_blás.wav",
    "2": "creep.wav",
    "3": "drum_go_dum.wav",
    "4": "barrido.wav",
}

# === INTERFAZ DE CONSOLA ===
while True:
    print("\n¿Qué filtro deseas aplicar?")
    print("1. Pasa Bajas")
    print("2. Pasa Altas")
    print("3. Pasa Bandas")
    print("4. Rechaza Banda")
    print("5. Salir")
    opcion_filtro = input("Ingresa el número de filtro: ").strip()

    if opcion_filtro == "5":
        print("Saliendo del programa.")
        break

    if opcion_filtro not in filtros:
        print("Opción de filtro inválida.")
        continue

    print("\n¿Qué archivo de audio deseas usar?")
    print("1. muelle_san_blás.wav")
    print("2. creep.wav")
    print("3. drum_go_dum.wav")
    print("4. barrido.wav")
    opcion_audio = input("Ingresa el número de archivo de audio: ").strip()

    if opcion_audio not in audios:
        print("Opcion de audio inválida.")
        continue

    ruta_filtro, nombre_filtro = filtros[opcion_filtro]
    ruta_audio = audios[opcion_audio]

    if not os.path.exists(ruta_filtro):
        print(f"No se encontró el archivo de filtro '{ruta_filtro}'.")
        continue
    if not os.path.exists(ruta_audio):
        print(f"No se encontró el archivo de audio '{ruta_audio}'.")
        continue

    procesar_audio(ruta_audio, ruta_filtro, nombre_filtro)

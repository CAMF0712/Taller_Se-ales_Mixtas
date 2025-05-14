import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfreqz
import scipy.io.wavfile as wav
import warnings
import os
import tkinter as tk
from tkinter import messagebox, simpledialog

# === UTILIDADES ===
def leer_fcf(path):
    sos, scales = [], []
    leyendo_sos = leyendo_scales = False
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
    return np.array(sos), np.array(scales)

def plot_fft(signal, fs, title):
    N = len(signal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    fft_mag = np.abs(np.fft.rfft(signal)) / N
    plt.plot(freqs, 20*np.log10(fft_mag + 1e-12), label=title)

def procesar_audio(audio_path, filtro_path, tipo_filtro):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs, audio = wav.read(audio_path)

    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]

    sos, scales = leer_fcf(filtro_path)
    sos_scaled = sos.copy()
    sos_scaled[:, 0:3] *= scales.reshape(-1, 1)

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

def filtro_configurable(audio_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs, audio = wav.read(audio_path)

    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]

    order = simpledialog.askinteger("Orden", "Ingrese el orden del filtro:")
    f_low = simpledialog.askfloat("Frecuencia Baja", "Ingrese frecuencia baja (Hz):")
    f_high = simpledialog.askfloat("Frecuencia Alta", "Ingrese frecuencia alta (Hz):")

    if order is None or f_low is None or f_high is None:
        return

    if f_low >= f_high or f_high >= fs / 2:
        messagebox.showerror("Error", "⚠ Frecuencias inválidas. Asegúrese de que f_low < f_high < fs/2.")
        return

    sos = butter(order, [f_low, f_high], btype='bandpass', fs=fs, output='sos')
    y = sosfilt(sos, audio)
    y_norm = y / np.max(np.abs(y))

    sd.play(audio / np.max(np.abs(audio)), fs)
    sd.wait()
    sd.play(y_norm, fs)
    sd.wait()

    w, h = sosfreqz(sos, worN=2000, fs=fs)
    plt.figure(figsize=(10, 4))
    plt.plot(w, 20 * np.log10(np.abs(h) + 1e-12))
    plt.title('Respuesta en frecuencia del filtro configurable')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Ganancia (dB)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plot_fft(audio, fs, 'Original')
    plot_fft(y, fs, 'Filtrado Configurable')
    plt.title('FFT antes y después del filtrado (configurable)')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (dB)')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, fs // 2)
    plt.show()

# === INTERFAZ GRÁFICA ===
filtros = {
    "Pasa Bajas": ("FPasaBajas.fcf", "Pasa Bajas"),
    "Pasa Altas": ("FPasaAltas.fcf", "Pasa Altas"),
    "Pasa Bandas": ("FPasaBandas.fcf", "Pasa Bandas"),
    "Rechaza Banda": ("FSuprimeBandas.fcf", "Rechaza Banda"),
}

audios = {
    "Muelle": "muelle_san_blás.wav",
    "Creep": "creep.wav",
    "Drum": "drum_go_dum.wav",
    "Barrido": "barrido.wav",
}

audio_seleccionado = [None]  # para usar dentro de funciones

def seleccionar_audio(nombre):
    ruta = audios[nombre]
    if not os.path.exists(ruta):
        messagebox.showerror("Error", f"No se encontró '{ruta}'")
        return
    audio_seleccionado[0] = ruta
    messagebox.showinfo("Audio", f"Seleccionado: {nombre}")

def aplicar_filtro(nombre):
    if audio_seleccionado[0] is None:
        messagebox.showerror("Error", "Seleccione un archivo de audio primero.")
        return
    ruta_filtro, nombre_filtro = filtros[nombre]
    if not os.path.exists(ruta_filtro):
        messagebox.showerror("Error", f"No se encontró '{ruta_filtro}'")
        return
    procesar_audio(audio_seleccionado[0], ruta_filtro, nombre_filtro)

def aplicar_configurable():
    if audio_seleccionado[0] is None:
        messagebox.showerror("Error", "Seleccione un archivo de audio primero.")
        return
    filtro_configurable(audio_seleccionado[0])

root = tk.Tk()
root.title("Procesador de Filtros de Audio Jason Bourne")
root.configure(bg="#d0e6f7")  # Color azul claro

frame = tk.Frame(root)
frame.pack(padx=100, pady=100)

tk.Label(frame, text="Selecciona un archivo de audio:").pack()
for nombre in audios:
    tk.Button(frame, text=nombre, width=20, command=lambda n=nombre: seleccionar_audio(n)).pack()

tk.Label(frame, text="\nAplicar filtro fijo:").pack()
for nombre in filtros:
    tk.Button(frame, text=nombre, width=20, command=lambda n=nombre: aplicar_filtro(n)).pack()

tk.Label(frame, text="\nFiltro Pasa Bandas Configurable:").pack()
tk.Button(frame, text="Configurable", width=20, command=aplicar_configurable).pack()

tk.Button(frame, text="Salir", command=root.quit).pack(pady=10)

root.mainloop()

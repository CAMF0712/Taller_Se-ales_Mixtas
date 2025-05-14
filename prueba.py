import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfreqz
import scipy.io.wavfile as wav
import warnings
import os
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, Listbox, MULTIPLE, Scrollbar, RIGHT, Y

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

def cargar_audio(ruta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs, audio = wav.read(ruta)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return fs, audio

def combinar_senales(rutas):
    señales = []
    fs = None
    for ruta in rutas:
        f, audio = cargar_audio(ruta)
        if fs is None:
            fs = f
        elif f != fs:
            raise ValueError("Los archivos deben tener la misma frecuencia de muestreo.")
        señales.append(audio)

    max_len = max(len(s) for s in señales)
    señales_padded = [np.pad(s, (0, max_len - len(s))) for s in señales]
    return fs, np.mean(señales_padded, axis=0)

def procesar_audio_combinado(rutas, filtro_path, tipo_filtro):
    fs, audio = combinar_senales(rutas)
    print(" Reproduciendo sonido original combinado...")
    sd.play(audio / np.max(np.abs(audio)), fs)
    sd.wait()

    sos, scales = leer_fcf(filtro_path)
    sos_scaled = sos.copy()
    sos_scaled[:, 0:3] *= scales.reshape(-1, 1)

    filtrado = sosfilt(sos_scaled, audio)
    filtrado_norm = filtrado / np.max(np.abs(filtrado))
    print(" Reproduciendo sonido filtrado...")
    sd.play(filtrado_norm, fs)
    sd.wait()

    plt.figure(figsize=(10, 6))
    plot_fft(audio, fs, 'Original')
    plot_fft(filtrado, fs, f'Filtrado ({tipo_filtro})')
    plt.title(f'FFT antes y después del filtrado ({tipo_filtro})')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (dB)')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, fs // 2)
    plt.show()

def filtro_configurable(rutas):
    fs, audio = combinar_senales(rutas)
    order = simpledialog.askinteger("Orden", "Ingrese el orden del filtro:")
    f_corte = simpledialog.askfloat("Frecuencia Corte", "Ingrese frecuencia de corte (Hz):")

    sos = butter(order, [f_corte], btype='lowpass', fs=fs, output='sos')
    y = sosfilt(sos, audio)
    y_norm = y / np.max(np.abs(y))

    print(" Reproduciendo sonido original combinado...")
    sd.play(audio / np.max(np.abs(audio)), fs)
    sd.wait()
    print(" Reproduciendo sonido filtrado...")
    sd.play(y_norm, fs)
    sd.wait()

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

# === INTERFAZ ===
root = tk.Tk()
root.title("Procesador de Filtros de Audio Jason Bourne")
root.configure(bg="#f0f8ff")

filtros = {
    "Pasa Bajas": ("FPasaBajas.fcf", "Pasa Bajas"),
    "Pasa Altas": ("FPasaAltas.fcf", "Pasa Altas"),
    "Pasa Bandas": ("FPasaBandas.fcf", "Pasa Bandas"),
    "Rechaza Banda": ("FSuprimeBandas.fcf", "Rechaza Banda"),
}

audios = {
    "Muelle San Blas": "muelle_san_blás.wav",
    "Creep": "creep.wav",
    "Drum": "drum_go_dum.wav",
    "Barrido": "barrido.wav",
}

# === WIDGETS ===
frame = tk.Frame(root)
frame.pack(padx=30, pady=30)

tk.Label(frame, text="Selecciona archivos de audio:").pack()

listbox = Listbox(frame, selectmode=MULTIPLE, width=40)
listbox.pack()

scrollbar = Scrollbar(frame, orient="vertical", command=listbox.yview)
scrollbar.pack(side=RIGHT, fill=Y)
listbox.config(yscrollcommand=scrollbar.set)

# Inicializar lista
for nombre in audios:
    listbox.insert(tk.END, nombre)

def agregar_archivos():
    archivos = filedialog.askopenfilenames(filetypes=[("Archivos WAV", "*.wav")])
    for ruta in archivos:
        nombre = os.path.basename(ruta)
        if nombre not in audios:
            audios[nombre] = ruta
            listbox.insert(tk.END, nombre)

def obtener_rutas_seleccionadas():
    seleccion = listbox.curselection()
    return [audios[listbox.get(i)] for i in seleccion]

def aplicar_filtro(nombre_filtro):
    rutas = obtener_rutas_seleccionadas()
    if not rutas:
        messagebox.showerror("Error", "Selecciona al menos un archivo de audio.")
        return
    ruta_filtro, tipo = filtros[nombre_filtro]
    if not os.path.exists(ruta_filtro):
        messagebox.showerror("Error", f"No se encontró '{ruta_filtro}'")
        return
    procesar_audio_combinado(rutas, ruta_filtro, tipo)

def aplicar_configurable():
    rutas = obtener_rutas_seleccionadas()
    if not rutas:
        messagebox.showerror("Error", "Selecciona al menos un archivo de audio.")
        return
    filtro_configurable(rutas)

tk.Button(frame, text="Agregar nuevo archivo .wav", command=agregar_archivos).pack(pady=5)

tk.Label(frame, text="\nAplicar filtro fijo:").pack()
for nombre in filtros:
    tk.Button(frame, text=nombre, width=25, command=lambda n=nombre: aplicar_filtro(n)).pack()

tk.Label(frame, text="\nFiltro Pasa Bajas Configurable:").pack()
tk.Button(frame, text="Configurable", width=25, command=aplicar_configurable).pack()

tk.Button(frame, text="Salir", command=root.quit).pack(pady=10)

root.mainloop()

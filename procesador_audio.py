import librosa
import numpy as np

def procesar_audio_tactico(ruta_archivo):
    """
    Convierte un audio en un tensor de Espectrograma de Mel estandarizado.
    Cumple con el Punto 4 de la Metodología.
    """
    # Parámetros del proyecto
    sr_objetivo = 22050
    duracion_objetivo = 4.0
    muestras_objetivo = int(sr_objetivo * duracion_objetivo) # 88,200 muestras
    
    # 1. Estandarización de Frecuencia y Canal Mono
    y, sr = librosa.load(ruta_archivo, sr=sr_objetivo, mono=True)
    
    # 2. Recorte y Zero-Padding Centrado
    if len(y) > muestras_objetivo:
        # Si viene de ESC-50 (5 segundos), recortamos a 4 segundos
        y = y[:muestras_objetivo]
    elif len(y) < muestras_objetivo:
        # Si es un transitorio corto de UrbanSound8K, rellenamos con ceros
        pad_total = muestras_objetivo - len(y)
        pad_izq = pad_total // 2
        pad_der = pad_total - pad_izq
        y = np.pad(y, (pad_izq, pad_der), mode='constant')
        
    # 3. Transformada a Espectrograma de Mel
    # Se utiliza una ventana de 1024 y un salto de 512 (aprox 50% de superposición)
    espectrograma_mel = librosa.feature.melspectrogram(
        y=y, 
        sr=sr_objetivo, 
        n_fft=1024, 
        hop_length=512, 
        n_mels=128
    )
    
    # Conversión a escala logarítmica (Decibelios)
    mel_db = librosa.power_to_db(espectrograma_mel, ref=np.max)
    
    # 4. Ajuste de dimensiones para Conv2D (Alto, Ancho, Canales)
    # Resultado esperado: (128, 173, 1)
    tensor_cnn = mel_db[..., np.newaxis]
    
    return tensor_cnn
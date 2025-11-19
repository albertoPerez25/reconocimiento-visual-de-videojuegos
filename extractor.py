import os
import subprocess
import json
from pathlib import Path

def get_video_duration(video_path):
    """Obtiene la duración del vídeo en segundos usando ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception as e:
        print(f"Error obteniendo duración de {video_path}: {e}")
        return None

def extract_frames(video_path, output_dir, video_number):
    """Extrae fotogramas del vídeo según su duración."""
    duration = get_video_duration(video_path)
    
    if duration is None:
        return
    
    # Determinar intervalo según duración (1 hora = 3600 segundos)
    if duration < 1800: # Menos de media hora
        interval = 10  # 10 segundos
    elif duration >= 1800 and duration < 7200: # Entre media hora y 2 horas
        interval = 30  # 30 segundos
    elif duration >= 7200 and duration < 14400: # Entre 2 horas y 4 horas
        interval = 60 # 1 minuto
    else:
        interval = 120 # 2 minutos
    
    print(f"Vídeo {video_number}: {os.path.basename(video_path)}")
    print(f"  Duración: {duration:.2f}s ({duration/60:.2f} min)")
    print(f"  Intervalo: {interval}s")
    
    # Calcular número de fotogramas a extraer
    num_frames = int(duration / interval)
    
    # Extraer fotogramas usando ffmpeg
    for frame_num in range(num_frames + 1):
        timestamp = frame_num * interval
        
        # Formato: XXXXXXX_Y.jpg
        output_filename = f"{frame_num:07d}_{video_number}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        cmd = [
            'ffmpeg',
            '-ss', str(timestamp),
            '-i', video_path,
            '-frames:v', '1',
            '-q:v', '2',  # Calidad alta
            '-y',  # Sobrescribir si existe
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"  ✓ Fotograma {frame_num} extraído ({timestamp}s)")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error extrayendo fotograma {frame_num}: {e}")
    
    print()

def main():
    # Directorio con los vídeos (directorio actual)
    video_dir = "."
    
    # Directorio de salida para las imágenes
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extensiones de vídeo comunes
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm'}
    
    # Obtener lista de vídeos
    video_files = []
    for file in os.listdir(video_dir):
        if Path(file).suffix.lower() in video_extensions:
            video_files.append(file)
    
    video_files.sort()  # Ordenar alfabéticamente
    
    if not video_files:
        print("No se encontraron vídeos en el directorio actual.")
        return
    
    print(f"Se encontraron {len(video_files)} vídeos.\n")
    
    # Procesar cada vídeo
    for idx, video_file in enumerate(video_files, start=1):
        video_path = os.path.join(video_dir, video_file)
        extract_frames(video_path, output_dir, idx)
    
    print(f"✓ Proceso completado. Imágenes guardadas en '{output_dir}/'")

if __name__ == "__main__":
    main()

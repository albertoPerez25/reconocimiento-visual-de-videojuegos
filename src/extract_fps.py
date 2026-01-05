import os
import sys
import subprocess
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Usando extractor como base, se ha modificado para hacerlo más eficiente

# ================= CONFIGURACIÓN =================
class Config:
    EXTERNAL_DIR = "../DatasetVideos/" # Carpeta donde guardar los frames
    INPUT_DIR = EXTERNAL_DIR       # Carpeta raíz donde buscar videos
    OUTPUT_SUFFIX = "_frames"    # Sufijo para la carpeta de salida
    MAX_WORKERS = 4              # Número de procesos simultáneos
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm'}
    FFMPEG_QUALITY = '2'         # 1-31 (1 es mejor calidad, 31 peor. 2 es muy alta)

# ================= LÓGICA DE ESTADO =================
class Stats:
    def __init__(self):
        self.lock = Lock()
        self.total_videos = 0
        self.processed = 0
        self.failed = 0

    def mark_success(self):
        with self.lock:
            self.processed += 1
            self.print_progress()

    def mark_fail(self, video_path, error):
        with self.lock:
            self.failed += 1
            print(f"\n[ERROR] {os.path.basename(video_path)}: {error}")
            self.print_progress()

    def print_progress(self):
        sys.stdout.write(f"\r[PROGRESO] Videos Procesados: {self.processed}/{self.total_videos} | Fallos: {self.failed}")
        sys.stdout.flush()

# ================= FUNCIONES CORE =================

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
        # En Windows a veces es necesario shell=False, en Linux no afecta mucho
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception as e:
        raise Exception(f"No se pudo obtener duración: {e}")

def determine_interval(duration):
    """Determina el intervalo de captura en segundos basado en la duración del video."""
    if duration < 1800:                  # < 30 min
        return 4
    elif 1800 <= duration < 7200:        # 30 min - 2 horas
        return 8
    elif 7200 <= duration < 14400:       # 2 horas - 4 horas
        return 15
    else:                                # > 4 horas
        return 30

def process_video(task, stats):
    """
    Procesa un único video: calcula intervalo y ejecuta ffmpeg UNA sola vez
    usando el filtro fps para máxima eficiencia.
    """
    video_path = task['path']
    video_id = task['id']
    
    # Definir estructura de salida:
    # Si video está en: .../Juego/Juego_videos/video.mp4
    # Salida será:      .../Juego/Juego_frames/video_nombre/XXXX.jpg
    
    parent_dir = os.path.dirname(video_path) # Carpeta .../Juego_videos
    grandparent_dir = os.path.dirname(parent_dir) # Carpeta .../Juego
    
    # Nombre base del video sin extensión para la subcarpeta
    video_basename = Path(video_path).stem
    
    # Crear carpeta hermana "Juego_frames" si detectamos esa estructura
    if parent_dir.endswith("_videos"):
        base_folder_name = os.path.basename(parent_dir).replace("_videos", "")
        output_root = os.path.join(grandparent_dir, f"{base_folder_name}{Config.OUTPUT_SUFFIX}")
    else:
        # Fallback genérico si no sigue la estructura anterior
        output_root = os.path.join(parent_dir, "extracted_frames")

    # Carpeta específica para este video (para no mezclar fotogramas de distintos videos)
    final_output_dir = os.path.join(output_root, video_basename)
    os.makedirs(final_output_dir, exist_ok=True)

    try:
        duration = get_video_duration(video_path)
        interval = determine_interval(duration)
        
        # Construcción del comando optimizado
        # Usamos el filtro 'fps=1/interval' para extraer un frame cada X segundos
        # %07d genera números como 0000001, 0000002, etc.
        output_pattern = os.path.join(final_output_dir, f"%07d_{video_id}.jpg")
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps=1/{interval}', # MAGIA AQUÍ: Extrae periódicamente sin loop Python
            '-q:v', Config.FFMPEG_QUALITY,
            '-loglevel', 'error',       # Menos ruido en consola
            '-n',                       # NO sobrescribir, saltar
            output_pattern
        ]

        subprocess.run(cmd, check=True)
        stats.mark_success()

    except Exception as e:
        stats.mark_fail(video_path, str(e))

def find_videos(root_dir):
    """Busca videos recursivamente."""
    video_list = []
    print(f"Buscando videos en '{root_dir}'...")
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if Path(file).suffix.lower() in Config.VIDEO_EXTENSIONS:
                full_path = os.path.join(root, file)
                video_list.append(full_path)
                
    return video_list

def main():
    print("=== Iniciando Extractor de Fotogramas Paralelo ===")
    
    if not os.path.exists(Config.INPUT_DIR):
        # Fallback al directorio actual si no existe dataset
        print(f"Advertencia: '{Config.INPUT_DIR}' no existe. Buscando en directorio actual.")
        search_dir = "."
    else:
        search_dir = Config.INPUT_DIR

    videos = find_videos(search_dir)
    
    stats = Stats()
    stats.total_videos = len(videos)
    
    if stats.total_videos == 0:
        print("No se encontraron videos.")
        return

    print(f"Se encontraron {stats.total_videos} videos. Iniciando extracción con {Config.MAX_WORKERS} hilos...")
    print("Nota: Se usará el método eficiente 'fps filter' de ffmpeg.")

    # Preparar tareas
    tasks = []
    for idx, video_path in enumerate(videos, start=1):
        tasks.append({
            'path': video_path,
            'id': idx
        })

    # Ejecución Paralela
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        futures = [executor.submit(process_video, task, stats) for task in tasks]
        
        for future in as_completed(futures):
            pass # El manejo de errores y progreso ya está dentro de process_video/stats

    print("\n\n=== Proceso Completado ===")
    print(f"Total Videos: {stats.total_videos}")
    print(f"Procesados Correctamente: {stats.processed}")
    print(f"Fallidos: {stats.failed}")

if __name__ == "__main__":
    # Verificar dependencias
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Error Crítico: ffmpeg o ffprobe no están instalados o no están en el PATH.")
        sys.exit(1)
        
    main()

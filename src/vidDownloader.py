import os
import sys
import yt_dlp
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ================= CONFIGURACIÓN =================
class Config:
    DATASET_DIR = "dataset"  # Carpeta raíz
    MAX_WORKERS = 4          # Número de descargas simultáneas (Núcleos/Hilos)
    VIDEO_HEIGHT = 480       # Calidad objetivo
    RETRIES = 3              # Intentos por video
    
    # Formato de yt-dlp para forzar <= 480p
    # Intenta bajar el mejor video que sea menor o igual a 480p.
    # Si no puede, baja el mejor formato combinado menor o igual a 480p.
    FORMAT_STR = f'bestvideo[height<={VIDEO_HEIGHT}]'
    EXTERNAL_OUTPUT_DIR = "/mnt/ntfs/kosos/alber/Linux/Dataset/"

# ================= LÓGICA DE ESTADO =================
class Stats:
    def __init__(self):
        self.lock = Lock()
        self.total = 0
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.failed_urls = []

    def increment_total(self):
        with self.lock:
            self.total += 1

    def mark_success(self):
        with self.lock:
            self.completed += 1
            self.print_progress()

    def mark_fail(self, url, game_name, error_msg):
        with self.lock:
            self.failed += 1
            self.failed_urls.append(f"[{game_name}] {url} -> {error_msg}")
            self.print_progress()

    def print_progress(self):
        # Sobrescribe la línea actual para mostrar progreso limpio
        processed = self.completed + self.failed
        sys.stdout.write(f"\r[PROGRESO] Procesados: {processed}/{self.total} | Éxitos: {self.completed} | Fallos: {self.failed}")
        sys.stdout.flush()

# ================= LÓGICA PRINCIPAL =================

def get_download_tasks(base_dir):
    """
    Recorre el directorio buscando la estructura:
    dataset/Juego/Juego (archivo)
    Retorna una lista de tareas (diccionarios).
    """
    tasks = []
    
    if not os.path.exists(base_dir):
        print(f"Error: La carpeta '{base_dir}' no existe.")
        return tasks

    # Listar carpetas dentro de dataset
    for game_folder in os.listdir(base_dir):
        game_path = os.path.join(base_dir, game_folder)
        
        if os.path.isdir(game_path):
            # El archivo de enlaces debe llamarse igual que la carpeta
            links_file = os.path.join(game_path, game_folder).rstrip('\n') + ".txt"
            
            if os.path.exists(links_file) and os.path.isfile(links_file):
                if not Config.EXTERNAL_OUTPUT_DIR:
                    # Carpeta destino: dataset/Juego/Juego_videos
                    output_dir = os.path.join(game_path, f"{game_folder}_videos")
                else:
                    output_dir = os.path.join(Config.EXTERNAL_OUTPUT_DIR, f"{game_folder}_videos")
                
                try:
                    with open(links_file, 'r', encoding='utf-8') as f:
                        urls = [line.strip() for line in f if line.strip()]
                        
                    for url in urls:
                        tasks.append({
                            'url': url,
                            'output_dir': output_dir,
                            'game_name': game_folder
                        })
                except Exception as e:
                    print(f"\nError leyendo archivo {links_file}: {e}")
            else:
                # Opcional: Avisar si una carpeta no cumple la estructura
                print(f"Saltando {game_folder}: No se encontró el archivo de enlaces '{game_folder}' dentro.")
                pass
                
    return tasks

def download_video(task, stats):
    """
    Función que ejecuta un hilo/proceso individual de yt-dlp.
    """
    url = task['url']
    output_dir = task['output_dir']
    game_name = task['game_name']

    # Opciones específicas para esta descarga
    ydl_opts = {
        'format': Config.FORMAT_STR,
        'paths': {'home': output_dir},
        # Plantilla de nombre: Título del video.extensión
        'outtmpl': '%(title)s.%(ext)s',
        'quiet': True,              # No imprimir salida estándar de yt-dlp
        'no_warnings': True,
        'ignoreerrors': False,      # Queremos capturar el error nosotros
        'retries': Config.RETRIES,
        'noprogress': True,         # Usamos nuestro propio progreso
        # Opción importante para evitar descargar lo mismo dos veces si se interrumpe
        'no_overwrites': True,
        'continuedl': True,
    }

    try:
        # Nos aseguramos que exista la carpeta
        os.makedirs(output_dir, exist_ok=True)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        stats.mark_success()
        
    except Exception as e:
        # Limpiamos el mensaje de error para que no sea kilométrico
        error_msg = str(e).split('\n')[0]
        stats.mark_fail(url, game_name, error_msg)

def main():
    print("=== Iniciando Script de Descarga Masiva ===")
    print(f"Directorio base: {Config.DATASET_DIR}")
    print(f"Calidad objetivo: {Config.VIDEO_HEIGHT}")
    print(f"Hilos paralelos: {Config.MAX_WORKERS}")
    print("-------------------------------------------")

    stats = Stats()
    tasks = get_download_tasks(Config.DATASET_DIR)
    
    # Establecer el total para la barra de progreso
    stats.total = len(tasks)
    
    if stats.total == 0:
        print("No se encontraron enlaces para descargar. Verifica la estructura de carpetas.")
        return

    print(f"Se encontraron {stats.total} videos para procesar. Comenzando descargas...")

    # Paralelización usando ThreadPoolExecutor
    # Usamos ThreadPool en vez de ProcessPool porque la descarga es I/O bound (red/disco)
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        futures = [executor.submit(download_video, task, stats) for task in tasks]
        
        # Esperar a que terminen todas
        for future in as_completed(futures):
            pass # La lógica de actualización ya está dentro de la función download_video

    print("\n\n=== Resumen Final ===")
    print(f"Total procesados: {stats.total}")
    print(f"Descargados/Existentes: {stats.completed}")
    print(f"Fallidos: {stats.failed}")
    
    if stats.failed > 0:
        print("\nDetalle de errores:")
        for fail in stats.failed_urls:
            print(f" - {fail}")
        
        # Opcional: Guardar log de errores
        with open("errores_descarga.log", "w", encoding="utf-8") as f:
            f.write("\n".join(stats.failed_urls))
        print("\nSe ha generado un archivo 'errores_descarga.log' con los detalles.")

if __name__ == "__main__":
    main()
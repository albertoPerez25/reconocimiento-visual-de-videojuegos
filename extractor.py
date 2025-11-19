import cv2
import os
from moviepy.editor import VideoFileClip

# Carpeta con los vídeos
VIDEO_FOLDER = "videos"
# Carpeta donde se guardarán las imágenes
OUTPUT_FOLDER = "images"

# Crear carpeta images si no existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Obtener lista de vídeos
videos = sorted([f for f in os.listdir(VIDEO_FOLDER)
                 if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".mpeg"))])

for video_index, video_name in enumerate(videos, start=1):
    video_path = os.path.join(VIDEO_FOLDER, video_name)

    # Obtener duración del vídeo
    clip = VideoFileClip(video_path)
    duration_seconds = clip.duration

    # Elegir intervalo
    if duration_seconds < 3600:
        interval = 30      # cada 30s
    else:
        interval = 60      # cada 1 min
    clip.close()

    # Cargar vídeo con OpenCV
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print(f"Error leyendo FPS en {video_name}")
        continue

    frame_interval = int(fps * interval)
    frame_counter = 0
    extracted_counter = 0

    print(f"\nProcesando vídeo {video_index}: {video_name}")
    print(f"Duración: {duration_seconds:.2f}s → Intervalo: {interval}s")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Guardar cada N frames
        if frame_counter % frame_interval == 0:
            extracted_counter += 1
            filename = f"{extracted_counter:07d}_{video_index}.jpg"
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            cv2.imwrite(output_path, frame)

        frame_counter += 1

    cap.release()

print("\nProceso completado.")

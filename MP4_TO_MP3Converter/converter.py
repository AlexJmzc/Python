# Código que convierte un archivo MP4 a MP3

# Instalar el convertidor con el comando -> pip install moviepy

from moviepy.editor import VideoFileClip

def audio(mp4_path, mp3_path):
    
    video = VideoFileClip(mp4_path)
    audio = video.audio
    audio.write_audiofile(mp3_path)

mp4_path = 'video.mp4'

mp3_path = 'audio.mp3'

audio(mp4_path, mp3_path)



from moviepy.editor import VideoFileClip

video_path = r"C:\Users\Bruker\Desktop\nb-whisper-demo\video\dovregubben_video.mp4"
audio_path = r"C:\Users\Bruker\Desktop\nb-whisper-demo\video_audio\dovregubben_audio.mp3"

video = VideoFileClip(video_path)
audio = video.audio
audio.write_audiofile(audio_path)

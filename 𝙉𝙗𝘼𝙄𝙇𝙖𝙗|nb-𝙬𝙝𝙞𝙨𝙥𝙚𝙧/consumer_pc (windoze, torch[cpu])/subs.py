import os
import torch
from transformers import WhisperTokenizer, WhisperForConditionalGeneration, WhisperProcessor #pipeline

# Make project directory
project_dir = r'C:\Users\Bruker\Desktop\nb-whisper-demo'
os.makedirs(project_dir, exist_ok=True)


# Make dir/interview samples
audio_dir = os.path.join(project_dir, 'interview_audio')
os.makedirs(audio_dir, exist_ok=True)

# Device
device = torch.device('cpu')
#torch_dtype = torch.float32

# Tokenizer/model
tokenizer = WhisperTokenizer.from_pretrained("NbAiLabBeta/nb-whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained(
    "NbAiLabBeta/nb-whisper-medium")
processor = WhisperProcessor.from_pretrained("NbAiLabBeta/nb-whisper-medium")

#video_audio
audio_file = "dovregubben_audio.mp3"
audio_path = os.path.join(audio_dir, audio_file)


# Create subs
with torch.no_grad():
    inputs = processor(audio_path, return_tensors="pt").to(device)
    generated_ids = model.generate(inputs["input_features"]) #num_beams=1, max_length=448
    subtitles = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Create directory
output_dir = os.path.join(project_dir, 'video_subtitles')
os.makedirs(output_dir, exist_ok=True)

# Save subtitles to file
output_file = os.path.splitext(audio_file)[0] + '.srt'
output_path = os.path.join(output_dir, output_file)
with open(output_path, 'w') as f:
    f.write(subtitles)



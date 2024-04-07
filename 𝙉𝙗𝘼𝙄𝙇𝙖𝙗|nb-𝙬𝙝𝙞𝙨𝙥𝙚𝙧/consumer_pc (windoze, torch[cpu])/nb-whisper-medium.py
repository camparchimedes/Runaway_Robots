import os
import torch
from transformers import pipeline, WhisperTokenizer, WhisperForConditionalGeneration, WhisperProcessor

# Make project directory
project_dir = r'C:\Users\Bruker\Desktop\nb-whisper-demo'
os.makedirs(project_dir, exist_ok=True)


# Make dir/interview samples
audio_dir = os.path.join(project_dir, 'interview_audio')
os.makedirs(audio_dir, exist_ok=True)

# Device
device = torch.device('cpu')
torch_dtype = torch.float32

# Tokenizer/model
tokenizer = WhisperTokenizer.from_pretrained("NbAiLabBeta/nb-whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("NbAiLabBeta/nb-whisper-medium")
processor = WhisperProcessor.from_pretrained("NbAiLabBeta/nb-whisper-medium")

# Generation kwargs
generate_kwargs = {
    "task": "transcribe",
    "language": "no"
}

# Initialize pipeline
asr = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, device=device, torch_dtype=torch_dtype)

# Input specify/audio file
audio_file = "king.mp3"

# ASR pipeline on audio
with torch.no_grad():
    output = asr(os.path.join(audio_dir, audio_file), chunk_length_s=28, generate_kwargs=generate_kwargs)

# Dir for output
output_dir = r'C:\Users\Bruker\Desktop\nb-whisper-demo\interview_text'
os.makedirs(output_dir, exist_ok=True)

# Create output file name from file name
output_file = os.path.splitext(audio_file)[0] + '.txt'

# Save to dir
output_path = os.path.join(output_dir, output_file)
with open(output_path, 'w') as f:
    f.write(str(output))

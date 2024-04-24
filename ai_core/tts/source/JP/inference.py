import locale
import time
from ai_core.tts.source.VI.inference import OUTPUT_DIR
from pydub import AudioSegment
import numpy as np
from helpers.common import split_paragraphs
locale.getpreferredencoding = lambda: "UTF-8"
from transformers import BarkModel, AutoProcessor
import torch
import soundfile as sf
model = BarkModel.from_pretrained("suno/bark") # Can be suno/bark
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

processor = AutoProcessor.from_pretrained("suno/bark")

def speak_JP(text: str, vocal = "male", speed = 1):
    text = text.replace("。", ". ")
    print(text)
    if vocal == "male":
        voice_preset = "v2/ja_speaker_2"
    elif vocal == "female":
        voice_preset = "v2/ja_speaker_7"
    k = 0
    paragraphs = text.split(".")
    for paragraph in paragraphs:
        k += 1
        output_file = OUTPUT_DIR + '/' + str(time.time()) + ".wav"
        inputs = processor(paragraph, voice_preset=voice_preset)
        speech_output = model.generate(**inputs.to(device))[0].cpu().numpy()
        if k == len(paragraphs): 
                silence = AudioSegment.silent(duration=2000)
                silence_array = np.array(silence.get_array_of_samples())
                if len(speech_output.shape) > 1:
                    speech_output = speech_output[:, 0] 
                if len(silence_array.shape) > 1:
                    silence_array = silence_array[:, 0]
                speech_output = np.concatenate([speech_output, silence_array])
        sf.write(output_file, speech_output, int(model.generation_config.sample_rate * speed))



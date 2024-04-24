from .EN.inference import OUTPUT_DIR, speak_EN
from .VI.inference import speak_VI
from .JP.inference import speak_JP

def speak(text, speed = 1, language = "VI", vocal = "female"):
    if language == "VI":
        speak_VI(text, speed=speed, vocal=vocal)
    elif language == "EN":
        speak_EN(text, speed, vocal)
    else:
        speak_JP(text, vocal=vocal, speed=speed)

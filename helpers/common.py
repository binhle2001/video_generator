
from configparser import ConfigParser
import os
import re

def split_paragraphs(text):
    # Thay thế dấu chấm liên tiếp bằng một dấu chấm duy nhất
    text = re.sub(r'\.+', '.', text)
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    if len(sentences) <= 1:
        return sentences

    sentence = ""
    paragraphs = []
    i = 0
    while i < len(sentences):
        if len(sentence.split()) < 5:
            sentence = " ".join([sentence, sentences[i]]).strip()
        else:
            paragraphs.append(sentence)
            sentence = sentences[i]
        i += 1
        print(sentence)
    if len(sentence) != 0:
        if len(sentence.split()) < 5:
            last_sentence = " ".join([paragraphs[-1], sentence]).strip()
            paragraphs = paragraphs[:-1]
            paragraphs.append(last_sentence)
        else: 
            paragraphs.append(sentence)
    return paragraphs

                


def get_env_var(group, var_name): 
    config = ConfigParser()
    file_path = ".env"
    if os.path.exists(file_path):
        config.read(file_path)
        return  config[group][var_name]
    return os.environ.get(var_name)
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import os
import subprocess

os.environ["SUNO_OFFLOAD_CPU"] = "False"
os.environ["SUNO_USE_SMALL_MODELS"] = "False"

# download and load all bark models
preload_models()

def bark_male(prompt: str, filename_out: str) -> None:
    text_prompt = prompt
    # generate audio from text
    audio_array = generate_audio(text_prompt, history_prompt="v2/de_speaker_4", text_temp=0.1, waveform_temp=0.7)
    # save audio to disk
    write_wav(filename_out, SAMPLE_RATE, audio_array)

def bark_female(prompt: str, filename_out: str) -> None:
    text_prompt = prompt
    # generate audio from text
    audio_array = generate_audio(text_prompt, history_prompt="v2/de_speaker_3", text_temp=0.1)
    # save audio to disk
    write_wav(filename_out, SAMPLE_RATE, audio_array)

def whisper_transcribe(filename_in: str, work_dir: str) -> str:
    # whisper --model large-v2 --output_format all --task transcribe --language de --device cuda --output_dir . Week10-2.mp3
    whisper = ["whisper", 
        "--model", "large-v2", 
        "--output_format", "txt", 
        "--task", "transcribe", 
        "--language", "de", 
        "--device", "cuda", 
        "--output_dir", work_dir, 
        filename_in ]
    subprocess.Popen(whisper).wait()
    transcribed_file = filename_in.split(os.sep)[-1][:-4] + ".txt"
    transcribed_file_full = os.path.join(work_dir, transcribed_file)
    prompt = ""
    with open(transcribed_file_full, "r", encoding="utf-8") as f:
        for line in f.readlines():
            prompt += " " + line.strip()
    os.remove(transcribed_file_full)
    return prompt
        
input_dir = "template"
output_dir = "out"
start_dir = "start"
middle_dir = "middle"
end_dir = "end"

whisper_work_dir = "whisper"

dirs = [start_dir, middle_dir, end_dir]

for dir in dirs:
    files = os.listdir(os.path.join(input_dir, dir))
    for file in files:
        print("file: " + file)
        content = whisper_transcribe(os.path.join(input_dir, dir, file), whisper_work_dir) 
        print("whisper transcribed: " + content)
        m_name_out = "m_" + "tts_" + file[8:-4] + ".wav"
        f_name_out = "w_" + "tts_" + file[8:-4] + ".wav"
        bark_out_dir = os.path.join(output_dir, dir)
        if not os.path.exists(bark_out_dir):
            os.makedirs(bark_out_dir)
        print("bark_male ...")
        m_full_out = os.path.join(bark_out_dir, m_name_out)
        bark_male(content, m_full_out)
        print("bark_female ...")
        f_full_out = os.path.join(bark_out_dir, f_name_out)
        bark_female(content, f_full_out)
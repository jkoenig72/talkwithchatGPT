import pyaudio
import wave
import whisper
import io
import time
import numpy as np
import tempfile
import os
import wave
import webrtcvad
import collections
from gtts import gTTS
import pygame
import openai


# const
seconds_of_silence = 3
vad_aggressiveness = 1
openai_key = "xyz"
chat_log = [] # To store the conversation history

# Initialize Whisper model
model = whisper.load_model("medium")

openai.api_key = openai_key

# Function to process audio with Whisper
def process_with_whisper(audio_data, sample_rate=16000, channels=1, sample_width=2):
    # Write audio data to a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
        wf = wave.open(temp_audio_file, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
        wf.close()
        temp_audio_file_path = temp_audio_file.name

    # Process the audio file with Whisper
    result = model.transcribe(temp_audio_file_path, language="de")

    # Delete the temporary file
    os.remove(temp_audio_file_path)

    return result["text"]

def speak_text(text):
    tts = gTTS(text, lang='de')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        tts.save(temp_file.name)
        temp_file_path = temp_file.name

    # Initialize pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load(temp_file_path)
    pygame.mixer.music.play()

    # Wait for the music to play
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
        time.sleep(1)
    
    # Properly quit the mixer
    pygame.mixer.quit()
    
    try: 
      os.remove(temp_file_path)
    except PermissionError as e:
      print(f"Error deleting temporary file: {e}")
    except Exception as e:
      print(f"Unexpected error: {e}")

def chat_with_openai(message, chat_log=None):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Replace with your preferred model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ] + ([] if chat_log is None else chat_log)
    )
    return response
            

# Initialize VAD
vad = webrtcvad.Vad(vad_aggressiveness)  # The argument is the aggressiveness from 0 to 3

# Initialize PyAudio
p = pyaudio.PyAudio()

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 320  

# Open stream
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Listening...")
speak_text("OK, Ich wäre soweit. Worüber möchtest Du reden?")    

# Buffer to hold recent audio frames
buffer = collections.deque(maxlen=5)  # Buffer for 5 seconds of audio
audio_data = []  # Store the continuous audio data
last_speech_time = time.time()

try:
    while True:
        frame = stream.read(CHUNK)
        is_speech = vad.is_speech(frame, RATE)

        if is_speech:
            last_speech_time = time.time()
            audio_data.append(frame)
        else:
            current_time = time.time()
            if current_time - last_speech_time > seconds_of_silence:  # 5 seconds of silence
                if audio_data:
                    print("Processing audio...")
                    # Combine frames
                    combined_audio = b''.join(audio_data)
                    # Process with Whisper
                    text = process_with_whisper(combined_audio)
                    print("Transcribed Text, wait 10 sec., then read it:", text)
                    
                    if text.lower() == "neue konversation":
                        chat_log = []  # Reset conversation history
                        print("Starting a new conversation.")
                        speak_text("OK, verstanden. Wir starten eine neue Konversation. Den alten Verlauf habe ich gelöscht.")    

                    response = chat_with_openai(text, chat_log)
                    ai_text = response.choices[0].message['content']
                    print("AI:", ai_text)
                    chat_log.append({"role": "user", "content": text})
                    chat_log.append({"role": "assistant", "content": ai_text})

                    print("GPT Response:", ai_text)
                    speak_text(ai_text)  # Add this line to speak the transcribed text
                    audio_data = []  # Clear the audio data for the next speech

            else:
                buffer.append(frame)

except KeyboardInterrupt:
    print("Stopping...")

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()


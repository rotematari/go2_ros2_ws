# import whisper
import pyaudio
import wave
import os
import time
from pynput import keyboard
# ANSI color codes for prettier output
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def on_press(key):
    global start_recording
    try:
        if key == keyboard.Key.enter:
            start_recording = True
            return False  # Stop the listener
    except AttributeError:
        pass

def record_audio(output_filename, record_seconds=1, sample_rate=44100, chunk_size=1024):
    global start_recording
    start_recording = False


    print(f"{BLUE}Press Enter to start recording...{RESET}")
    input()  # Wait for Enter key press - simpler than using pynput
    
    # print(f"{YELLOW}Say your command in...{RESET}")
    # for i in range(3, 0, -1):
    #     print(f"{YELLOW}{i}{RESET}")
    #     time.sleep(1)

    # Open stream
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    print(f"{GREEN}Listening...{RESET}")

    frames = []

    for _ in range(0, int(sample_rate / chunk_size * record_seconds)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Turning your command into text...")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a WAV file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

def speech_to_text(model, record_seconds=6):
    # Example usage
    record_audio("output.wav", record_seconds=record_seconds)

    # Load and transcribe the audio file
    
    result = model.transcribe("output.wav")

    text = result["text"]
    # Remove the WAV file after transcription
    os.remove("output.wav")
    return text

import pyaudio
import wave
import numpy as np
import time

# Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 4096
SILENCE_THRESHOLD = 500  # Adjust this threshold based on your environment
SILENCE_DURATION = 1  # Duration of silence to consider speech ended (in seconds)
RECORD_SECONDS = 10  # Maximum duration to record (in seconds)
DEVICE_INDEX = 4

WAVE_OUTPUT_FILENAME = "speech_output.wav"

p = pyaudio.PyAudio()

# Function to check if the sound level is above a threshold
def is_speaking(data):
    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.int32)
    rms = np.sqrt(np.mean(np.square(audio_data)))
    print(f"RMS: {rms:.2f}")
    return rms > SILENCE_THRESHOLD

# Open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=DEVICE_INDEX,
                frames_per_buffer=CHUNK)

print("Listening for speech...")

frames = []
recording = False
silence_start_time = None

while True:
    data = stream.read(CHUNK, exception_on_overflow=False)

    if is_speaking(data):
        if not recording:
            print("Speech detected, starting recording...")
            recording = True
            silence_start_time = None
        frames.append(data)
    elif recording:
        if silence_start_time is None:
            silence_start_time = time.time()
        elif time.time() - silence_start_time > SILENCE_DURATION:
            print("Silence detected, stopping recording...")
            break
    if recording and len(frames) >= int(RATE / CHUNK * RECORD_SECONDS):
        print("Maximum recording duration reached, stopping recording...")
        break

print("Done recording.")

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded audio to a file
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

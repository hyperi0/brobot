import pyaudio
import wave
import numpy as np
import time

class Listener():
    # Configuration
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 4096
    SILENCE_THRESHOLD = 1000
    SILENCE_DURATION = 1
    RECORD_SECONDS = 10
    DEVICE_INDEX = 4
    WAVE_OUTPUT_LOCATION = "audio_files/"

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.n_recordings = 0

    def start(self):
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=self.DEVICE_INDEX,
            frames_per_buffer=self.CHUNK
        )

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def listen(self):
        frames = []
        recording = False
        silence_start_time = None
        print("Listening...")
        while True:
            data = self.stream.read(4096, exception_on_overflow=False)

            if self.is_speaking(data):
                if not recording:
                    recording = True
                    silence_start_time = None
                    print("Speech detected, starting recording...")
                frames.append(data)
            elif recording:
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > self.SILENCE_DURATION:
                    print("Silence detected, stopping recording...")
                    break
        return frames
        
    def record(self, frames):
        # Save the recorded audio to a file
        wf = wave.open(f'{self.WAVE_OUTPUT_LOCATION}speech_{self.n_recordings}.wav', 'wb')
        self.n_recordings += 1
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    def is_speaking(self, data):
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.int32)
        rms = np.sqrt(np.mean(np.square(audio_data)))
        return rms > self.SILENCE_THRESHOLD
    
if __name__ == "__main__":
    listener = Listener()
    listener.start()
    frames = listener.listen()
    listener.record(frames)
    listener.stop()
    print("Done recording.")
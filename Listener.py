import pyaudio
import wave
import numpy as np
import time
from transformers import pipeline
import redis

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
        filename = f'{self.WAVE_OUTPUT_LOCATION}speech_{self.n_recordings}.wav'
        wf = wave.open(filename, 'wb')
        self.n_recordings += 1
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        return filename

    def is_speaking(self, data):
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.int32)
        rms = np.sqrt(np.mean(np.square(audio_data)))
        return rms > self.SILENCE_THRESHOLD


class SentimentAnalyzer():
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis")
        self.speech_recognizer = pipeline('automatic-speech-recognition')

    def analyze_sentiment(self, audio_file):
        text = self.speech_recognizer(audio_file)['text']
        if text == "":
            return None
        sentiment = self.classifier(text)
        score = sentiment[0]['score']
        if sentiment[0]['label'] == 'NEGATIVE':
            score *= -1
        print(f'Heard "{text}" with sentiment score {score}')
        return score
    

if __name__ == "__main__":
    listener = Listener()
    listener.start()
    analyzer = SentimentAnalyzer()
    r = redis.Redis(host='localhost', port=6379, db=0)

    try:
        while True:
            frames = listener.listen()
            audio_file = listener.record(frames)
            sentiment_score = analyzer.analyze_sentiment(audio_file)
            if sentiment_score is not None:
                r.rpush('sentiment_scores', sentiment_score)
    except KeyboardInterrupt:
        listener.stop()
        print("Listener stopped.")
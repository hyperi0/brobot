from transformers import pipeline

class SentimentAnalyzer():
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis")
        self.speech_recognizer = pipeline('automatic-speech-recognition')

    def analyze_sentiment(self, audio_file):
        text = self.speech_recognizer(audio_file)['text']
        sentiment = self.classifier(text)
        score = sentiment[0]['score']
        if sentiment[0]['label'] == 'NEGATIVE':
            score *= -1
        return score
    
if __name__ == "__main__":
    sa = SentimentAnalyzer()
    print(sa.analyze_sentiment("audio_files/speech_0.wav"))
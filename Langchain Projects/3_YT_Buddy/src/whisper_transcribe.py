from pytube import YouTube
from haystack.nodes.audio import WhisperTranscriber

class TranscriptGeneration:
    def __init__(self):
        self.whisper = WhisperTranscriber()

    def ytvideo2audio (self, url):
        yt = YouTube(url)
        video = yt.streams.filter(abr='160kbps').last()
        return video.download()  
    
    def gen_transcript(self, video_url):
        aud_path = self.ytvideo2audio(video_url)
        transcription = self.whisper.transcribe(aud_path)
        with open("transcription.txt", "w+") as f:
            f.write(transcription["text"])
        f.close()
        return transcription["text"]
    
if __name__ == "__main__":
    transcript_generator = TranscriptGeneration()
    yt_url = input("Enter YT URL: ")
    transcription = transcript_generator.gen_transcript(yt_url)
    print(transcription)
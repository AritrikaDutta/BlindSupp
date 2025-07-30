import os
import time
import subprocess
from gtts import gTTS
from pydub import AudioSegment

class VoiceAlertManager:
    def __init__(self, temp_audio_dir="voice_cache", export_wav_path="C:/temp/voice.wav",
                 ffplay_path="C:/Users/HP/ffmpeg-7.1.1-essentials_build/ffmpeg-7.1.1-essentials_build/bin/ffplay.exe"):
        self.last_state = None   # "Move" or "Stop"
        self.last_time = 0       # last time a message was spoken
        self.cooldown = 8        # repeat after 8 seconds

        # Allow custom temp dir & export path
        self.audio_folder = temp_audio_dir
        self.export_wav_path = export_wav_path
        os.makedirs(self.audio_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self.export_wav_path), exist_ok=True)

        # Path to ffplay executable
        self.ffplay_path = ffplay_path

    def generate_audio(self, message):
        """Generate and cache audio for the given message."""
        filepath = os.path.join(self.audio_folder, f"{message}.mp3")
        if not os.path.exists(filepath):
            try:
                tts = gTTS(text=message, lang='en')
                tts.save(filepath)
                print(f"[VoiceInfo] Generated new audio: {filepath}")
            except Exception as e:
                print(f"[VoiceError] Failed to generate audio for '{message}': {e}")
        return filepath

    def speak(self, message):
        """Play the cached or newly generated audio message."""
        mp3_path = self.generate_audio(message)
        try:
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(self.export_wav_path, format="wav")

            subprocess.run(
                [self.ffplay_path, "-nodisp", "-autoexit", self.export_wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"[VoiceDebug] Played message: {message}")
        except Exception as e:
            print(f"[VoiceError] Failed to play audio: {e}")

    def update_and_speak(self, is_safe, timestamp):
        """Decide whether to play 'Move' or 'Stop' based on state changes."""
        now = time.time()
        label = "Move" if is_safe else "Stop"
        print(f"[VoiceDebug] is_safe={is_safe}, label={label}, now={timestamp:.2f}s")

        # Speak if state changes or cooldown exceeded
        if self.last_state != label or now - self.last_time > self.cooldown:
            self.speak(label)
            self.last_time = now
            self.last_state = label

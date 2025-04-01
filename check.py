import os
import wave

directory = "/home/ubuntu/kit_wavs"
invalid_files = []

for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        filepath = os.path.join(directory, filename)
        try:
            with wave.open(filepath, 'rb') as wf:
                wf.readframes(wf.getnframes())  # Try reading all frames
        except Exception as e:
            print(f"Invalid WAV: {filename} - {e}")
            invalid_files.append(filename)

print(f"\nTotal invalid files: {len(invalid_files)}")

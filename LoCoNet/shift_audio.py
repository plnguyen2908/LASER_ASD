import argparse
import pandas as pd
import os
import numpy as np
import random
from pydub import AudioSegment

parser = argparse.ArgumentParser()
parser.add_argument("--second", type = float, default=0, required=True)
args = parser.parse_args()

df = pd.read_csv("ASW_Dataset/csv/test_orig.csv")
loader = loader = open(f"ASW_Dataset/csv/test_loader.csv").read().splitlines()

# Grouping by the 'instance_id' column
grouped = df.groupby('entity_id')
cnt = 0
# You can then iterate over the groups or perform operations on them
shift_ms = args.second * 1000

ids_to_remove = set()

for instance_id, group in grouped:
    image_data = f"ASW_Dataset/clips_videos/test/{group['video_id'].iloc[0]}/{instance_id}"
    audio_data = f"ASW_Dataset/clips_audios/test/{group['video_id'].iloc[0]}/{instance_id}.wav"
    audio_save_path = f"ASW_Dataset/clips_audios_shifted_{args.second}s/test_shift_1/{group['video_id'].iloc[0]}"

    audio = AudioSegment.from_file(audio_data)


    if audio.duration_seconds <= args.second:
        ids_to_remove.add(instance_id)

        identityPos = -1
            
        # print(identity)
        for index, target_video in enumerate(loader):
            data = target_video.split('\t')
            target_entity = data[0]
            if target_entity == instance_id:
                identityPos = index
                break
        assert identityPos != -1

        loader.pop(identityPos)
        continue


    os.makedirs(audio_save_path, exist_ok=True)

    audio = AudioSegment.from_wav(audio_data)
    
    if shift_ms > 0:
        # Delay the audio by adding silence at the start
        silence = AudioSegment.silent(duration=shift_ms)
        shifted_audio = silence + audio
        shifted_audio = shifted_audio[:len(audio)]  # Trim to match original length

        shifted_audio.export(audio_save_path + f"/{instance_id}.wav", format="wav")
        cnt += 1

df = df[~df['entity_id'].isin(ids_to_remove)]
df.loc[:, 'label'] = "NOT_SPEAKING"
assert (df['label'] == "NOT_SPEAKING").all()
assert (len(df.groupby('entity_id')) == len(loader))

with open(f"ASW_Dataset/csv/test_loader_shifted_{args.second}s.csv", 'w') as f:
        for line in loader:
            f.write(line + "\n")

csv_save_path = f"ASW_Dataset/csv/test_orig_shifted_{args.second}s.csv"
df.to_csv(csv_save_path)
    
print(cnt)
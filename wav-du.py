#!/usr/bin/env python

import pandas as pd
import librosa as R

from pathlib import Path
from tqdm import tqdm
from typing import Optional

def wavdu(wav_dir: str, sample_rate: Optional[int] = 16000, workers: int = 4):
    wav_dir = Path(wav_dir)

    durations = []
    waves = [ str(p) for p in wav_dir.rglob('*.wav')]


    for wavpath in tqdm(waves):
        duration = 0.
        try:
            wav, sr = R.load(wavpath, sr=sample_rate)
            duration = len(wav) / sr
        except Exception as e:
            print(wavpath, e)
        finally:
            durations.append(duration)

    dataset = pd.DataFrame(data=dict(fpath=waves, duration=durations))
    print(dataset.describe())
    dataset.to_csv(wav_dir.with_name(f'{wav_dir.stem}-duration.csv'), index=False, header=False)


# ./calculate-durations --root ./egs/gao-luo/gaoxiaosong
if __name__ == '__main__':
    from fire import Fire
    Fire(wavdu)

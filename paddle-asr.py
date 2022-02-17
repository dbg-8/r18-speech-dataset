import pandas as pd
from multiprocessing import Pool, set_start_method
from pathlib import Path
from fire import Fire
from tqdm import tqdm
from paddle_speech import ASRExecutor
from paddle_text import TextExecutor


def main_init(func):  
    asr = ASRExecutor()
    asr.xinit()
    punc = TextExecutor()
    punc.xinit()
    func.asr, func.punc = asr, punc
    print('subprocess inited')    


def doasr(wpath):
    asr, punc = doasr.asr, doasr.punc
    txt = asr.xinfer(wpath)
    if txt:
        txt = punc.xinfer(txt)
    return txt


def main(wav_dir: str, text_file: str = None):
    wav_dir = Path(wav_dir)
    text_file = Path(text_file) if text_file else wav_dir.with_name(
        f'{wav_dir.name}.csv'
    )

    asr = ASRExecutor()
    asr.xinit()
    punc = TextExecutor()
    punc.xinit()

    suffixes = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.wma', '.ogg']

    wavs = [
        p for p in wav_dir.rglob('*.*') if p.suffix.lower() in suffixes
    ]

    text = []
    for wav in tqdm(wavs):
        txt = ''
        try:
            txt = asr.xinfer(wav)
            if txt:
                txt = punc.xinfer(txt)
        except Exception as e:
            print(wav, e)
        finally:
            text.append(txt)
    dataset = pd.DataFrame(data = dict(fpath = wavs, text = text))
    dataset.to_csv(text_file, index = False, header = False)


if __name__ == '__main__':
    Fire(main)

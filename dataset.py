from torch.utils.data import Dataset
import os
import numpy as np
import torch
import glob
import torchaudio
import torchaudio.transforms as T

RESAMPLE_RATE = 44100
AUDIO_DURATION = 7
SAMPLES = RESAMPLE_RATE*AUDIO_DURATION

class AudioDataset(Dataset):
    def __init__(self, src, repro = False):
        self.src = src
        self.repro = repro
        self.filelist = self.get_files()
    
    def get_files(self):
        r"""Return a list of filepaths to evaluate PAM on. User implemented."""
        raise NotImplementedError

    def __getitem__(self, index):
        r"""Retrieve audio file and return processed audio."""
        file =  self.filelist[index]
        audio = self.readaudio(file)
        return file, audio
    
    def process_audio(self, audio):
        r"""Process audio to be a multiple of 7 seconds."""
        audio = audio.reshape(-1)
        if SAMPLES >= audio.shape[0]:
            repeat_factor = int(np.ceil((SAMPLES) /
                                        audio.shape[0]))
            # Repeat audio_time_series by repeat_factor to match audio_duration
            audio = audio.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio = audio[0:SAMPLES]
        else:
            if self.repro:
                # retain only first 7 seconds
                start_index = 0
                audio = audio[start_index:start_index + SAMPLES]
            else: 
                cutoff = int(np.floor(audio.shape[0]/SAMPLES))
                # cutoff audio
                initial_audio_series = audio[0:cutoff*SAMPLES]
                # remaining audio repeat and cut off
                remaining = audio[cutoff*SAMPLES:]
                if remaining.shape[0] != 0:
                    remaining = audio[-SAMPLES:]
                    audio = torch.cat([initial_audio_series, remaining])
                else:
                    audio = initial_audio_series

        return audio
    
    def readaudio(self, file):
        r"""Loads audio file and returns raw audio."""
        audio, sample_rate = torchaudio.load(file)
        
        # Resample audio if needed
        if RESAMPLE_RATE != sample_rate:
            resampler = T.Resample(sample_rate, RESAMPLE_RATE)
            audio = resampler(audio)
        
        # process audio to be a multiple of 7 seconds
        audio = self.process_audio(audio)
        return audio
    
    def collate(self, batch):
        r"""Collate batch and generate chunk pointers."""
        # Assign a reference variable to identify the file associated with each chunk
        files = [x[0] for x in batch]
        sample_len = [0] + [int(len(x[1])/SAMPLES) for x in batch]
        sample_index = [sum(sample_len[0:i+1]) for i in range(len(sample_len))]
        
        # Create chunks
        batch = torch.cat([x[1] for x in batch])
        batch_chunks = [batch[SAMPLES*i:SAMPLES*i+SAMPLES].reshape(1,-1) for i in range(0,int(batch.shape[0]/SAMPLES))]
        batch_chunks = torch.cat(batch_chunks,axis=0)

        return files, batch_chunks, sample_index

    def __len__(self):
        r"""Size of dataset."""
        return len(self.filelist)

class ExampleDatasetFolder(AudioDataset):
    def __init__(self, src, repro = False):
        self.src = src
        self.repro = repro
        self.filelist = self.get_files()
        super().__init__(src,repro)
    
    def get_files(self):
        return glob.glob(os.path.join(self.src,"**.wav"))
    
class ExampleDatasetFiles(AudioDataset):
    def __init__(self, src, repro = False):
        self.src = src
        self.repro = repro
        super().__init__(src,repro)
    
    def get_files(self):
        return self.src

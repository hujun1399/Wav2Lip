from glob import glob
from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from hparams import hparams, get_image_list

import audio
import cv2


class MemoryDataHandler(object):
    def __init__(self, data_root, dataset_type):
        self.all_videos = get_image_list(data_root, dataset_type)
        self.all_images = self._load_image_in_memory()
        self.all_audios_mel = self._load_audio_in_memory()

    def get_image(self, file_name):
        """
            file_name: image file path
        """
        if file_name not in self.all_images:
            return None
        return self.all_images[file_name]

    def get_audio_mel(self, vidname):
        if vidname not in self.all_audios_mel:
            return None
        return self.all_audios_mel[vidname]

    def _load_image_in_memory(self):
        all_images = {}
        print("[MemoryDataHandler] start to load image into memory...")
        for idx in tqdm(range(len(self.all_videos))):
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            for img_name in img_names:
                img = cv2.imread(img_name)
                if img is not None:
                    all_images[img_name] = img
        return all_images

    def _load_audio_in_memory(self):
        all_audios_mel = {}
        print("[MemoryDataHandler] start to load audio into memory...")
        for idx in tqdm(range(len(self.all_videos))):
            vidname = self.all_videos[idx]
            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
                all_audios_mel[vidname] = orig_mel
            except Exception as e:
                continue
        return all_audios_mel

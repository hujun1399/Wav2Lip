
import argparse
import audio
import copy
import cv2
import os
import subprocess
import torch
import time
import platform

import numpy as np

from tqdm import tqdm
from models import Wav2Lip
from infer_charactor_builder import InferCharactorBuilder

parser = argparse.ArgumentParser(
    description='Inference server using Wav2Lip models')
args = parser.parse_args()

args.img_size = 96
args.static = False
args.fps = 25
args.checkpoint_path = "./checkpoints/wav2lip.pth"
args.wav2lip_batch_size = 32


from gfpgan import GFPGANer
restorer = GFPGANer(
    model_path="/home/james/workspace/GFPGAN/gfpgan/weights/GFPGANv1.3.pth",
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None,
    device='cuda' if torch.cuda.is_available() else 'cpu')

class InferCharactorModel(object):

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} for inference.'.format(self.device))

        self.mel_step_size = 16

        # charactor info index
        self.index_builder = InferCharactorBuilder(identity_list=['kiki', 'guilin'])

        # generator model
        self.model = self.load_model(args.checkpoint_path)

    def load_model(self, path):
        model = Wav2Lip()
        print("Load checkpoint from: {}".format(path))
        checkpoint = self._load(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()

    def _load(self, checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def datagen(self, frames, boxes, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        for i, m in enumerate(mels):
            idx = 0 if args.static else i % len(frames)
            frame = frames[idx].copy()
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (args.img_size, args.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frames[idx])
            coords_batch.append(boxes[idx])

            if len(img_batch) >= args.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(
                    img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, args.img_size//2:] = 0

                img_batch = np.concatenate(
                    (img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(
                    mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def inference(self, identity_name, audio_file=None, video_name='gen'):
        if audio_file is None:
            print("[ERROR] audio_file must not be empty!")
            return
        print("Start to inferece audio_file={}".format(audio_file))

        result = self.index_builder.get_identity_info(identity_name)
        if result is None:
            print("[ERROR] Failed to load charator identity_name={} from local directory.".format(
                identity_name))
            return

        fps = args.fps

        full_frames, coords = result[0], result[1]
        print("Number of frames available for inference: " + str(len(full_frames)))

        if not audio_file.endswith('.wav'):
            print('Extracting raw audio...')
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(
                audio_file, 'temp/temp.wav')

            subprocess.call(command, shell=True)
            audio_file = 'temp/temp.wav'

        wav = audio.load_wav(audio_file, 16000)
        mel = audio.melspectrogram(wav)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80./fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(
                mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        start_time = time.time()

        full_frames = full_frames[:len(mel_chunks)]
        batch_size = args.wav2lip_batch_size
        gen = self.datagen(copy.deepcopy(full_frames), coords, mel_chunks)
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(
                gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
            if i == 0:
                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter('temp/result.avi',
                                      cv2.VideoWriter_fourcc(*'FFV1'), 
                                      fps, 
                                      (frame_w, frame_h), 
                                      isColor=True) # 无损压缩

            img_batch = torch.FloatTensor(np.transpose(
                img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(
                mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)
 
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            for p, f, c in zip(pred, frames, coords):
                # y1, y2, x1, x2 = c
                x1, y1, x2, y2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)

            # pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            # with torch.no_grad():
            #     for p, f, c in zip(pred, frames, coords):
            #         x1, y1, x2, y2 = c
            #         p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            #         # p = cv2.resize(p, (x2 - x1, y2 - y1))
            #         f[y1:y2, x1:x2] = p
            #         cropped_faces, restored_faces, restored_img = restorer.enhance(
            #             f,
            #             has_aligned=False,
            #             only_center_face=True,
            #             paste_back=True,
            #             weight=0.5)

            #         restored_img = cv2.resize(restored_img, (1080,1920))
            #         out.write(restored_img)


        end_time = time.time()
        print("sub-total cost time: ", end_time - start_time)

        out.release()

        # command = 'ffmpeg -y -i {} -i {} -c:v libx264 -crf 18 -preset slow -strict -2 -q:v 1 {}'.format(
        #     audio_file, 'temp/result.avi', os.path.join('./results', video_name))
        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(
            audio_file, 'temp/result.avi', os.path.join('./results', video_name))
        # command = 'ffmpeg -re -i {} -i {} -c:v copy -c:a aac -strict experimental -f flv {}'.format(
        #     'temp/result.avi', audio_file, 'rtmp://192.168.110.212/live/STREAM_NAME'
        # )
        subprocess.call(command, shell=platform.system() != 'Windows')


if __name__ == '__main__':
    model = InferCharactorModel()
    model.inference("kiki", audio_file="./results/kiki_10s.wav",
                    video_name="kiki_out_0717.mp4")

import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

from os import listdir, path

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import audio
from hparams import hparams as hp

import mediapipe as mp
import gc

parser = argparse.ArgumentParser()

parser.add_argument('--worker_num', help='Number of thread to run in parallel', default=1, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)

args = parser.parse_args()


template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'

def process_video_file(vfile, args, thread_id):
	video_stream = cv2.VideoCapture(vfile)
	
	frames = []
	while 1:
		still_reading, frame = video_stream.read()
		if not still_reading:
			video_stream.release()
			break
		frames.append(frame)
	
	vidname = os.path.basename(vfile).split('.')[0]
	dirname = vfile.split('/')[-2]

	fulldir = path.join(args.preprocessed_root, dirname, vidname)
	os.makedirs(fulldir, exist_ok=True)

	face_detection = mp.solutions.face_detection.FaceDetection(
		model_selection=0, min_detection_confidence=0.7)
	
	for idx, frame in enumerate(frames):
		results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		H, W, _ = frame.shape
		if results.detections is None:
			print("[FaceDetection ERORR] no face detected. vidname={}, idx={}".format(vidname, idx))
			continue

		for detection in results.detections:
			if len(detection.score) != 1 or detection.score[0] < 0.7:
				continue
			
			bbox = detection.location_data.relative_bounding_box
			left = int(bbox.xmin * W)
			right = int((bbox.xmin + bbox.width) * W)
			up = int(bbox.ymin * H)
			down = int((bbox.ymin + bbox.height) * H)

			try:
				cv2.imwrite(path.join(fulldir, '{}.jpg'.format(idx)), frame[up:down, left:right])
			except Exception as e:
				break
			break

		if idx % (25 * 60) == 0:
			print("[SubProcess] thread_id={}, idx={}, vfile={}".format(thread_id, idx, vfile))

	face_detection.close()

def process_audio_file(vfile, args):
	vidname = os.path.basename(vfile).split('.')[0]
	dirname = vfile.split('/')[-2]

	fulldir = path.join(args.preprocessed_root, dirname, vidname)
	os.makedirs(fulldir, exist_ok=True)

	wavpath = path.join(fulldir, 'audio.wav')

	command = template.format(vfile, wavpath)
	subprocess.call(command, shell=True)
	
def mp_handler(job):
	vfile, args, thread_id = job
	try:
		process_video_file(vfile, args, thread_id)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()
		
def main(args):
	filelist = glob(path.join(args.data_root, '*/*.mp4'))
	jobs = [(vfile, args, i%args.worker_num) for i, vfile in enumerate(filelist)]
	p = ThreadPoolExecutor(args.worker_num)
	futures = [p.submit(mp_handler, j) for j in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

	print('Dumping audios...')

	for vfile in tqdm(filelist):
		try:
			process_audio_file(vfile, args)
		except KeyboardInterrupt:
			exit(0)
		except:
			traceback.print_exc()
			continue

if __name__ == '__main__':
	main(args)
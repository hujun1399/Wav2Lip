import os
import cv2
import mediapipe
import numpy as np

from glob import glob


class InferCharactorBuilder(object):

    def __init__(self, nosmooth=False, identity_list=None):
        self.nosmooth = nosmooth
        self.data = {}

        # # load kiki as default
        # kiki = self.load_identity_files("kiki")
        # if kiki:
        #     print("Succeed to load kiki identity files!")
        #     self.data['kiki'] = kiki

        if identity_list is not None:
            for name in identity_list:
                result = self.load_identity_files(name)
                if result:
                    print("Succeed to load {} identity files!".format(name))
                    self.data[name] = result

    def get_identity_info(self, name="kiki"):
        if name not in self.data:
            print("Failed to find identity={} in local directory.".format(name))
            return None
        return self.data[name]

    def process_and_save_video_identity(self, video_path, identity_name, manual_height_bias=0):
        # read video stream
        print("Start read video stream and detect face boxes. video_path={}".format(
            video_path))
        video_stream = cv2.VideoCapture(video_path)

        frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            frames.append(frame)

        # detect face
        face_detection = mediapipe.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.7)

        results = []
        images = []
        fulldir = os.path.join('./results', identity_name)
        print("fulldir:", fulldir)
        for idx, frame in enumerate(frames):
            fd_results = face_detection.process(frame)
            H, W, _ = frame.shape
            if fd_results.detections is None:
                print("[FaceDetection ERORR] no face detected. video_path={}, idx={}".format(
                    video_path, idx))
                continue

            manual_height_bias = manual_height_bias
            for detection in fd_results.detections:
                if len(detection.score) != 1 or detection.score[0] < 0.7:
                    continue

                bbox = detection.location_data.relative_bounding_box
                x1 = bbox.xmin * W
                x2 = (bbox.xmin + bbox.width) * W
                y1 = bbox.ymin * H
                y2 = (bbox.ymin + bbox.height) * H
                y2 = min(y2 + manual_height_bias, H)
                results.append([x1, y1, x2, y2])
                # cv2.imwrite(os.path.join("./results/face", '{}.png'.format(idx)), 
                #             frame[int(y1):int(y2), int(x1):int(x2)], 
                #             [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(os.path.join("./results/face", '{}.jpg'.format(idx)), 
                            frame[int(y1):int(y2), int(x1):int(x2)])
                images.append(frame)

        face_detection.close()

        boxes = np.array(results)
        if not self.nosmooth:
            boxes = self._get_smoothened_boxes(boxes, T=5)
        boxes = np.rint(boxes).astype(int)
        # print("boxes shape 2:", boxes.shape)
        # print("image number:", len(images))
        # for idx, frame in enumerate(images):
        #     x1, y1, x2, y2 = boxes[idx]
        #     cv2.imwrite(os.path.join(fulldir, 'smooth_{}.jpg'.format(idx)), images[idx][y1:y2, x1:x2])

        def _save_identity_file(fulldir, images, boxes):
            if len(images) != len(boxes):
                print("ERROR:save identity file failed, for len(images) != len(boxes).")
                return False
            for idx, image in enumerate(images):
                try:
                    cv2.imwrite(os.path.join(
                        fulldir, '{}.jpg'.format(idx)), image)
                        # fulldir, '{}.png'.format(idx)), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                except Exception as e:
                    break
            np.savetxt(os.path.join(fulldir, 'boxes.txt'), boxes)

        # save identity files and face boxes
        fulldir = os.path.join('./results', identity_name)
        os.makedirs(fulldir, exist_ok=True)
        _save_identity_file(fulldir, images, boxes)

        print("Succeed to process video and save result into dir={}".format(fulldir))

        return True

    def load_identity_files(self, identity_name):
        fulldir = os.path.join('./results', identity_name)
        if not os.path.exists(fulldir):
            print("[load_identity_files] fulldir={} do not exits".format(fulldir))
            return False

        img_names = list(glob(os.path.join(fulldir, '*.jpg')))
        if len(img_names) <= 3 * 5:
            return False

        def _read_window(window_fnames):
            if window_fnames is None:
                return None
            window = []
            for img_file in window_fnames:
                img = cv2.imread(img_file)
                if img is None:
                    return None
                img_name = os.path.basename(img_file).split('.')[0]
                window.append([int(img_name), img])
            return window

        frames = _read_window(img_names)
        frames.sort(key=lambda item: item[0], reverse=False)
        boxes = np.loadtxt(os.path.join(fulldir, "boxes.txt")).astype(np.int64)
        if len(boxes) != len(frames):
            print("[ERROR] read identity files failed: len(boxes) != len(frames). ")
            return False

        return [pair[1] for pair in frames], boxes

    def _get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes


if __name__ == '__main__':
    charactor_builder = InferCharactorBuilder()
    # result = charactor_builder.get_identity_info("guilin")
    # print(len(result[0]), result[1].shape)
    # print(result[1][61])
    charactor_builder.process_and_save_video_identity(
        "/home/james/workspace/Wav2Lip/results/kiki_sdr_high.mp4", "kiki", manual_height_bias=10)
    charactor_builder.process_and_save_video_identity(
        "/home/james/workspace/Wav2Lip/results/guilin_20s.mp4", "guilin", manual_height_bias=0)
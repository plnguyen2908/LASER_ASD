import sys
import time
import os
import tqdm
import torch
import argparse
import glob
import subprocess
import warnings
import cv2
import pickle
import numpy
import pdb
import math
import python_speech_features
from pydub import AudioSegment

from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
import soundfile as sf
from scipy.interpolate import interp1d
# from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from loss_multi import lossAV, lossA, lossV
from model.loconet_encoder import locoencoder
from torchvggish import vggish_input, vggish_params, mel_features
from loconet import loconet
import random
import numpy as np

from utils.tools import *
from dlhammer.dlhammer import bootstrap

warnings.filterwarnings("ignore")


def args_create():

    parser = argparse.ArgumentParser(
        description="TalkNet Demo or Columnbia ASD Evaluation")

    parser.add_argument('--videoName',             type=str,
                        default="demo_6",   help='Demo video name')
    parser.add_argument('--audioName',             type=str,
                        default="demo_6",   help='Demo video name')
    parser.add_argument('--videoFolder',           type=str,
                        default="demo",  help='Path for inputs, tmps and outputs')
    parser.add_argument('--pretrainModel',         type=str,
                        default="loconet_ava_best.model",   help='Path for the pretrained LoCoNet model')

    parser.add_argument('--nDataLoaderThread',     type=int,
                        default=10,   help='Number of workers')
    parser.add_argument('--facedetScale',          type=float, default=0.25,
                        help='Scale factor for face detection, the frames will be scale to 0.25 orig')
    parser.add_argument('--minTrack',              type=int,
                        default=20,   help='Number of min frames for each shot')
    parser.add_argument('--numFailedDet',          type=int,   default=12,
                        help='Number of missed detections allowed before tracking is stopped')
    parser.add_argument('--minFaceSize',           type=int,
                        default=1,    help='Minimum face size in pixels')
    parser.add_argument('--cropScale',             type=float,
                        default=0.40, help='Scale bounding box')

    parser.add_argument('--start',                 type=int,
                        default=0,   help='The start time of the video')
    parser.add_argument('--duration',              type=int, default=0,
                        help='The duration of the video, when set as 0, will extract the whole video')

    args = parser.parse_args()

    args.videoPath = glob.glob(os.path.join(
        args.videoFolder, args.videoName + '.*'))[0]
    args.audioPath = glob.glob(os.path.join(
        args.videoFolder, args.audioName + '.*'))[0]
    args.savePath = os.path.join(args.videoFolder, args.videoName)

    return args


def scene_detect(args):
    # CPU: Scene detection, output is the list of each shot's time duration
    videoManager = VideoManager([args.videoFilePath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source=videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    savePath = os.path.join(args.pyworkPath, 'scene.pckl')
    if sceneList == []:
        sceneList = [(videoManager.get_base_timecode(),
                      videoManager.get_current_timecode())]
    with open(savePath, 'wb') as fil:
        pickle.dump(sceneList, fil)
        sys.stderr.write('%s - scenes detected %d\n' %
                         (args.videoFilePath, len(sceneList)))
    return sceneList


def inference_video(args):
    # GPU: Face detection, output is the list contains the face location and score in this frame
    DET = S3FD(device='cuda')
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.9)
        dets.append([])
        for bbox in bboxes:
            # dets has the frames info, bbox info, conf info
            dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]
                                                     ).tolist(), 'conf': bbox[-1]})
        sys.stderr.write('%s-%05d; %d dets\r' %
                         (args.videoFilePath, fidx, len(dets[-1])))
    savePath = os.path.join(args.pyworkPath, 'faces.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(dets, fil)
    return dets


def bb_intersection_over_union(boxA, boxB, evalCol=False):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol == True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def track_shot(args, sceneFaces):
    # CPU: Face tracking
    iouThres = 0.5     # Minimum IOU between consecutive face detections
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(
                        face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            frameNum = numpy.array([f['frame'] for f in track])
            bboxes = numpy.array([numpy.array(f['bbox']) for f in track])
            frameI = numpy.arange(frameNum[0], frameNum[-1]+1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = numpy.stack(bboxesI, axis=1)
            if max(numpy.mean(bboxesI[:, 2]-bboxesI[:, 0]), numpy.mean(bboxesI[:, 3]-bboxesI[:, 1])) > args.minFaceSize:
                tracks.append({'frame': frameI, 'bbox': bboxesI})
    return tracks

# Source: https://github.com/cs-giung/face-detection-pytorch/blob/master/utils/bbox.py


def crop_thumbnail(image, bounding_box, padding=1, size=100):

    # infos in original image
    w, h = image.shape[1], image.shape[0]
    x1, y1, x2, y2 = bounding_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r = max(x2 - x1, y2 - y1) * padding

    # get thumbnail
    p1x = int(cx - r)
    p1y = int(cy - r)
    p2x = int(cx + r)
    p2y = int(cy + r)

    img = image.copy()

    if p1x < 0:
        img = cv2.copyMakeBorder(img, top=0, bottom=0, left=-p1x,
                                 right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        x1 -= p1x
        x2 -= p1x
        p2x -= p1x
        p1x -= p1x
    if p1y < 0:
        img = cv2.copyMakeBorder(img, top=-p1y, bottom=0, left=0,
                                 right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        y1 -= p1y
        y2 -= p1y
        p2y -= p1y
        p1y -= p1y
    if p2x > w:
        img = cv2.copyMakeBorder(img, top=0, bottom=0, left=0, right=p2x-w,
                                 borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    if p2y > h:
        img = cv2.copyMakeBorder(img, top=0, bottom=p2y-h, left=0,
                                 right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    output = img[p1y:p2y, p1x:p2x]
    output = cv2.resize(output, (size, size), interpolation=cv2.INTER_LINEAR)

    # infos in thumbnail
    s_x = size / (p2x - p1x)
    s_y = size / (p2y - p1y)
    new_bbox = [(x1 - p1x) * s_x, (y1 - p1y) * s_y,
                (x2 - p1x) * s_x, (y2 - p1y) * s_y]

    return output, new_bbox


def side_padding(input_tensor, amount, side):
    result = torch.zeros_like(input_tensor)[0]
    result = torch.stack([result for i in range(amount)], dim=0)
    if side == 'left':
        return torch.cat((result, input_tensor), dim=0)
    else:
        return torch.cat((input_tensor, result), dim=0)


def padding(input_tensor, start_input, end_input, start, end):
    result = input_tensor[(max(start_input, start) - start_input):(min(end_input, end) - start_input + 1), :]
    # print(result.shape[0])
    if start_input > start:
        result = side_padding(result, start_input - start, 'left')
    if end_input < end:
        result = side_padding(result, end - end_input, 'right')
    return result


def prepare_input(args, tracks):
    # read in the frames of the video:
    flist = glob.glob(os.path.join(
        args.pyframesPath, '*.jpg'))  # Read the frames
    flist.sort()

    # prepare inputs
    visual_info = [{'frame': [], 'faceCrop': []} for i in range(len(tracks))]

    H = 112

    # loop through each face tracks
    for tidx, track in enumerate(tracks):
        for fidx, frame_num in enumerate(track['frame']):
            # read in the frame that contains the current track
            frameFile = os.path.join(
                args.pyframesPath, f'{frame_num + 1:06}.jpg')
            frame = cv2.imread(frameFile)

            # crop face
            faceCrop, _ = crop_thumbnail(
                frame, track['bbox'][fidx], padding=0.775, size=H)
            faceCrop = cv2.cvtColor(faceCrop, cv2.COLOR_BGR2GRAY)
            faceCrop = torch.from_numpy(faceCrop)

            # store information
            visual_info[tidx]['frame'].append(frame_num)
            visual_info[tidx]['faceCrop'].append(faceCrop)
        visual_info[tidx]['faceCrop'] = torch.stack(
            visual_info[tidx]['faceCrop'], dim=0)

    visual_feature, audio_feature = [], []

    # loop through each tracked identity
    for person_id in range(len(visual_info)):
        # get list of people that have at least half time with person_id
        candidate = []

        for i in range(len(visual_info)):
            if i == person_id:
                continue

            intersect = set(visual_info[i]['frame']).intersection(
                set(visual_info[person_id]['frame']))
            if len(intersect) >= len(visual_info[person_id]['frame']) / 2:
                candidate.append(
                    {'id': i, 'start': visual_info[i]['frame'][0], 'end': visual_info[i]['frame'][-1]})

        visualFeature = None

        # extract visual input
        if len(candidate) == 0:
            visualFeature = torch.stack([visual_info[person_id]['faceCrop'], visual_info[person_id]
                                        ['faceCrop'], visual_info[person_id]['faceCrop']], dim=0)
        elif len(candidate) == 1:
            context = padding(visual_info[candidate[0]['id']]['faceCrop'], candidate[0]['start'], candidate[0]
                              ['end'], visual_info[person_id]['frame'][0], visual_info[person_id]['frame'][-1])
            visualFeature = torch.stack(
                [visual_info[person_id]['faceCrop'], context, visual_info[person_id]['faceCrop']], dim=0)
        else:
            random.shuffle(candidate)
            candidate1 = padding(visual_info[candidate[0]['id']]['faceCrop'], candidate[0]['start'],
                                 candidate[0]['end'], visual_info[person_id]['frame'][0], visual_info[person_id]['frame'][-1])
            candidate2 = padding(visual_info[candidate[-1]['id']]['faceCrop'], candidate[-1]['start'],
                                 candidate[-1]['end'], visual_info[person_id]['frame'][0], visual_info[person_id]['frame'][-1])
            visualFeature = torch.stack(
                [visual_info[person_id]['faceCrop'], candidate1, candidate2], dim=0)

        visual_feature.append(visualFeature.unsqueeze(0))

        # extract audio
        audioTmp = os.path.join(args.pycropPath, f'audio{person_id:06d}.wav')
        audioStart = tracks[person_id]['frame'][0] / 25
        audioEnd = tracks[person_id]['frame'][-1] / 25
        command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" %
                   (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp))
        subprocess.call(command, shell=True, stdout=False)
        sr, wav_data = wavfile.read(audioTmp)
        audioFeature = vggish_input.waveform_to_examples(
            wav_data, sr, visualFeature.shape[1], 25, False)

        # audio feature has shape (B, C, 4*T, M)
        audioFeature = torch.from_numpy(audioFeature).unsqueeze(0).unsqueeze(0)
        audio_feature.append(audioFeature)

    return visual_feature, audio_feature


def inference(args, cfg, visual_feature, audio_feature, lenTracks):
    # initialize model
    model = loconet(cfg)
    model.loadParameters('')
    model = model.to(device='cuda')
    model.eval()

    with torch.no_grad():
        result = [None for i in range(lenTracks)]

        # TODO: make a forward pass to make prediction
        for i in tqdm.tqdm(range(lenTracks)):
            visual = visual_feature[i]

            # get visual feature with size
            visualFeature = visual.to(device='cuda')
            b, s, t = visualFeature.shape[0], visualFeature.shape[1], visualFeature.shape[2]

            # get audio feature
            audioFeature = audio_feature[i].to(dtype=torch.float, device='cuda')

            # run frontend part of the model
            audioEmbed = model.model.forward_audio_frontend(audioFeature)
            visualEmbed = model.model.forward_visual_frontend(
                visualFeature.view(b * s, *visualFeature.shape[2:]))
            audioEmbed = audioEmbed.repeat(s, 1, 1).to(device = 'cuda')
            audioEmbed, visualEmbed = model.model.forward_cross_attention(
                audioEmbed, visualEmbed)
            outsAV = model.model.forward_audio_visual_backend(
                audioEmbed, visualEmbed, b, s)
            outsAV = outsAV.reshape(b, s, t, -1)[:, 0, :, :].reshape(b * t, -1)
            predScore = model.lossAV(outsAV)
            result[i] = predScore
            print(sum(predScore < 0))

    return result


def visualization(args, pred, tracks):
    # get list of frames
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()

    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = pred[tidx]
        for fidx, frame in enumerate(track['frame'].tolist()):
            # s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
            # s = numpy.mean(s)
            faces[frame].append(
                {'track': tidx, 'score': score[fidx], 'bbox': track['bbox'][fidx]})

    # begin writing result video
    firstImage = cv2.imread(flist[0])
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(os.path.join(
        args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw, fh))

    colorDict = {0: 0, 1: 255}
    l = []
    for fidx, frame in tqdm.tqdm(enumerate(flist), total=len(flist)):
        image = cv2.imread(frame)
        for face in faces[fidx]:
            clr = colorDict[int((face['score'] >= 0.0))]
            if face['score'] >= 0:
                l.append(fidx)
            txt = round(face['score'], 2)
            p1 = (int(face['bbox'][0]), int(face['bbox'][1]))
            p2 = (int(face['bbox'][2]), int(face['bbox'][3]))
            cv2.rectangle(image, p1, p2, (0, clr, 255-clr), 3)
            cv2.imwrite(os.path.join(args.pyframesViz, frame.split('/')[-1]), image)
            cv2.putText(image, '%s' % (txt), p1,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, clr, 255-clr), 2)
        vOut.write(image)
    l = list(set(l))
    # print(l)
    vOut.release()
    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" %
               (os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'),
                args.nDataLoaderThread, os.path.join(args.pyaviPath, 'video_out.avi')))
    output = subprocess.call(command, shell=True, stdout=None)

# Main function


def main():
    # This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
    # ```
    # .
    # ├── pyavi
    # │   ├── audio.wav (Audio from input video)
    # │   ├── video.avi (Copy of the input video)
    # │   ├── video_only.avi (Output video without audio)
    # │   └── video_out.avi  (Output video with audio)
    # ├── pycrop (The detected face videos and audios)
    # │   ├── 000000.avi
    # │   ├── 000000.wav
    # │   ├── 000001.avi
    # │   ├── 000001.wav
    # │   └── ...
    # ├── pyframes (All the video frames in this video)
    # │   ├── 000001.jpg
    # │   ├── 000002.jpg
    # │   └── ...
    # └── pywork
    #     ├── faces.pckl (face detection result)
    #     ├── scene.pckl (scene detection result)
    #     ├── scores.pckl (ASD result)
    #     └── tracks.pckl (face tracking result)
    # ```

    # Initialization
    args = args_create()
    args.pyaviPath = os.path.join(args.savePath, 'pyavi')
    args.pyframesPath = os.path.join(args.savePath, 'pyframes')
    args.pyworkPath = os.path.join(args.savePath, 'pywork')
    args.pycropPath = os.path.join(args.savePath, 'pycrop')
    args.pyframesViz = os.path.join(args.savePath, 'pyframesViz')
    if os.path.exists(args.savePath):
        rmtree(args.savePath)
    # The path for the input video, input audio, output video
    os.makedirs(args.pyaviPath, exist_ok=True)
    os.makedirs(args.pyframesPath, exist_ok=True)  # Save all the video frames
    # Save the results in this process by the pckl method
    os.makedirs(args.pyworkPath, exist_ok=True)
    # Save the detected face clips (audio+video) in this process
    os.makedirs(args.pycropPath, exist_ok=True)
    os.makedirs(args.pyframesViz, exist_ok=True)

    # setup configuration
    default_config = {'cfg': './configs/multi.yaml'}
    cfg = bootstrap(default_cfg=default_config, print_cfg=True)

    warnings.filterwarnings("ignore")

    # Extract video
    args.videoFilePath = os.path.join(args.pyaviPath, args.videoName + ".avi")
    # If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
    if args.duration == 0:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" %
                   (args.videoPath, args.nDataLoaderThread, args.videoFilePath))
    else:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" %
                   (args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.videoFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Extract the video and save in %s \r\n" % (args.videoFilePath))
    
    args.videoFilePath1 = os.path.join(args.pyaviPath, args.audioName + ".avi")
    # If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
    if args.duration == 0:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" %
                   (args.audioPath, args.nDataLoaderThread, args.videoFilePath1))
    else:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" %
                   (args.audioPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.videoFilePath1))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Extract the video and save in %s \r\n" % (args.videoFilePath1))

    # Extract audio
    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" %
               (args.videoFilePath1, args.nDataLoaderThread, args.audioFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Extract the audio and save in %s \r\n" % (args.audioFilePath))

    audio = AudioSegment.from_wav(args.audioFilePath)

    shift_ms = 0
    swap = False
    if shift_ms > 0:
        # Delay the audio by adding silence at the start
        silence = AudioSegment.silent(duration=shift_ms)
        shifted_audio = silence + audio
        shifted_audio = shifted_audio[:len(audio)]  # Trim to match original length
        shifted_audio.export(args.audioFilePath, format="wav")
    elif swap:
        l, r = 0, len(audio)
        mid = (l + r) // 2
        shifted_audio = audio[mid:] + audio[:mid]
        shifted_audio.export(args.audioFilePath, format="wav")
    # Extract the video frames
    command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" %
               (args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg')))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Extract the frames and save in %s \r\n" % (args.pyframesPath))

    # Scene detection for the video frames
    scene = scene_detect(args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Scene detection and save in %s \r\n" % (args.pyworkPath))

    # Face detection for the video frames
    faces = inference_video(args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Face detection and save in %s \r\n" % (args.pyworkPath))

    # Face tracking
    allTracks = []
    for shot in scene:
        # Discard the shot frames less than minTrack frames
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack:
            # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
            allTracks.extend(track_shot(
                args, faces[shot[0].frame_num:shot[1].frame_num]))
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Face track and detected %d tracks \r\n" % len(allTracks))

    visual_feature, audio_feature = prepare_input(args, allTracks)

    result = inference(args, cfg, visual_feature,
                       audio_feature, lenTracks=len(allTracks))

    visualization(args, result, allTracks)


if __name__ == '__main__':
    main()

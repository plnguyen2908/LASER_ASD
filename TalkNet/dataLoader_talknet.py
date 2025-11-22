import os, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile
from torchvision.transforms import RandomCrop
import json

def generate_audio_set(dataPath, batchList):
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        videoName = data[0][:11]
        dataName = data[0]
        _, audio = wavfile.read(os.path.join(dataPath, videoName, dataName + '.wav'))
        audioSet[dataName] = audio
    return audioSet

def overlap(dataName, audio, audioSet):   
    noiseName =  random.sample(set(list(audioSet.keys())) - {dataName}, 1)[0]
    noiseAudio = audioSet[noiseName]    
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = numpy.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * numpy.log10(numpy.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * numpy.log10(numpy.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = numpy.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio    
    return audio.astype(numpy.int16)

def load_audio(data, dataPath, numFrames, audioAug, audioSet = None):
    dataName = data[0]
    fps = float(data[2])    
    audio = audioSet[dataName]   
    if audioAug == True:
        augType = random.randint(0,1)
        if augType == 1:
            audio = overlap(dataName, audio, audioSet)
        else:
            audio = audio
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    if audioSet[dataName] is None or len(audioSet[dataName]) == 0:
        # print(audio)
        # print(dataName)
        raise ValueError(f"Audio data for '{dataName}' is empty.")
    audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio:
        shortage    = maxAudio - audio.shape[0]
        audio     = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:int(round(numFrames * 4)),:]  
    return audio

def load_visual(data, dataPath, numFrames, visualAug): 
    dataName = data[0]
    videoName = data[0][:11]
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    faces = []
    H = 112
    if visualAug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H,H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H,H)))
    faces = numpy.array(faces)
    return faces

def load_landmark(data, dataPath, visualPath, numFrames): 
    dataName = data[0]
    videoName = data[0][:11]
    landmark_json_path = os.path.join(dataPath, videoName, dataName + '.json')

    with open(landmark_json_path, 'r') as f:
        landmark_map = json.load(f) # t, 82, 2

    faceFolderPath = os.path.join(visualPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    landmarks = []
    for faceFile in sortedFaceFiles[:numFrames]:
        landmarks.append(numpy.array(landmark_map[faceFile.split('/')[-1][:-4]]))
        # landmarks.append(numpy.full((82, 2), -1))
        
    landmarks = numpy.array(landmarks)
    return landmarks

def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res

class train_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, landmark_path, batchSize, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = []      
        self.landmark_path = landmark_path
        mixLst = open(trialFileName).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)         
        start = 0        
        while True:
          length = int(sortedMixLst[start].split('\t')[1])
          end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
          self.miniBatch.append(sortedMixLst[start:end])
          if end == len(sortedMixLst):
              break
          start = end     

    def __getitem__(self, index):
        batchList    = self.miniBatch[index]
        numFrames   = int(batchList[-1].split('\t')[1])
        audioFeatures, visualFeatures, labels, landmarkFeatures = [], [], [], []
        audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        for line in batchList:
            data = line.split('\t')            
            audioFeatures.append(load_audio(data, self.audioPath, numFrames, audioAug = True, audioSet = audioSet))  
            visualFeatures.append(load_visual(data, self.visualPath,numFrames, visualAug = True))
            landmarkFeatures.append(load_landmark(data, self.landmark_path, self.visualPath, numFrames))
            labels.append(load_label(data, numFrames))
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.FloatTensor(numpy.array(landmarkFeatures)), \
               torch.LongTensor(numpy.array(labels))        

    def __len__(self):
        return len(self.miniBatch)


class val_loader(object):
    def __init__(self, landmark_path, trialFileName, audioPath, visualPath, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()
        self.landmark_path = landmark_path

    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')
        # print(data)
        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
        visualFeatures = [load_visual(data, self.visualPath,numFrames, visualAug = False)]
        landmarkFeatures = [load_landmark(data, self.landmark_path, self.visualPath, numFrames)]
        labels = [load_label(data, numFrames)]         
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.FloatTensor(numpy.array(landmarkFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)
    
def load_label_ASW(data, numFrames):
    res = []
    labels = data[3:]
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res

class val_loader_ASW(object):
    def __init__(self, landmark_path, trialFileName, audioPath, visualPath, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()
        self.landmark_path = landmark_path

    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')
        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
        visualFeatures = [load_visual(data, self.visualPath,numFrames, visualAug = False)]
        landmarkFeatures = [load_landmark(data, self.landmark_path, self.visualPath, numFrames)]
        labels = [load_label_ASW(data, numFrames)]         
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.FloatTensor(numpy.array(landmarkFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)
    
def load_dataName(data):
    for i in range(len(data[0]) - 1, -1, -1):
        if data[0][i] == ':':
            videoName = data[0][:i]
            break
    return videoName

def generate_audio_set_talkies(dataPath, batchList):
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        videoName = load_dataName(data)
        dataName = data[0]
        _, audio = wavfile.read(os.path.join(dataPath, videoName, dataName + '.wav'))
        if len(audio) == 0:
            # print(f"File {file_path} is empty. Assigning fallback.")
            audioSet[dataName] = numpy.zeros((1,), dtype=numpy.int16)  # Fallback
            continue
        audioSet[dataName] = audio
    return audioSet

def load_visual_talkies(data, dataPath, numFrames, visualAug): 
    dataName = data[0]
    videoName = load_dataName(data)
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    faces = []
    H = 112
    if visualAug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H,H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H,H)))
    faces = numpy.array(faces)
    return faces

def load_landmark_talkies(data, dataPath, visualPath, numFrames): 
    dataName = data[0]
    videoName = load_dataName(data)
    landmark_json_path = os.path.join(dataPath, videoName, dataName + '.json')

    with open(landmark_json_path, 'r') as f:
        landmark_map = json.load(f) # t, 82, 2

    faceFolderPath = os.path.join(visualPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    landmarks = []
    for faceFile in sortedFaceFiles[:numFrames]:
        landmarks.append(numpy.array(landmark_map[faceFile.split('/')[-1][:-4]]))
        
    landmarks = numpy.array(landmarks)
    return landmarks

class val_loader_talkies(object):
    def __init__(self, landmark_path, trialFileName, audioPath, visualPath, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()
        self.landmark_path = landmark_path

    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set_talkies(self.audioPath, line)        
        data = line[0].split('\t')
        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
        visualFeatures = [load_visual_talkies(data, self.visualPath,numFrames, visualAug = False)]
        landmarkFeatures = [load_landmark_talkies(data, self.landmark_path, self.visualPath, numFrames)]
        labels = [load_label_ASW(data, numFrames)]         
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.FloatTensor(numpy.array(landmarkFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)
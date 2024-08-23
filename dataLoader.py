import os, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile
from scipy import signal
import csv
import imageio
from torchvision.transforms import RandomCrop
CSV__TRAIN_FILE={
   
    'FAV':'FAV_train.csv',
    'My_test':'My_test.csv',
    'W2L':"W2L.csv",
    "W2L_Self":"W2L_Self.csv",
    "MIT":"MakeItTalk.csv",
    "MIT_Self":"MakeItTalk_Self.csv",
    "A2H":"Audio2Head.csv",
    "A2H_Self":"Audio2Head_Self.csv",
    "VRT":"VideoReTalk.csv",
    "VRT_Self":"VideoReTalk_Self.csv"

}


CSV__TEST_FILE={
   
    'FAV':'FAV_test.csv',
    'My_test':'My_test.csv',
    'W2L':"W2L.csv",
    "W2L_Self":"W2L_Self.csv",
    "MIT":"MakeItTalk.csv",
    "MIT_Self":"MakeItTalk_Self.csv",
    "A2H":"Audio2Head.csv",
    "A2H_Self":"Audio2Head_Self.csv",
    "VRT":"VideoReTalk.csv",
    "VRT_Self":"VideoReTalk_Self.csv"

   

}



def read_csv_files(csv_path,Csv_dict,specific_data=None):
    combined_data = []
    header_saved = False
    headers=[]
    print (specific_data)
    if specific_data==None:
        for file_key in Csv_dict:
            file_path = os.path.join(csv_path, Csv_dict[file_key])
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                print ("---------loading data-------",csvfile)

                
                if not header_saved:
                    header = next(csvreader)
                    #combined_data.append(header)
                    headers=header
                    header_saved = True
                else:
                 
                    next(csvreader)

           
                for row in csvreader:
                    combined_data.append(row)
    else:
        file_path = os.path.join(csv_path, Csv_dict[specific_data])
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            print ("---------loading data-------",csvfile)

            
            if not header_saved:
                header = next(csvreader)
                #combined_data.append(header)
                headers=header
                header_saved = True
            else:
              
                next(csvreader)

           
            for row in csvreader:
                combined_data.append(row)

    return combined_data

def generate_audio_set(dataPath, batchList):
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        videoName = data[0][:11]
        dataName = data[0]
        _, audio = wavfile.read(os.path.join(dataPath, videoName, dataName + '.wav'))
        audioSet[dataName] = audio
    return audioSet


def load_audio_features(data, numFrames, audioAug):
    stftWindow = "hamming"
    stftWinLen = 0.040
    stftOverlap = 0.030 

    dataName = data[0]
    fps = float(data[-2])    
    
    audiopath=data[1]
  


    
    sampFreq, audio = wavfile.read(audiopath)
    #pad the audio to get atleast 4 STFT vectors
    # if len(audio) < sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)):
    #     padding = int(np.ceil((sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)) - len(audio))/2))
    #     inputAudio = np.pad(audio, padding, "constant")
    inputAudio = audio/numpy.max(numpy.abs(audio))            
    inputAudio = inputAudio/numpy.sqrt(numpy.sum(inputAudio**2)/len(inputAudio))

    #computing STFT and taking only the magnitude of it
   
    # _, _, stftVals = signal.stft(inputAudio, sampFreq, window=stftWindow, nperseg=sampFreq*stftWinLen, \
    #     noverlap=sampFreq*(stftWinLen-(stftWinLen-stftOverlap)*25/fps),boundary=None, padded=False)
    _, _, stftVals = signal.stft(inputAudio, sampFreq, window=stftWindow, nperseg=sampFreq*stftWinLen, \
        noverlap=sampFreq*stftOverlap* 25 / fps, boundary=None, padded=False)
    audInp = numpy.abs(stftVals)
    audInp = audInp.T

    audio = python_speech_features.mfcc(audio, sampFreq, numcep = 13,nfft=int(sampFreq*stftWinLen), winlen = 0.040 * 25 / fps, winstep = 0.010 * 25 / fps)
    #audio = python_speech_features.mfcc(audio, 16000, numcep = 13,nfft=640, winlen = 0.040 , winstep = 0.010 )
    #print (audInp.shape,audio.shape)
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio or audInp.shape[0] <maxAudio:
        shortage    = maxAudio - audio.shape[0]
        audio     = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
        shortage    = maxAudio - audInp.shape[0]
        audInp     = numpy.pad(audInp, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:int(round(numFrames * 4)),:]  
    audInp = audInp[:int(round(numFrames * 4)),:]  
    #print ("aliment",audInp.shape,audio.shape,data[-4])
    return audio,audInp

def load_visual(data,  numFrames, visualAug): 
    dataName = data[0]
    #videoName = data[0][:11]
    videopath=data[2]
    

    lipInp = numpy.load(videopath[:-4]+".npy")
  
    vid = imageio.get_reader(videopath)
    
    
    faces = []
    H = 112
    if visualAug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'
    #for faceFile in sortedFaceFiles[:numFrames]:
    #print (videopath,data[-4],numFrames)
    try:
        for i in range(numFrames):

            
        
            #if ret == False:break
            face = vid.get_data(i)               
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
        #cap.release()
    except RuntimeError:
            return None,None
    vid.close()
    #print ("lenth",len(faces),numFrames,data[-4])
    faces = numpy.array(faces)
    if (faces.shape[0]!=lipInp[:numFrames].shape[0]): print ("id name",faces.shape,lipInp.shape,numFrames,dataName)
    return faces, lipInp[:numFrames]


def load_label(data, numFrames):
    Entire_Label=numpy.repeat(int(data[3]),numFrames,axis=0)
    Audio_Label=numpy.repeat(int(data[4]),numFrames,axis=0)
    Video_Label=numpy.repeat(int(data[5]),numFrames,axis=0)
    Ident_Label=numpy.repeat(int(data[6]),numFrames,axis=0)
    Sync_Label=numpy.repeat(int(data[7]),numFrames,axis=0)
    
    #res = numpy.array(res[:numFrames])
    return Entire_Label,Audio_Label,Video_Label,Ident_Label,Sync_Label

class train_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, batchSize, traindata=None,**kwargs):
       
        # self.audioPath  = audioPath
        self.traindata = traindata
        self.miniBatch = []      
        if self.traindata=="FAV_LeavOneOut": 
            print ("come")
            mixLst = read_csv_files(trialFileName,CSV__TEST_FILE)
        else:
            mixLst = read_csv_files(trialFileName,CSV__TRAIN_FILE,specific_data=self.traindata)
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst[1:], key=lambda x: int(float(x[-4])), reverse=True)

        #mixLst = open(trialFileName).read().splitlines()
        
        
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        #sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)         
        start = 0        
        while True:
          length = int(float(sortedMixLst[start][-4]))
          
          end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
          self.miniBatch.append(sortedMixLst[start:end])
          if end == len(sortedMixLst):
              break
          start = end     

    def __getitem__(self, index):
        batchList    = self.miniBatch[index]
        numFrames   = int(float(batchList[-1][-4]))
        Fnm=numFrames if numFrames<=1000 else 1000
        audioFeatures, visualFeatures, EntireLabels = [], [], []
        lipaudioFeatures,lipvideoFeatures=[],[]
        AudioLabel,VideoLabel,IdentLabel,SyncLabel=[],[],[],[]
        #audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        for line in batchList:
           
            #print (line,len(line))
            vTalk,vLip=load_visual(line,Fnm, visualAug = True)
            if vTalk is None: 
                print("Failed to load the frames for the video:", data[1])
                continue
            aTalk,aLip=load_audio_features(line, Fnm, audioAug = True)
            
            audioFeatures.append(aTalk)  
            visualFeatures.append(vTalk)
            lipaudioFeatures.append(aLip)
            lipvideoFeatures.append(vLip)
            eLabel,aLabel,vLabel,iLabel,sLabel=load_label(line, Fnm)
            #print (aTalk.shape,aLip.shape,vTalk.shape,vLip.shape)
            EntireLabels.append(eLabel)
            AudioLabel.append(aLabel)
            VideoLabel.append(vLabel)
            IdentLabel.append(iLabel)
            SyncLabel.append(sLabel)
        return  torch.FloatTensor(numpy.array(audioFeatures)), \
                torch.FloatTensor(numpy.array(visualFeatures)), \
                torch.FloatTensor(numpy.array(lipaudioFeatures)), \
                torch.FloatTensor(numpy.array(lipvideoFeatures)),\
                torch.LongTensor(numpy.array(EntireLabels)), \
                torch.LongTensor(numpy.array(AudioLabel)),\
                torch.LongTensor(numpy.array(VideoLabel)), \
                torch.LongTensor(numpy.array(IdentLabel)), \
                torch.LongTensor(numpy.array(SyncLabel))

    def __len__(self):
        return len(self.miniBatch)


class val_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, testdata=None,**kwargs):
        
        #self.miniBatch = open(trialFileName).read().splitlines()
        self.miniBatch  = read_csv_files(trialFileName,CSV__TEST_FILE,specific_data=testdata)

    def __getitem__(self, index):
        line       = [self.miniBatch[index]][0]
        #print(line)
        Fnm  = int(float(line[-4]))
        #audioSet   = generate_audio_set(self.audioPath, line)      
        #self.miniBat  
        aTalk,aLip=load_audio_features(line, Fnm, audioAug = False)
        vTalk,vLip=load_visual(line,Fnm, visualAug = False)
        eLabel,aLabel,vLabel,iLabel,sLabel=load_label(line, Fnm)
        audioFeatures = [aTalk]
        visualFeatures = [vTalk]
        lipaudioFeatures=[aLip]
        lipvideoFeatures=[vLip]
        EntireLabels=[eLabel]
        AudioLabel=[aLabel]
        VideoLabel=[vLabel]
        IdentLabel=[iLabel]
        SyncLabel=[sLabel]
        
        return  torch.FloatTensor(numpy.array(audioFeatures)), \
                torch.FloatTensor(numpy.array(visualFeatures)), \
                torch.FloatTensor(numpy.array(lipaudioFeatures)), \
                torch.FloatTensor(numpy.array(lipvideoFeatures)),\
                torch.LongTensor(numpy.array(EntireLabels)), \
                torch.LongTensor(numpy.array(AudioLabel)),\
                torch.LongTensor(numpy.array(VideoLabel)), \
                torch.LongTensor(numpy.array(IdentLabel)), \
                torch.LongTensor(numpy.array(SyncLabel))

    def __len__(self):
        return len(self.miniBatch)

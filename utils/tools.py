import os, subprocess, glob, pandas, tqdm, cv2, numpy
from scipy.io import wavfile

def init_args(args):
    # The details for the following folders/files can be found in the annotation of the function 'preprocess_AVA' below
    args.modelSavePath    = os.path.join(args.savePath, args.trainData,'model')
    args.scoreSavePath    = os.path.join(args.savePath, args.trainData, 'score.txt')
    args.resutlSavePath    = os.path.join(args.savePath, args.trainData,'result_{}.txt')
    args.trialPathAVA     = os.path.join(args.dataPathAVA, 'csv')
    #args.trialPathAVA     = "/data5/caiyu/code/MisMatch/JsonMisMatch/csvfile"
    args.audioOrigPathAVA = os.path.join(args.dataPathAVA, 'orig_audios')
    args.visualOrigPathAVA= os.path.join(args.dataPathAVA, 'orig_videos')
    args.audioPathAVA     = os.path.join(args.dataPathAVA, 'clips_audios')
    args.visualPathAVA    = os.path.join(args.dataPathAVA, 'clips_videos')
    args.trainTrialAVA    = "./csvfile"
    #"/data5/caiyu/code/MisMatch/JsonMisMatch/csvfile"
    args.evalTrialAVA = "./csvfile"#"/data5/caiyu/code/MisMatch/JsonMisMatch/csvfile"

  
    
    os.makedirs(args.modelSavePath, exist_ok = True)
    os.makedirs(args.dataPathAVA, exist_ok = True)
    return args
 

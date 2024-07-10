import time, os, torch, argparse, warnings, glob

from dataLoader import train_loader, val_loader
from utils.tools import *

from MTL_DDF import MTL_DDF

import numpy as np

import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def main():
    #the structure of this code is learnt from https://github.com/clovaai/voxceleb_trainer
    warnings.filterwarnings("ignore")
    set_seed(898) 

    parser = argparse.ArgumentParser(description = "TalkNet Training")
    # Training details
    parser.add_argument('--lr',           type=float, default=0.00001,help='Learning rate')
    parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
    parser.add_argument('--maxEpoch',     type=int,   default=100,    help='Maximum number of epochs')
    parser.add_argument('--maxFramLen',     type=int,   default=250,    help='Maximum number of frames per video')
    parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize',    type=int,   default=2000,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    parser.add_argument('--nDataLoaderThread', type=int, default=4,  help='Number of loader threads')
    # # Data path
    parser.add_argument('--dataPathAVA',  type=str, default=".", help='')
    #base_Ident_Sync
    parser.add_argument('--savePath',     type=str, default="final/MTL_DDF")
   
    parser.add_argument('--trainData', type=str, default="FAV", help='to choose the dataset for evaluation, val or test')#My_test
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help='to choose the dataset for evaluation, val or test')
  
    args = parser.parse_args()
    # Data loader
    args = init_args(args)

    
    if args.trainData=="All": args.trainData=None
   
    loader = train_loader(trialFileName = args.trainTrialAVA, \
                          audioPath      = os.path.join(args.audioPathAVA , 'train'), \
                          visualPath     = os.path.join(args.visualPathAVA, 'train'), \
                          traindata  =args.trainData, \
                          **vars(args))
    trainLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = True, num_workers = args.nDataLoaderThread)

    
   
    epoch = 1
   
    s=MTL_DDF(epoch = epoch, **vars(args))
    s.loadParameters('pretrain_TalkSet.model')
   
       
    

        

    mAPs = []
    scoreFile = open(args.scoreSavePath, "a+")

    while(1):        
        loss, lr = s.train_network(epoch = epoch, loader = trainLoader, **vars(args))
        
        if epoch % args.testInterval == 0:        
            s.saveParameters(args.modelSavePath + "/model_%04d.model"%epoch)
            # mAPs.append(s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args)))
            # print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, mAP %2.2f%%, bestmAP %2.2f%%"%(epoch, mAPs[-1], max(mAPs)))
            scoreFile.write("%d epoch, LR %f, LOSS %f\n"%(epoch, lr, loss))
            # scoreFile.flush()

        if epoch >= args.maxEpoch:
            quit()

        epoch += 1

if __name__ == '__main__':
    
    main()

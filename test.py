import time, os, torch, argparse, warnings, glob

from dataLoader import train_loader, val_loader
from utils.tools import *

from MTL_DDF import MTL_DDF

def main():
    # The structure of this code is learnt from https://github.com/clovaai/voxceleb_trainer
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = "DDF Testing")
    # Training details
    parser.add_argument('--lr',           type=float, default=0.0001,help='Learning rate')
    parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
    parser.add_argument('--maxEpoch',     type=int,   default=80,    help='Maximum number of epochs')
    parser.add_argument('--maxFramLen',     type=int,   default=1000,    help='Maximum number of frames per video')
    parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize',    type=int,   default=2000,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    parser.add_argument('--nDataLoaderThread', type=int, default=4,  help='Number of loader threads')
    # Data path
    parser.add_argument('--dataPathAVA',  type=str, default=".", help='')
    parser.add_argument('--savePath',     type=str, default="final/MTL_DDF")
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help=' to choose the dataset for evaluation, val or test')
    #My_test;Res_LHG;FAV
    parser.add_argument('--trainData', type=str, default="FAV", help=' to choose the dataset for evaluation, val or test')#My_test
    # Data selection
  
    parser.add_argument('--testData', type=str, default="W2L", help='[W2L, A2H,VRT, MIT,W2L_Self, A2H_Self,VRT_Self, MIT_Self]')#My_test
    # parser.add_argument('--testMode', type=str, default="E_V", help=' to choose the dataset for evaluation, val or test')
 
    
    args = parser.parse_args()
    # Data loader
    args = init_args(args)

    

    if args.testData=="All": args.testData=None
    loader = val_loader(trialFileName = args.evalTrialAVA, \
                        audioPath     = os.path.join(args.audioPathAVA , args.evalDataType), \
                        visualPath    = os.path.join(args.visualPathAVA, args.evalDataType), \
                        testdata  =args.testData, \
                        **vars(args))
    valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = 16)



 
    ACCs = []
    resutlsFile = args.resutlSavePath.format(args.testData)
 

    epoch=1
    s=MTL_DDF(epoch = epoch, **vars(args))
       


    results=[]


    
    modelpath="best.model"
    print (modelpath)
    s.loadParameters(modelpath)
    result=s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args))
    print (result)
    results.append(result + '\n')
    print("\nTesting Done.\n")
    
    with open(resutlsFile,'w') as f:
         f.writelines(results)



       

       
if __name__ == '__main__':
    main()

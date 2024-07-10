import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm

from loss_MTL import lossAV, MetricCont_Spea
from model.talkNetModel import talkNetModel
from model.audio_net import AudioNet
from model.video_net import VideoNet
from sklearn import metrics
from utils.metrics import *

class MTL_DDF(nn.Module):
    def __init__(self, lr = 0.0001, lrDecay = 0.95, **kwargs):
        super(MTL_DDF, self).__init__()   
    
        self.model = talkNetModel().cuda()
        self.ASRmodel = AudioNet(512,8,6,2500,321,2048,0.1,40).cuda()
        self.VSRmodel = VideoNet(512,8,6,2500,2048,0.1,40).cuda()
        self.ASRmodel.load_state_dict(torch.load( "teacher_model_weights/audio-only.pt", map_location="cuda"), strict=False)
        self.VSRmodel.load_state_dict(torch.load( "teacher_model_weights/video-only.pt", map_location="cuda"), strict=False)
       
        self.lossAV = lossAV().cuda()
       
        self.lossCont_Spea = MetricCont_Spea().cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        for param in self.ASRmodel.parameters():
            param.requires_grad = False
        for param in self.VSRmodel.parameters():
            param.requires_grad = False
        print (" ASR & VSR frozen")
   
   

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        #self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']        
        for num, (audioFeature, visualFeature, lipaudioFeatures,lipvideoFeatures,\
            EntireLabel,AudioLabel,VideoLabel,IdentLabel,SyncLabel) in enumerate(loader, start=1):
            self.zero_grad()
            
            #av[0].shape  BTH (torch.Size([1, 4000, 13]), torch.Size([1, 1000, 112, 112]), 
            #lipav[0].shape BTH torch.Size([1, 4000, 321]), torch.Size([1, 1000, 512]))
            audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda()) # torch.Size([1, 1000, 128])
            visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda()) ## torch.Size([1, 1000, 128])
            ##torch.Size([1, 1000, 128]);torch.Size([1, 1000, 128])
            audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
           
             #B*T*256
            outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  #torch.Size([1000, 256])
            outsA = self.model.forward_audio_backend(audioEmbed) #torch.Size([1000, 256])
            outsV = self.model.forward_visual_backend(visualEmbed) #torch.Size([1000, 256])
            # #LipoutsA T*B*512；LipoutV T*B*512
            LipOutsA= self.ASRmodel(lipaudioFeatures[0].cuda().detach())
            LipOutsV= self.VSRmodel(lipvideoFeatures[0].cuda().detach())
            Loss_similarity,Metric_loss= self.lossCont_Spea(LipOutsA,outsA,LipOutsV,outsV,EntireLabel[0].cuda())
            EntireLabel = EntireLabel[0].reshape((-1)).cuda() # Loss
            
            # AudioLabel = AudioLabel[0].reshape((-1)).cuda() # Loss
            # VideoLabel = VideoLabel[0].reshape((-1)).cuda() # Loss
            outsAV = torch.reshape(outsAV, (-1, 256))
            # outsA = torch.reshape(outsA, (-1, 128))
            # outsV = torch.reshape(outsV, (-1, 128))
            nlossAV, _, _, prec = self.lossAV.forward(outsAV, EntireLabel)
          
            nloss = nlossAV+Loss_similarity+Metric_loss
            
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            
            index += len(EntireLabel)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " LossTotal: %.5f,DFloss: %.5f, MatchSoftLabelLoss: %.5f,Metricloss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), nlossAV, Loss_similarity,Metric_loss, 100 * (top1/index)))
            
           # " Loss: %.5f,ALoss: %.5f,VLoss: %.5f,ANCELoss: %.5f,VNCELoss: %.5f, ACC: %2.2f%% \r"        %(nlossAV/(num),nlossA/(num),nlossV/(num), Ainfo/(num),Vinfo/(num),100 * (top1/index)))
            sys.stderr.flush()  
        self.scheduler.step()
        sys.stdout.write("\n")      
        return loss/num, lr

    def evaluate_network(self, epoch,loader,  **kwargs):
        self.eval()
        # all_outputs =[]
        # all_outsigmoid=[]
        all_preds = []
        all_labels = []
        all_pos_scores = []
        all_predsA = []
        all_audlabels = []
        all_pos_scoresA = []
        all_predsV = []
        all_vidlabels = []
        all_pos_scoresV = []
        results = []
        for num, (audioFeature, visualFeature, lipaudioFeatures,lipvideoFeatrues,\
            EntireLabel,AudioLabel,VideoLabel,IdentLabel,SyncLabel) in enumerate(loader, start=1):
            with torch.no_grad():                
                audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].cuda())
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
                outsA = self.model.forward_audio_backend(audioEmbed)
                outsV = self.model.forward_visual_backend(visualEmbed)
                EntireLabels = EntireLabel[0].reshape((-1)).cuda()[0]      
                AudioLabels = AudioLabel[0].reshape((-1)).cuda()[0]    
                VideoLabels = VideoLabel[0].reshape((-1)).cuda()[0]       
                outsAV = torch.reshape(outsAV, (-1, 256))
               
                output = self.lossAV.forward(outsAV)    
             
                #print (output.shape)
                # import pdb
                # pdb.set_trace()
                _, predicted = torch.max(output,0)
               
                #pdb.set_trace()
                outputs =output[-1].view(-1)
              
                #print (outputBatch.shape)[12,2]
                # import pdb
                # pdb.set_trace()
                #all_outputs.extend(output.detach().cpu().numpy())
                #all_outsigmoid.extend(torch.sigmoid(output).detach().cpu().numpy().tolist())
                all_pos_scores.extend(outputs.detach().cpu().numpy().tolist())
               
                #all_preds.extend(predicted.cpu().numpy().tolist())
                all_preds.append(predicted.cpu().numpy().tolist())
                
                all_labels.append(EntireLabels.cpu().numpy().tolist())
              
        # import pdb
        # pdb.set_trace()
        acc = get_acc(all_labels, all_preds)
        F1  = get_f1(all_labels, all_preds)
        bacc, roc_auc = evaluate_auc(all_labels, all_preds, all_pos_scores)
        TN, FP, FN, TP, = confusion_matrix(all_labels, all_preds).ravel()
        real_recall = TN / (TN + FP)
        fake_recall = TP / (TP + FN)
        eer = get_eer(all_labels, all_pos_scores)

        far = FP / (FP + TN)
        frr = FN / (FN + TP)
        hter = (far + frr) / 2
        
        # Aacc = get_acc(all_audlabels, all_predsA)
        # AF1  = get_f1(all_audlabels, all_predsA)
        # if len(numpy.unique(all_audlabels))<=1:Abacc, Aroc_auc,Aeer = float('-inf'), float('-inf'), float('-inf')
        # else: 
        #     Abacc, Aroc_auc = evaluate_auc(all_audlabels, all_predsA, all_pos_scoresA)
        #     Aeer = get_eer(all_audlabels, all_pos_scoresA)
        # ATN, AFP, AFN, ATP, = confusion_matrix(all_audlabels, all_predsA).ravel()
        # Areal_recall = ATN / (ATN + AFP)
        # Afake_recall = ATP / (ATP + AFN)
       

        # Afar = AFP / (AFP + ATN)
        # Afrr = AFN / (AFN + ATP)   
        # Ahter = (Afar + Afrr) / 2

        # Vacc = get_acc(all_vidlabels, all_predsV)
        # VF1  = get_f1(all_vidlabels, all_predsV)
        # Vbacc, Vroc_auc = evaluate_auc(all_vidlabels, all_predsV, all_pos_scoresV)
        # VTN, VFP, VFN, VTP, = confusion_matrix(all_vidlabels, all_predsV).ravel()
        # Vreal_recall = VTN / (VTN + VFP)
        # Vfake_recall = VTP / (VTP + VFN)
        # Veer = get_eer(all_vidlabels, all_pos_scoresV)

        # Vfar = VFP / (VFP + VTN)
        # Vfrr = VFN / (VFN + VTP)
        # Vhter = (Vfar + Vfrr) / 2
        result = 'model:{},Total images:{},acc:{:.6f},F1:{:.6f},bACC:{:.6f},RR:{:.6f},FR:{:.6f},ROC_AUC:{:.6f},EER:{:.6f},HTER:{:.6f},TN:{},FN:{},TP:{},FP:{}'\
        .format(str(epoch), len(all_labels), acc, F1, bacc, real_recall, fake_recall, roc_auc, eer, hter, TN, FN, TP, FP)
        #print(step,result))
        #results.append(result + '\n')    
        return result


    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
        print("model loading down")

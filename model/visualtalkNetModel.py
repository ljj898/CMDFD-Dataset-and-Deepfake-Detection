import torch
import torch.nn as nn

from model.audioEncoder      import audioEncoder
from model.visualEncoder     import visualFrontend, visualTCN, visualConv1D
from model.attentionLayer    import attentionLayer

class visualtalkNetModel(nn.Module):
    def __init__(self):
        super(visualtalkNetModel, self).__init__()
        # Visual Temporal Encoder
        self.visualFrontend  = visualFrontend() # Visual Frontend 
        # self.visualFrontend.load_state_dict(torch.load('visual_frontend.pt', map_location="cuda"))
        # for param in self.visualFrontend.parameters():
        #     param.requires_grad = False       
        self.visualTCN       = visualTCN()      # Visual Temporal Network TCN
        self.visualConv1D    = visualConv1D()   # Visual Temporal Network Conv1d

      
      
        self.crossV2V = attentionLayer(d_model = 128, nhead = 8)

        # Audio-visual Self Attention
        self.selfAV = attentionLayer(d_model = 128, nhead = 8)

    def forward_visual_frontend(self, x):
        # import pdb
        # pdb.set_trace()
        B, T, W, H = x.shape  #x.shape,torch.Size([5, 342, 112, 112])
        x = x.view(B*T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)        
        x = x.transpose(1,2)     
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1,2)
        return x

  

    def forward_cross_attention(self, x):
        
        x2_c = self.crossV2V(src = x, tar = x)        
        return x2_c

    def forward_audio_visual_backend(self, x): 
       
        x = self.selfAV(src = x, tar = x)       
        #x = torch.reshape(x, (-1, 256))
        return x    

  

    def forward_visual_backend(self,x):
        #x = torch.reshape(x, (-1, 128))
        return x


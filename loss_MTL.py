import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def calculate_js_divergence(p, q, eps=1e-10):
	m = (p + q) / 2
	#js_div = (F.kl_div(p.log(), m, reduction='none') + F.kl_div(q.log(), m, reduction='none')) / 2
	js_div = (F.kl_div((p + eps).log(), m, reduction='none') + F.kl_div((q + eps).log(), m, reduction='none')) / 2
	js_div1 = js_div.sum(-1)  # Sum across the last dimension (class/probability distribution dimension)
	js_div2=js_div.mean(-1) 
	#print (js_div)
	return js_div1,js_div2
def calculate_js_divergence2(p, q, EPISILON=1e-10):
	loss1 = p * torch.log(p / q + EPISILON)
	loss1 = loss1.mean(2)
	loss2 = q * torch.log(q / p + EPISILON)
	loss2 = loss2.mean(2)


	loss = (loss1 + loss2)/2
	
	return loss

class lossCont_Spea(nn.Module):
	def __init__(self,dim=256):
		super(lossCont_Spea, self).__init__()
		self.projection_A = nn.Linear(512, dim)
		self.projection_B = nn.Linear(128, dim)
		self.temperature = 0.5
		self.softmax = nn.Softmax(dim=1)
		# Create linear layers for projection
	def forward(self, LipOutsA, OutsA, LipOutsV, OutsV ):		
		
		lipA=LipOutsA.transpose(0,1)
		lipV=LipOutsV.transpose(0,1)
		#pdb.set_trace()
		
		#lip_js_div,lip_js_div2 = calculate_js_divergence(lipA, lipV)#B*T*Dim
		
		lip_js_div = calculate_js_divergence2(lipA, lipV)#B*T*Dim
		# Normalize the features to get unit-length embeddings
		outsA_normalized = F.normalize(OutsA, p=2, dim=2)
		outsV_normalized = F.normalize(OutsV, p=2, dim=2)
		cos_sim = F.cosine_similarity(outsA_normalized, outsV_normalized, dim=-1)
		cos_sim_scaled = (cos_sim + 1) / 2
		loss = F.binary_cross_entropy(cos_sim_scaled, 1-lip_js_div)
		#print (loss.shape,cos_sim_scaled.shap#) 
		
		return  cos_sim_scaled.mean()#(1-lip_js_div).mean()#


class MetricCont_Spea(nn.Module):
	def __init__(self,dim=256):
		super(MetricCont_Spea, self).__init__()
		self.projection_A = nn.Linear(512, dim)
		self.projection_B = nn.Linear(128, dim)
		self.temperature = 0.5
		self.margin = 1.0
		self.softmax = nn.Softmax(dim=1)
		# Create linear layers for projection
	def forward(self, LipOutsA, OutsA, LipOutsV, OutsV,Label ):		
			# Project both features to the same dimension
		
		lipA=LipOutsA.transpose(0,1)
		lipV=LipOutsV.transpose(0,1)
		#pdb.set_trace()
		
		#lip_js_div,lip_js_div2 = calculate_js_divergence(lipA, lipV)#B*T*Dim
		
		lip_js_div = calculate_js_divergence2(lipA, lipV)#B*T*Dim
		# Normalize the features to get unit-length embeddings
		outsA_normalized = F.normalize(OutsA, p=2, dim=2)
		outsV_normalized = F.normalize(OutsV, p=2, dim=2)
		cos_sim = F.cosine_similarity(outsA_normalized, outsV_normalized, dim=-1)

# Scale cosine similarity to be between 0 and 1
		cos_sim_scaled = (cos_sim + 1) / 2
		loss = F.binary_cross_entropy(cos_sim_scaled, 1-lip_js_div)
		pre=1-lip_js_div
		labels=Label[:, 0]
		
		#print (loss.shape,cos_sim_scaled.shap#) 
		distance = (pre - cos_sim_scaled).pow(2).sum(-1).sqrt() 
		distance = torch.sigmoid(distance)
		loss_close = (1 - labels) * distance.pow(2)
		loss_far = labels * torch.clamp(self.margin - distance, min=0.0).pow(2)
		metric= torch.mean(loss_close + loss_far)

		
		return  loss,metric#cos_sim_scaled.mean(),(1-lip_js_div).mean()



class lossAV(nn.Module):
	def __init__(self,dim=256):
		super(lossAV, self).__init__()
		self.criterion = nn.CrossEntropyLoss()#nn.CrossEntropyLoss(reduction='none')#
		self.FC        = nn.Linear(dim, 2)
		
	def forward(self, x, labels=None):	
		x = x.squeeze(1)
		x = self.FC(x)# T*2
		
		if labels == None:
			whole_pre=torch.mean(x,dim=0)
  
			# predScore = x[:,1]
			# predScore = predScore.t()
			# predScore = predScore.view(-1).detach().cpu().numpy()
			return whole_pre
		else:
			nloss = self.criterion(x, labels)#.detach()
			# import pdb
			# pdb.set_trace()
			predScore = F.softmax(x, dim = -1)
			predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
			correctNum = (predLabel == labels).sum().float()
			#pdb.set_trace()
			return nloss, predScore, predLabel, correctNum

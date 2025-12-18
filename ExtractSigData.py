import matplotlib.pyplot as plt
import numpy as np
from unsloth import FastLanguageModel
import torch
from scipy import spatial
import seaborn as sns
import torch.nn.functional as F

def calc_svd(tensor,r):
    t = tensor.float()
    _,s,_ = torch.linalg.svd(t, full_matrices=False)
    ts = s.detach().numpy()
    return ts[:r]


def getdistance(m_name,disfname,rank=16):
    max_seq_length = 2048
    # dtype = None # Auto detection of dtype
    model, _ = FastLanguageModel.from_pretrained(
    model_name=m_name,
    load_in_4bit = False,
    max_seq_length = max_seq_length,
    )
    layernum = len(model.model.layers)
    w_arry = None
    for i in range(layernum):
        if w_arry is None:
           w_arry =  calc_svd((model.model.layers[i].self_attn.q_proj.weight).cpu(),rank)
        else:
            w_arry=np.vstack((w_arry,calc_svd((model.model.layers[i].self_attn.q_proj.weight).cpu(),rank)))
        w_arry=np.vstack((w_arry,calc_svd((model.model.layers[i].self_attn.k_proj.weight).cpu(),rank)))
        w_arry=np.vstack((w_arry,calc_svd((model.model.layers[i].self_attn.v_proj.weight).cpu(),rank)))
        w_arry=np.vstack((w_arry,calc_svd((model.model.layers[i].self_attn.o_proj.weight).cpu(),rank)))
        w_arry=np.vstack((w_arry,calc_svd((model.model.layers[i].mlp.gate_proj.weight).cpu(),rank)))
        w_arry=np.vstack((w_arry,calc_svd((model.model.layers[i].mlp.up_proj.weight).cpu(),rank)))                          
        w_arry=np.vstack((w_arry,calc_svd((model.model.layers[i].mlp.down_proj.weight).cpu(),rank)))   
    print(w_arry.shape)     
    np.save(disfname,w_arry)
    return

def getcos(tensor1,tensor2,k):
    t1 = tensor1.float()
    t2 = tensor2.float()
    _,s1,_ = torch.linalg.svd(t1, full_matrices=False)
    _,s2,_ = torch.linalg.svd(t2, full_matrices=False)
    ts1 = s1.detach().numpy()
    ts2 = s2.detach().numpy()
    dis = spatial.distance.cosine(ts1[:k], ts2[:k])
    # U, s, V = torch.linalg.svd(tensor1, full_matrices=False)
    # m =  U[:, :k] @ torch.diag_embed(s[:k]) @ V[:k, :]  
    return round(dis,8)

    

getdistance("/src/models/smo135","smo135_2025.npy",rank=16)



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import seaborn as sns
os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


def plotwt():
    sn1='qwen05down'
    sn2='qwen15down'
    sn3='qwen7bdown'
    sn4=''
    if sn1 !='':
        s1 = np.load('weight/'+sn1+'.npy')
        s1 = np.random.choice(s1,size=2000,replace=False)
    if sn2 !='':
        s2 = np.load('weight/'+sn2+'.npy')
        s2 = np.random.choice(s2,size=2000,replace=False)   
    if sn3 !='':
        s3 = np.load('weight/'+sn3+'.npy')
        s3 = np.random.choice(s3,size=2000,replace=False)
    if sn4 !='':
        s4 = np.load('weight/'+sn4+'.npy')
        s4 = np.random.choice(s4,size=2000,replace=False)          
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams["font.size"] = 16     
    if sn1 !='':
        sns.kdeplot(data=s1,
                    fill=False,
                    color="darkorange",
                    label="Qwen2-0.5B",
                    bw_adjust=1,
                    common_norm = False,
                    linewidth= 2,
                    alpha=1)
    if sn2 !='':    
        sns.kdeplot(data=s2,
                    fill=False,
                    color="dodgerblue",
                    label="Qwen2-1.5B",
                    bw_adjust=1,
                    linewidth= 2,
                    alpha=1)  
    if sn3 !='':    
        sns.kdeplot(data=s3,
                    fill=False,
                    color="green",
                    label="Qwen2-7B",
                    bw_adjust=1,
                    linewidth= 2,
                    alpha=1)     
    if sn4 !='':
        sns.kdeplot(data=s4,
                    fill=False,
                    color="hotpink",
                    label="Output",
                    bw_adjust=1,
                    linewidth= 2,
                    alpha=1) 
    ax = plt.gca()
    ax.set_xlim(-0.2, 0.2) 
    # plt.gca().set_xticklabels([])
    # plt.gca().set_yticklabels([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylabel('')
    plt.title('Down-Matrix')
    plt.legend(title='')
    plt.show()
    # plt.savefig('./img/down-qwen017b.png',bbox_inches='tight')


def plotcluster(x,x_kmeans,centers):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(x)
    # 对聚类中心进行降维处理
    centers_pca = pca.transform(centers)
    # print(data_pca)
    # 可视化降维后的数据和聚类中心
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams["font.size"] = 16  
    ax = plt.gca()
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=x_kmeans, s=60, cmap='viridis')
    # plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.5, marker='*')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.title('LLaMA3-1B-1k')
    plt.show()
    # plt.savefig('./img/cluster-la1b3.png',bbox_inches='tight')




def plotline():
    x=['Q','K','V','O','Gate','Up','Down']
    ys=np.zeros((22,7))
    ys[0]=[0.57 ,0.27 ,0.97 ,0.7 , 0.23, 0.87 ,0.17]
    ys[1]=[0.83, 0.93,0.07,0.37,0.87,0.13,0.8 ]
    ys[2]=[0.88 ,0.25,0.96,0.04,0.08,0.79,0.29]
    ys[3]=[0.58 ,0.04,0.96,0.04,0.08,0.83,0.29]
    ys[4]=[0.93 ,1,   1,   0.61, 0,   0.82, 0.21]
    ys[5]=[0.79 ,0.71 ,1,   0.5,  0,   0.82, 0.21]
    ys[6]=[0.79 ,1,   0.96, 0.39, 0,   0.75, 0.39]
    ys[7]=[0.75, 1,   0.96, 0.39, 0,   0.79, 0.32]
    ys[8]=[1   ,1 ,  0,   0.31, 0.94 ,0.25 ,0.25]
    ys[9]=[1   ,1,   0,   0.31, 1,   0.25 ,0.31]
    ys[10]=[1   ,1 ,  0,   0.54 ,0.96 ,0.21 ,0.29]
    ys[11]=[1   ,1 ,  0,   0.39, 1,   0.18, 0.32]
    ys[12]=[1   ,1 ,  0.03, 0.56, 1  , 0.19, 0.34]
    ys[13]=[1   ,1 ,  0.03, 0.5 , 1 ,  0.19, 0.41]
    ys[14]=[1, 1, 0, 0.31, 0.94 ,0.25, 0.25]
    ys[15]=[1, 1, 0, 0.31, 1, 0.25, 0.31]    
    ys[16]=[1, 1, 0, 0.31, 0.94, 0.25, 0.25]
    ys[17]=[1, 1, 0, 0.31, 1, 0.25, 0.31]
    ys[18]=[0.88, 0.25, 0.96, 0.04, 0.08, 0.79, 0.29]
    ys[19]=[0.58, 0.04, 0.96, 0.04, 0.08, 0.83, 0.29]    
    ys[20]=[0.88, 0.25, 0.96, 0.04, 0.08, 0.79, 0.29]
    ys[21]=[0.58, 0.04, 0.96, 0.04, 0.08, 0.83, 0.29]           
    plt.clf()
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams["font.size"] = 16 
    ax = plt.gca()
    # y=np.zeros((14,7))
    # y[:,[0]]=ys[:,[6]]
    # y[:,[1]]=ys[:,[3]]
    # y[:,[2]]=ys[:,[0]]
    # y[:,[3]]=ys[:,[4]]
    # y[:,[4]]=ys[:,[5]]
    # y[:,[5]]=ys[:,[2]]
    # y[:,[6]]=ys[:,[1]]     
    plt.plot(x, ys[0],label='SmolLM135-16')
    # plt.plot(x, ys[1],label='SmolLM135-64')
    # plt.plot(x, ys[2],label='SmolLM1.7B-16')
    # plt.plot(x, ys[3],label='SmolLM1.7B-64')
    # plt.plot(x, ys[4],label='Qwen1.5B-16')
    # plt.plot(x, ys[5],label='Qwen1.5B-64')
    # plt.plot(x, ys[6],label='Qwen7B-16')
    # plt.plot(x, ys[7],label='Qwen7B-64')
    # plt.plot(x, ys[8],label='LLaMA1B-16')
    # plt.plot(x, ys[9],label='LLaMA1B-64')
    # plt.plot(x, ys[10],label='LLaMA3B-16')
    # plt.plot(x, ys[11],label='LLaMA3B-64')
    # plt.plot(x, ys[12],label='LLaMA8B-16') 
    # plt.plot(x, ys[13],label='LLaMA8B-64')
    # plt.plot(x, ys[14],label='LLaMA-1k-16')
    # plt.plot(x, ys[15],label='LLaMA-1k-64')      
    # plt.plot(x, ys[16],label='LLaMA-Gpt-16')
    # plt.plot(x, ys[17],label='LLaMA-Gpt-64')     
    # plt.plot(x, ys[18],label='SmolLM-1k-16')
    # plt.plot(x, ys[19],label='SmolLM-1k-64')      
    # plt.plot(x, ys[20],label='SmolLM-Gpt-16')
    # plt.plot(x, ys[21],label='SmolLM-Gpt-64')         
    plt.title('LoRA Cluster')
    plt.legend(title='')  
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  
    plt.show()
    # plt.savefig('./img/cluster-lora2.png',bbox_inches='tight')


def nargmax(array, n):
    flattened_array = array.flatten()
    max_indices = np.argsort(flattened_array)[-n:]
    return max_indices

def get_vindex(x_kmeans):
    m = int(len(x_kmeans)/7)
    x = np.reshape(x_kmeans,(m,7))
    v = np.mean(x,axis=0)
    if v[0]<=0.5:
        x[x==0]=2
        x[x==1]=0
        x[x==2]=1
        v = np.mean(x,axis=0)
    return np.round(v,6) 

def getindex(x,labels,center):
    dis =np.zeros(len(labels))
    for i in range(len(labels)):
        if labels[i]==0:
            dis[i]=np.linalg.norm(x[i] - center[0])
        else:
            dis[i]=np.linalg.norm(x[i] - center[1])
    print(dis)
    return nargmax(dis,1)


def cluster(fname):
    if fname !='':
        s1 = np.load('./data/'+fname+'.npy')        
    for i in range(s1.shape[0]):
        s1[i]=s1[i]/np.max(s1[i])
    kmeans = KMeans(n_clusters=2, random_state=0)
    x_kmeans = kmeans.fit_predict(s1)
    centers = kmeans.cluster_centers_
    return s1,x_kmeans,centers

def plot_std():
    y = np.loadtxt('std.txt',delimiter=',')
    x = ['Q','K','V','O','Gate','Up','Down']
    for i in range(y.shape[0]):
        y[[i],:] = y[[i],:]/np.max(y[[i],:])
    plt.clf()
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams["font.size"] = 16 
    ax = plt.gca()
    plt.plot(x,y[0],marker='o',linewidth= 3,label='SmolLM2-135M')
    # plt.plot(x,y[1],marker='*',linewidth= 3,label='SmolLM2-1.7B')
    # plt.plot(x,y[2],marker='o',linewidth= 3,label='LLaMA3-1B')
    # plt.plot(x,y[3],marker='*',linewidth= 3,label='LLaMA3-3B')
    # plt.plot(x,y[4],marker='^',linewidth= 3,label='LLaMA3-8B')
    # plt.plot(x,y[5],marker='o',linewidth= 3,label='Qwen2-0.5B')   
    # plt.plot(x,y[6],marker='^',linewidth= 3,label='Qwen2-1.5B')       
    # plt.plot(x,y[7],marker='*',linewidth= 3,label='Qwen2-7B')
    # plt.plot(x,y[8],marker='*',linewidth= 3,label='LoRA-LLaMA-1K')
    # plt.plot(x,y[9],marker='*',linewidth= 3,label='LoRA-LLaMA-Gpt')
    # plt.plot(x,y[10],marker='^',linewidth= 3,label='LoRA-SmolLM-1K')
    # plt.plot(x,y[11],marker='^',linewidth= 3,label='LoRA-SmolLM-Gpt')    
    plt.title('Weights Std')
    plt.legend(title='')  #loc=4
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)          
    plt.show()
    # plt.savefig('./img/std_la_vs_lora.png',bbox_inches='tight')

def plotheat(data):
    m = int(len(data)/7)
    x = np.reshape(data,(m,7))
    v = np.mean(x,axis=0)
    if v[0]<=0.5:
        x[x==0]=2
        x[x==1]=0
        x[x==2]=1
    plt.clf()
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams["font.size"] = 16
    # sns.heatmap(x,xticklabels=[],yticklabels=[])
    sns.heatmap(x,xticklabels=['Q','K','V','O','Gate','Up','Down'],yticklabels=[], cbar=False)
    plt.title('SmolLM-1K-16 cluster', pad=10)
    plt.show()    
    # plt.savefig('./img/heat-smo1k.png',bbox_inches='tight')

sn1='la1b_16'
x,x_kmeans,centers = cluster(sn1) #threadpoolctl==3.1.0, otherwise raise error  'NoneType' object has no attribute 'split'
# print(get_vindex(x_kmeans))
# t = getindex(x,x_kmeans,centers)
plotcluster(x,x_kmeans,centers)

# plotheat(x_kmeans)

# plotwt()

# plotline()

# plot_std()

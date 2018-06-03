import numpy as np
import pickle
import operator
class knn:
    def __init__(self,name):
        self.name=name
        
    def train(self,dataSet,label):
        dataSet=np.array(dataSet)
        if not(dataSet.shape[0]==label.shape[0]):
            raise Exception(f"No. of rows in dataSet ({dataSet.shape[0]}) and No. of label ({dataSet.shape[0]}) should be the same")
        minimum=[]
        maximum=[]
        for row in dataSet.T:
            minimum.append(min(row))
            maximum.append(max(row))
        minimum=np.array(minimum)
        maximum=np.array(maximum)
        dataSet=(dataSet-minimum)/(maximum-minimum)
        train=True        
        f=open(self.name,'wb')
        pickle.dump(dataSet,f)
        pickle.dump(minimum,f)
        pickle.dump(maximum,f)
        pickle.dump(label,f)
        f.close
    def classify(self,inputVect,k):
        try:
            f=open(self.name,'rb')
        except:
            raise Exception(f"{name} dosent exist")
        dataSet=pickle.load(f)
        minimum=pickle.load(f)
        maximum=pickle.load(f)
        label=pickle.load(f)
        if(k>label.shape[0]):
            raise Exception(f"k={k} cant be greater than no. of dataset entries({label.shape[0]})")
        inputVect=np.array(inputVect)
        dist=(inputVect-dataSet)**2
        dist=(dist.sum(axis=1))**0.5
        dist=zip(dist,label)
        dist=sorted(dist,key=operator.itemgetter(1))
        sort={}
        for i in range(k):
            if dist[i][1] not in sort.keys():
                sort[dist[i][1]]=0
            sort[dist[i][1]]+=1
        sort=sorted(sort.items(),key=operator.itemgetter(1),reverse=True)
        return sort[0][0]

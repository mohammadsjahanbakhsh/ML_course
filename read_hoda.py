from cv2 import resize
import numpy as np
from scipy import io
from pandas import DataFrame , concat

def load(path="./dataset/handwrite_farsi/Data_hoda_full.mat",size=25,return_X_y=True,save_csv=False):
    dataset = io.loadmat(path)
    X = np.array([resize(img,dsize=(size,size)) for img in np.squeeze(dataset["Data"])])
    
    y = np.squeeze(dataset['labels'])
    
    if return_X_y:
        return X , y
    elif save_csv:
        size1=X.shape[0]
        
        concat([DataFrame(X.resize(size1,size**2),columns=range(size**2))] ,DataFrame(y ,columns=["label"]) , axis=1).to_csv("hoda.csv",index=False)
        
import numpy as np

def sigmoid(x):
    s=np.array(x)
    return(1/(1+np.exp(-s)))
    # Write code here
   
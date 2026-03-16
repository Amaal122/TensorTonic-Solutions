import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    #ite code here Wr
    st=np.array(s)
    wt=np.array(w)
    gt=np.array(g)
    snew=beta*st+(1-beta)*(gt**2)
    wnew=wt-((lr*gt)/(np.sqrt(snew+eps)))
    return(wnew,snew)
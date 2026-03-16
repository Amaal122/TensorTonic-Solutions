import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    mt=np.array(m)
    vt=np.array(v)
    gt=np.array(grad)
    mnew=beta1*mt+(1-beta1)*gt
    vnew=beta2*vt+(1-beta2)*(gt**2)
    mes=mnew/(1-(beta1**t))
    ves=vnew/(1-(beta2**t))
    paramnew=param-lr*(mes/((np.sqrt(ves))+eps))
    
    return(paramnew ,mnew,vnew)
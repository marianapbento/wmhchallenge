from numpy import *
#import matplotlib.pyplot as plt
import nibabel as nib
from numpy import *
import siamxt
import os
from scipy.ndimage.morphology import grey_erosion,binary_dilation

def ianormalize(f, range=[0,255]):
    f = asarray(f)
    range = asarray(range)
    if f.dtype.char in ['D', 'F']:
        raise Exception, 'error: cannot normalize complex data'
    faux = ravel(f).astype(float)
    minimum = faux.min()
    maximum = faux.max()
    lower = range[0]
    upper = range[1]
    if upper == lower:
        g = ones(f.shape) * maximum
    if minimum == maximum:
        g = ones(f.shape) * (upper + lower) / 2.
    else:
        g = (faux-minimum)*(upper-lower) / (maximum-minimum) + lower
    g = reshape(g, f.shape)
        
    if f.dtype == uint8:
        if upper > 255: 
            raise Exception,'ianormalize: warning, upper valuer larger than 255. Cannot fit in uint8 image'
    g = g.astype(f.dtype) # set data type of result the same as the input image
    return g

def iaconv(f, h):
    f, h = asarray(f), asarray(h,float)
    if len(f.shape) == 1: f = f[newaxis,:]
    if len(h.shape) == 1: h = h[newaxis,:]
    if f.size < h.size:
        f, h = h, f
    g = zeros(array(f.shape) + array(h.shape) - 1)
    if f.ndim == 2:
        H,W = f.shape
        for (r,c) in transpose(nonzero(h)):
            g[r:r+H, c:c+W] += f * h[r,c]

    if f.ndim == 3:
        D,H,W = f.shape
        for (d,r,c) in transpose(nonzero(h)):
            g[d:d+D, r:r+H, c:c+W] += f * h[d,r,c]

    return g

def iabinary(f, k1=1):
    f = asarray(f)
    y = f >= k1            
    return y

def iaisbinary(f):
        
    return f.dtype == bool

def ialimits(f):
    code = f.dtype
    if   code == bool:   y=array([0,1],'bool')
    elif code == uint8:  y=array([0,255],'uint8')
    elif code == uint16: y=array([0,(2**16)-1],'uint16')
    elif code == int32:  y=array([-((2**31)-1),(2**31)-1],'int32')
    elif code == int64:  y=array([-((2**63)-1), (2**63)-1],'int64')
    elif code == float64:y=array([-Inf,Inf],'float64')
    else:
        assert 0,'ialimits: Does not accept this typecode:%s' % code
    return y

def ianeg(f):

    if ialimits(f)[0].astype(int) == (- ialimits(f)[1].astype(int)):
        y = -f
    else:
        y = ialimits(f)[0] + ialimits(f)[1] - f
    y = y.astype(f.dtype)     
    return y

def iaintersec(f1, f2, *args):
        
    y = minimum(f1,f2)
    for f in args:
        y = minimum(y,f)
    return y.astype(f1.dtype)

def iasubm(f1, f2):

    if type(f2) is array:
        assert f1.dtype == f2.dtype, 'Cannot have different datatypes:'
    k1,k2 = ialimits(f1)
    y = clip(f1.astype(int32)-f2, k1, k2)
    y = y.astype(f1.dtype)
    return y

def iase2off(Bc,option='neigh'):

    h,w = Bc.shape
    hc,wc = h/2,w/2
    B = Bc.copy()
    B[hc,wc] = 0  # remove origin
    off = transpose(B.nonzero()) - array([hc,wc])
    if option == 'neigh':
        return off  # 2 columns x n. of neighbors rows
    elif option == 'fw':
        i = off[:,0] * w + off[:,1] 
        return off[i>0,:]  # only neighbors higher than origin in raster order
    elif option == 'bw':
        i = off[:,0] * w + off[:,1] 
        return off[i<0,:]  # only neighbors less than origin in raster order
    else:
        assert 0,'options are neigh, fw or bw. It was %s'% option
        return None
    
def iaadd4dil(f, c):

    if not c:
        return f
    if f.dtype == 'float64':
        y = f + c
    else:
        y = asarray(f,int64) + c
        k1,k2 = ialimits(f)
        y = ((f==k1) * k1) + ((f!=k1) * y)
        y = clip(y,k1,k2)
    a = y.astype(f.dtype)
    return a


def iaseunion(B1, B2):

    if B1.dtype != B2.dtype:
        print 'B1=',B1
        print 'B2=',B2
    assert B1.dtype == B2.dtype, \
        'iaseunion: Cannot have different datatypes: \
        %s and %s' % (str(B1.dtype), str(B2.dtype))
    type1 = B1.dtype
    #if len(B1) == 0: return B2
    if len(B1.shape) == 1: B1 = B1[newaxis,:]
    if len(B2.shape) == 1: B2 = B2[newaxis,:]
    if B1.shape <> B2.shape:
        inf = ialimits(B1)[0]
        h1,w1 = B1.shape
        h2,w2 = B2.shape
        H,W = max(h1,h2),max(w1,w2)
        Hc,Wc = (H-1)/2,(W-1)/2    # center
        BB1,BB2 = asarray(B1),asarray(B2)
        B1, B2  = inf * ones((H,W)), inf *ones((H,W))
        dh1s , dh1e = (h1-1)/2 , (h1-1)/2 + (h1+1)%2 # deal with even and odd dimensions
        dw1s , dw1e = (w1-1)/2 , (w1-1)/2 + (w1+1)%2
        dh2s , dh2e = (h2-1)/2 , (h2-1)/2 + (h2+1)%2
        dw2s , dw2e = (w2-1)/2 , (w2-1)/2 + (w2+1)%2
        B1[ Hc-dh1s : Hc+dh1e+1  ,  Wc-dw1s : Wc+dw1e+1 ] = BB1
        B2[ Hc-dh2s : Hc+dh2e+1  ,  Wc-dw2s : Wc+dw2e+1 ] = BB2
    B = maximum(B1,B2).astype(type1)
    return B


def iamat2set(A):

    if len(A.shape) == 1: A = A[newaxis,:]
    offsets = nonzero(ravel(A).astype(int)-ialimits(A).astype(int)[0])
    if type(offsets) == type(()):
        offsets = offsets[0]        # for compatibility with numarray
    if len(offsets) == 0: return ([],[])
    (h,w) = A.shape
    x = range(2)
    x[0] = offsets/w - (h-1)/2
    x[1] = offsets%w - (w-1)/2
    x = transpose(x)
    CV = x,ravel(A)[offsets]           
    return CV

def iaset2mat(A):

    if len(A) == 2:
        x, v = A
        v = asarray(v)
    elif len(A) == 1:
        x = A[0]
        v = ones((len(x),),bool)
    else:
        raise TypeError, 'Argument must be a tuple of length 1 or 2'
    if len(x) == 0:  return array([0]).astype(v.dtype)
    if len(x.shape) == 1: x = x[newaxis,:]
    dh, dw = abs(x).max(0)
    h,w = (2*dh) + 1, (2*dw) + 1 
    M=ones((h, w)) * ialimits(v)[0]
    offset = x[:,0] * w + x[:,1] + (dh*w + dw)
    M.flat[offset] = v
    M = M.astype(v.dtype)
                    
    return M

def iasetrans(Bi, t):
  
    x,v=iamat2set(Bi)
    Bo = iaset2mat((x+t,v))
    Bo = Bo.astype(Bi.dtype)
                    
    return Bo
    
def iagray(f, TYPE="uint8", k1=None):
    ff = array([0],TYPE)
    kk1,kk2 = ialimits(ff)
    if k1!=None:
        kk2=k1
    if   TYPE == 'uint8'  : y = where(f,kk2,kk1).astype(uint8)
    elif TYPE == 'uint16' : y = where(f,kk2,kk1).astype(uint16)
    elif TYPE == 'int32'  : y = where(f,kk2,kk1).astype(int32)
    elif TYPE == 'int64'  : y = where(f,kk2,kk1).astype(int64)
    elif TYPE == 'float64': y = where(f,kk2,kk1).astype(float64)
    else:
        assert 0, 'type not supported:'+TYPE
    return y

def iasedil(B1, B2):
        
    assert (iaisbinary(B1) or (B1.dtype == int32) or (B1.dtype == int64) or (B1.dtype == float64)) and \
               (iaisbinary(B2) or (B2.dtype == int32) or (B2.dtype == int64) or (B2.dtype == float64)), \
               'iasedil: s.e. must be binary, int32, int64 or float64'
    if len(B1.shape) == 1: B1 = B1[newaxis,:]
    if len(B2.shape) == 1: B2 = B2[newaxis,:]
    if B1.dtype=='bool' and B2.dtype == 'bool':
        Bo = iabinary([0])
    else:
        Bo = array(ialimits(B1)[0]).reshape(1)
        if iaisbinary(B1):
            Bo = array(ialimits(B2)[0]).reshape(1)
            B1 = iagray(B1,B2.dtype,0)
        if iaisbinary(B2):
            Bo = array(ialimits(B1)[0]).reshape(1)
            B2 = iagray(B2,B1.dtype,0)
    x,v = iamat2set(B2)
    if len(x):
        for i in range(x.shape[0]):
            s = iaadd4dil(B1,v[i])
            st= iasetrans(s,x[i])
            Bo = iaseunion(Bo,st)
    return Bo

def iasesum(B, N=1): 
    if N==0:
        if iaisbinary(B): return iabinary([1])
        else:             return array([0],int32) # identity
    NB = B
    for i in range(N-1):
        NB = iasedil(NB,B)
    return NB    
    
def iasecross(r=1):
    B = iasesum( iabinary([[0,1,0],
                           [1,1,1],
                           [0,1,0]]),r)      
    return B
  
def iadil(f, b=None):

    if b is None: b = iasecross()
        
    if len(f.shape) == 1: f = f[newaxis,:]
    h,w = f.shape
    x,v = iamat2set(b)
    if len(x)==0:
        y = (ones((h,w)) * ialimits(f)[0]).astype(f.dtype)
    else:
        if iaisbinary(v):
            v = iaintersec( iagray(v,'int32'),0)
        mh,mw = max(abs(x)[:,0]),max(abs(x)[:,1])
        y = (ones((h+2*mh,w+2*mw)) * ialimits(f)[0]).astype(f.dtype)
        for i in range(x.shape[0]):
            if v[i] > -2147483647:
                y[mh+x[i,0]:mh+x[i,0]+h, mw+x[i,1]:mw+x[i,1]+w] = maximum(
                    y[mh+x[i,0]:mh+x[i,0]+h, mw+x[i,1]:mw+x[i,1]+w], iaadd4dil(f,v[i]))
        y = y[mh:mh+h, mw:mw+w]
            
    return y


def iaserot(B, theta=45, DIRECTION="CLOCKWISE"):
               
    if DIRECTION == "ANTI-CLOCKWISE":
        theta = -theta
    SA = iamat2set(B)
    theta = pi * theta/180
    (y,v)=SA
    if len(y)==0: return iabinary([0])
    x0 = y[:,1] * cos(theta) - y[:,0] * sin(theta)
    x1 = y[:,1] * sin(theta) + y[:,0] * cos(theta)
    x0 = int32((x0 +0.5)*(x0>=0) + (x0-0.5)*(x0<0))
    x1 = int32((x1 +0.5)*(x1>=0) + (x1-0.5)*(x1<0))
    x = transpose(array([transpose(x1),transpose(x0)]))
    BROT = iaset2mat((x,v))   
    return BROT

def iasereflect(Bi):
    Bo = iaserot(Bi, 180)
    return Bo

def iaero(f, b=None):

    if b is None: b = iasecross()
    y = ianeg( iadil( ianeg(f),iasereflect(b)))
    return y


def iaNlut(s,offset):

    H,W = s
    n = H*W
    hi = arange(H).reshape(-1,1)
    wi = arange(W).reshape(1,-1) 
    hoff = offset[:,0]
    woff = offset[:,1]
    h = hi + hoff.reshape(-1,1,1)
    w = wi + woff.reshape(-1,1,1)
    h[(h<0) | (h>=H)] = n
    w[(w<0) | (w>=W)] = n
    Nlut = clip(h * W + w,0,n)
    return Nlut.reshape(offset.shape[0],-1).transpose()

def find_area(area, i):
    lista = []
    while area[i] >= 0:
        lista.append(i)
        i = area[i]
    area[lista] = i
    return i

def gradimg(f):

    if len(f.shape) == 2:                                    # 2D case
        h1 = array([[0,1,0],                                 # Defining the horizontal mask
                    [0,0,0],
                    [0,-1,0]])
        h2 = array([[0,0,0],                                 # Defining the vertical mask
                    [1,0,-1],
                    [0,0,0]])
    
        aux1 = iaconv(f,h1)[1:-1,1:-1].astype(int)        # Make the convolution between horizontal mask and image
        aux2 = iaconv(f,h2)[1:-1,1:-1].astype(int)        # Make the convolution between vertical mask and image
        g = sqrt(aux1**2 + aux2**2)                       # Use the equation to compute the gradient of an image
            
        return g
            
    else:                                                    # 3D case
        h1 = array([[[0,0,0],                                # Defining the horizontal mask 
                     [0,0,0],
                     [0,0,0]],
                    [[0,1,0],
                     [0,0,0],
                     [0,-1,0]],
                    [[0,0,0],
                     [0,0,0],
                     [0,0,0]]])
        h2 = array([[[0,0,0],                                # Defining the vertical mask
                     [0,0,0],
                     [0,0,0]],
                    [[0,0,0],
                     [1,0,-1],
                     [0,0,0]],
                    [[0,0,0],
                     [0,0,0],
                     [0,0,0]]])
        h3 = array([[[0,0,0],                                # Defining the depth mask
                     [0,1,0],
                     [0,0,0]],
                    [[0,0,0],
                     [0,0,0],
                     [0,0,0]],
                    [[0,0,0],
                     [0,-1,0],
                     [0,0,0]]])  
    
        aux1 = iaconv(f,h1)[1:-1,1:-1,1:-1].astype(int)    # Make the convolution between horizontal mask and image 
        aux2 = iaconv(f,h2)[1:-1,1:-1,1:-1].astype(int)    # Make the convolution between vertical mask and image
        aux3 = iaconv(f,h3)[1:-1,1:-1,1:-1].astype(int)    # Make the convolution between depth mask and image
        grad = sqrt(aux1**2 + aux2**2 + aux3**2)              # Use the equation to compute the gradient of an image
        return grad

def iagradm(f, Bdil=None, Bero=None):

    if Bdil is None: Bdil = iasecross()
    if Bero is None: Bero = iasecross()
        
    y = iasubm( iadil(f,Bdil),iaero(f,Bero))
    return y

def iaareaopen(f,a,Bc=iasecross()):
    a = -a
    s = f.shape
    g = zeros_like(f).ravel()
    f1 = concatenate((f.ravel(),array([0])))
    area = -ones((f1.size,), int32)
    N = iaNlut(s, iase2off(Bc))
    pontos = f1.nonzero()[0]
    pontos = pontos[lexsort((arange(0,-len(pontos),-1),f1[pontos]))[::-1]]
    for p in pontos:
        for v in N[p]:
            if f1[p] < f1[v] or (f1[p] == f1[v] and v < p):
                rv = find_area(area, v)
                if rv != p:
                    if area[rv] > a or f1[p] == f1[rv]:
                        area[p] = area[p] + area[rv]
                        area[rv] = p
                    else:
                        area[p] = a
    for p in pontos[::-1]:
        if area[p] >= 0:
            g[p] = g[area[p]]
        else:
            if area[p] <= a:
                g[p] = f1[p]
    return g.reshape(s)

def wmseg(img_path):

    # Structuring elemnt C6
    Bc = zeros((3,3,3),dtype =bool)
    Bc[1,1,:] = True
    Bc[1,:,1] = True
    Bc[:,1,1] = True

    img = nib.load(img_path)
    affine = img.affine 
    sx,sy,sz = img.header['pixdim'][1:4]
    img = img.get_data()
  
    img = grey_erosion(img,footprint = ones((3,3,3),dtype = bool))
    mxt = siamxt.MaxTreeAlpha(img,Bc)
    mxt.areaOpen(250)

    min_vol,max_vol=200000,500000
    volume = mxt.node_array[3,:]*sx*sy*sz
    RR = mxt.computeRR()


    indexes = (volume > min_vol) & (volume < max_vol)
    aux = nonzero(indexes)[0]
    try:
        f = 1.0*RR #(avg_gray*dims[:,0]*dims[:,1])/((var_gray+1-6)*dims[:,2]**2)*RR
        f = f[indexes]
        node = argmax(f)
        node = aux[node]
        node = mxt.getBifAncestor(node)
        f = mxt.recConnectedComponent(node)
        f = binary_dilation(f,ones((3,3,3)))
    except:
        f = mxt.recConnectedComponent(node)
    return f.transpose(2,1,0)

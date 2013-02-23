
from numeric import array,zeros,circconv
import math

import nef.simplenode
class DirectConvolution(nef.simplenode.SimpleNode):
    def __init__(self,name,dimensions,invert_first=False,invert_second=False,pstc_gate=0.01,pstc_input=0):
        self.invert_first=invert_first
        self.invert_second=invert_second
        self.dimension = dimensions
        self.A=[0]*dimensions
        self.B=[0]*dimensions
        self.gate=0
        nef.simplenode.SimpleNode.__init__(self,name)
        self.getTermination('A').setDimensions(dimensions)
        self.getTermination('B').setDimensions(dimensions)
        self.getTermination('gate').setTau(pstc_gate)
        if pstc_input>0:
            self.getTermination('A').setTau(pstc_input)
            self.getTermination('B').setTau(pstc_input)
    def termination_gate(self,value):
        self.gate=value[0]
    def termination_A(self,value):
        if self.invert_first:
            self.A=[value[-i] for i in range(len(value))]
        else:
            self.A=value
    def termination_B(self,value):
        if self.invert_second:
            self.B=[value[-i] for i in range(len(value))]
        else:
            self.B=value
    def origin_C(self):
        if self.gate>0.1:
            return [0]*len(self.A)
        else:
            return circconv(self.A,self.B)
    def reset(self,randomize=False):
        self.A=[0]*len(self.A)
        self.B=[0]*len(self.B)
        self.gate=0
        


# DxD discrete fourier transform matrix            
def discrete_fourier_transform(D):
    m=[]
    for i in range(D):
        row=[]
        for j in range(D):            
            row.append(complex_exp((-2*math.pi*1.0j/D)*(i*j)))
        m.append(row)
    return m

# DxD discrete inverse fourier transform matrix            
def discrete_fourier_transform_inverse(D):
    m=[]
    for i in range(D):
        row=[]
        for j in range(D):            
            row.append(complex_exp((2*math.pi*1.0j/D)*(i*j))/D)
        m.append(row)
    return m

# formula for e^z for complex z
def complex_exp(z):
    a=z.real
    b=z.imag
    return math.exp(a)*(math.cos(b)+1.0j*math.sin(b))

def product(x):
    return x[0]*x[1]

def make_array(net,name,N_per_D,dimensions,quick=True,encoders=[[1,1],[1,-1],[-1,1],[-1,-1]],radius=3):
    return net.make_array(name,N_per_D,(dimensions/2+1)*4,dimensions=2,quick=quick,encoders=encoders,radius=radius)


def output_transform(dimensions):
    ifft=array(discrete_fourier_transform_inverse(dimensions))

    def makeifftrow(D,i):
        if i==0 or i*2==D: return ifft[i]
        if i<=D/2: return ifft[i]+ifft[-i].real-ifft[-i].imag*1j
        return zeros(dimensions)
    ifftm=array([makeifftrow(dimensions,i) for i in range(dimensions/2+1)])
    
    ifftm2=[]
    for i in range(dimensions/2+1):
        ifftm2.append(ifftm[i].real)
        ifftm2.append(-ifftm[i].real)
        ifftm2.append(-ifftm[i].imag)
        ifftm2.append(-ifftm[i].imag)
    ifftm2=array(ifftm2)

    return ifftm2.T

def input_transform(dimensions,first,invert=False):
    fft=array(discrete_fourier_transform(dimensions))

    M=[]
    for i in range((dimensions/2+1)*4):
        if invert: row=fft[-(i/4)]
        else: row=fft[i/4]
        if first:
            if i%2==0:
                row2=array([row.real,zeros(dimensions)])
            else:
                row2=array([row.imag,zeros(dimensions)])
        else:
            if i%4==0 or i%4==3:
                row2=array([zeros(dimensions),row.real])
            else:    
                row2=array([zeros(dimensions),row.imag])
        M.extend(row2)
    return M
    
    
        
               
def make_convolution(self,name,A,B,C,N_per_D,quick=False,encoders=[[1,1],[1,-1],[-1,1],[-1,-1]],pstc_out=0.01,pstc_in=0.01,pstc_gate=0.01,invert_first=False,invert_second=False,mode='default',output_scale=1,radius=3):
    if isinstance(A,str):
        A=self.network.getNode(A)
    if isinstance(B,str):
        B=self.network.getNode(B)
    if isinstance(C,str):
        C=self.network.getNode(C)

    dimensions=C.dimension
    if (B is not None and B.dimension!=dimensions) or (A is not None and A.dimension!=dimensions):
        raise Exception('Dimensions not the same for convolution (%d,%d->%d)'%(A.dimension,B.dimension,C.dimension))
        
    if mode=='direct':
        D=DirectConvolution(name,dimensions,invert_first,invert_second)
        self.add(D)
        D.getTermination('A').setTau(pstc_in)
        D.getTermination('B').setTau(pstc_in)
        D.getTermination('gate').setTau(pstc_gate)
        if A is not None:
            self.connect(A,D.getTermination('A'))
        if B is not None:
            self.connect(B,D.getTermination('B'))
        self.connect(D.getOrigin('C'),C,pstc=pstc_out,weight=output_scale)
    else:
        D=make_array(self,name,N_per_D,dimensions,quick=quick,encoders=encoders,radius=radius)

        A2=input_transform(dimensions,True,invert_first)
        B2=input_transform(dimensions,False,invert_second)
        
        D.addDecodedTermination('A',A2,pstc_in,False)
        D.addDecodedTermination('B',B2,pstc_in,False)
        
        if A is not None:
            self.connect(A,D.getTermination('A'))
        if B is not None:
            self.connect(B,D.getTermination('B'))


        ifftm2=output_transform(dimensions)
        
        self.connect(D,C,func=product,transform=ifftm2*output_scale,pstc=pstc_out)
        
    return D

    

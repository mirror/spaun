import nef
import re
import hrr

class FloatNode(nef.SimpleNode):
    def __init__(self,name,data,time):
        self.time=[]
        self.data=[]
        nef.SimpleNode.__init__(self,name)
        self.data=data
        self.time=time
    def origin_X(self):
        if len(self.time)==0: return [0]
        index=0
        while index<len(self.time) and self.time[index]<self.t_start:
            index+=1
        if index>=len(self.time): index=len(self.time)-1                
        return [self.data[index]]    
class VectorNode(nef.SimpleNode):
    def __init__(self,name,data,time):
        self.time=[]
        self.data=[]
        nef.SimpleNode.__init__(self,name)
        self.data=data
        self.time=time
    def origin_X(self):
        if len(self.time)==0: return [0]
        index=0
        while index<len(self.time) and self.time[index]<self.t_start:
            index+=1
        if index>=len(self.time): index=len(self.time)-1                
        return self.data[index]
class HRRNode(nef.SimpleNode):
    def __init__(self,name,data,vocab,time):
        self.vocab=vocab
        self.time=[]
        self.data=[]
        nef.SimpleNode.__init__(self,name)
        self.data=data
        self.time=time
        self.dimension=self.vocab.dimensions
    def origin_X(self):
        data=[0]*self.vocab.dimensions
    
        if len(self.time)==0: return data
        index=0
        while index<len(self.time) and self.time[index]<self.t_start:
            index+=1
        if index>=len(self.time): index=len(self.time)-1      
        
        if isinstance(self.data[index],float):
            return [self.data[index]]*self.vocab.dimensions
        
        
        for weight,key in self.data[index]:
            data[self.vocab.keys.index(key)]=weight
        return data
            
        text='+'.join(['%1.2f*%s'%x for x in self.data[index]])
                  
        return self.vocab.parse(text).v
        
        


class LogParser:
    def __init__(self,filename):
        self.filename=filename
        self.vocab=[]
        self.data=[self.parse_line(x) for x in file(filename).readlines()]
        self.data=[x for x in self.data if len(x)>0]
        self.time=[d[0] for d in self.data[1:]]
        self.vocab.sort()
        
    def parse_line(self,line):
        line=line.strip()
        if len(line)==0: return []
        if line[0]=='#': return []
        
        
        data=[]
        for d in line.split(','):
            if len(d)==0:
                continue
            elif d[0]=='[' and d[-1]==']':
                x=[float(xx) for xx in d[1:-1].split('; ')]
            else:
                try:
                    x=float(d)
                except:
                    x=[]
                    for v in d.split('+'):
                        m=re.match(r'(\d+\.\d+)(.*)',v)
                        if m is None:
                            x.append(v)
                        else:
                            term=m.group(2)
                            term=term.replace("'",'').replace('*','_')
                            if term not in self.vocab: self.vocab.append(term)
                            x.append((float(m.group(1)),term))
            data.append(x)            
        return data
            
    def columns(self):
        return self.data[0]
    
    def create_network(self,name=None):
        if name is None: name=self.filename
        net=nef.Network(name)
        
        v=hrr.Vocabulary(len(self.vocab)+1,randomize=False)
        for term in self.vocab:
            v.parse(term.replace('*','_'))
        
        for i,n in enumerate(self.columns()):
            data=[d[i] for d in self.data[1:]]
            n=n[0]
            if isinstance(data[-1],float):
                net.add(FloatNode(n,data,self.time))
            elif isinstance(data,list):
                if isinstance(data[-1][0],float):
                    net.add(VectorNode(n,data,self.time))
                elif isinstance(data[-1][0],tuple):
                    net.add(HRRNode(n,data,v,self.time))
                else:
                    print 'unknown data',n,data[10]
        return net            
            
        
        
log=LogParser('log_M(6, 0)_111109132106.csv')
net=log.create_network()
net.view()        
        
        

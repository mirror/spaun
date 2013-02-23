import nef
import spa
import numeric

from ca.nengo.model import Units
from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl
from ca.nengo.model.impl import EnsembleTermination
from ca.nengo.math.impl import PostfixFunction
from ca.nengo.model.impl import FunctionInput

from selector import Selector
from cleanup_mem import CleanupMem
from util_funcs import *
from random import *
from math import *
from copy import deepcopy

from java.lang.System.err import println


class VisualHeirachy(NetworkImpl):
    def __init__(self, name = "Vision", file_path = "", in_dim = 28*28, tau_psc = 0.0001, \
                 mu_filename = "", output_vecs = None, mut_inhib_scale = 2, net = None, \
                 en_bypass = False, en_norm = False, en_neuron_rep = False, \
                 sim_mode = SimulationMode.DEFAULT, sim_mode_cleanup = None, quick = True):
        
        if( str(sim_mode).lower() == 'ideal' ):
            sim_mode = SimulationMode.DIRECT
        if( sim_mode == SimulationMode.DIRECT ):
            N = 1
            N2 = 1
        else:
            N  = 3  # Number of neurons for layer 1 to 3
            N2 = 10 # Number of neurons for layer 4
        N3 = 20 # Number of neurons (per dim) for layer N
        NN = 20

        R1 = 7#50#10
        R2 = 7#20#2
        R3 = 7#15#5
        R4 = 1.5#2
        RN = 1.5

        I1 = (-0.3,1)#(-0.1,0.5)
        I2 = (-0.3,1)#(-0.125,0.75)
        I3 = (-0.3,1)#(-0.15,0.8)
        I4 = (-1,1)
        IN = (-1,1)

        E1 = [[1]]
        E2 = [[1]]
        E3 = [[1]]
        E4 = None
        EN = None
        
        NetworkImpl.__init__(self)
        if( net is None ):
            net = nef.Network(self, quick)
        self.setName(name)

        def transform(x):
            return 1.0/(1 + exp(-x[0]))

#        params = dict(max_rate = (100,200), quick = quick, encoders=[[1]], intercept=(0,0.8), mode = sim_mode)
#        params = dict(max_rate = (50,100), quick = quick, encoders=[[1]], intercept=(-0.1,0.8), mode = sim_mode)

#        params = dict(max_rate = (50,60), mode = sim_mode, tau_ref=0.005)
        params = dict(max_rate = (50,60), mode = SimulationMode.DIRECT, tau_ref=0.005)   ## HARDCODED DIRECT SIM MODE

        in_ens = net.make('Input', 1, in_dim)
        net.network.exposeTermination(in_ens.addDecodedTermination("Input", eye(in_dim), 0.0001, False), "Input")
        in_ens.setMode(SimulationMode.DIRECT)
        in_ens.fixMode()

        w1 = read_csv(file_path + 'mat_1_w.csv')
        b1 = read_csv(file_path + 'mat_1_b.csv')

        layer1 = net.make_array('layer1', N, len(w1[0]), radius = R1, intercept = I1, encoders = E1, **params)
        bias1  = net.make_input('bias1', b1[0])
        net.connect(bias1, layer1)
        net.connect(in_ens, layer1, transform = numeric.array(w1).T, pstc = tau_psc)

        w2 = read_csv(file_path + 'mat_2_w.csv')
        b2 = read_csv(file_path + 'mat_2_b.csv')

        layer2 = net.make_array('layer2', N, len(w2[0]), radius = R2, intercept = I2, encoders = E2, **params)
        bias2  = net.make_input('bias2', b2[0])
        net.connect(bias2, layer2)
        net.connect(layer1, layer2, func = transform, transform = numeric.array(w2).T, pstc = tau_psc)

        w3 = read_csv(file_path + 'mat_3_w.csv')
        b3 = read_csv(file_path + 'mat_3_b.csv')

        layer3 = net.make_array('layer3', N, len(w3[0]), radius = R3, intercept = I3, encoders = E3, **params)
        bias3  = net.make_input('bias3', b3[0])
        net.connect(bias3, layer3)
        net.connect(layer2, layer3, func = transform, transform = numeric.array(w3).T, pstc = tau_psc)

        w4 = read_csv(file_path + 'mat_4_w.csv')
        b4 = read_csv(file_path + 'mat_4_b.csv')

        layer4 = net.make_array('layer4', N2, len(w4[0]), radius = R4, intercept = I4, encoders = E4, **params)
        bias4  = net.make_input('bias4', b4[0])
        net.connect(bias4, layer4)
        net.connect(layer3, layer4, func = transform, transform = numeric.array(w4).T, pstc = tau_psc)

        if( en_norm ):
            if( sim_mode == SimulationMode.DIRECT ):
                sim_mode_N = SimulationMode.RATE
            else:
                sim_mode_N = sim_mode
    #        layerN = net.make('layerN', N3 * len(w4[0]), len(w4[0]), max_rate = (50,60), radius = RN, intercept = IN, encoders = EN, quick = quick, mode = sim_mode_N)
            layerN = net.make('layerN', N3 * len(w4[0]), len(w4[0]), radius = RN, intercept = IN, encoders = EN, quick = quick, **params)
            net.connect(layer4, layerN, pstc = tau_psc)
            layerN.fixMode()
        else:
            layerN = layer4
        
        if( en_neuron_rep ):
            layerNeur = net.make_array('layerNeur', NN * len(w4[0]), len(w4[0]), \
                                       radius = RN, intercept = IN, encoders = EN, quick = quick, \
                                       max_rate = (100,200), mode = SimulationMode.DEFAULT, tau_ref=0.005)
            net.connect(layer4, layerNeur, pstc = tau_psc)
            layerNeur.fixMode()

        if( output_vecs is None or mu_filename == "" ):
            net.network.exposeOrigin(layerN.getOrigin("X"), "X")
            self.dimension = len(w4[0])
        else:
            if( sim_mode_cleanup is None ):
                sim_mode_cleanup = sim_mode
            
            visual_am = make_VisHeir_AM(net, "Vision Assoc Mem", file_path, mu_filename, output_vecs, \
                                        mut_inhib_scale, sim_mode_cleanup, quick)

            net.connect(layerN.getOrigin("X"), visual_am.getTermination("Input"))
            if( en_bypass ):
                net.network.exposeOrigin(layerN.getOrigin("X"), "Vis Raw")
    
            net.network.exposeOrigin(visual_am.getOrigin("X"), "X")
            self.dimension = len(output_vecs[0])

        self.setMode(sim_mode)
        if( sim_mode == SimulationMode.DIRECT):
            self.fixMode()
        self.releaseMemory()


    def releaseMemory(self):
        for node in self.getNodes():
            if( not isinstance(node, FunctionInput) ):
                node.releaseMemory()


def make_VisHeir_AM(net = None, name = "Vision Assoc Mem", file_path = "", mu_filename = "", \
                    output_vecs = None, mut_inhib_scale = 2, sim_mode = SimulationMode.DEFAULT, \
                    en_X_out = False, quick = True):

        mu = read_csv(file_path + mu_filename)
        if( len(mu) != len(output_vecs) ):
            print("Warning - VisualHeirachy: Number of items mismatch between mu and output_vecs\n")
        num_items = min(len(mu), len(output_vecs))
        mu_vecs   = mu[0:num_items]
        out_vecs  = output_vecs[0:num_items]

        ## Preset thresholds. Modify as necessary
        threshold = [0.6] * num_items
        threshold[1]  = 0.2
        threshold[2]  = 0.4
        threshold[5]  = 0.4
        threshold[6]  = 0.2
        threshold[9]  = 0.5
        if( num_items > 10 ):
            threshold[10] = 0.8
        if( num_items > 20 ):
            threshold[22] = 0.8
            threshold[23] = 0.4

        visual_am = CleanupMem(name, mu_vecs, out_vecs, en_mut_inhib = True, \
                               mut_inhib_scale = mut_inhib_scale, \
                               threshold = threshold, en_X_out = en_X_out, \
                               sim_mode = sim_mode, quick = quick)
        try:
            net.add(visual_am)
        except:
            net.addNode(visual_am)
        return visual_am           

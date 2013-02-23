import nef

from ca.nengo.model import Units
from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl
from ca.nengo.model.impl import EnsembleTermination
from ca.nengo.math.impl import IndicatorPDF
from ca.nengo.math.impl import PostfixFunction
from ca.nengo.math import PDFTools

from util_nodes import *
from util_funcs import *
from math import *
from copy import deepcopy
from random import *
from java.lang.System.err import println


class Detector(NetworkImpl):
    def __init__(self, name = "Detector", detect_vec = None, inhib_vec = None, tau_in = 0.005, \
                 en_inhib = False, en_inhibN = None, tau_inhib = 0.005, in_scale = 1.0, inhib_scale = 2.0,\
                 en_out = True, en_N_out = False, en_X_out = False, num_neurons = 20, detect_threshold = 0.4, \
                 sim_mode = SimulationMode.DEFAULT, quick = True, rand_seed = 0, net = None, input_name = "Input"):
   
        self.dimension = 1
        NetworkImpl.__init__(self)
        ens_name = name
        if( not isinstance(net, nef.Network) ):
            if( not net is None ):
                net = nef.Network(net, quick)
            else:
                ens_name = "detect"
                net = nef.Network(self, quick)
        self.setName(name)

        if( detect_vec is None ):
            detect_vec = [1]

        vec_dim = len(detect_vec)
        detect_vec_scale = [detect_vec[n] * in_scale for n in range(vec_dim)]
        if( en_inhib ):
            if( inhib_vec is None ):
                inhib_vec = [1]
            inhib_dim = len(inhib_vec)
        if( en_inhibN is None ):
            en_inhibN = en_inhib

        max_rate  = (100,200)
        max_rateN = (300,400)
        detect_threshold = max(min(detect_threshold, 0.8), 0.2)
        intercepts  = [detect_threshold + n * (1-detect_threshold)/(num_neurons) for n in range(num_neurons)]
        interceptsN = [-(n * (detect_threshold)/(num_neurons)) for n in range(num_neurons)]
        params  = dict(intercept = intercepts , max_rate = max_rate , quick = quick)
        paramsN = dict(intercept = interceptsN, max_rate = max_rateN, quick = quick)

        out_func  = FilteredStepFunction(shift = detect_threshold, mirror = False)
        out_funcN = FilteredStepFunction(shift = detect_threshold, mirror = True)
        
        if( rand_seed >= 0 ):
            PDFTools.setSeed(rand_seed)
            seed(rand_seed)

        params["encoders"]  = [[1]] * num_neurons
        paramsN["encoders"] = [[-1]] * num_neurons

        pdf  = IndicatorPDF(detect_threshold + 0.1, 1.1)
        pdfN = IndicatorPDF(-0.1, detect_threshold - 0.1)
        params["eval_points"]  = [[pdf.sample()[0]] for _ in range(1000)]
        paramsN["eval_points"] = [[pdfN.sample()[0]] for _ in range(1000)]
        
        if( en_out ):
            if( sim_mode == SimulationMode.DIRECT or str(sim_mode).lower() == 'ideal' ):
                detect = SimpleNEFEns(ens_name, 1, input_name = "")
                net.add(detect)
            else:
                detect = net.make(ens_name, num_neurons, 1, **params)
            if( not input_name is None ):
                detect.addDecodedTermination(input_name, [detect_vec_scale], tau_in, False)
            if( en_inhib ):
                inhib_vec_scale = [inhib_vec[n] * -inhib_scale for n in range(inhib_dim)]
                detect.addTermination("Inhib", [inhib_vec_scale] * num_neurons, tau_inhib, False)
            
            detect.removeDecodedOrigin("X")
            detect.addDecodedOrigin("X", [out_func], "AXON")

            if( en_X_out ):
                detect.addDecodedOrigin("x0", [PostfixFunction("x0", 1)], "AXON")
                self.exposeOrigin(detect.getOrigin("x0"), "x0")

        if( en_N_out ):
            if( sim_mode == SimulationMode.DIRECT or str(sim_mode).lower() == 'ideal' ):
                detectN = SimpleNEFEns(ens_name + "N", 1, input_name = "")
                net.add(detectN)
            else:
                detectN = net.make(ens_name + "N", num_neurons, 1, **paramsN)
            if( not input_name is None ):
                detectN.addDecodedTermination(input_name, [detect_vec_scale], tau_in, False)
            if( en_inhibN ):
                detectN.addTermination("Inhib", [inhib_vec_scale] * num_neurons, tau_inhib, False)
        
            detectN.removeDecodedOrigin("X")
            detectN.addDecodedOrigin("X", [out_funcN], "AXON")

            if( en_X_out ):
                detectN.addDecodedOrigin("x0", [PostfixFunction("x0", 1)], "AXON")
                self.exposeOrigin(detectN.getOrigin("x0"), "x0N")
                
        input_terms = []
        inhib_terms = []
        
        if( en_out ):
            if( not input_name is None ):
                input_terms.append(detect.getTermination(input_name))
            self.exposeOrigin(detect.getOrigin("X"), name)
            if( en_inhib ):
                inhib_terms.append(detect.getTermination("Inhib"))                
        if( en_N_out ):
            if( not input_name is None ):
                input_terms.append(detectN.getTermination(input_name))
            self.exposeOrigin(detectN.getOrigin("X"), str(name + "N"))
            if( en_inhibN ):
                inhib_terms.append(detectN.getTermination("Inhib"))

        if( len(input_terms) > 0 ):
            input_term = EnsembleTermination(self, input_name, input_terms)
            self.exposeTermination(input_term, input_name)
        if( len(inhib_terms) > 0 ):
            inhib_term = EnsembleTermination(self, "Inhib", inhib_terms)
            self.exposeTermination(inhib_term, "Inhib")

        if( str(sim_mode).lower() == 'ideal' ):
            sim_mode = SimulationMode.DIRECT
        NetworkImpl.setMode(self, sim_mode)
        if( sim_mode == SimulationMode.DIRECT ):
            self.fixMode()

        self.releaseMemory()

            
    def setMode(self, sim_mode):
        if( sim_mode == SimulationMode.DIRECT ):
            sim_mode = SimulationMode.RATE
        NetworkImpl.setMode(self, sim_mode)


    def releaseMemory(self):
        for node in self.getNodes():
            node.releaseMemory()


    def addDecodedTermination(self, name, matrix, tauPsc, isModulatory = False):
        try:
            node = self.getNode(self.getName())
        except:
            node = self.getNode("gate")
        node.addDecodedTermination(name, matrix, tauPsc, isModulatory)
        self.exposeTermination(node.getTermination(name), name)

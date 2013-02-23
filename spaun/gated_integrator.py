import nef
import spa

from ca.nengo.model import Units
from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl
from ca.nengo.model.impl import EnsembleTermination
from ca.nengo.math.impl import PostfixFunction
from ca.nengo.util  import MU

from util_nodes import *
from util_funcs import *
from math import *
from copy import deepcopy
from random import *
from threshold_detect import Detector
from selector import Selector
from cleanup_mem import CleanupMem

class GatedIntNode(nef.SimpleNode):
    def __init__(self, name, num_dim, tau_in, en_reset = False, reset_vec = None, en_cyc_in = True, cyc_opt = 0):
        self.stored_val = zeros(1, num_dim)
        self.input_terms = []
        self.input_mats = dict()
        self.dimension = num_dim
        
        if( not reset_vec is None ):
            self.reset_vec = [reset_vec[n] for n in range(num_dim)]
        else:
            self.reset_vec = zeros(1, num_dim)

        self.cyc_opt = cyc_opt

        nef.SimpleNode.__init__(self, name)
        self.addDecodedTermination("Input", eye(num_dim), tau_in, False)
        self.getTermination("Cycle").setTau(0.005)
        self.getTermination("Reset").setTau(0.005)
        if( not en_reset ):
            self.removeTermination("Reset")
        if( en_cyc_in ):
            self.removeTermination("CycleN")

    def addDecodedTermination(self, name, matrix, tauPsc, isModulatory = False):
        nef.SimpleNode.create_termination(self, name, self.template_Termination)
        self.getTermination(name).setTau(tauPsc)
        self.getTermination(name).setDimensions(self.dimension)
        self.input_terms.append(name)
        if( isScalarMatrix(matrix) ):
            self.input_mats[name] = matrix[0][0]
        else:
            self.input_mats[name] = matrix
        return self.getTermination(name)

    def reset(self, randomize = False):
        self.stored_val = zeros(1, self.dimension)
        nef.SimpleNode.reset(self, randomize)

    def origin_X(self):
        return self.stored_val
    
    def template_Termination(self, x):
        return

    def termination_Reset(self, x):
        self.reset_val = x[0]
        if( self.reset_val > 0.5 ):
            self.stored_val = self.reset_vec
   
    def termination_Cycle(self, x):
        x = x[0]
        if( self.cyc_opt ):
            x = 1 - x
        if( x < 0.025 ):
            if( self.reset_val < 0.5 ):
                input_total = zeros(1, self.dimension)
                for term_name in self.input_terms:
                    termination = self.getTermination(term_name)
                    term_matrix = self.input_mats[term_name]
                    term_output = termination.getOutput()
                    if( isinstance(term_matrix, (int,float,long)) ):
                        input_total = [input_total[n] + term_matrix * term_output[n] for n in range(self.dimension)]
                    else:
                        #term_value = numeric.dot(numeric.array(term_output, typecode='f'), self.input_mats[term_name])
                        term_value  = MU.prod(self.input_mats[term_name], term_output)
                        input_total = [input_total[n] + term_value[n] for n in range(self.dimension)]
                self.stored_val = deepcopy(input_total)

    def termination_CycleN(self, x):
        # Dummy termination... doesn't actually do anything
        return

    def releaseMemory(self):
        pass


class GatedInt(NetworkImpl):
    def __init__(self, name = "Gated Integrator", num_dim = 1, neurons_per_dim = 25, \
                 tau_fb = 0.05, tau_in = 0.010, tau_buf_in = 0.01, tau_inhib = 0.005, \
                 in_scale = 1.0, fb_scale = 1.00, inhib_scale = 2.0, input_name = "Input", \
                 en_reset = False, reset_vec = None, en_cyc_in = True, cyc_opt = 0, \
                 sim_mode = SimulationMode.DEFAULT, quick = True, mode = 1, rand_seed = 0, \
                 cleanup_vecs = None):
   
        self.dimension = num_dim
        NetworkImpl.__init__(self)
        self.setName(name)
        
        if( rand_seed >= 0 ):
            seed(rand_seed)

        if( not reset_vec is None ):
            en_reset = True
            self.init_opt = True
        else:
            self.init_opt = False

        if( str(sim_mode).lower() == 'ideal' ):
            node = GatedIntNode(name, num_dim, tau_in, en_reset, reset_vec, en_cyc_in, cyc_opt)
            self.addNode(node)
            if( not(input_name is None or input_name == "") ):
                self.exposeTermination(node.getTermination("Input"), input_name)
            else:
                node.removeTermination("Input")
            self.exposeTermination(node.getTermination("Cycle"), "Cycle")
            if( en_reset ):
                self.exposeTermination(node.getTermination("Reset"), "Reset")
            if( not en_cyc_in ):
                self.exposeTermination(node.getTermination("CycleN"), "CycleN")
            self.exposeOrigin(node.getOrigin("X"), "X")

            ## TODO
            if( cleanup_vecs is None ):
                print("GINT - Cleanupvecs not implemented yet")
        else:
            net = nef.Network(self, quick)
            nn_per_dim = neurons_per_dim

            if( mode == 1 ):
                radius = 1/sqrt(num_dim) * 3.5
            else:
                radius = 1
            if( mode == -1 ):
                eval_points = [[1 - random() * 0.6 + 0.15] for _ in range(2000)]
                encoders    = [[1]]
                intercept   = (0.25,1)
            else:
                eval_points = None
                encoders    = None
                intercept   = (-1,1)

            params = dict(max_rate = (100,200), radius = radius, quick = quick, \
                          intercept = intercept, encoders = encoders, eval_points = eval_points)
            
            if( sim_mode == SimulationMode.DIRECT ):
                inhib_mat = [[-inhib_scale]]
                if( cleanup_vecs is None ):
                    buffer = SimpleNEFEns("buffer", num_dim, input_name = "")
                else:
                    buffer = CleanupMem("buffer", cleanup_vecs, num_neurons_per_vec = 1, \
                                        tau_in = tau_buf_in, tau_inhib = tau_inhib, \
                                        en_mut_inhib = True, inhib_scale = inhib_scale, \
                                        en_inhib = en_reset and not self.init_opt, \
                                        threshold = 0.5, sim_mode = sim_mode)
                feedback   = SimpleNEFEns("feedback", num_dim, input_name = "")
                net.add(buffer)
                net.add(feedback)
            else:
                inhib_mat = [[[-inhib_scale]] * nn_per_dim] * num_dim
                if( cleanup_vecs is None ):
                    buffer = net.make_array("buffer", nn_per_dim, num_dim, 1, **params)
                else:
                    buffer = CleanupMem("buffer", cleanup_vecs, num_neurons_per_vec = nn_per_dim, \
                                        tau_in = tau_buf_in, tau_inhib = tau_inhib, \
                                        en_mut_inhib = True, inhib_scale = inhib_scale, \
                                        en_inhib = en_reset and not self.init_opt, threshold = 0.5, \
                                        sim_mode = sim_mode, rand_seed = rand_seed, quick = quick)
                    net.add(buffer)
                feedback   = net.make_array("feedback", nn_per_dim, num_dim, 1, **params)
            
            if( cleanup_vecs is None ):
                buffer.addDecodedTermination("Input", eye(num_dim), tau_buf_in, False)
            buffer.addDecodedTermination("Feedback", eye(num_dim), 0.005, False)
            if( en_reset and not self.init_opt ):
                if( cleanup_vecs is None ):
                    buffer.addTermination("Inhib", inhib_mat, tau_inhib, False)
                net.network.exposeTermination(buffer.getTermination("Inhib"), "Reset")

            feedback.addDecodedTermination("Input", diag(num_dim, value = fb_scale), tau_fb, False)
            feedback.addTermination("Inhib", inhib_mat, tau_inhib, False)
            
            if( input_name is None or input_name == "" ):
                self.num_inputs = 0
            else:
                self.num_inputs = 1

            if( not self.init_opt ):
                if( sim_mode == SimulationMode.DIRECT ):
                    gate = SimpleNEFEns("gate"  , num_dim, input_name = "")
                    net.add(gate)
                else:
                    gate = net.make_array("gate", nn_per_dim, num_dim, 1, **params)
                if( self.num_inputs ):
                    gate.addDecodedTermination("Input", diag(num_dim, value = in_scale), tau_in, False)
                    net.network.exposeTermination(gate.getTermination("Input"), input_name)
                gate.addTermination("Inhib", inhib_mat, tau_inhib, False)
                gate_inhib_name = "Inhib"
            else:
                gate = Selector("gate", num_dim, nn_per_dim, num_dim, tau_in = [0.005,tau_in], in_scale = in_scale, \
                                inhib_scale = inhib_scale, **params)
                gate.addSuppressTerminations([1])
                feedback.addTermination("Reset", inhib_mat, 0.005, False)
                reset_net = Detector("Reset", en_N_out = True, sim_mode = sim_mode, rand_seed = rand_seed)
                net.add(reset_net)
                net.add(gate)
                net.network.exposeTermination(reset_net.getTermination("Input"), "Reset")
                if( self.num_inputs ):
                    net.network.exposeTermination(gate.getTermination("Input 1"), input_name)
                init_val_in = net.make_input("init_val", reset_vec)
                net.connect(init_val_in                  , gate.getTermination("Input 2"))
                net.connect(reset_net.getOrigin("Reset") , gate.getTermination("Suppress 1_2"))
                net.connect(reset_net.getOrigin("ResetN"), gate.getTermination("Suppress 2"))
                net.connect(reset_net.getOrigin("Reset") , feedback.getTermination("Reset"))
                gate_inhib_name = "Suppress 1"

            net.connect(gate.getOrigin("X")    , buffer.getTermination("Input"))
            net.connect(buffer.getOrigin("X")  , feedback.getTermination("Input"))
            net.connect(feedback.getOrigin("X"), buffer.getTermination("Feedback"))

            net.network.exposeOrigin(buffer.getOrigin("X"), "X")

            if( cyc_opt ):
                gate_inhib_str = ("CycleN")
                fb_inhib_str = ("Cycle")
            else:
                gate_inhib_str = ("Cycle")
                fb_inhib_str = ("CycleN")
                
            if( en_cyc_in ):
                cyc_net  = Detector("Cycle", en_N_out = True, sim_mode = sim_mode, rand_seed = rand_seed)
                net.add(cyc_net)
                net.connect(cyc_net.getOrigin(gate_inhib_str), gate.getTermination(gate_inhib_name))
                net.connect(cyc_net.getOrigin(fb_inhib_str)  , feedback.getTermination("Inhib"))
                net.network.exposeTermination(cyc_net.getTermination("Input"), "Cycle")
            else:
                net.network.exposeTermination(gate.getTermination(gate_inhib_name), gate_inhib_str)
                net.network.exposeTermination(feedback.getTermination("Inhib")    , fb_inhib_str)

            self.releaseMemory()

        if( str(sim_mode).lower() == 'ideal' ):
            sim_mode = SimulationMode.DIRECT
        NetworkImpl.setMode(self, sim_mode)
        if( sim_mode == SimulationMode.DIRECT ):
            self.fixMode()

            
    def releaseMemory(self):
        for node in self.getNodes():
            try:
                node.releaseMemory()
            except:
                pass


    def addDecodedTermination(self, name, matrix, tauPsc, isModulatory = False):
        try:
            node = self.getNode(self.getName())
        except:
            node = self.getNode("gate")
        
        if( self.init_opt ):
            if( self.num_inputs ):
                node.addDecodedTerminations([1], [matrix], [tauPsc], [isModulatory])
                self.exposeTermination(node.getTermination("Input 1_" + str(self.num_inputs+1)), name)
            else:
                self.exposeTermination(node.getTermination("Input 1", name))
            self.num_inputs += 1
        else:
            node.addDecodedTermination(name, matrix, tauPsc, isModulatory)
            self.exposeTermination(node.getTermination(name), name)
        return self.getTermination(name)

    
    def addAxonOrigin(self):
        if( self.getMode() != SimulationMode.DIRECT ):
            self.getNode("buffer").createEnsembleOrigin("AXON")
            self.exposeOrigin(self.getNode("buffer").getOrigin("AXON"), "AXON")


    def setMode(self, sim_mode):
        if( sim_mode == SimulationMode.DIRECT ):
            sim_mode = SimulationMode.RATE
        NetworkImpl.setMode(self, sim_mode)


class GatedIntModule(spa.module.Module):
    def create(self, dimensions = 1, N_per_D = 25, \
               tau_fb = 0.05, tau_in = 0.01, tau_inhib = 0.005, \
               in_scale = 1.0, fb_scale = 1.0, inhib_scale = 2.0, \
               en_reset = False, en_cyc_in = True, cyc_opt = 0, \
               sim_mode = SimulationMode.DEFAULT, quick = True, hrrMode = True):

        gint = GatedInt("GINT", dimensions, N_per_D, tau_fb, tau_in, tau_inhib, in_scale, \
                        fb_scale, inhib_scale, en_reset, en_cyc_in, cyc_opt, sim_mode, quick, hrrMode)
        self.net.add(gint)

        self.add_source(gint.getOrigin("X"))
        self.add_sink(gint)

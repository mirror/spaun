import numeric
import nef
import spa

from ca.nengo.model import Units
from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl
from ca.nengo.model.impl import EnsembleTermination
from ca.nengo.math.impl import PostfixFunction

from gated_integrator import GatedInt
from util_funcs import *
from math import *
from copy import deepcopy
from threshold_detect import Detector

class MemBlockNode(nef.SimpleNode):
    def __init__(self, name, num_dim, tau_in = 0.005, reset_opt = 0, reset_vec = None, cyc_opt = 0, en_gint_out = False, in_scale = 1):
        self.input_terms = []
        self.input_mats = dict()
        self.MB_val1 = zeros(1, num_dim)
        self.MB_val2 = zeros(1, num_dim)

        self.reset_opt = reset_opt # 0 - no reset, 1 - reset GINT1, 2 - reset GINT2, 3 - reset both
        self.reset_val = 0
        self.reset_vec_opt = not reset_vec is None
        if( reset_vec is None ):
            self.reset_vec = zeros(1, num_dim)
        else:
            self.reset_vec = [reset_vec[n] for n in range(num_dim)]
        self.dimension = num_dim

        self.cyc_opt = cyc_opt # 0 - store in GINT1 when cyc = 0, 1 - store in GINT1 when cyc = 1
        self.cyc_val = 0

        nef.SimpleNode.__init__(self, name)
        self.addDecodedTermination("Input", diag(num_dim, value = in_scale), tau_in)
        self.getTermination("Input").setTau(tau_in)
        self.getTermination("Reset").setTau(tau_in)
        self.getTermination("Cycle").setTau(tau_in)

        if( not en_gint_out ):
            self.removeOrigin("GINT1")
            self.removeOrigin("GINT2")
    
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
        self.MB_val1 = zeros(1, self.dimension)
        self.MB_val2 = zeros(1, self.dimension)
        nef.SimpleNode.reset(self, randomize)

    def template_Termination(self, x):
        return

    def termination_Reset(self, x):
        self.reset_val = x[0]
        if( self.reset_val > 0.9 and self.reset_opt & 1 ):
            self.MB_val1 = self.reset_vec 
        if( self.reset_val > 0.9 and self.reset_opt & 2 ):
            self.MB_val2 = zeros(1, self.dimension)

    def termination_Cycle(self, x):
        x = x[0]
        if( self.cyc_opt ):
            x = 1 - x
        if( x > 0.95 ):
            if( self.reset_val < 0.1 or self.reset_vec_opt ):
                self.MB_val2 = deepcopy(self.MB_val1)
        elif( x < 0.025 ):
            if( self.reset_val < 0.1 ):
                input_total = zeros(1, self.dimension)
                for term_name in self.input_terms:
                    termination = self.getTermination(term_name)
                    term_matrix = self.input_mats[term_name]
                    term_output = termination.getOutput()
                    if( isinstance(term_matrix, (int,float,long)) ):
                        input_total = [input_total[n] + term_matrix * term_output[n] for n in range(self.dimension)]
                    else:
                        # Warning: This is SUPER SLOW!
                        #term_value = numeric.dot(numeric.array(term_output, typecode='f'), self.input_mats[term_name])
                        term_value  = MU.prod(self.input_mats[term_name], term_output)
                        input_total = [input_total[n] + term_value[n] for n in range(self.dimension)]
                self.MB_val1 = deepcopy(input_total)
    
    def origin_X(self):
        return self.MB_val2

    def origin_GINT1(self):
        return self.MB_val1

    def origin_GINT2(self):
        return self.MB_val2

    def releaseMemory(self):
        pass
    

class MemBlock(NetworkImpl):
    def __init__(self, name = "Memory Block", num_dim = 1, neurons_per_dim = 25, tau_in = 0.005, \
                 in_scale = 1.0, fb_scale = 1.00, inhib_scale = 2.0, input_name = "Input", \
                 reset_opt = 0, reset_vec = None, cyc_opt = 0, en_gint_out = False, tau_buf_in = 0.01,\
                 sim_mode = SimulationMode.DEFAULT, quick = True, mode = 1, rand_seed = 0, cleanup_vecs = None):
        # mode: 1 - hrr mode (radius is scaled to num_dim), 0 - normal mode, 
        #      -1 - aligned mode (eval points chosen around 1 and 0)

        self.dimension = num_dim
        self.sim_mode = sim_mode
        NetworkImpl.__init__(self)
        self.setName(name)

        if( not reset_vec is None ):
            reset_opt = reset_opt | 1

        if( str(sim_mode).lower() == 'ideal' ):
            node = MemBlockNode(name, num_dim, tau_in, reset_opt, reset_vec, cyc_opt, en_gint_out, in_scale)
            self.addNode(node)
            if( not(input_name is None or input_name == "") ):
                self.exposeTermination(node.getTermination("Input"), input_name)
            self.exposeTermination(node.getTermination("Cycle"), "Cycle")
            if( not reset_opt == 0 ):
                self.exposeTermination(node.getTermination("Reset"), "Reset")
            self.exposeOrigin(node.getOrigin("X"), "X")
            if( en_gint_out ):
                self.exposeOrigin(node.getOrigin("GINT1"), "GINT1")
                self.exposeOrigin(node.getOrigin("GINT2"), "GINT2")
        else:
            net = nef.Network(self, quick)
            
            gint1 = GatedInt("GINT1", num_dim, neurons_per_dim, in_scale = in_scale, fb_scale = fb_scale, tau_in = tau_in, \
                             inhib_scale = inhib_scale, en_reset = reset_opt & 1, reset_vec = reset_vec, en_cyc_in = False, \
                             cyc_opt = cyc_opt, mode = mode, quick = quick, rand_seed = rand_seed, input_name = input_name, \
                             sim_mode = sim_mode, tau_buf_in = tau_buf_in, cleanup_vecs = cleanup_vecs)
            gint2 = GatedInt("GINT2", num_dim, neurons_per_dim, fb_scale = fb_scale, inhib_scale = inhib_scale, \
                             en_reset = reset_opt & 2, en_cyc_in = False, cyc_opt = cyc_opt, mode = mode, \
                             quick = quick, rand_seed = rand_seed, sim_mode = sim_mode, cleanup_vecs = cleanup_vecs)
            
            net.add(gint1)
            net.add(gint2)
            net.connect(gint1.getOrigin("X"), gint2.getTermination("Input"))
        
            if( not(input_name is None or input_name == "") ):
                net.network.exposeTermination(gint1.getTermination("Input"), input_name)
            net.network.exposeOrigin(gint2.getOrigin("X"), "X")

            if( en_gint_out ):
                net.network.exposeOrigin(gint1.getOrigin("X"), "GINT1")
                net.network.exposeOrigin(gint2.getOrigin("X"), "GINT2")
    
            if( reset_opt > 0 ):
                rst_terms = []
                if( reset_opt & 1 ):
                    rst_terms.append(gint1.getTermination("Reset"))
                if( reset_opt & 2 ):
                    rst_terms.append(gint2.getTermination("Reset"))              
                rst_term = EnsembleTermination(net.network, "Reset", rst_terms)
                net.network.exposeTermination(rst_term, "Reset")

            cyc_net  = Detector("Cycle", en_N_out = True, sim_mode = sim_mode, rand_seed = rand_seed)
            net.add(cyc_net)
            net.connect(cyc_net.getOrigin("Cycle") , gint1.getTermination("Cycle"))
            net.connect(cyc_net.getOrigin("Cycle") , gint2.getTermination("CycleN"))
            net.connect(cyc_net.getOrigin("CycleN"), gint1.getTermination("CycleN"))
            net.connect(cyc_net.getOrigin("CycleN"), gint2.getTermination("Cycle"))
            net.network.exposeTermination(cyc_net.getTermination("Input"), "Cycle")

            self.releaseMemory()

        if( str(sim_mode).lower() == 'ideal' ):
            sim_mode = SimulationMode.DIRECT
        self.setMode(sim_mode)
        if( sim_mode == SimulationMode.DIRECT ):
            self.fixMode()

    
    def addDecodedTermination(self, name, matrix, tauPsc, isModulatory = False):
        try:
            node = self.getNode(self.getName())
        except:
            node = self.getNode("GINT1")
        node.addDecodedTermination(name, matrix, tauPsc, isModulatory)
        self.exposeTermination(node.getTermination(name), name)
        return self.getTermination(name)


    def addAxonOrigin(self):
        if( self.getMode() != SimulationMode.DIRECT ):
            self.getNode("GINT2").addAxonOrigin()
            self.exposeOrigin(self.getNode("GINT2").getOrigin("AXON"), "AXON")


    def releaseMemory(self):
        for node in self.getNodes():
            node.releaseMemory()


class MemBlockModule(spa.module.Module):
    def create(self, dimensions = 1, N_per_D = 25, \
               in_scale = 1.0, fb_scale = 1.00, inhib_scale = 2.0, \
               reset_opt = 0, cyc_opt = 0, en_gint_out = False, \
               sim_mode = SimulationMode.DEFAULT, quick = True, hrrMode = True):

        mb = MemBlock("MB", dimensions, N_per_D, in_scale, fb_scale, inhib_scale, reset_opt, cyc_opt, \
                      cyc_opt, en_gint_out, sim_mode, quick, hrrMode)
        self.net.add(mb)

        self.add_source(mb.getOrigin("X"))
        self.add_sink(mb)
            

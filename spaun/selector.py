import nef

from ca.nengo.model import Units
from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl
from ca.nengo.model.impl import EnsembleOrigin
from ca.nengo.model.impl import EnsembleTermination
from ca.nengo.math.impl import PostfixFunction

from util_nodes import *
from util_funcs import *
from random import *
from math import *
from copy import deepcopy


class Selector(NetworkImpl):
    def __init__(self, name = "Selector", num_dim = 1, neurons_per_dim = 25,
                 ens_per_array = 0, num_items = 2, in_scale = 1.0, \
                 tau_in = 0.005, tau_inhib = 0.005, inhib_scale = 2.0, \
                 inhib_vecs = [], en_sum_out = True, \
                 sim_mode = SimulationMode.DEFAULT, quick = True, **params):
        # Note: tau_in and tau_inhib and inhib_scale can be lists.

        self.dimension = num_dim
        self.num_items = num_items
        
        if( ens_per_array == 0 ):
            ens_per_array = 8
        
        if( str(sim_mode).lower() == 'ideal' ):
            sim_mode = SimulationMode.DIRECT

        # Deepcopy inhib_vec list... otherwise append function will affect other classes.
        inhib_vecs = [inhib_vecs[n] for n in range(len(inhib_vecs))] 
        len_diff = num_items - len(inhib_vecs)
        if( len_diff != 0 ):
            if( len(inhib_vecs) > 0 ):
                print(len(inhib_vecs))
                print(str(inhib_vecs))
                print("Selector.__init__ [" + name + "] - inhib_vec length and num_item mismatch")
            for n in range(len_diff):
                inhib_vecs.append([1])
        inhib_dim = len(inhib_vecs[0])

        NetworkImpl.__init__(self)
        net = nef.Network(self, quick)
        self.setName(name)

        self.ens_per_array   = min(num_dim, ens_per_array)
        self.dim_per_ens     = num_dim / self.ens_per_array
        self.neurons_per_ens = neurons_per_dim * self.dim_per_ens

        self.make_mode         = sim_mode

        enss = []
        if( en_sum_out ):
            out_relay = SimpleNEFEns("Output", self.dimension, pstc = 0.0001, input_name = "")
            net.add(out_relay)

        if( not isinstance(tau_in, list) ):
            tau_in = [tau_in] * num_items
        if( not isinstance(tau_inhib, list) ):
            tau_inhib = [tau_inhib] * num_items
        if( not isinstance(inhib_scale, list) ):
            inhib_scale = [inhib_scale] * num_items

        self.inhib_scale = inhib_scale
        self.tau_inhib   = tau_inhib
        self.tau_in      = tau_in

        if( not "max_rate" in params ):
            params["max_rate"] = (100,200)
        if( not "quick" in params ):
            params["quick"] = quick

        for item in range(num_items):
            inhib_vec_scale = [inhib_vecs[item][n] * -inhib_scale[item] for n in range(inhib_dim)]
            if( sim_mode == SimulationMode.DIRECT ):
                ens = SimpleNEFEns("Item " + str(item+1), self.dimension, pstc = tau_in[item], input_name = None)
                net.add(ens)
                inhib_mat = [inhib_vec_scale]
            else:
                ens = net.make_array("Item " + str(item+1), self.neurons_per_ens, self.ens_per_array, \
                                     self.dim_per_ens, **params)
                inhib_mat = [[inhib_vec_scale] * self.neurons_per_ens] * self.ens_per_array
            in_term = ens.addDecodedTermination("Input", diag(num_dim, value = in_scale), tau_in[item], False)
            inhib_term = ens.addTermination("Inhib", inhib_mat, tau_inhib[item], False)
            enss.append(ens)

            net.network.exposeTermination(in_term, "Input " + str(item+1))
            net.network.exposeTermination(inhib_term, "Suppress " + str(item+1))
            if( not en_sum_out ):
                net.network.exposeOrigin(ens.getOrigin("X"), "Output " + str(item+1))
            else:
                out_relay.addDecodedTermination("Item" + str(item+1), None, 0.0001, False)
                net.connect(ens.getOrigin("X"), out_relay.getTermination("Item" + str(item+1)))

        if( en_sum_out ):
            net.network.exposeOrigin(out_relay.getOrigin("X"), "X")

        NetworkImpl.setMode(self, sim_mode)
        if( sim_mode == SimulationMode.DIRECT ):
            self.fixMode()
        self.releaseMemory()


    def setMode(self, sim_mode):
        if( sim_mode == SimulationMode.DIRECT ):
            sim_mode = SimulationMode.RATE
        NetworkImpl.setMode(self, sim_mode)


    def addDecodedTerminations(self, input_list = [], matrix_list = [], pstc_list = [], mod_list = []):
        ret_vals = zeros(1, self.num_items)

        for i,input_num in enumerate(input_list):
            if( input_num <= self.num_items ):
                item_node             = self.getNode("Item " + str(input_num))
                term_list             = item_node.getTerminations()
                term_names            = [term_list[n].name for n in range(len(term_list))]
                num_terms             = sum(["Input" in term_names[n] for n in range(len(term_names))]) + 1
                ret_vals[input_num-1] = num_terms

                if( len(pstc_list) < i ):
                    pstc = pstc_list[i]
                else:
                    pstc = self.tau_in[input_num-1]
                
                if( len(mod_list) < i ):
                    mod  = mod_list[i]
                else:
                    mod  = False

                if( len(matrix_list) < i ):
                    matrix = matrix_list[i]
                else:
                    matrix = eye(self.dimension)
                in_term = item_node.addDecodedTermination("Input" + str(num_terms), matrix, pstc, mod)
                self.exposeTermination(in_term, "Input " + str(input_num) + "_" + str(num_terms))
        # Return the input numbers of the new terminations
        return ret_vals


    def addSuppressTerminations(self, input_list = [], inhib_vecs = []):
        if( len(input_list) == 0 ):
            if( len(inhib_vecs) == 0 ):
                input_list = range(1,self.num_items+1)
            else:
                input_list = range(1,len(inhib_vecs)+1)                

        ret_vals = zeros(1, self.num_items)

        # Deepcopy inhib_vec list... otherwise append function will affect other classes.
        inhib_vecs = [inhib_vecs[n] for n in range(len(inhib_vecs))] 
        len_diff = len(input_list) - len(inhib_vecs)
        if( len_diff != 0 ):
            if( len(inhib_vecs) > 0 ):
                print("Selector.addSuppressTerminations - inhib_vec length and num_item mismatch")
            for n in range(len_diff):
                inhib_vecs.append([1])
        inhib_dim = len(inhib_vecs[0])

        for i,input_num in enumerate(input_list):
            if( input_num <= self.num_items ):
                item_node             = self.getNode("Item " + str(input_num))
                term_list             = item_node.getTerminations()
                term_names            = [term_list[n].name for n in range(len(term_list))]
                num_terms             = sum(["Inhib" in term_names[n] for n in range(len(term_names))]) + 1
                ret_vals[input_num-1] = num_terms

                inhib_vec_scale = [inhib_vecs[i][n] * -self.inhib_scale[input_num-1] \
                                   for n in range(inhib_dim)] 
                if( self.make_mode == SimulationMode.DIRECT ):
                    inhib_mat = [inhib_vec_scale]
                else:
                    inhib_mat =  [[inhib_vec_scale] * self.neurons_per_ens] * self.ens_per_array
                inhib_term = item_node.addTermination("Inhib" + str(num_terms), inhib_mat, \
                                                      self.tau_inhib[input_num-1], False)
                self.exposeTermination(inhib_term, "Suppress " + str(input_num) + "_" + str(num_terms))
        # Return the input numbers of the new terminations
        return ret_vals


    def releaseMemory(self):
        for node in self.getNodes():
            node.releaseMemory()

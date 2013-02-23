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

class SpaunMotorNode(nef.SimpleNode):
    def __init__(self, name, num_dim, num_items = 10, unk_vec = None, go_threshold = 0.9, \
                 motor_init = 0, out_file = "", raw_file = "", motor_valid = None, motor_raw_scale = 15, \
                 pstc = 0.005, pstc_raw = 0.01):
        self.pstc = pstc
        self.pstc_raw = pstc_raw
        self.dimension = num_dim
        self.done = True
        self.out_file = out_file
        self.raw_file = raw_file

        self.num_items = num_items
        self.out_index = -1
        self.out_value = zeros(1,self.dimension)
        
        if( unk_vec is None ):
            self.unk_vec = zeros(1,self.dimension)
        else:
            self.unk_vec = unk_vec

        self.motor_next = -1
        self.motor_init = motor_init
        self.motor_valid = motor_valid
        self.motor_raw_scale = motor_raw_scale

        self.suppress_output = False

        self.go_threshold = go_threshold

        nef.SimpleNode.__init__(self, name)
        self.getTermination("Input").setDimensions(num_dim)
        self.getTermination("Input").setTau(pstc_raw)
        self.getTermination("Index").setDimensions(num_items)
        self.getTermination("Index").setTau(pstc_raw)
        self.getTermination("Go").setDimensions(1)

    def reset(self,randomize=False):
        self.done = True
        nef.SimpleNode.reset(self, randomize)

    def origin_X(self):
        self.done = ((self.t_start >= self.motor_next) or (self.motor_next < 0))
        if( self.suppress_output ):
            return zeros(1,self.dimension)
        else:
            return self.out_value

    def origin_Done(self):
        return [self.done]
    
    def origin_Busy(self):
        return [1 - self.done]
    
    def origin_Plan(self):
        return [self.out_index]
    
    def termination_Suppress(self, x):
        self.suppress_output = (x[0] > 0.5)

    def termination_Reset(self, x):
        self.done = self.done or (x[0] > 0.5)

    def termination_Index(self, x):
        if( sum(x) < 0.5 ):
            self.out_index = -1
        else:
            sum_index = [(x[i] > 0.5) * i for i in range(len(x))]
            self.out_index = sum(sum_index)
            if( self.out_index > self.num_items ):
                self.out_index = -1

    def termination_Input(self, x):
        self.out_value = x
        
    def termination_Go(self, x):
        if( x[0] >= self.go_threshold and self.done ):
            self.write_to_file()
            self.done = False
            self.motor_next = self.t_start + self.motor_init
        return

    def write_to_file(self):
        if( not self.suppress_output ):
            if( self.out_index >= 0 ):
                self.write_str_to_file(self.out_file, str(self.out_index))
                raw_value = self.out_value
            else:
                self.write_str_to_file(self.out_file, "X")
                raw_value = self.unk_vec
            
            # Process motor raw vector out
            # - Scale output by motor_out_scale
            # - Has to be integer
            #raw_value = [int(round(raw_value[i] * (self.motor_raw_scale-1) + 1)) for i in range(self.dimension)]
            if( not self.motor_valid is None and self.out_index >= 0 ):
                raw_valid = self.motor_valid[self.out_index]
                raw_value = raw_value[:raw_valid]
            self.write_str_to_file(self.raw_file, str(raw_value))
            self.write_str_to_file(self.raw_file, "\n")
        return
    
    def write_str_to_file(self, file_name, str_val):
        if( not file_name == "" ):
            file_handle = open(file_name, 'a')
            file_handle.write(str_val)
            file_handle.close()

    def write_debug(self, timestamp = 0, label = "", value = 0):
        file_handle = open(self.out_file + "debug.txt", 'a')
        file_handle.write("%10.8f" % timestamp + ": [" + label + "] " + str(value))
        file_handle.write("\n")
        file_handle.close()    


class TraceMotorTransform(NetworkImpl):
    def __init__(self, name = "MotorTransform", mtr_filepath = "", valid_strs = [], vis_dim = 0, \
                 mtr_dim = 0, neurons_per_dim = 50, inhib_scale = 10.0, tau_in = 0.05, tau_inhib = 0.05, \
                 in_strs = None, quick = True):
        
        NetworkImpl.__init__(self)
        self.setName(name)
        net = nef.Network(self, quick)
        
        self.dimension = mtr_dim
        
        in_terms = []
        inhib_terms = []

        out_relay = SimpleNEFEns("Output", mtr_dim, pstc = 0.0001, input_name = "")
        net.add(out_relay)

        for i,in_str in enumerate(in_strs):
            x_transform = read_csv(mtr_filepath + in_str + "_x.csv")
            y_transform = read_csv(mtr_filepath + in_str + "_y.csv")
            xy_transform = []
            
            transform_w = len(x_transform[0])
            transform_h = len(x_transform)

            for row in range(transform_h):
                xy_transform.append(x_transform[row])
                xy_transform.append(y_transform[row])
            # append ignore rows
            for row in range(mtr_dim - (transform_h * 2)):
                xy_transform.append(zeros(1,transform_w))

            inhib_vec = num_vector(-inhib_scale, 1, len(valid_strs))
            inhib_vec[valid_strs.index(in_str)] = 0

            ens = net.make_array(in_str, neurons_per_dim, mtr_dim, 1, max_rate = (100,200), quick = quick, storage_code = "%d")
            ens.addDecodedTermination("Input", xy_transform, tau_in, False)
            ens.addTermination("Inhib", [[inhib_vec] * neurons_per_dim] * mtr_dim, tau_inhib, False)

            out_relay.addDecodedTermination(in_str, None, 0.0001, False)
            net.connect(ens.getOrigin("X"), out_relay.getTermination(in_str))

            in_terms.append(ens.getTermination("Input"))
            inhib_terms.append(ens.getTermination("Inhib"))

        in_term    = EnsembleTermination(net.network, "Input", in_terms)
        inhib_term = EnsembleTermination(net.network, "Inhib", inhib_terms)

        net.network.exposeTermination(in_term, "Input")
        net.network.exposeTermination(inhib_term, "Inhib")
        net.network.exposeOrigin(out_relay.getOrigin("X"), "X")
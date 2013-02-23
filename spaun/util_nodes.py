import nef
import util_funcs

reload(nef)
reload(util_funcs)

from ca.nengo.model import SimulationMode
from ca.nengo.model import Termination
from ca.nengo.model import Network
from ca.nengo.model.impl import BasicOrigin
from ca.nengo.util  import MU

from java.lang.System.err import println
import datetime
import random
import math
import os.path
import copy

class MemoryInput(nef.SimpleNode):
    def __init__(self, name, vec_val):
        self.v = vec_val
        self.dimension = len(self.v)
        nef.SimpleNode.__init__(self, name)
    
    def origin_X(self):
        return self.v


class MemorySelInput(nef.SimpleNode):
    def __init__(self, name, vecs, sel_vals, threshold = 0.1, tau_selval = 0.005, pstc = 0.005):
        self.pstc = pstc
        self.vecs = vecs
        self.sel_vals = sel_vals
        self.threshold = threshold  
        self.pstc = tau_selval
        for n in range(len(self.sel_vals)-1):
            self.threshold = min(abs(self.sel_vals[n]-self.sel_vals[n+1]) / 2.0, self.threshold)
        self.dimension = len(self.vecs[0])
        self.out_sel = -1
        nef.SimpleNode.__init__(self, name)
            
    def termination_SelVal(self, x):
        out_sel = -1
        for n in range(len(self.sel_vals)):
            if( x[0] <= (self.sel_vals[n] + self.threshold) and \
                x[0] >= (self.sel_vals[n] - self.threshold) ):
                out_sel = n
        self.out_sel = out_sel

    def origin_X(self):
        if( self.out_sel >= 0 ):
            return self.vecs[self.out_sel]
        else:
            return util_funcs.zeros(1, self.dimension)


class CycleInput(nef.SimpleNode):
    def __init__(self, name, interval, pstc = 0.005):
        self.pstc = pstc
        self.dimension = 1
        self.interval = interval
        self.next_int = interval
        self.v = 0
        nef.SimpleNode.__init__(self, name)
    
    def origin_X(self):
        if( self.t_start >= self.next_int ):
            self.next_int += self.interval
            self.v = 1 - self.v
        return [self.v]


class MotorNode(nef.SimpleNode):
    def __init__(self, name, vocab, motor_hrrs, num_dim, check_interval = 0.005, threshold = 0.25, go_threshold = 0.9, \
                 motor_init = 0, motor_interval = 0, out_file = "", pstc = 0.005):
        self.pstc = pstc
        self.out_state = [0,0]
        self.dimension = 2
        self.done = True
        self.out_file = out_file

        self.check_interval = check_interval
        self.next_check = check_interval

        self.suppress_output = False

        self.debug = False

        self.motor_cmds = [[[-0.5,0.8],[0.5,0.8],[0.5,-0.8],[-0.5,-0.8],[-0.5,0.8]], \
                           [[-0.5,0.4],[0,0.8],[0,-0.8]], \
                           [[-0.5,0.8],[0.5,0.8],[0.5,0],[-0.5,0],[-0.5,-0.8],[0.5,-0.8]], \
                           [[-0.5,0.8],[0.5,0.4],[-0.5,0],[0.5,-0.4],[-0.5,-0.8]], \
                           [[0,0.8],[-0.5,0],[0.5,0],[0.5,0.8],[0.5,-0.8]], \
                           [[0.5,0.8],[-0.5,0.8],[-0.5,0],[0.5,0],[0.5,-0.8],[-0.5,-0.8]], \
                           [[0.5,0.8],[-0.5,0.8],[-0.5,-0.8],[0.5,-0.8],[0.5,0],[-0.5,0]], \
                           [[-0.5,0.8],[0.5,0.8],[0,-0.8]], \
                           [[0,0.8],[-0.5,0.4],[0.5,-0.4],[0,-0.8],[-0.5,-0.4],[0.5,0.4],[0,0.8]], \
                           [[0.5,0],[-0.5,0],[-0.5,0.8],[0.5,0.8],[0.5,-0.8]], \
                           [[0,0],[-0.8,0.8],[0,0],[-0.8,-0.8],[0.8,0.8],[0,0],[0.8,-0.8],[0,0]]]
        self.motor_hrrs = motor_hrrs
        self.motor_num_hrrs = len(self.motor_hrrs)
        self.motor_plan = None
        self.motor_state = 0
        if( motor_init == 0 and motor_interval == 0 ):
            if( len(out_file) > 0 ):
                self.motor_init = 0.5
                self.motor_interval = 0
            else:
                self.motor_init = 0.5
                self.motor_interval = 0.1
        else:
            self.motor_init = motor_init
            self.motor_interval = motor_interval
        self.motor_next = 0

        self.vocab = vocab
        self.threshold = threshold

        self.go_threshold = go_threshold

        nef.SimpleNode.__init__(self, name)
        self.getTermination("Input").setDimensions(num_dim)
        self.getTermination("Go").setDimensions(1)

    def reset(self,randomize=False):
        self.next_check = self.check_interval
        self.motor_next = 0
        self.motor_state = 0
        self.done = True

        nef.SimpleNode.reset(self, randomize)

    def origin_X(self):
        if( self.t_start >= self.motor_next and not self.done and not(self.motor_plan is None) ):
            self.motor_next += self.motor_interval
            self.out_state = self.motor_cmds[self.motor_plan][self.motor_state]
            self.motor_state += 1
            if( self.motor_state >= len(self.motor_cmds[self.motor_plan]) ):
                self.done = True
            if( self.debug ):
                self.write_debug(self.t_start, "X - Out State", str(self.out_state))
        
        if( self.suppress_output ):
            return [0,0]
        else:
            return self.out_state

    def origin_Done(self):
        return [self.done]
    
    def origin_Busy(self):
        return [1 - self.done]
    
    def origin_State(self):
        return [self.motor_state]
    
    def origin_Plan(self):
        if( self.motor_plan is None ):
            return [-10]
        else:
            return [self.motor_plan]
    
    def termination_Suppress(self, x):
        self.suppress_output = (x[0] > 0.9)

    def termination_Reset(self, x):
        if( x[0] > 0.9 ):
            self.motor_next = 0
            self.motor_state = 0
            self.done = True

    def termination_Input(self, x):
        if( self.t_start >= self.next_check and self.done ):            
            dot_prods = [util_funcs.dot(x, self.vocab.hrr[self.motor_hrrs[d]].v, False) for d in range(self.motor_num_hrrs)]
            max_dot = max(dot_prods)
            if( max_dot >= self.threshold ): 
                max_dots = [(dot_prods[d] == max_dot) * (d + 1) for d in range(self.motor_num_hrrs)]
                max_item = sum(max_dots)
            else:
                max_item = 0
            
            if( max_item > self.motor_num_hrrs or max_item <= 0 ):
                self.motor_plan = len(self.motor_cmds) - 1
            elif( (max_item - 1) != self.motor_plan ):
                self.motor_plan = max_item - 1
                self.motor_state = 0
                self.done = True
            self.next_check += self.check_interval
            
            if( self.debug ):
                self.write_debug(self.t_start, "Inp - Motor Plan", str(self.motor_plan))
                self.write_debug(self.t_start, "Inp - Max Item  ", str(max_item))
        return
    
    def termination_Go(self, x):
        if( x[0] >= self.go_threshold and self.done and not(self.motor_plan is None) ):
            self.write_to_file()
            self.done = False
            self.motor_state = 0
            self.motor_next = self.t_start + self.motor_init
            self.out_state = self.motor_cmds[self.motor_plan][self.motor_state]
            if( self.debug ):
                self.write_debug(self.t_start, "Go - Motor Next", str(self.motor_next))
        return

    def write_to_file(self):
        if( not self.out_file == "" and not self.suppress_output and not self.motor_plan is None ):
            file_handle = open(self.out_file, 'a')
            if( self.motor_plan >= 0 and self.motor_plan < self.motor_num_hrrs ):
                file_handle.write(str(self.motor_plan))
                if( self.debug ):  ## Debug
                    self.write_debug(self.t_start, "W2F - Motor Plan", str(self.motor_plan))
            else:
                file_handle.write("X")
                if( self.debug ):  ## Debug
                    self.write_debug(self.t_start, "W2F - Motor Plan", "X")
            file_handle.close()
        return
    
    def write_debug(self, timestamp = 0, label = "", value = 0):
        file_handle = open(self.out_file + "debug.txt", 'a')
        file_handle.write("%10.8f" % timestamp + ": [" + label + "] " + str(value))
        file_handle.write("\n")
        file_handle.close()    


class SetInitialValuesNode(nef.SimpleNode):
    def __init__(self, name, setup_time = 0.1):
        self.setup_time = setup_time
        self.init_values = dict()
        nef.SimpleNode.__init__(self, name)
    
    def addSetInitialOrigin(self, name, init_value):
        self.init_values[name] = init_value
        self.addOrigin(BasicOrigin(self, name, len(init_value), Units.UNK))
    
    def run(self,start,end):
        if( start <= 0 ):
            for origin in self.getOrigins():
                origin.setValues(start, end, self.init_values[origin.getName()])
        if( start > self.setup_time ):
            for origin in self.getOrigins():
                origin.setValues(start, end, [0])


class AveragerNode(nef.SimpleNode):
    def __init__(self, name, num_dim, input_scale = -1, recur_scale = -1, pstc = 0.005):
        self.pstc = pstc
        self.dimension = num_dim
        self.num_items = 0.0
        self.last_num_items = 0.0
        self.input_value = util_funcs.zeros(1, num_dim)
        self.ave_value = util_funcs.zeros(1, num_dim)
        self.input_scale = input_scale
        self.recur_scale = recur_scale

        nef.SimpleNode.__init__(self, name)
        self.getTermination("Input").setDimensions(num_dim)
        self.getTermination("Input2").setDimensions(num_dim)
    
    def reset(self, randomize = False):
        self.num_items = 0.0
        self.last_num_items = 0.0
        self.ave_value = util_funcs.zeros(1, self.dimension)
        nef.SimpleNode.reset(self, randomize)

    def termination_Reset(self, x):
        if( x[0] > 0.9 ):
            self.reset(False)

    def termination_Input(self, x):
        self.input_value = x

    def termination_Input2(self, x):
        return

    def termination_Cycle(self, x):
        if( self.last_num_items == self.num_items and x[0] > 0.9 ):
            # Calculate average 
            self.num_items += 1.0
            if( self.input_scale > 0 ):
                input_scale = self.input_scale
            else:
                input_scale = 1/self.num_items
            if( self.recur_scale > 0 ):
                recur_scale = self.recur_scale
            else:
                recur_scale = 1 - input_scale
            self.ave_value = [input_scale * self.input_value[n] + recur_scale * self.ave_value[n] for n in range(self.dimension)]
        elif( x < 0.1 ):
            self.last_num_items = self.num_items
    
    def origin_X(self):
        return self.ave_value


class SimpleNEFEns(nef.SimpleNode):
    def __init__(self, name, num_dim, radius = 1, pstc = 0.005, input_name = "Input"):
        self.pstc      = pstc
        self.dimension = num_dim
        self.radius    = radius
        
        self.inhib_terms  = {}
        self.input_terms  = {}
        self.return_val   = util_funcs.zeros(1, num_dim)

        nef.SimpleNode.__init__(self, name)
        self.addDecodedOrigin("X", None, None)
        if( isinstance(input_name, str) and len(input_name) > 0 ):
            self.addDecodedTermination("Input", None, pstc, False)

    def reset(self, randomize = False):
        self.return_val = util_funcs.zeros(1, self.dimension)
        nef.SimpleNode.reset(self, randomize)
    
    def template_Termination(self, x):
        return
    
    def addDecodedOrigin(self, name, func, node_origin):
        def template_Origin(self = self, func = func):
            if( func is None ): 
                return self.return_val
            elif( isinstance(func,list) ):
                return [func[n].map(self.return_val) for n in range(len(func))]
            else:
                return func(self.return_val)
        
        self.create_origin(name, template_Origin)       
        return self.getOrigin(name)

    def removeDecodedOrigin(self, name):
        self.removeOrigin(name)

    def addDecodedTermination(self, name, matrix, pstc, isModulatory = False):
        self.input_terms[name] = matrix
        self.create_termination(name, self.template_Termination)

        if( matrix is None ):
            self.getTermination(name).setDimensions(self.dimension)
        else:
            self.getTermination(name).setDimensions(len(matrix[0]))
        self.getTermination(name).setTau(pstc)
        return self.getTermination(name)
    
    def addTermination(self, name, matrix = None, pstc = 0, isModulatory = False):
        if( isinstance(name, Termination) ):
            nef.SimpleNode.addTermination(self, name)
        else:
            self.inhib_terms[name] = matrix[0]
            self.create_termination(name, self.template_Termination)

            self.getTermination(name).setDimensions(len(matrix[0]))
            self.getTermination(name).setTau(pstc)
            return self.getTermination(name)
    
    def removeTermination(self, name):
        if( name in self.inhib_terms.keys() ):
            del self.inhib_terms[name]
        elif( name in self.input_terms.keys() ):
            del self.input_terms[name]
        nef.SimpleNode.removeTermination(self, name)

    def removeDecodedTermination(self, name):
        self.removeTermination(name)

    def run(self, start, end):
        nef.SimpleNode.run(self, start, end)
        # Get total values from input terminations
        total_input = util_funcs.zeros(1,self.dimension)
        for term_str in self.input_terms.keys():
            term_obj = self.getTermination(term_str)
            term_out = term_obj._filtered_values
            term_mat = self.input_terms[term_str]
            if( term_mat is None ):
                term_val = term_out
            else:
                term_val = MU.prod(term_mat, term_out)
            total_input = [total_input[n] + term_val[n] for n in range(self.dimension)]

        # Get total inhibitory input
        total_inhib = 0
        for term_str in self.inhib_terms.keys():
            term_obj = self.getTermination(term_str)
            term_out = term_obj._filtered_values
            term_mat = self.inhib_terms[term_str]
            term_val = MU.prod(term_mat, term_out)
            total_inhib = total_inhib + term_val
        
        # Calculate return value
        input_mag  = util_funcs.norm(total_input)
        input_sign = cmp(input_mag, 0)
        inhibd_mag = max(abs(input_mag) + (total_inhib * self.radius), 0) * input_sign
        if( input_mag != 0 ):
            self.return_val = [total_input[n] * inhibd_mag / input_mag for n in range(self.dimension)]
        else:
            self.return_val = util_funcs.zeros(1, self.dimension)
        
        return
    
    def fixMode(self):
        pass
    
    def releaseMemory(self):
        pass


class CconvNode(nef.SimpleNode):
    def __init__(self, name, num_dim, pstc = 0.005, invert_first = False, invert_second = False):
        self.dimension = num_dim
        self.pstc = pstc

        self.invert_first  = invert_first
        self.invert_second = invert_second

        self.A = util_funcs.zeros(1,num_dim)
        self.B = util_funcs.zeros(1,num_dim)
        nef.SimpleNode.__init__(self, name)
        self.getTermination("A").setDimensions(num_dim)
        self.getTermination("B").setDimensions(num_dim)
    
    def reset(self, randomize = False):
        self.A = util_funcs.zeros(1,self.dimension)
        self.B = util_funcs.zeros(1,self.dimension)
        nef.SimpleNode.reset(self, randomize)
    
    def termination_A(self, x):
        self.A = x
    
    def termination_B(self, x):
        self.B = x
    
    def origin_X(self):
        return util_funcs.cconv(self.A, self.B, self.invert_first, self.invert_second)
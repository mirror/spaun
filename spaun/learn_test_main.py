#import random
#
#import sys
#import nef
#import nps
#import hrr
#import datetime
#import timeview.components.core as core
#import timeview.view
#import os
#from ca.nengo.math.impl import ConstantFunction
#from ca.nengo.model.impl import FunctionInput
#from ca.nengo.model import Units
#from ca.nengo.model import SimulationMode
#
#from util_nodes import SimpleNEFEns
#import conf
#
##directory = 'trevor/bandit'
#directory = None
#e_name = 'Normal'
#
#delay_dur = 0.01
#approach_dur = 0.1
#reward_dur = 0.1
#
#D = 2
#N = 40
#
#def inhibitize(input_vec, inhib_scale = 2):
#    return [input_vec[d] * -inhib_scale for d in range(len(input_vec))]
#
#def create_ens(name, ef = None, network = None, num_neurons = 25, num_dim = 1, input_name = "Input", \
#               in_matrix = None, in_psc = conf.pstc, en_inhib = True, inhib_name = "Inhib", inhib_vec = [1],
#               inhib_scale = 2, inhib_psc = conf.pstc, sim_mode = SimulationMode.DEFAULT, quick = True):
#    if( str(sim_mode).lower() == 'ideal' ):
#        sim_mode = SimulationMode.DIRECT
#    if( isinstance(network, nef.Network) ):
#        net = network
#    elif( ef is None ):
#        net = nef.Network(network)
#    if( in_matrix is None ):
#        in_matrix = eye(num_dim)
#    if( sim_mode == SimulationMode.DIRECT ):
#        num_neurons = 1
#
#    in_dim = len(in_matrix[0])
#    inhib_dim = len(inhib_vec)
#    inhib_vec_scale = [inhib_vec[n] * -inhib_scale for n in range(inhib_dim)]
#    en_input = (not input_name is None) and (len(input_name) > 0)
#
#    if( num_dim > 1 and sim_mode != SimulationMode.DIRECT ):
#        neurons_per_ens = num_neurons / num_dim
#        ens = net.make_array(name, neurons_per_ens, num_dim, quick = quick)
#        if( en_inhib ):
#            ens.addTermination(inhib_name, [[inhib_vec_scale] * neurons_per_ens] * num_dim, inhib_psc, False)
#    else:
#        if( sim_mode == SimulationMode.DIRECT ):
#            ens = SimpleNEFEns(name, num_dim)
#            if( not network is None ):
#                network.add(ens)
#        else:
#            ens = net.make(name, num_neurons, num_dim)
#        if( en_inhib ):
#            ens.addTermination(inhib_name, [inhib_vec_scale] * num_neurons, inhib_psc, False)
#    if( en_input ):
#        ens.addDecodedTermination(input_name, in_matrix, in_psc, False)
#    ens.setMode(sim_mode)
#
#    if( sim_mode == SimulationMode.DIRECT ):
#        ens.fixMode()
#
#    return ens
#
#
#def create_reset_integrator(name, ef = None, network = None, num_neurons = 30, num_dim = 1, \
#                            in_matrix = [[1]], in_scale = 1, tau_feedback = 0.1, \
#                            feedback_scale = 1.0, inhib_scale = 2.0, sim_mode = SimulationMode.DEFAULT):
#    in_dim = len(in_matrix[0])
#
#    in_mat = [[in_matrix[r][c] * in_scale * tau_feedback for c in range(in_dim)] for r in range(num_dim)]
#    fb_mat = diag(num_dim, in_dim, value = feedback_scale)
#
#    ens = create_ens(name, ef, network, num_neurons, num_dim, in_matrix = in_mat, inhib_scale = inhib_scale, \
#                     sim_mode = sim_mode)
#    ens.addDecodedTermination("Feedback", fb_mat, tau_feedback, False)
#
#    network.connect(ens.getOrigin("X"), ens.getTermination("Feedback"))
#
#    return ens
#
#
#class BanditTask(nef.SimpleNode):
#    def __init__(self,name,dims,trials_per_block=40,block_rewards=[[0.21,0.63],[0.63,0.21],[0.12,0.72],[0.72,0.12]]):
#        # parameters
#        self.dimension = dims
#        self.trials_per_block = trials_per_block
#        if len(block_rewards[0]) != dims:
#            raise Exception('block_reward dimensionality must match dims')
#        self.block_rewards = block_rewards
#        
#        # vars and constants
#        self.trial_num = 0
#        self.delay_t = 0.0
#        self.approach_t = 0.0
#        self.reward_t = 0.0
#        self.reward = [0.0] * dims
#        self.thalamus_sum = [0.0] * dims
#        self.thalamus_choice = 0
#        self.rewarded = 0
#        self.reward_val = 1.0
#        self.gate_val = [1.0]
#        self.vstr_gate_val = [1.0]
#        self.data_log = []
#        
#        self.state = 'delay'
#
#        nef.SimpleNode.__init__(self,name)
#
#    def reset(self, randomize = False):
#        self.trial_num = 0
#        self.delay_t = 0.0
#        self.approach_t = 0.0
#        self.reward_t = 0.0
#        self.reward = [0.0] * self.dimension
#        self.thalamus_sum = [0.0] * self.dimension
#        self.thalamus_choice = 0
#        self.rewarded = 0
#        self.reward_val = 1.0
#        self.gate_val = [1.0]
#        self.vstr_gate_val = [1.0]
#        self.data_log = []
#        nef.SimpleNode.reset(self, randomize)
#
#    def get_experiment_length(self):
#        leeway = 0.003
#        return (delay_dur + approach_dur + reward_dur +
#                leeway) * self.trials_per_block * len(self.block_rewards)
#
#    def origin_cortex(self):
#        return [1.0]
#
#    def origin_cortex_gate(self):
#        return self.gate_val
#    
#    def origin_vstr_gate(self):
#        return self.vstr_gate_val
#    
#    def origin_reward(self):
#        return self.reward
#
#    def termination_thalamus(self,x):
#        t = self.t_start
#        
#        if self.state == 'delay':
#            self.gate_val = [self.gate_val[0]+0.001]
#            self.vstr_gate_val = [self.vstr_gate_val[0]+0.001]
#            self.reward = [0.0] * self.dimension
#            if t >= self.delay_t + delay_dur:
#                self.state = 'go'
#        elif self.state == 'go':
#            self.gate_val = [0.0]
#            self.thalamus_sum = [0.0] * self.dimension
#            self.trial_num += 1
#            self.approach_t = t
#            self.state = 'approach'
#        elif self.state == 'approach':
#            for i in range(self.dimension):
#                self.thalamus_sum[i] += x[i]
#            
#            if t >= self.approach_t + approach_dur:
#                thalamus_min = max(self.thalamus_sum)
#                for i in range(len(self.thalamus_sum)):
#                    if self.thalamus_sum[i] == thalamus_min:
#                        self.thalamus_choice = i
#                
#                block = (self.trial_num-1) / self.trials_per_block
#                if (block >= len(self.block_rewards)):
#                    self.state = 'reward'
#                    return
#                
#                ##########
#                ## NB!! ##
#                ##########
#                rand = random.random()
##                if( not self.thalamus_choice ):                  
##                if( self.thalamus_choice ):
#                if rand <= self.block_rewards[block][self.thalamus_choice]:
#                    self.rewarded = 1
#                    self.reward = [-1.0*self.reward_val] * self.dimension
#                    self.reward[self.thalamus_choice] = self.reward_val
#                else:
#                    self.rewarded = 0
#                    self.reward = [self.reward_val] * self.dimension
#                    self.reward[self.thalamus_choice] = -1.0 * self.reward_val
#                
#                # out_file structure:
#                #  trial number, choice, rewarded, thalamus_sums
#                out_l = str(self.trial_num)+', '+str(self.thalamus_choice)+', '+str(self.rewarded)
#                for i in range(len(self.thalamus_sum)):
#                    out_l += ', '+str(self.thalamus_sum[i])
#                self.data_log.append(out_l)
#                
#                self.reward_t = t
#                self.state = 'reward'
#        elif self.state == 'reward':
#            self.vstr_gate_val = [0.0]
#            if t >= self.reward_t + reward_dur:
#                self.gate_val = [1.0]
#                self.delay_t = t
#                self.state = 'delay'
#    
#    def write_data_log(self, filename):
#        """Attempts to write the contents of self.data_log to
#        the file pointed to by the consumed string, filename.
#        If there is an error writing to that file,
#        the contents of self.data_log are printed to console instead.
#        """
#        try:
#            f = open(filename, 'a+')
#        except:
#            print "Error opening %s" % filename
#            return self.print_data_log()
#        
#        for line in self.data_log:
#            f.write("%s\n" % line)
#        f.close()
#    
#    def print_data_log(self):
#        """Prints the contents of self.data_log to the console."""
#        for line in self.data_log:
#            print line
#
#class BanditWatch:
#    def __init__(self,objs):
#        self.objs=objs
#    def check(self,obj):
#        return obj in self.objs
#    def measure(self,obj):
#        r=[]
#        r.append(obj.trial_num)
#        r.append(obj.state)
#        r.append(obj.thalamus_choice)
#        r.append(obj.rewarded)
#        for sum in obj.thalamus_sum:
#            r.append(sum)
#        
#        return r
#    def views(self,obj):
#        return [('bandit task',BanditView,dict(func=self.measure,label="Bandit Task"))]
#
#from javax.swing.event import *
#from java.awt import *
#from java.awt.event import *
#class BanditView(core.DataViewComponent):
#    def __init__(self,view,name,func,args=(),label=None):
#        core.DataViewComponent.__init__(self,label)
#        self.view=view
#        self.name=name
#        self.func=func
#        self.data=self.view.watcher.watch(name,func,args=args)
#
#        self.setSize(200,100)
#
#    def paintComponent(self,g):
#        core.DataViewComponent.paintComponent(self,g)
#        
#        f_size = g.getFont().size
#        x_offset = 5
#
#        try:    
#            data=self.data.get(start=self.view.current_tick,count=1)[0]
#        except:
#            return
#        
#        cur_y = f_size*3
#        g.drawString("Trail "+str(data[0]),x_offset,cur_y)
#        cur_y += f_size
#        g.drawString("State: "+data[1],x_offset,cur_y)
#        cur_y += f_size
#        g.drawString("Thalamus sum",x_offset,cur_y)
#        cur_y += f_size
#        cur_x = x_offset
#        for sum in data[4:]:
#            g.drawString(str(round(sum*100)/100),cur_x,cur_y)
#            cur_x += 40
#        cur_y += f_size
#        g.drawString("Choice: "+str(data[2]),x_offset,cur_y)
#        cur_y += f_size
#        if data[3]: r_s = "Yes"
#        else:       r_s = "No"
#        g.drawString("Rewarded: "+r_s,x_offset,cur_y)
#
#def gate_weights(w, val = -0.0002):
#    for i in range(len(w)):
#        for j in range(len(w[0])):
#            w[i][j] = val
#			# -2, -0.008
#    return w
#
#def rand_weights(w):
#    for i in range(len(w)):
#        for j in range(len(w[0])):
#            w[i][j] = random.uniform(-1e-3,1e-3)
#    return w
#
#alpha = 1.0
#def pred_error(x):
#    # for each action, prediction error is
#    #         a   [        R            +   g   * V(S) -   V(S(t-1))]
#    return [alpha * (x[2] - x[0]), alpha * (x[3] - x[1])]
#
#def build_network():
#    
#    net = nef.Network('BanditTask_o')
#    
#    experiment = BanditTask('ExperimentRunner',D)
#    experiment.getTermination('thalamus').setDimensions(D)
#    net.add(experiment)
#    timeview.view.watches.append(BanditWatch([experiment]))
#    
#    cortex = net.make('Cortex',N,1)
#    net.connect(experiment.getOrigin('cortex'),cortex)
#    
#    cortex_gate = net.make('CortexGate',100,1,encoders=[[1.0]],intercept=(-0.2,0.3),max_rate=(50,150))
#    def cortex_gate_weights(w):
#        return gate_weights(w, -0.0005)
#
#    net.connect(experiment.getOrigin('cortex_gate'),cortex_gate)
#    net.connect(cortex_gate,cortex,weight_func=cortex_gate_weights)
#    
#    thalamus_bias = net.make_input("Thalamus Bias", [1])
#    thalamus_prod = SimpleNEFEns("Thalamus Prod", D)
#    thalamus_out  = SimpleNEFEns("Thalamus Out", D)
#    thalamus_mut_inhib = True
#    net.add(thalamus_prod)
#    net.add(thalamus_out)
# 
#    for d in range(D):
#        thalamus = net.make('Thalamus'+str(d+1),N,1,intercept=(0.2,1),encoders=[[1]])
#        net.connect(thalamus_bias, thalamus)
#        inhib_mat = [[-2*(n == d) for n in range(D)]] * N
#        thalamus.addTermination("Thalamus Prod", inhib_mat, 0.005, False)
#        net.connect(thalamus_prod, thalamus.getTermination("Thalamus Prod"), pstc = 0.0001)
#        output_mat = [[(n == d)] for n in range(D)]
#        net.connect(thalamus, thalamus_out, transform = output_mat)
#
#    if( thalamus_mut_inhib ):
#        for d in range(D):
#            thalamus = net.get('Thalamus'+str(d+1))
#            for n in range(D):
#                if( d != n ):
#                    thalamus.addTermination("Thalamus"+str(n+1), [[-2]] * N, 0.01, False)
#                    net.connect("Thalamus"+str(n+1), thalamus.getTermination("Thalamus"+str(n+1)))
#
#    net.connect(thalamus_out,experiment.getTermination('thalamus'))
#    
#    nps.basalganglia.make_basal_ganglia(net,cortex,thalamus_prod,D,N,learn=True)
#
#    StrD1 = net.network.getNode('StrD1')
#    StrD2 = net.network.getNode('StrD2')
#    
#    vStr = net.make('Ventral Striatum',N*D*2,D*2,max_rate=(100,200))
#    net.connect(cortex,vStr,index_post=[0,1],weight_func=rand_weights)
##    net.connect(cortex,vStr,index_post=[0,1])
#    net.connect(experiment.getOrigin('reward'),vStr,index_post=[2,3])
#    
#    net.connect(vStr,vStr,func=pred_error,modulatory=True, index_post = [0,1])
#    net.connect(vStr,StrD1,func=pred_error,modulatory=True)
#    net.connect(vStr,StrD2,func=pred_error,modulatory=True)
#    
#    l_args = {'stpd':False, 'rate':1e-7}
#    net.learn(vStr,'Cortex','Ventral Striatum',**l_args)
#    net.learn_array(StrD1,'Cortex','Ventral Striatum',**l_args)
#    net.learn_array(StrD2,'Cortex','Ventral Striatum',**l_args)
#    
#    vStr_gate = net.make('vStrGate',100,1,encoders=[[1.0]],intercept=(-0.2,0.3),max_rate=(50,150))
#    net.connect(experiment.getOrigin('vstr_gate'),vStr_gate)
#    net.connect(vStr_gate,vStr,weight_func=gate_weights)
#    
#    return net
#
#def run(world, OS = "WIN", test_type = 6, test_option = None, num_test_run = 2, num_subjects = 1, \
#        multi_thread = False, en_logging = False, rand_type = 0, #rand_type = 110518233715, \
#        perfect_MB = True, perfect_cconv = True, CUthreshold = 0.4, CUNumsThreshold = 0.3, CUinScale = 1.0, \
#        tranf_scale = 0.451, learn_alpha = 1.0, learn_actions = 3, auto_run = 2):
#    net = build_network()
#    experiment = net.network.getNode('ExperimentRunner')   
#    net.add_to_nengo()



# Import vocab string data and control module
from spaun.cleanup_mem import CleanupMem
from spaun.gated_integrator import GatedInt
from spaun.mem_block import MemBlock
from spaun.threshold_detect import Detector
from spaun.selector import Selector
from spaun.visual_heir import VisualHeirachy
from spaun.visual_heir import make_VisHeir_AM

import vocabs
import conf

import spa
import nef
import hrr

from ca.nengo.model import Units
from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import FunctionInput
from ca.nengo.model.impl import BasicOrigin
from ca.nengo.model.impl import NetworkImpl
from ca.nengo.model.impl import EnsembleTermination
from ca.nengo.model.neuron import Neuron
from ca.nengo.model.nef.impl import NEFEnsembleFactoryImpl
from ca.nengo.math.impl import AbstractFunction
from ca.nengo.math.impl import PostfixFunction
from ca.nengo.math.impl import ConstantFunction
from ca.nengo.math.impl import PiecewiseConstantFunction
from ca.nengo.util.impl import NodeThreadPool
from ca.nengo.util.impl import NEFGPUInterface
from ca.nengo.math import PDFTools
from util_nodes import *
from util_funcs import *
from spa import *
from nef.convolution import make_convolution

from java.lang import System
from java.lang.System.err import println
import datetime
import random
import os.path
import math
import copy


params_cconv_out = dict(num_neurons = 1, en_inhib = False, input_name = None, sim_mode = SimulationMode.DIRECT)
params_learn = dict(stpd = False, rate = 1e-7)

def inhibitize(input_vec, inhib_scale = 2):
    return [input_vec[d] * -inhib_scale for d in range(len(input_vec))]

def create_ens(name, ef = None, network = None, num_neurons = 25, num_dim = 1, input_name = "Input", \
               in_matrix = None, in_psc = conf.pstc, en_inhib = True, inhib_name = "Inhib", inhib_vec = [1], \
               inhib_scale = 2, inhib_psc = conf.pstc, sim_mode = SimulationMode.DEFAULT, use_array = True, \
               quick = True):
    if( str(sim_mode).lower() == 'ideal' ):
        sim_mode = SimulationMode.DIRECT
    if( isinstance(network, nef.Network) ):
        net = network
    elif( ef is None ):
        net = nef.Network(network)
    if( in_matrix is None ):
        in_matrix = eye(num_dim)
    if( sim_mode == SimulationMode.DIRECT ):
        num_neurons = 1

    in_dim = len(in_matrix[0])
    inhib_dim = len(inhib_vec)
    inhib_vec_scale = [inhib_vec[n] * -inhib_scale for n in range(inhib_dim)]
    en_input = (not input_name is None) and (len(input_name) > 0)

    if( num_dim > 1 and sim_mode != SimulationMode.DIRECT and use_array ):
        neurons_per_ens = num_neurons / num_dim
        ens = net.make_array(name, neurons_per_ens, num_dim, quick = quick)
        if( en_inhib ):
            ens.addTermination(inhib_name, [[inhib_vec_scale] * neurons_per_ens] * num_dim, inhib_psc, False)
    else:
        if( sim_mode == SimulationMode.DIRECT ):
            ens = SimpleNEFEns(name, num_dim)
            if( not network is None ):
                network.add(ens)
        else:
            ens = net.make(name, num_neurons, num_dim)
        if( en_inhib ):
            ens.addTermination(inhib_name, [inhib_vec_scale] * num_neurons, inhib_psc, False)
    if( en_input ):
        ens.addDecodedTermination(input_name, in_matrix, in_psc, False)
    ens.setMode(sim_mode)

    if( sim_mode == SimulationMode.DIRECT ):
        ens.fixMode()

    return ens


def create_reset_integrator(name, ef = None, network = None, num_neurons = 30, num_dim = 1, \
                            in_matrix = [[1]], in_scale = 1, tau_feedback = 0.1, \
                            feedback_scale = 1.0, inhib_scale = 2.0, sim_mode = SimulationMode.DEFAULT):
    in_dim = len(in_matrix[0])

    in_mat = [[in_matrix[r][c] * in_scale * tau_feedback for c in range(in_dim)] for r in range(num_dim)]
    fb_mat = diag(num_dim, in_dim, value = feedback_scale)

    ens = create_ens(name, ef, network, num_neurons, num_dim, in_matrix = in_mat, inhib_scale = inhib_scale, \
                     sim_mode = sim_mode)
    ens.addDecodedTermination("Feedback", fb_mat, tau_feedback, False)

    network.connect(ens.getOrigin("X"), ens.getTermination("Feedback"))

    return ens


class ControlModule(spa.module.Module):
    def create(self):
        interval     = conf.present_int
        max_time_out = 10

        control = vocabs.ControlInput("Control", interval, conf.num_test_run, conf.test_type, \
                                      conf.test_option, conf.out_file, max_time_out, conf.vocab_data,
                                      conf.rand_seed, self.spa.net)
        for term in control.getTerminations():
            term.setTau(conf.pstc)
        self.net.add(control)

        conf.est_runtime = control.getEstRuntime(interval * 3)

        # Set default spa vocabulary
        self.spa.default_vocabs[(conf.num_dim, False)] = conf.vocab_data.vis_vocab
        self.spa.default_vocabs[(conf.num_dim, True)]  = conf.vocab_data.vis_vocab

        self.add_source(control.getOrigin("X"))
        self.exposeTermination(control.getTermination("Cont"), "stimulus_cont")
        self.exposeTermination(control.getTermination("MotorOut"), "stimulus_motorout")
        return


class VisionModule(spa.module.Module):
    def create(self):
        vocab     = conf.vocab_data.vis_vocab

        list_strs = read_csv(conf.vis_filepath + conf.sym_list_filename, True)

        def check_input( str_list, vocab ):
            return_val = []
            vocab_keys = vocab.hrr.keys()
            for str in str_list:
                if( str in vocab_keys ):
                    return_val.append(vocab.hrr[str].v)
                else:
                    return_val.append(None)
            return return_val
        
        # vocab_vecs = [vis_vocab.hrr[str[0]].v for str in list_strs]
        vocab_vecs = check_input( [str[0] for str in list_strs], vocab )

        vis_node = VisualHeirachy(file_path = conf.vis_filepath, mu_filename = conf.mu_filename, \
                                  output_vecs = vocab_vecs, mut_inhib_scale = 2.0, en_bypass = True, \
                                  sim_mode = conf.MB_mode, sim_mode_cleanup = SimulationMode.DEFAULT)
        vis_node.fixMode()
        self.net.add(vis_node)

        self.add_source(vis_node.getOrigin("X"))
        self.exposeOrigin(vis_node.getOrigin("Vis Raw"), "vis_raw")
        self.exposeTermination(vis_node.getTermination("Input"), "Input")

        self.spa.vocabs["vis"] = conf.vocab_data.vis_vocab

        self.net.releaseMemory()
        return

    def connect(self):
        self.spa.net.connect(self.spa.getModuleOrigin("stimulus", "stimulus"), \
                             self.spa.getModuleTermination(self.name, "Input"))
        return



class ProdSysBufferModule(spa.module.Module):
    def create(self):
        params_MB  = dict(reset_opt = 2, cyc_opt = 1, mode = -1, neurons_per_dim = 50, input_name = None, \
                          tau_in = conf.pstc * 2, in_scale = 1.5, rand_seed = conf.rand_seed)
        params_GI  = dict(cyc_opt = 1, mode = -1, neurons_per_dim = 50, en_cyc_in = False, in_scale = 2.0, \
                          tau_in = conf.pstc, tau_in = 0.025, rand_seed = conf.rand_seed)
        params_det = dict(net = self.net, num_neurons = 30, tau_in = conf.pstc, tau_inhib = conf.pstc, \
                          sim_mode = conf.det_mode, rand_seed = conf.rand_seed, detect_threshold = 0.5)

        state_Buf   = MemBlock("state"   , conf.vocab_data.state_dim  , **params_MB)
        state_Buf.addAxonOrigin()   # For learning task
        task_Buf    = MemBlock("task"    , conf.vocab_data.task_dim   , **params_MB)
        subtask_Buf = GatedInt("subtask" , conf.vocab_data.subtask_dim, **params_GI)
        subtask_in  = self.net.make("subtask in", 1, conf.vocab_data.subtask_dim, mode = 'direct')
        subtask_in.fixMode()
        self.net.add(state_Buf)
        self.net.add(task_Buf)
        self.net.add(subtask_Buf)

        Detector("Nums"                 , detect_vec = conf.vocab_data.sum_num_vecs, en_out = False, en_N_out = True, **params_det)
        Detector("TaskInit"             , detect_vec = conf.vocab_data.task_vocab.hrr["X"].v   , **params_det)
        self.net.get("TaskInit").addTermination("Inhib", [[-2]] * params_det["num_neurons"], conf.pstc_inhib, False)
        Detector("PS Reset"             , detect_vec = conf.vocab_data.ps_reset_vecs           , **params_det)
        Detector("PS State Reset"       , detect_vec = conf.vocab_data.ps_state_reset_vecs     , **params_det)
        Detector("PS Cycle"             , detect_vec = conf.vocab_data.ps_cycle_vecs           , **params_det)
        self.net.get("PS Cycle").addDecodedTermination("TaskInit", [[1]], conf.pstc, False)
        task_Cyc    = create_ens("Task Cyc", network = self.net, num_neurons = params_det["num_neurons"], \
                                 en_inhib = False, num_dim = 1, sim_mode = conf.det_mode, quick = True)
        task_Cyc.addDecodedTermination("TaskInit", [[1]], conf.pstc, False)
        self.net.connect(self.net.get("NumsN").getOrigin("X"), self.net.get("TaskInit").getTermination("Inhib"))

        subtask_mag = Detector("SubtaskMag" , detect_vec = num_vector(params_GI["in_scale"],1,conf.vocab_data.subtask_dim), \
                               en_N_out = True, sim_mode = SimulationMode.DEFAULT, detect_threshold = 0.4, \
                               tau_in = conf.pstc, num_neurons = 20, rand_seed = conf.rand_seed)
        self.net.add(subtask_mag)

        ps_vis_terms = [self.net.get("NumsN").getTermination("Input"), \
                        self.net.get("PS Reset").getTermination("Input"), \
                        self.net.get("PS State Reset").getTermination("Input"), \
                        self.net.get("PS Cycle").getTermination("Input")]
        ps_vis_term  = EnsembleTermination(self.net.network, "vis", ps_vis_terms)
        self.exposeTermination(ps_vis_term, "vis")

        self.net.connect(task_Buf.getOrigin("X")                , self.net.get("TaskInit").getTermination("Input"))
        self.net.connect(self.net.get("TaskInit").getOrigin("X"), self.net.get("PS Cycle").getTermination("TaskInit"))
        self.net.connect(self.net.get("TaskInit").getOrigin("X"), self.net.get("Task Cyc").getTermination("TaskInit"))

        self.net.connect(self.net.get("PS Cycle").getOrigin("X"), state_Buf.getTermination("Cycle"))
        self.net.connect(self.net.get("PS Reset").getOrigin("X"), task_Cyc.getTermination("Input"))
        self.net.connect(task_Cyc.getOrigin("X")                , task_Buf.getTermination("Cycle"))

        self.net.connect(self.net.get("PS Reset").getOrigin("X"), task_Buf.getTermination("Reset"))
        self.net.connect(self.net.get("PS State Reset").getOrigin("X"), state_Buf.getTermination("Reset"))

        self.net.connect(subtask_in.getOrigin("X"), subtask_Buf.getTermination("Input"))
        self.net.connect(subtask_in.getOrigin("X"), subtask_mag.getTermination("Input"))
        self.net.connect(subtask_mag.getOrigin("SubtaskMag"), subtask_Buf.getTermination("Cycle"))
        self.net.connect(subtask_mag.getOrigin("SubtaskMagN"), subtask_Buf.getTermination("CycleN"))

        self.add_source(state_Buf.getOrigin("X"), "stateo")
        self.add_source(state_Buf.getOrigin("AXON"), "statea")
        self.add_source(task_Buf.getOrigin("X") , "tasko")
        self.add_source(subtask_Buf.getOrigin("X") , "subtasko")
        self.add_sink(state_Buf  , "state")
        self.add_sink(task_Buf   , "task")
        self.add_sink(subtask_in , "subtask")

        self.exposeTermination(self.net.get("TaskInit").getTermination("Inhib"), "NumsN")
        self.exposeOrigin(self.net.get("PS Reset").getOrigin("X"), "ps_reset")

        self.spa.vocabs["ps_state"]   = conf.vocab_data.state_vocab
        self.spa.vocabs["ps_stateo"]  = conf.vocab_data.state_vocab
        self.spa.vocabs["ps_task"]    = conf.vocab_data.task_vocab
        self.spa.vocabs["ps_tasko"]   = conf.vocab_data.task_vocab
        self.spa.vocabs["ps_subtask"] = conf.vocab_data.subtask_vocab

        self.net.releaseMemory()
        return

    def connect(self):
        self.spa.net.connect(self.spa.getModuleOrigin("vis", "vis"), \
							 self.spa.getModuleTermination(self.name, "vis"))
        return


alpha = 0.5
def pred_error(x):
    # for each action, prediction error is
    #         a   [        R            +   g   * V(S) -   V(S(t-1))]
    return [alpha * (x[2] - x[0]), alpha * (x[3] - x[1])]
#def pred_error(x):
#    return [alpha * x[0], alpha * x[1]]

def product(x):
    return x[0] * x[1]

def rand_weights(w):
    for i in range(len(w)):
        for j in range(len(w[0])):
            w[i][j] = random.uniform(-1e-3,1e-3)
    return w

class vStrModule(spa.module.Module):
    def create(self):
        neur_per_dim = 40
        num_actions  = 2

        params_det = dict(net = self.net, num_neurons = 30, tau_in = conf.pstc, tau_inhib = conf.pstc, \
                          sim_mode = conf.det_mode, detect_threshold = 0.35)
        Detector("Rewarded", detect_vec = conf.vocab_data.vis_vocab.hrr["ONE"].v    , **params_det)
        Detector("NoReward", detect_vec = conf.vocab_data.vis_vocab.hrr["ZER"].v    , **params_det)
        Detector("Action1" , detect_vec = [conf.vocab_data.subtask_vocab.hrr["A1"].v[n] / 2 \
                 for n in range(conf.vocab_data.subtask_dim)] , **params_det)
        Detector("Action2" , detect_vec = [conf.vocab_data.subtask_vocab.hrr["A2"].v[n] / 2 \
                 for n in range(conf.vocab_data.subtask_dim)] , **params_det)

        rewardA1_ens = self.net.make("RewardA1", neur_per_dim * 2, 2, radius = sqrt(2), \
                                     encoders = [[1,1],[1,-1],[-1,1],[-1,-1]], quick = True)
        self.net.connect(self.net.get("Rewarded"), rewardA1_ens, weight = 1 , index_post = [0])
        self.net.connect(self.net.get("NoReward"), rewardA1_ens, weight = -1, index_post = [0])
        self.net.connect(self.net.get("Action1") , rewardA1_ens, weight = 1 , index_post = [1])
        self.net.connect(self.net.get("Action2") , rewardA1_ens, weight = -1, index_post = [1])
        rewardA2_ens = self.net.make("RewardA2", neur_per_dim * 2, 2, radius = sqrt(2), \
                                     encoders = [[1,1],[1,-1],[-1,1],[-1,-1]], quick = True)
        self.net.connect(self.net.get("Rewarded"), rewardA2_ens, weight = 1 , index_post = [0])
        self.net.connect(self.net.get("NoReward"), rewardA2_ens, weight = -1, index_post = [0])
        self.net.connect(self.net.get("Action1") , rewardA2_ens, weight = -1, index_post = [1])
        self.net.connect(self.net.get("Action2") , rewardA2_ens, weight = 1 , index_post = [1])

        eval_ens  = create_ens("Evaluator", network = self.net, num_neurons = neur_per_dim * num_actions * 2, \
                               num_dim = num_actions * 2, sim_mode = conf.det_mode, use_array = False, \
                               en_inhib = False, input_name = None)
        self.net.connect(rewardA1_ens, eval_ens, func = product, index_post = [2])
        self.net.connect(rewardA2_ens, eval_ens, func = product, index_post = [3])
#        eval_ens  = create_ens("Evaluator", network = self.net, num_neurons = neur_per_dim * num_actions, \
#                               num_dim = num_actions, sim_mode = conf.det_mode, use_array = False, \
#                               en_inhib = False, input_name = None)
#        self.net.connect(rewardA1_ens, eval_ens, func = product, index_post = [0])
#        self.net.connect(rewardA2_ens, eval_ens, func = product, index_post = [1])
        pred_ori,pred_term = self.net.connect(eval_ens, eval_ens, func = pred_error, modulatory = True, \
                                              create_projection = False)
        self.net.connect(pred_ori, pred_term)

        vis_terms     = [self.net.get("Rewarded").getTermination("Input"), \
                         self.net.get("NoReward").getTermination("Input")]
        subtask_terms = [self.net.get("Action1").getTermination("Input"), \
                         self.net.get("Action2").getTermination("Input")]
        vis_term      = EnsembleTermination(self.net.network, "vis", vis_terms)
        subtask_term  = EnsembleTermination(self.net.network, "stt", subtask_terms)

        self.add_source(pred_ori, "pred_error")

        self.exposeTermination(vis_term    , "vis")
        self.exposeTermination(subtask_term, "subtask")
        return

    def connect(self):
        self.spa.net.connect(self.spa.getModuleOrigin("vis", "vis"), \
                             self.spa.getModuleTermination(self.name, "vis"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_subtasko"), \
                             self.spa.getModuleTermination(self.name, "subtask"))

        statea_origin = self.spa.getModuleOrigin("ps", "ps_statea")
        eval_ens = self.net.get("Evaluator")
        axon_dim = statea_origin.dimensions
        term_dim = eval_ens.getNodeCount()
        eval_ens.addTermination("statea", rand_weights(zeros(term_dim, axon_dim)), conf.pstc, False)
        self.exposeTermination(eval_ens.getTermination("statea"), "ps_statea")
        self.spa.net.connect(statea_origin, self.spa.getModuleTermination(self.name, "ps_statea"))
        self.net.learn(eval_ens, eval_ens.getTermination("statea"), eval_ens.name, **params_learn)
        return


class MotorModule(spa.module.Module):
    def create(self):
        params_det = dict(net = self.net, num_neurons = 30, tau_in = conf.pstc, tau_inhib = conf.pstc, \
                          sim_mode = conf.det_mode)
        motor_init     = conf.present_int
        motor_interval = 0

        motor_node = MotorNode("Motor Node", conf.vocab_data.nums_vocab, conf.vocab_data.num_strs, \
                               conf.vocab_data.nums_dim, out_file = conf.out_file, motor_init = motor_init, \
                               motor_interval = motor_interval, pstc = conf.pstc)
        self.net.add(motor_node)

# ========== Temp Stuff ===============
        cleanupmem = CleanupMem("Cleanup Memory", \
                                [conf.vocab_data.subtask_vocab.hrr["A1"].v, conf.vocab_data.subtask_vocab.hrr["A2"].v],
                                [conf.vocab_data.nums_vocab.hrr["ZER"].v, conf.vocab_data.nums_vocab.hrr["ONE"].v],
                                en_mut_inhib = True, tau_in = 0.01, in_scale = conf.CUinScale, \
                                threshold = conf.CUthreshold)
        self.net.add(cleanupmem)
        self.exposeTermination(cleanupmem.getTermination("Input"), "subtask")
        self.net.connect(cleanupmem.getOrigin("X"), motor_node.getTermination("Input"))
# ========== Temp Stuff ===============

        self.add_source(motor_node.getOrigin("X"))
        self.add_source(motor_node.getOrigin("Busy"), "busy")
        self.add_source(motor_node.getOrigin("Done"), "done")
        self.add_source(motor_node.getOrigin("Plan"), "plan")

#        self.exposeTermination(motor_in.getTermination("Input 1")   , "motor_input")
#        self.exposeTermination(motor_in.getTermination("Input 2")   , "motor_raw")
        self.exposeTermination(motor_node.getTermination("Go")      , "motor_go")
        self.exposeTermination(motor_node.getTermination("Suppress"), "motor_suppress")
        self.exposeTermination(motor_node.getTermination("Reset")   , "motor_reset")
        return

    def connect(self):
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_reset"), \
							 self.spa.getModuleTermination(self.name, "motor_reset"))
        self.spa.net.connect(self.spa.getModuleOrigin(self.name, "motor_plan"), \
                             self.spa.getModuleTermination("stimulus", "stimulus_motorout"))
#        self.spa.net.connect(self.spa.getModuleOrigin("dec", "cnt"), \
#							 self.spa.getModuleTermination(self.name, "motor_suppress"))
#        self.spa.net.connect(self.spa.getModuleOrigin("trans", "motor_raw"), \
#							 self.spa.getModuleTermination(self.name, "motor_raw"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_subtasko"), \
                             self.spa.getModuleTermination(self.name, "subtask"))
        return


def run(world, OS = "WIN", test_type = 7, test_option = None, num_test_run = 2, num_subjects = 1, \
        multi_thread = False, en_logging = False, rand_type = 0, #rand_type = 110518233715, \
        perfect_MB = True, perfect_cconv = True, CUthreshold = 0.4, CUNumsThreshold = 0.3, CUinScale = 1.0, \
        tranf_scale = 0.451, learn_alpha = 1.0, auto_run = 2):

    learn_actions = 2
    if( test_option is None ):
        if( test_type == 2 ):
#            test_option = [[0.21,0.63,5],[0.63,0.21,5],[0.12,0.72,5],[0.72,0.12,5]]
            test_option = [[100,0,20]]
            learn_actions = 2
        if( test_type == 3 ):
            test_option = (4,0)
        if( test_type == 4 ):
            test_option = (None,None)
        if( test_type == 5 ):
            test_option = (5,'K')
        if( test_type == 6 ):
            test_option = (4,"AAxB","xB")
        if( test_type == 7 ):
            test_option = [3,2,1]

    conf.OS = OS;                     conf.test_type = test_type;             conf.test_option = test_option
    conf.num_test_run = num_test_run; conf.num_subjects = num_subjects;       conf.learn_alpha = learn_alpha
    conf.CUthreshold = CUthreshold;   conf.CUNumsThreshold = CUNumsThreshold; conf.CUinScale = CUinScale;         
    conf.tranf_scale = tranf_scale;   conf.learn_actions = learn_actions

## ------------------------------------- DEFINE RULE SET ------------------------------------------
    class SpaUNRules:
        def task_init(vis = "A"):
            set(ps_task = "X")

    #    def task_r_init(vis = "R"):
    #        set(ps_task = "R", ps_state = "TRANS1", ps_subtask = "MF")
    #    def task_r_store(ps_tasko = "R", ps_stateo = "TRANS1", scale = 0.5):
    #        set(ps_state = "TRANS1")
    #
    #    def task_v_init(vis = "V"):
    #        set(ps_task = "V", ps_state = "SKIP", ps_subtask = "NON")
    #    def task_v_tr1_2_skp(ps_tasko = "V", ps_stateo = "TRANS1", scale = 0.5):
    #        set(ps_state = "SKIP")
    #
    #    def task_f_init(vis = "F"):
    #        set(ps_task = "F", ps_state = "SKIP", ps_subtask = "NON")
    #    def task_f_tr1_2_tr2(ps_tasko = "F", ps_stateo = "TRANS1", scale = 0.5):
    #        set(ps_state = "TRANS2")
    #    def task_f_tr2_2_skp(ps_tasko = "F", ps_stateo = "TRANS2", scale = 0.5):
    #        set(ps_state = "SKIP")
    #
    #    def task_m_init(vis = "M"):
    #        set(ps_task = "M", ps_state = "TRANS1", ps_subtask = "MF")
    #    def task_m_keep_tr1(ps_tasko = "M", ps_stateo = "TRANS1", scale = 0.5):
    #        set(ps_state = "TRANS1")
    #    def task_m_set_fwd(vis = "P", ps_tasko = "M", scale = 0.5):
    #        set(ps_subtask = "MF")
    #    def task_m_set_bck(vis = "K", ps_tasko = "M", scale = 0.5):
    #        set(ps_subtask = "MB")
    #
    #    def task_r_init(vis = "R"):
    #        set(ps_task = "R", ps_state = "TRANS1", ps_subtask = "MF")
    #    def task_r_keep_tr1(ps_tasko = "R", ps_stateo = "TRANS1", scale = 0.5):
    #        set(ps_state = "TRANS1")
    #
    #    def task_a_init(vis = "A"):
    #        set(ps_task = "A", ps_state = "SKIP", ps_subtask = "NON")
    #    def task_a_tr1_2_tr2(ps_tasko = "A", ps_stateo = "TRANS1", scale = 0.5):
    #        set(ps_state = "TRANS2")
    #    def task_a_keep_tr2(ps_tasko = "A", ps_stateo = "TRANS2", scale = 0.5):
    #        set(ps_state = "TRANS2")
    #    def task_a_set_k(ps_tasko = "A", vis = "K", scale = 0.5):
    #        set(ps_subtask = "AK")
    #    def task_a_set_p(ps_tasko = "A", vis = "P", scale = 0.5):
    #        set(ps_subtask = "AP")
    #
    #    def task_c_init(vis = "C"):
    #        set(ps_task = "C", ps_state = "SKIP", ps_subtask = "NON")
    #    def task_c_set_cnt(ps_tasko = "C", ps_stateo = "TRANS1", scale = 0.5):
    #        set(ps_state = "CNT", ps_subtask = "CNT")
    #    def task_c_nomatch(ps_stateo = "CNT", scale = 1.0):
    ##        match(mem_MB2 != mem_MBCnt)
    #        set(ps_subtask = "CNT")
    #    def task_c_match(ps_stateo = "CNT", scale = 0.5):
    ##        match(mem_MB2 == mem_MBCnt)
    #        set(ps_subtask = "MF")
    #
    #    def task_w_init(vis = "W"):
    #        set(ps_task = "W", ps_state = "VIS", ps_subtask = "MF")
    #

        def task_qm(vis = "QM"):
            set(ps_task = "DEC")
        def task_skp_2_tr1(ps_stateo = "SKIP"):
            set(ps_state = "TRANS1")

        def task_l_init(vis = "TWO", ps_tasko = "X", scale = 0.5):
            set(ps_task = "L", ps_state = "LEARN", ps_subtask = "NON")

        for i in range(conf.learn_actions):
            code = """def task_l_a%d(ps_stateo = "LEARN-TRANS1-TRANS2-SKIP", scale = %f, rand_weights = rand_weights):
                          learn(ps_statea = rand_weights, pred_error = vstr_pred_error)
                          set(ps_subtask = "A%d")""" % (i+1,0.35,i+1)
            exec(code)

## ------------------------------------- END DEFINE RULE SET ------------------------------------------
## ------------------------------------- DEFINE SPA NETWORK ------------------------------------------

    class SpaUNLearn(spa.core.SPA):
        dimensions = conf.num_dim
        align_hrrs = True

        stimulus = ControlModule()
        vis      = VisionModule()
        ps       = ProdSysBufferModule()
    #    mem      = MemoryBufferModule()
    #    trans    = TransformModule()
    #    enc      = EncodingModule()
    #    dec      = DecodingModule()
        vstr     = vStrModule()
        motor    = MotorModule()

        BG       = spa.bg.BasalGanglia(SpaUNRules(), pstc_input = 0.01)
        thalamus = spa.thalamus.Thalamus(bg = BG, pstc_route_input = 0.01, pstc_gate = 0.001, route_scale = 1, \
                                         pstc_inhibit = 0.01, pstc_output = 0.011, pstc_route_output = 0.01, \
                                         mutual_inhibit = 2, quick = False)

## ------------------------------------- END DEFINE SPA NETWORK ------------------------------------------

    if( perfect_MB ):
        conf.MB_mode = "ideal"
    if( perfect_cconv ):
        conf.cconv_mode = "direct"

    if( OS == "WIN" ):
        conf.root_path    = "D:\\fchoo\\Documents\\My Dropbox\\SPA\\Code\\Spaun\\"
        conf.vis_filepath = "D:\\fchoo\\Documents\\My Dropbox\\SPA\\Code\\Digits\\Matlab\\"
    elif( OS == "LIN_G" ):
        conf.root_path    = "/home/ctnuser/fchoo/code/"
        conf.vis_filepath = "/home/ctnuser/fchoo/code/Digits/Matlab/"
    elif( OS == "LIN" ):
        conf.root_path    = "/home/fchoo/Dropbox/SPA/Code/data/"
        conf.vis_filepath = "/home/fchoo/Dropbox/SPA/Code/Digits/Matlab/"

    rand_seed = rand_type
    if( not rand_type == 0 ):
        if( rand_type == 1 ):
            rand_seed = eval(datetime.datetime.today().strftime("%y%m%d%H%M"))
    else:
        rand_seed = eval(datetime.datetime.today().strftime("%y%m%d%H%M%S"))
    PDFTools.setSeed(rand_seed)
    random.seed(rand_seed)

    if( not multi_thread ):
        NodeThreadPool.turnOffMultithreading()
    else:
        NodeThreadPool.setNumThreads(multi_thread)

    datetime_str = datetime.datetime.today().strftime("%y%m%d%H%M%S")
    filename  = "task_"  + vocabs.task_strs[test_type] + str(test_option) + "_" + datetime_str + ".txt"
    logname   = "log_"   + vocabs.task_strs[test_type] + str(test_option) + "_" + datetime_str + ".csv"

    if( not OS == "" ):
        conf.out_file = conf.root_path + filename
    else:
        conf.out_file = ""

    if( en_logging and not OS == "" ):
        conf.log_file = conf.root_path + logname
    else:
        conf.log_file = ""

    for i in range(learn_actions):
        vocabs.subtask_strs.append("A" + str(i+1))

    conf.vocab_data = vocabs.VocabData()
    spaun = SpaUNLearn()

    println(conf.est_runtime)

    # Set default vocabularies (for interactive mode)
    hrr.Vocabulary.defaults[conf.num_dim]           = conf.vocab_data.vis_vocab
    hrr.Vocabulary.defaults[conf.vocab_data.nums_dim]    = conf.vocab_data.nums_vocab
    hrr.Vocabulary.defaults[conf.vocab_data.state_dim]   = conf.vocab_data.state_vocab
    hrr.Vocabulary.defaults[conf.vocab_data.task_dim]    = conf.vocab_data.task_vocab
    hrr.Vocabulary.defaults[conf.vocab_data.subtask_dim] = conf.vocab_data.subtask_vocab

    # Raw visual output vocab (for debug)
    vis_raw_vocab = hrr.Vocabulary(conf.vis_dim, max_similarity = 0.05, include_pairs = False)
    list_strs = read_csv(conf.vis_filepath + conf.sym_list_filename, True)
    raw_vecs  = read_csv(conf.vis_filepath + conf.mu_filename)
    for i,list_str in enumerate(list_strs):
        vis_raw_vocab.add(list_str[0], hrr.HRR(data = raw_vecs[i]))
    hrr.Vocabulary.defaults[conf.vis_dim]           = vis_raw_vocab

    if( en_logging ):
        wtfNode = nef.WriteToFileNode("wtf", conf.log_file, spaun.net, \
                                      conf.vocab_data.vis_vocab, log_interval = 0.01, pstc = 0.01)
        wtfNode.addValueTermination("dec", "motor_go")
        wtfNode.addValueTermination("dec", "stimulus_cont")

    if( auto_run > 0 ):
        if( auto_run > 1 ):
            spaun.net.view(play = conf.est_runtime)
        else:
            spaun.net.network.simulator.resetNetwork(False, False)
            spaun.net.network.simulator.run(0, conf.est_runtime, 0.001, False)

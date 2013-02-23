# Import vocab string data and control module
from cleanup_mem import CleanupMem
from gated_integrator import GatedInt
from mem_block import MemBlock
from threshold_detect import Detector
from selector import Selector
from visual_heir import VisualHeirachy
from visual_heir import make_VisHeir_AM
from motor_node import SpaunMotorNode
from motor_node import TraceMotorTransform

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
import time
import socket

params_BG = dict(same_neurons = False)
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
                            in_matrix = [[1]], in_scale = 1, tau_feedback = 0.1, quick = True, \
                            feedback_scale = 1.0, inhib_scale = 2.0, sim_mode = SimulationMode.DEFAULT):
    in_dim = len(in_matrix[0])

    in_mat = [[in_matrix[r][c] * in_scale * tau_feedback for c in range(in_dim)] for r in range(num_dim)]
    fb_mat = diag(num_dim, in_dim, value = feedback_scale)

    ens = create_ens(name, ef, network, num_neurons, num_dim, in_matrix = in_mat, inhib_scale = inhib_scale, \
                     sim_mode = sim_mode, quick = quick)
    ens.addDecodedTermination("Feedback", fb_mat, tau_feedback, False)

    network.connect(ens.getOrigin("X"), ens.getTermination("Feedback"))

    return ens


class ControlModule(spa.module.Module):
    def create(self):
        interval     = conf.present_int
        max_time_out = 10

        control = vocabs.ControlInput("Control", interval, conf.num_test_run, conf.test_type, \
                                      conf.test_option, conf.out_file, conf.mtr_file, max_time_out, \
                                      conf.vocab_data, conf.rand_seed, self.spa.net.network, \
                                      ave_motor_digit_time = interval * 3.2)
        conf.ctrl_node = control
        for term in control.getTerminations():
            term.setTau(conf.pstc)
        self.net.add(control)

        conf.est_runtime = control.getEstRuntime()

        # Set default spa vocabulary
        self.spa.default_vocabs[(conf.num_dim, False)] = conf.vocab_data.vis_vocab
        self.spa.default_vocabs[(conf.num_dim, True)]  = conf.vocab_data.vis_vocab

        self.add_source(control.getOrigin("X"))
        self.exposeTermination(control.getTermination("Cont"), "stimulus_cont")
        self.exposeTermination(control.getTermination("MotorOut"), "stimulus_motorout")
        return


class VisionModule(spa.module.Module):
    def create(self):
        vocab  = conf.vocab_data.vis_vocab

        list_strs = read_csv(conf.vis_filepath + conf.sym_list_filename, True)

        def check_input( str_list, vocab ):
            return_val = []
            valid_strs = []
            vocab_keys = vocab.hrr.keys()
            for str in str_list:
                if( str in vocab_keys ):
                    return_val.append(vocab.hrr[str].v)
                    valid_strs.append(str)
                else:
                    return_val.append(None)
            return (return_val, valid_strs)
        
        vocab_vecs, conf.valid_vis_strs = check_input( [str[0] for str in list_strs], vocab )

        vis_node = VisualHeirachy(file_path = conf.vis_filepath, mu_filename = conf.mu_filename, \
                                  output_vecs = vocab_vecs, mut_inhib_scale = 2.0, en_bypass = True, \
                                  sim_mode = "ideal", sim_mode_cleanup = SimulationMode.DEFAULT, \
                                  en_neuron_rep = True)
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

        Detector("TaskInit"             , detect_vec = conf.vocab_data.task_vocab.hrr["X"].v   , **params_det)
        self.net.get("TaskInit").addTermination("Inhib", [[-2]] * params_det["num_neurons"], conf.pstc_inhib, False)
        Detector("PS Reset"             , detect_vec = conf.vocab_data.ps_reset_vecs           , **params_det)
        Detector("PS State Reset"       , detect_vec = conf.vocab_data.ps_state_reset_vecs     , **params_det)
        Detector("PS Cycle"             , detect_vec = conf.vocab_data.ps_cycle_vecs           , **params_det)
        self.net.get("PS Cycle").addDecodedTermination("TaskInit", [[1]], conf.pstc, False)
        task_Cyc    = create_ens("Task Cyc", network = self.net, num_neurons = params_det["num_neurons"], \
                                 en_inhib = False, num_dim = 1, sim_mode = conf.det_mode, quick = True)
        task_Cyc.addDecodedTermination("TaskInit", [[1]], conf.pstc, False)

        subtask_mag = Detector("SubtaskMag" , detect_vec = num_vector(params_GI["in_scale"],1,conf.vocab_data.subtask_dim), \
                               en_N_out = True, sim_mode = SimulationMode.DEFAULT, detect_threshold = 0.4, \
                               tau_in = conf.pstc, num_neurons = 20, rand_seed = conf.rand_seed)
        self.net.add(subtask_mag)

        ps_vis_terms = [self.net.get("PS Reset").getTermination("Input"), \
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

        self.add_source(state_Buf.getOrigin("X")   , "stateo")
        self.add_source(state_Buf.getOrigin("AXON"), "statea")
        self.add_source(task_Buf.getOrigin("X")    , "tasko")
        self.add_source(subtask_Buf.getOrigin("X") , "subtasko")
        self.add_sink(state_Buf  , "state")
        self.add_sink(task_Buf   , "task")
        self.add_sink(subtask_in , "subtask")

        self.exposeTermination(self.net.get("TaskInit").getTermination("Inhib"), "NumsN")
        self.exposeOrigin(self.net.get("PS Reset").getOrigin("X"), "ps_reset")

        self.spa.vocabs["ps_state"]    = conf.vocab_data.state_vocab
        self.spa.vocabs["ps_stateo"]   = conf.vocab_data.state_vocab
        self.spa.vocabs["ps_task"]     = conf.vocab_data.task_vocab
        self.spa.vocabs["ps_tasko"]    = conf.vocab_data.task_vocab
        self.spa.vocabs["ps_subtask"]  = conf.vocab_data.subtask_vocab
        self.spa.vocabs["ps_subtasko"] = conf.vocab_data.subtask_vocab

        self.net.releaseMemory()
        return

    def connect(self):
        self.spa.net.connect(self.spa.getModuleOrigin("vis", "vis"), \
							 self.spa.getModuleTermination(self.name, "vis"))
        return


class MemoryBufferModule(spa.module.Module):
    def create(self):
        params_det = dict(net = self.net, num_neurons = 30, tau_in = conf.pstc, tau_inhib = conf.pstc, \
                          sim_mode = conf.det_mode, rand_seed = conf.rand_seed, detect_threshold = 0.5)
        params_MB  = dict(num_dim = conf.num_dim, mode = 1, neurons_per_dim = 50, sim_mode = conf.MB_mode, \
                          rand_seed = conf.rand_seed, tau_in = conf.pstc, inhib_scale = conf.inhib_scale)

        Detector("Tr1"        , detect_vec = conf.vocab_data.vec_trans1              , **params_det)
        Detector("Tr2 Skp Cnt", detect_vec = conf.vocab_data.vec_trans2_skip_cnt     , **params_det)
        Detector("Skp"        , detect_vec = conf.vocab_data.vec_skip                , **params_det)
        Detector("Tasks"      , detect_vec = conf.vocab_data.sum_task_vis_vecs       , **params_det)
        Detector("AP_"        , detect_vec = conf.vocab_data.subtask_vocab["AP"].v   , en_out = False, en_N_out = True, **params_det)
        Detector("AK_"        , detect_vec = conf.vocab_data.subtask_vocab["AK"].v   , en_out = False, en_N_out = True, **params_det)
        Detector("AP+AK"      , detect_vec = conf.vocab_data.vec_ap_ak               , **params_det)
        Detector("Nums"       , detect_vec = conf.vocab_data.sum_num_vecs            , en_N_out = True, **params_det)
        Detector("DEC"        , detect_vec = conf.vocab_data.task_vocab.hrr["DEC"].v , **params_det)
        Detector("stCNT"      , detect_vec = conf.vocab_data.state_vocab.hrr["CNT"].v, en_out = False, en_N_out = True, **params_det)
        self.net.get("DEC").addTermination("CNT", [[-2]] * params_det["num_neurons"], conf.pstc_inhib, False)

        Detector("MB1 Rst"    , detect_vec = conf.vocab_data.item_reset_vecs      , **params_det)
        self.net.get("MB1 Rst").addTermination("tr1", [[-2]] * params_det["num_neurons"], conf.pstc_inhib, False)
        Detector("MB1 Cyc"    , detect_vec = conf.vocab_data.sum_num_vecs         , **params_det)
        self.net.get("MB1 Cyc").addDecodedTermination("cnt_cyc", [[1]], conf.pstc, False)
        self.net.get("MB1 Cyc").addTermination("tr1", [[-2]] * params_det["num_neurons"], conf.pstc_inhib, False)
        self.net.get("MB1 Cyc").addTermination("dec", [[-2]] * params_det["num_neurons"], conf.pstc_inhib, False)
        Detector("MB2 Rst"    , detect_vec = conf.vocab_data.item_reset_vecs      , **params_det)
        self.net.get("MB2 Rst").addTermination("tr2_skp_cnt", [[-2]] * params_det["num_neurons"], conf.pstc_inhib, False)
        Detector("MB2 Cyc"    , detect_vec = conf.vocab_data.sum_num_vecs         , **params_det)
        self.net.get("MB2 Cyc").addTermination("tr2_skp_cnt", [[-2]] * params_det["num_neurons"], conf.pstc_inhib, False)
        self.net.get("MB2 Cyc").addTermination("dec"        , [[-2]] * params_det["num_neurons"], conf.pstc_inhib, False)
        self.net.connect(self.net.get("Tr1").getOrigin("X")        , self.net.get("MB1 Cyc").getTermination("tr1"))
        self.net.connect(self.net.get("DEC").getOrigin("X")        , self.net.get("MB1 Cyc").getTermination("dec"))
        self.net.connect(self.net.get("Tr1").getOrigin("X")        , self.net.get("MB1 Rst").getTermination("tr1"))
        self.net.connect(self.net.get("Tr2 Skp Cnt").getOrigin("X"), self.net.get("MB2 Cyc").getTermination("tr2_skp_cnt"))
        self.net.connect(self.net.get("DEC").getOrigin("X")        , self.net.get("MB2 Cyc").getTermination("dec"))
        self.net.connect(self.net.get("Tr2 Skp Cnt").getOrigin("X"), self.net.get("MB2 Rst").getTermination("tr2_skp_cnt"))

        MB_1   = MemBlock("MB1"   , reset_opt = 3, en_gint_out = True, cyc_opt = 1, in_scale = 1.0, **params_MB)
        MB_1.addDecodedTermination("Input2", diag(conf.num_dim, value = 1 + conf.fdbk_val), conf.pstc, False)
        MB_2   = MemBlock("MB2"   , reset_opt = 3, en_gint_out = True, cyc_opt = 1, in_scale = 1.0, **params_MB)
        MB_2.addDecodedTermination("Input2", diag(conf.num_dim, value = 1 + conf.fdbk_val), conf.pstc, False)
        self.net.add(MB_1)
        self.net.add(MB_2)
        self.net.connect(self.net.get("MB1 Cyc").getOrigin("X"), MB_1.getTermination("Cycle"))
        self.net.connect(self.net.get("MB1 Rst").getOrigin("X"), MB_1.getTermination("Reset"))
        self.net.connect(self.net.get("MB2 Cyc").getOrigin("X"), MB_2.getTermination("Cycle"))
        self.net.connect(self.net.get("MB2 Rst").getOrigin("X"), MB_2.getTermination("Reset"))
        self.net.connect(MB_2.getOrigin("X"), MB_2.getTermination("Input2"))

        MB1_Fdbk  = create_ens("MB1 Fdbk", network = self.net, num_neurons = params_MB['neurons_per_dim'] * conf.num_dim, \
                               num_dim = conf.num_dim, en_inhib = True, sim_mode = conf.MB_mode, \
                               inhib_scale = conf.inhib_scale, quick = True)
        self.net.connect(MB_1.getOrigin("X"), MB1_Fdbk.getTermination("Input"))
        self.net.connect(MB1_Fdbk.getOrigin("X"), MB_1.getTermination("Input2"))

        MB1_Out   = create_ens("MB1 Out", network = self.net, num_neurons = 1, in_psc = conf.pstc, \
                               num_dim = conf.num_dim, en_inhib = False, sim_mode = SimulationMode.DIRECT)
        MB1_Out.addDecodedTermination("Input2", eye(conf.num_dim), conf.pstc, False)
        self.net.connect(MB_1.getOrigin("X"), MB1_Out.getTermination("Input"))

        MB2_Out = Selector("MB2 Out", num_dim = conf.num_dim, num_items = 3, sim_mode = conf.MB_mode, \
                           inhib_scale = conf.inhib_scale)

        AM1     = CleanupMem("AM1"  , conf.vocab_data.posxnum0_vec_list, conf.vocab_data.num_vec_list, \
                             en_mut_inhib = True, tau_in = 0.01, in_scale = conf.CUinScale, \
                             threshold = conf.CUthreshold)
        AM2     = CleanupMem("AM2"  , conf.vocab_data.posxnum1_vec_list, conf.vocab_data.pos_vec_list, \
                             en_mut_inhib = True, tau_in = 0.01, in_scale = conf.CUinScale, \
                             threshold = conf.CUthreshold)
        self.net.add(MB2_Out)
        self.net.add(AM1)
        self.net.add(AM2)
        self.net.connect(MB_2.getOrigin("X") , AM1.getTermination("Input"))
        self.net.connect(MB_2.getOrigin("X") , AM2.getTermination("Input"))
        self.net.connect(MB_2.getOrigin("X") , MB2_Out.getTermination("Input 1"))
        self.net.connect(AM1.getOrigin("X")  , MB2_Out.getTermination("Input 2"))
        self.net.connect(AM2.getOrigin("X")  , MB2_Out.getTermination("Input 3"))
        self.net.connect(self.net.get("AP+AK").getOrigin("X"), MB2_Out.getTermination("Suppress 1"))
        self.net.connect(self.net.get("AK_N").getOrigin("X") , MB2_Out.getTermination("Suppress 2"))
        self.net.connect(self.net.get("AP_N").getOrigin("X") , MB2_Out.getTermination("Suppress 3"))

        if( conf.en_buf ):
            MB_1b  = MemBlock("MB1B"   , reset_opt = 3, en_gint_out = True, fb_scale = conf.decay_val, cyc_opt = 1, in_scale = 1.5, **params_MB)
            MB_1b.addDecodedTermination("Input2", diag(conf.num_dim, value = 1), conf.pstc, False)
            MB_2b  = MemBlock("MB2B"   , reset_opt = 3, en_gint_out = True, fb_scale = conf.decay_val, cyc_opt = 1, in_scale = 1.5, **params_MB)
            MB_2b.addDecodedTermination("Input2", diag(conf.num_dim, value = 1), conf.pstc, False)
            self.net.add(MB_1b)
            self.net.add(MB_2b)
            self.net.connect(self.net.get("MB1 Cyc").getOrigin("X"), MB_1b.getTermination("Cycle"))
            self.net.connect(self.net.get("MB1 Rst").getOrigin("X"), MB_1b.getTermination("Reset"))
            self.net.connect(self.net.get("MB2 Cyc").getOrigin("X"), MB_2b.getTermination("Cycle"))
            self.net.connect(self.net.get("MB2 Rst").getOrigin("X"), MB_2b.getTermination("Reset"))
            self.net.connect(MB_2b.getOrigin("X"), MB_2b.getTermination("Input2"))

            MB1b_Fdbk = create_ens("MB1B Fdbk", network = self.net, num_neurons = 30 * conf.num_dim, \
                                   num_dim = conf.num_dim, en_inhib = True, sim_mode = conf.MB_mode, \
                                   inhib_scale = conf.inhib_scale, quick = True)
            self.net.connect(MB_1b.getOrigin("X"), MB1b_Fdbk.getTermination("Input"))
            self.net.connect(MB1b_Fdbk.getOrigin("X"), MB_1b.getTermination("Input2"))
            self.net.connect(MB_1b.getOrigin("X"), MB1_Out.getTermination("Input2"))

            MB2_Out.addDecodedTerminations(input_list = [1])
            self.net.connect(MB_2b.getOrigin("X"), MB2_Out.getTermination("Input 1_2"))

        Detector("Ave Cyc", detect_vec = conf.vocab_data.vis_vocab.hrr["CLOSE"].v, **params_det)
        self.net.get("Ave Cyc").addTermination("skp", [[-2]] * 30, conf.pstc_inhib, False)

        tranf_scale = conf.tranf_scale
        ave_mult_new = tranf_scale
        ave_mult_old = 1 - tranf_scale
        
        MB_Ave = MemBlock("MB Ave", reset_opt = 3, in_scale = ave_mult_new, cyc_opt = 1, **params_MB)
        MB_Ave.addDecodedTermination("Old", diag(conf.num_dim, value = ave_mult_old), 0.005, False)
        self.net.add(MB_Ave)
        self.net.connect(self.net.get("Skp").getOrigin("X")    , self.net.get("Ave Cyc").getTermination("skp"))
        self.net.connect(self.net.get("Ave Cyc").getOrigin("X"), MB_Ave.getTermination("Cycle"))
        self.net.connect(self.net.get("Tasks").getOrigin("X")  , MB_Ave.getTermination("Reset"))
        self.net.connect(MB_Ave.getOrigin("X")                 , MB_Ave.getTermination("Old"))

        Detector("Cnt Cyc"   , **params_det)
        self.net.get("Cnt Cyc").addTermination("ps_reset", [[-2]] * 30, conf.pstc_inhib, False)
        self.net.get("Cnt Cyc").addTermination("cntn"    , [[-2]] * 30, conf.pstc_inhib, False)
        self.net.get("Cnt Cyc").addTermination("decn"    , [[-2]] * 30, conf.pstc_inhib, False)
        MB_Cnt  = MemBlock("MB Cnt" , reset_opt = 3, reset_vec = cconv(conf.vocab_data.vis_vocab.hrr["POS1"].v, \
                           conf.vocab_data.vis_vocab.hrr["ZER"].v), cyc_opt = 1, **params_MB)
        MB_Cnt_Out = create_ens("MBCnt Out", network = self.net, num_neurons = params_MB['neurons_per_dim'] * conf.num_dim, \
                               num_dim = conf.num_dim, en_inhib = True, inhib_scale = conf.inhib_scale, sim_mode = conf.MB_mode, quick = True)
        self.net.add(MB_Cnt)
#        self.net.connect(self.net.get("Tasks").getOrigin("X")  , MB_Cnt.getTermination("Reset"))
        self.net.connect(self.net.get("Cnt Cyc").getOrigin("X"), MB_Cnt.getTermination("Cycle"))
        self.net.connect(self.net.get("Cnt Cyc").getOrigin("X"), self.net.get("MB1 Cyc").getTermination("cnt_cyc"))
        self.net.connect(MB_Cnt.getOrigin("X")                 , MB_Cnt_Out.getTermination("Input"))
        self.net.connect(self.net.get("stCNTN").getOrigin("X") , MB_Cnt_Out.getTermination("Inhib"))

#        MB_Vis  = GatedInt("MB Vis", en_reset = True, en_cyc_in = False, cyc_opt = 1, num_dim = conf.vis_dim, \
#                           mode = 0, neurons_per_dim = params_MB['neurons_per_dim'], tau_in = 0.025, tau_buf_in = 0.06)
        MB_Vis  = GatedInt("MB Vis", en_reset = True, en_cyc_in = False, cyc_opt = 1, num_dim = conf.vis_dim, \
                           mode = 0, neurons_per_dim = params_MB['neurons_per_dim'], tau_in = 0.05, tau_buf_in = 0.01)
        self.net.add(MB_Vis)
        self.net.connect(self.net.get("Tasks").getOrigin("X")  , MB_Vis.getTermination("Reset"))
        self.net.connect(self.net.get("Nums").getOrigin("X")   , MB_Vis.getTermination("Cycle"))
        self.net.connect(self.net.get("NumsN").getOrigin("X")  , MB_Vis.getTermination("CycleN"))

        vis_terms        = [self.net.get("MB1 Rst").getTermination("Input"), \
                            self.net.get("MB1 Cyc").getTermination("Input"), \
                            self.net.get("MB2 Rst").getTermination("Input"), \
                            self.net.get("MB2 Cyc").getTermination("Input"), \
                            self.net.get("Ave Cyc").getTermination("Input"), \
                            self.net.get("Tasks").getTermination("Input"), \
                            self.net.get("Nums").getTermination("Input"), \
                            self.net.get("NumsN").getTermination("Input")]
        state_terms      = [self.net.get("Tr1").getTermination("Input"), \
                            self.net.get("Tr2 Skp Cnt").getTermination("Input"), \
                            self.net.get("Skp").getTermination("Input"), \
                            self.net.get("stCNTN").getTermination("Input")]
        subtask_terms    = [self.net.get("AP_N").getTermination("Input"), \
                            self.net.get("AK_N").getTermination("Input"), \
                            self.net.get("AP+AK").getTermination("Input")]
        trans_out2_terms = [MB_1.getTermination("Input") , MB_2.getTermination("Input")]
        if( conf.en_buf ):
            trans_out2_terms.extend([MB_1b.getTermination("Input"), MB_2b.getTermination("Input")])
        ps_reset_terms   = [self.net.get("Cnt Cyc").getTermination("ps_reset"), MB_Cnt.getTermination("Reset")] ##
        cnt_terms        = [MB1_Fdbk.getTermination("Inhib"), self.net.get("DEC").getTermination("CNT")]
        if( conf.en_buf ):
            cnt_terms.append(MB1b_Fdbk.getTermination("Inhib"))
        cntn_terms       = [self.net.get("Cnt Cyc").getTermination("cntn")]

        vis_term        = EnsembleTermination(self.net.network, "vis", vis_terms)
        state_term      = EnsembleTermination(self.net.network, "state", state_terms)
        subtask_term    = EnsembleTermination(self.net.network, "subtask", subtask_terms)
        trans_out2_term = EnsembleTermination(self.net.network, "trans_out2", trans_out2_terms)
        ps_reset_term   = EnsembleTermination(self.net.network, "ps_reset", ps_reset_terms) ##
        cnt_term        = EnsembleTermination(self.net.network, "cnt", cnt_terms)
        cntn_term       = EnsembleTermination(self.net.network, "cntn", cntn_terms)

        self.add_source(MB1_Out.getOrigin("X")   , "MB1")
        self.add_source(MB2_Out.getOrigin("X")   , "MB2")
        self.add_source(MB_2.getOrigin("X")      , "MB2_Norm")
        self.add_source(MB_Ave.getOrigin("X")    , "MBAve")
        self.add_source(MB_Cnt_Out.getOrigin("X"), "MBCnt")
        self.add_source(MB_Vis.getOrigin("X")    , "MBVis")

        self.exposeTermination(vis_term       , "vis")
        self.exposeTermination(self.net.get("DEC").getTermination("Input"), "task")
        self.exposeTermination(state_term     , "state")
        self.exposeTermination(subtask_term   , "subtask")
        self.exposeTermination(trans_out2_term, "trans_out2")
        self.exposeTermination(ps_reset_term  , "ps_reset")
        self.exposeTermination(cntn_term      , "cntn")
        self.exposeTermination(cnt_term       , "cnt")
        self.exposeTermination(MB_Ave.getTermination("Input"), "trans_cconv1")
        self.exposeTermination(MB_Cnt.getTermination("Input"), "trans_cconv2")
        self.exposeTermination(MB_Vis.getTermination("Input"), "vis_raw")
        self.exposeTermination(self.net.get("Cnt Cyc").getTermination("Input")   , "motor_busy")
        self.exposeTermination(self.net.get("Cnt Cyc").getTermination("decn")    , "decn")
        self.exposeOrigin(self.net.get("Tr1").getOrigin("X")        , "tr1")
        self.exposeOrigin(self.net.get("Tr2 Skp Cnt").getOrigin("X"), "tr2_skp_cnt")
        self.exposeOrigin(self.net.get("NumsN").getOrigin("X")      , "NumsN")
        self.net.releaseMemory()

        return

    def connect(self):
        self.spa.net.connect(self.spa.getModuleOrigin("vis", "vis"), \
							 self.spa.getModuleTermination(self.name, "vis"))
        self.spa.net.connect(self.spa.getModuleOrigin("vis", "vis_raw"), \
							 self.spa.getModuleTermination(self.name, "vis_raw"))
        self.spa.net.connect(self.spa.getModuleOrigin(self.name, "NumsN"), \
							 self.spa.getModuleTermination("ps", "NumsN"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_tasko"), \
							 self.spa.getModuleTermination(self.name, "task"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_stateo"), \
							 self.spa.getModuleTermination(self.name, "state"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_subtasko"), \
                             self.spa.getModuleTermination(self.name, "subtask"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_reset"), \
                             self.spa.getModuleTermination(self.name, "ps_reset"))
        self.spa.net.connect(self.spa.getModuleOrigin("dec", "cnt"), \
                             self.spa.getModuleTermination(self.name, "cnt"))
        self.spa.net.connect(self.spa.getModuleOrigin("dec", "cntn"), \
                             self.spa.getModuleTermination(self.name, "cntn"))
        self.spa.net.connect(self.spa.getModuleOrigin("dec", "decn"), \
                             self.spa.getModuleTermination(self.name, "decn"))
        self.spa.net.connect(self.spa.getModuleOrigin("motor", "motor_busy"), \
                             self.spa.getModuleTermination(self.name, "motor_busy"))
        return


class TransformModule(spa.module.Module):
    def create(self):
        params_det = dict(net = self.net, num_neurons = 30, tau_in = conf.pstc, tau_inhib = conf.pstc, \
                          sim_mode = conf.det_mode, rand_seed = conf.rand_seed, detect_threshold = 0.75)

        In_Sel1 = Selector("Input Sel1", num_dim = conf.num_dim, sim_mode = conf.MB_mode, inhib_scale = conf.inhib_scale)
        In_Sel2 = Selector("Input Sel2", num_dim = conf.num_dim, num_items = 3, sim_mode = conf.MB_mode, inhib_scale = conf.inhib_scale)
        In_Sel2.addSuppressTerminations([1,2])
        In_Sel3 = Selector("Input Sel3", num_dim = conf.num_dim, sim_mode = conf.MB_mode, inhib_scale = conf.inhib_scale)
        In_Sel4 = Selector("Input Sel4", num_dim = conf.num_dim, sim_mode = conf.MB_mode, inhib_scale = conf.inhib_scale)
        self.net.add(In_Sel1)
        self.net.add(In_Sel2)
        self.net.add(In_Sel3)
        self.net.add(In_Sel4)
        self.net.connect(In_Sel2.getOrigin("X"), In_Sel4.getTermination("Input 1"))

        cconv_t = create_ens("CConv Trans Out", network = self.net, num_dim = conf.num_dim, \
                             **params_cconv_out)
        cconv_trans = make_convolution(self.net, "CConv Trans", None, None, cconv_t, conf.neur_cconv, \
                                       radius = 6, invert_second = True, quick = True, mode = conf.cconv_mode)
        self.net.connect(In_Sel1.getOrigin("X"), cconv_trans.getTermination("A"))
        self.net.connect(In_Sel4.getOrigin("X"), cconv_trans.getTermination("B"))

        cconv_a = create_ens("CConv Ans Out", network = self.net, num_dim = conf.num_dim, \
                             **params_cconv_out)
        cconv_ans = make_convolution(self.net, "CConv Ans", None, None, cconv_a, conf.neur_cconv, \
                                     radius = 6, quick = True, mode = conf.cconv_mode)
        self.net.connect(In_Sel2.getOrigin("X"), cconv_ans.getTermination("A"))
        self.net.connect(In_Sel3.getOrigin("X"), cconv_ans.getTermination("B"))

        vec_subtask_KP  = [conf.vocab_data.subtask_vocab.hrr["AP"].v[n] + \
                           conf.vocab_data.subtask_vocab.hrr["AK"].v[n]  for n in range(conf.vocab_data.subtask_dim)]
        vec_subtask_FBC = [conf.vocab_data.subtask_vocab.hrr["MF"].v[n] + \
                           conf.vocab_data.subtask_vocab.hrr["MB"].v[n] + \
                           conf.vocab_data.subtask_vocab.hrr["CNT"].v[n] for n in range(conf.vocab_data.subtask_dim)]
        vec_subtask_As  = zeros(1, conf.vocab_data.subtask_dim)
        for i in range(conf.learn_actions):
            vec_subtask_As = [vec_subtask_As[n] + conf.vocab_data.subtask_vocab.hrr["A%i" % (i+1)].v[n] for n in range(conf.vocab_data.subtask_dim)]

        Detector("SubTsk KP" , detect_vec = vec_subtask_KP , **params_det)
        Detector("SubTsk FBC", detect_vec = vec_subtask_FBC, **params_det)
        Detector("SubTsk As" , detect_vec = vec_subtask_As , **params_det)
        Detector("NON"       , detect_vec = conf.vocab_data.subtask_vocab.hrr["NON"].v, **params_det)
        Detector("AK"        , detect_vec = conf.vocab_data.subtask_vocab.hrr["AK"].v , en_out = False, \
                 en_N_out = True, **params_det)
        Detector("AP"        , detect_vec = conf.vocab_data.subtask_vocab.hrr["AP"].v , en_out = False, \
                 en_N_out = True, **params_det)
        Out_Sel1 = Selector("Output Sel1", num_dim = conf.num_dim, num_items = 4, sim_mode = conf.MB_mode, inhib_scale = conf.inhib_scale)
        Out_Sel1.addDecodedTerminations([3])
        # Out Sel1 inhib signals
        # ======================
        # Ind tasks (suppress1) = vec_subtask_KP  + vec_subtask_FBC + vec_subtask_As + state_LEARN
        # Mem tasks (suppress2) = vec_subtask_KP  + vec_subtask_As  + "NON" + state_LEARN
        # QA  tasks (suppress3) = vec_subtask_FBC + vec_subtask_As  + "NON" + state_LEARN
        # Learn task(suppress4) = vec_subtask_KP  + vec_subtask_FBC + "NON"

        #Out_Sel1.addSuppressTerminations([1,2,4])# Subtask_KP  terminations (1,2,4) (using default terminations)
        Out_Sel1.addSuppressTerminations([1,4])   # Subtask_FBC terminations (1,3,4) (3 is using default termination)
        Out_Sel1.addSuppressTerminations([1,2,3]) # Subtask_As  terminations (1,2,3)
        Out_Sel1.addSuppressTerminations([2,3,4]) # Subtask_NON terminations (2,3,4)
        Out_Sel1.addSuppressTerminations([1,2,3]) # State_LEARN terminations (1,2,3)
        Out_Sel2 = Selector("Output Sel2", num_dim = conf.num_dim, inhib_scale = conf.inhib_scale, sim_mode = conf.MB_mode)
        Out_AM_K = CleanupMem("Out AM K", conf.vocab_data.pos_vec_list, conf.vocab_data.posxnum1_vec_list, \
                              en_inhib = True, en_mut_inhib = True, tau_in = 0.01, in_scale = conf.CUinScale, \
                              inhib_scale = conf.inhib_scale, threshold = conf.CUthreshold) # AM for "Where is the X?" questions
        Out_AM_P = CleanupMem("Out AM P", conf.vocab_data.num_vec_list, conf.vocab_data.posxnum0_vec_list, \
                              en_inhib = True, en_mut_inhib = True, tau_in = 0.01, in_scale = conf.CUinScale, \
                              inhib_scale = conf.inhib_scale, threshold = conf.CUthreshold) # AM for "What is in pos X?" questions
        self.net.add(Out_Sel1)
        self.net.add(Out_Sel2)
        self.net.add(Out_AM_K)
        self.net.add(Out_AM_P)
        self.net.connect(cconv_t.getOrigin("X") , Out_AM_K.getTermination("Input"))
        self.net.connect(cconv_t.getOrigin("X") , Out_AM_P.getTermination("Input"))
        self.net.connect(cconv_a.getOrigin("X") , Out_Sel1.getTermination("Input 1"))
        self.net.connect(In_Sel1.getOrigin("X") , Out_Sel1.getTermination("Input 2"))
        self.net.connect(Out_AM_K.getOrigin("X"), Out_Sel1.getTermination("Input 3"))
        self.net.connect(Out_AM_P.getOrigin("X"), Out_Sel1.getTermination("Input 3_2"))
        self.net.connect(self.net.get("SubTsk KP").getOrigin("X") , Out_Sel1.getTermination("Suppress 1"))
        self.net.connect(self.net.get("SubTsk KP").getOrigin("X") , Out_Sel1.getTermination("Suppress 2"))
        self.net.connect(self.net.get("SubTsk KP").getOrigin("X") , Out_Sel1.getTermination("Suppress 4"))
        self.net.connect(self.net.get("SubTsk FBC").getOrigin("X"), Out_Sel1.getTermination("Suppress 1_2"))
        self.net.connect(self.net.get("SubTsk FBC").getOrigin("X"), Out_Sel1.getTermination("Suppress 3"))
        self.net.connect(self.net.get("SubTsk FBC").getOrigin("X"), Out_Sel1.getTermination("Suppress 4_2"))
        self.net.connect(self.net.get("SubTsk As").getOrigin("X") , Out_Sel1.getTermination("Suppress 1_3"))
        self.net.connect(self.net.get("SubTsk As").getOrigin("X") , Out_Sel1.getTermination("Suppress 2_2"))
        self.net.connect(self.net.get("SubTsk As").getOrigin("X") , Out_Sel1.getTermination("Suppress 3_2"))
        self.net.connect(self.net.get("NON").getOrigin("X")       , Out_Sel1.getTermination("Suppress 2_3"))
        self.net.connect(self.net.get("NON").getOrigin("X")       , Out_Sel1.getTermination("Suppress 3_3"))
        self.net.connect(self.net.get("NON").getOrigin("X")       , Out_Sel1.getTermination("Suppress 4_3"))
        self.net.connect(self.net.get("AKN").getOrigin("X")       , Out_AM_K.getTermination("Inhib"))
        self.net.connect(self.net.get("APN").getOrigin("X")       , Out_AM_P.getTermination("Inhib"))
        self.net.connect(cconv_t.getOrigin("X"), Out_Sel2.getTermination("Input 2"))

        add_vec  = conf.vocab_data.vis_vocab.hrr["ADD"].v
        add_vec_scaled = [add_vec[n] * conf.add_scale for n in range(len(add_vec))]
        add_in   = MemoryInput("ADD", add_vec_scaled)
        add_i_in = MemoryInput("ADD_I", invol(add_vec_scaled))
        self.net.add(add_in)
        self.net.add(add_i_in)
        self.net.connect(add_in.getOrigin("X")  , In_Sel2.getTermination("Input 3"))
        self.net.connect(add_i_in.getOrigin("X"), In_Sel4.getTermination("Input 2"))

        w_vecs     = eye(len(conf.vocab_data.num_strs))
        w_classify = make_VisHeir_AM(self.net, "W Classifier", conf.vis_filepath, conf.mu_filename, w_vecs)
        w_trans    = TraceMotorTransform("W Trans", conf.mtr_filepath, conf.vocab_data.num_strs, conf.vis_dim, conf.motor_dim, \
                                         tau_in = conf.pstc, tau_inhib = conf.pstc, in_strs = conf.vocab_data.num_strs, \
                                         inhib_scale = 10.0)
        self.net.add(w_trans)
        self.net.connect(w_classify.getOrigin("X"), w_trans.getTermination("Inhib"))

        mem_MB1_terms     = [In_Sel1.getTermination("Input 1")   , In_Sel2.getTermination("Input 1")]
        mem_MB2_terms     = [In_Sel1.getTermination("Input 2")   , In_Sel2.getTermination("Input 2")]
        mem_MBVis_terms   = [w_classify.getTermination("Input")  , w_trans.getTermination("Input")]
        tr1_terms         = [In_Sel1.getTermination("Suppress 1"), In_Sel2.getTermination("Suppress 2")]
        tr2_skp_cnt_terms = [In_Sel1.getTermination("Suppress 2"), In_Sel2.getTermination("Suppress 1")]
        subtask_terms     = [self.net.get("SubTsk FBC").getTermination("Input"), \
                             self.net.get("SubTsk KP").getTermination("Input"), \
                             self.net.get("SubTsk As").getTermination("Input"), \
                             self.net.get("NON").getTermination("Input"), \
                             self.net.get("AKN").getTermination("Input"), \
                             self.net.get("APN").getTermination("Input")]
        st_learn_terms    = [Out_Sel1.getTermination("Suppress 1_4"), Out_Sel1.getTermination("Suppress 2_4"),
                             Out_Sel1.getTermination("Suppress 3_4")]
        cnt_terms         = [In_Sel2.getTermination("Suppress 1_2"), In_Sel2.getTermination("Suppress 2_2"), \
                             In_Sel3.getTermination("Suppress 1")  , In_Sel4.getTermination("Suppress 1"), \
                             Out_Sel2.getTermination("Suppress 1")]
        cntn_terms        = [In_Sel2.getTermination("Suppress 3")  , In_Sel3.getTermination("Suppress 2"), \
                             In_Sel4.getTermination("Suppress 2")  , Out_Sel2.getTermination("Suppress 2")]
        mem_MB1_term      = EnsembleTermination(self.net.network, "mMB1t"  , mem_MB1_terms)
        mem_MB2_term      = EnsembleTermination(self.net.network, "mMB2t"  , mem_MB2_terms)
        mem_MBVis_term    = EnsembleTermination(self.net.network, "mMBVt"  , mem_MBVis_terms)
        tr1_term          = EnsembleTermination(self.net.network, "mt1t"   , tr1_terms)
        tr2_skp_cnt_term  = EnsembleTermination(self.net.network, "mt2sct" , tr2_skp_cnt_terms)
        subtask_term      = EnsembleTermination(self.net.network, "sbtt"   , subtask_terms)
        st_learn_term     = EnsembleTermination(self.net.network, "stlearn", st_learn_terms)
        cnt_term          = EnsembleTermination(self.net.network, "cntt"   , cnt_terms)
        cntn_term         = EnsembleTermination(self.net.network, "cntnt"  , cntn_terms)

        self.exposeTermination(mem_MB1_term                          , "mem_MB1")
        self.exposeTermination(mem_MB2_term                          , "mem_MB2")
        self.exposeTermination(In_Sel3.getTermination("Input 1")     , "mem_MBAve")
        self.exposeTermination(In_Sel3.getTermination("Input 2")     , "mem_MBCnt")
        self.exposeTermination(mem_MBVis_term                        , "mem_MBVis")
        self.exposeTermination(tr1_term                              , "tr1")
        self.exposeTermination(tr2_skp_cnt_term                      , "tr2_skp_cnt")
        self.exposeTermination(subtask_term                          , "subtask")
        self.exposeTermination(st_learn_term                         , "learn")
        self.exposeTermination(cnt_term                              , "cnt")
        self.exposeTermination(cntn_term                             , "cntn")
        self.exposeTermination(Out_Sel1.getTermination("Input 4")    , "action_out")
        self.exposeTermination(Out_Sel2.getTermination("Input 1")    , "enc_positm")

        self.exposeOrigin(Out_Sel2.getOrigin("X")               , "trans_out2")
        self.exposeOrigin(cconv_t.getOrigin("X")                , "trans_cconv1")
        self.exposeOrigin(cconv_a.getOrigin("X")                , "trans_cconv2")
        self.exposeOrigin(Out_Sel1.getOrigin("X")               , "trans_out1")
        self.exposeOrigin(w_trans.getOrigin("X")                , "motor_raw")

        self.net.releaseMemory()
        return

    def connect(self):
        self.spa.net.connect(self.spa.getModuleOrigin("mem", "mem_MB1"), \
                             self.spa.getModuleTermination(self.name, "mem_MB1"))
        self.spa.net.connect(self.spa.getModuleOrigin("mem", "mem_MB2"), \
                             self.spa.getModuleTermination(self.name, "mem_MB2"))
        self.spa.net.connect(self.spa.getModuleOrigin("mem", "mem_MBAve"), \
                             self.spa.getModuleTermination(self.name, "mem_MBAve"))
        self.spa.net.connect(self.spa.getModuleOrigin("mem", "mem_MBCnt"), \
                             self.spa.getModuleTermination(self.name, "mem_MBCnt"))
        self.spa.net.connect(self.spa.getModuleOrigin("mem", "mem_MBVis"), \
                             self.spa.getModuleTermination(self.name, "mem_MBVis"))
        self.spa.net.connect(self.spa.getModuleOrigin("mem", "tr1"), \
                             self.spa.getModuleTermination(self.name, "tr1"))
        self.spa.net.connect(self.spa.getModuleOrigin("mem", "tr2_skp_cnt"), \
                             self.spa.getModuleTermination(self.name, "tr2_skp_cnt"))
        self.spa.net.connect(self.spa.getModuleOrigin(self.name, "trans_out2"), \
                             self.spa.getModuleTermination("mem", "trans_out2"))
        self.spa.net.connect(self.spa.getModuleOrigin(self.name, "trans_cconv1"), \
                             self.spa.getModuleTermination("mem", "trans_cconv1"))
        self.spa.net.connect(self.spa.getModuleOrigin(self.name, "trans_cconv2"), \
                             self.spa.getModuleTermination("mem", "trans_cconv2"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_subtasko"), \
							 self.spa.getModuleTermination(self.name, "subtask"))
        self.spa.net.connect(self.spa.getModuleOrigin(self.name, "trans_out1"), \
                             self.spa.getModuleTermination("dec", "Input"))
        self.spa.net.connect(self.spa.getModuleOrigin("dec", "cnt"), \
                             self.spa.getModuleTermination(self.name, "cnt"))
        self.spa.net.connect(self.spa.getModuleOrigin("dec", "cntn"), \
                             self.spa.getModuleTermination(self.name, "cntn"))
        self.spa.net.connect(self.spa.getModuleOrigin("vstr", "learn"), \
                             self.spa.getModuleTermination(self.name, "learn"))
        return


class EncodingModule(spa.module.Module):
    def create(self):
        params_det = dict(net = self.net, num_neurons = 30, tau_inhib = conf.pstc, \
                          sim_mode = conf.det_mode, rand_seed = conf.rand_seed)
        params_MB  = dict(num_dim = conf.num_dim, mode = 1, neurons_per_dim = 50, sim_mode = conf.MB_mode, \
                          tau_in = conf.pstc * 2, rand_seed = conf.rand_seed)

        Detector("Tr1-DEC"     , detect_vec = conf.vocab_data.vec_trans1             , **params_det)
        dec_minus_vec = [-1 * conf.vocab_data.task_vocab.hrr["DEC"].v[n] for n in range(conf.vocab_data.task_dim)]
        self.net.get("Tr1-DEC").addDecodedTermination("DEC", [dec_minus_vec], params_MB['tau_in'], False)

        Detector("stMB", detect_vec = conf.vocab_data.subtask_vocab.hrr["MB"].v, en_N_out = True, **params_det)
        self.net.get("stMB").addTermination("Inhib", [[-2]] * 30, conf.pstc_inhib, False)
        self.net.connect(self.net.get("Tr1-DEC").getOrigin("X"), self.net.get("stMB").getTermination("Inhib"))

        Detector("Pos Cyc"     , detect_vec = conf.vocab_data.pos_cyc_vecs       , tau_in = conf.pstc, **params_det)
        self.net.get("Pos Cyc").addDecodedTermination("motor_busy", [[1.2]], conf.pstc, False)
        self.net.get("Pos Cyc").addTermination("cnt_cyc", [[-2]] * 30, conf.pstc_inhib, False)
        Detector("Pos Rst"     , detect_vec = conf.vocab_data.pos_reset_vecs     , tau_in = conf.pstc * 2, **params_det)
        self.net.get("Pos Rst").addTermination("stMB"   , [[-2]] * 30, conf.pstc_inhib, False)
        Detector("Pos Cnt Cyc" , **params_det)
        self.net.get("Pos Cnt Cyc").addTermination("ps_reset", [[-2]] * 30, conf.pstc_inhib, False)
        self.net.connect(self.net.get("stMB").getOrigin("X"), \
                         self.net.get("Pos Rst").getTermination("stMB"))
        self.net.connect(self.net.get("Pos Cnt Cyc").getOrigin("X"), \
                         self.net.get("Pos Cyc").getTermination("cnt_cyc"))

        pos_cu_vecs = [conf.vocab_data.vis_vocab[pos_str].v for pos_str in conf.vocab_data.pos_strs]
        Pos_MB  = MemBlock("Pos MB", cyc_opt = 1, reset_vec = conf.vocab_data.vis_vocab.hrr["POS1"].v, \
                           cleanup_vecs = pos_cu_vecs, **params_MB)
        self.net.add(Pos_MB)
        self.net.connect(self.net.get("Pos Cyc").getOrigin("X"), Pos_MB.getTermination("Cycle"))
        self.net.connect(self.net.get("Pos Rst").getOrigin("X"), Pos_MB.getTermination("Reset"))

        inc___in = MemoryInput("Inc_input"  , conf.vocab_data.vis_vocab.hrr["INC"].v)
        inc_i_in = MemoryInput("Inc_I_input", invol(conf.vocab_data.vis_vocab.hrr["INC"].v))
        inc_in   = Selector("Inc_sel", neurons_per_dim = 30, num_dim = conf.num_dim, sim_mode = SimulationMode.DIRECT, inhib_scale = conf.inhib_scale)
        inc_in.addSuppressTerminations([2])
        self.net.add(inc___in)
        self.net.add(inc_i_in)
        self.net.add(inc_in)
        self.net.connect(inc___in.getOrigin("X"), inc_in.getTermination("Input 1"))
        self.net.connect(inc_i_in.getOrigin("X"), inc_in.getTermination("Input 2"))
        self.net.connect(self.net.get("stMB").getOrigin("X") , inc_in.getTermination("Suppress 1"))
        self.net.connect(self.net.get("stMBN").getOrigin("X"), inc_in.getTermination("Suppress 2"))
        self.net.connect(self.net.get("Tr1-DEC").getOrigin("X"), inc_in.getTermination("Suppress 2_2"))

        cconv_pi = create_ens("CConv Pos*Inc Out", network = self.net, num_dim = conf.num_dim, \
                              **params_cconv_out)
        cconv_pos_inc = make_convolution(self.net, "CConv Pos*Inc", None, None, cconv_pi, conf.neur_cconv, \
                                         quick = True) ##
        cconv_pm = create_ens("CConv Pos*Item Out", network = self.net, num_dim = conf.num_dim, \
                              **params_cconv_out)
        cconv_pos_itm = make_convolution(self.net, "CConv Pos*Item", None, None, cconv_pm, conf.neur_cconv, \
                                         quick = True, mode = conf.cconv_mode)
        self.net.connect(Pos_MB.getOrigin("X")  , cconv_pos_inc.getTermination("A"))
        self.net.connect(inc_in.getOrigin("X")  , cconv_pos_inc.getTermination("B"))
        self.net.connect(Pos_MB.getOrigin("X")  , cconv_pos_itm.getTermination("A"))
        self.net.connect(cconv_pi.getOrigin("X"), Pos_MB.getTermination("Input"))

        vis_terms        = [self.net.get("Pos Cyc").getTermination("Input"), \
                            self.net.get("Pos Rst").getTermination("Input"), \
                            cconv_pos_itm.getTermination("B")]
        subtask_terms    = [self.net.get("stMB").getTermination("Input"), \
                            self.net.get("stMBN").getTermination("Input")]

        vis_term        = EnsembleTermination(self.net.network, "vis", vis_terms)
        subtask_term    = EnsembleTermination(self.net.network, "subtask", subtask_terms)

        self.exposeTermination(vis_term       , "vis")
        self.exposeTermination(subtask_term   , "subtask")
        self.exposeTermination(self.net.get("Tr1-DEC").getTermination("Input"), "state")
        self.exposeTermination(self.net.get("Tr1-DEC").getTermination("DEC"), "task")
        self.exposeTermination(self.net.get("Pos Cnt Cyc").getTermination("Input")   , "cnt")
        self.exposeTermination(self.net.get("Pos Cnt Cyc").getTermination("ps_reset"), "ps_reset")
        self.exposeTermination(self.net.get("Pos Cyc").getTermination("motor_busy")  , "motor_busy")
        self.exposeOrigin(cconv_pm.getOrigin("X"), "enc_positm")
        self.exposeOrigin(Pos_MB.getOrigin("X")  , "MB Pos")

        self.net.releaseMemory()
        return

    def connect(self):
        self.spa.net.connect(self.spa.getModuleOrigin("vis", "vis"), \
							 self.spa.getModuleTermination(self.name, "vis"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_tasko"), \
							 self.spa.getModuleTermination(self.name, "task"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_stateo"), \
							 self.spa.getModuleTermination(self.name, "state"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_subtasko"), \
							 self.spa.getModuleTermination(self.name, "subtask"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_reset"), \
							 self.spa.getModuleTermination(self.name, "ps_reset"))
        self.spa.net.connect(self.spa.getModuleOrigin(self.name, "enc_positm"), \
                             self.spa.getModuleTermination("trans", "enc_positm"))
        self.spa.net.connect(self.spa.getModuleOrigin("motor", "motor_busy"), \
                             self.spa.getModuleTermination(self.name, "motor_busy"))
        self.spa.net.connect(self.spa.getModuleOrigin("dec", "cnt"), \
                             self.spa.getModuleTermination(self.name, "cnt"))
        return


class DecodingModule(spa.module.Module):
    def create(self):
        params_det = dict(net = self.net, num_neurons = 30, tau_in = conf.pstc, tau_inhib = conf.pstc, \
                          sim_mode = conf.det_mode, rand_seed = conf.rand_seed)

        cconv_ep = create_ens("CConv Enc*Pos' Out", network = self.net, num_dim = conf.num_dim, \
                              **params_cconv_out)
        cconv_enc_pos = make_convolution(self.net, "CConv Enc*Pos'", None, None, cconv_ep, conf.neur_cconv, \
                                         radius = 6, invert_second = True, quick = True)

        Detector("DEC"       , detect_vec = conf.vocab_data.task_vocab.hrr["DEC"].v, en_out = True, en_N_out = True, \
                 **params_det)
        Detector("Dec Phase" , detect_vec = conf.vocab_data.task_vocab.hrr["DEC"].v, **params_det)
        self.net.get("Dec Phase").addTermination("ps_reset", [[-1.2]] * 30, conf.pstc_inhib, False)
        Detector("Dec Phase", detect_vec = [1], en_out = False, en_N_out = True, **params_det)
        self.net.connect(self.net.get("Dec Phase").getOrigin("X"), \
                         self.net.get("Dec PhaseN").getTermination("Input"))

        delay = create_reset_integrator("Dec Delay", network = self.net, num_neurons = 30, \
                                        in_scale = 3.0, inhib_scale = 5.0, tau_feedback = 0.05, \
                                        quick = True)
        delay.addTermination("motor_busy", [[-5]] * 30, conf.pstc_inhib, False)
        delay.addTermination("dec_end", [[-5]] * 30, conf.pstc_inhib, False)
        self.net.connect(self.net.get("Dec PhaseN").getOrigin("X"), delay.getTermination("Inhib"))

        cleanup_in_vecs  = [conf.vocab_data.vis_vocab.hrr[num_str].v for num_str in conf.vocab_data.num_strs]
        cleanup_out_vecs = conf.vocab_data.motor_vecs[0:10]
        cleanupmem = CleanupMem("Cleanup Memory", cleanup_in_vecs, cleanup_out_vecs, en_inhib = True, \
                                en_mut_inhib = True, tau_in = 0.01, in_scale = conf.CUinScale, \
                                inhib_scale = conf.inhib_scale, num_neurons_per_vec = 30, threshold = conf.CUthreshold, \
                                en_X_out = True)
        self.net.add(cleanupmem)
        self.net.connect(cconv_ep.getOrigin("X"), cleanupmem.getTermination("Input"))
        self.net.connect(self.net.get("Dec PhaseN").getOrigin("X"), cleanupmem.getTermination("Inhib"))

        Detector("CNT", conf.vocab_data.subtask_vocab.hrr["CNT"].v, en_N_out = True, **params_det)

        Detector("Vec Nums"  , input_name = None, en_out = False, en_N_out = True, \
                 detect_threshold = conf.CUNumsThreshold, **params_det)

        for num in conf.vocab_data.num_strs:
            Detector("Vec " + num, detect_vec = conf.vocab_data.vis_vocab.hrr[num].v, en_N_out = False, \
                     detect_threshold = conf.CUNumsThreshold, **params_det)
            self.net.get("Vec NumsN").addDecodedTermination("Vec " + num, [[1]], conf.pstc, False)
            self.net.connect(cconv_ep.getOrigin("X")                  , self.net.get("Vec " + num).getTermination("Input"))
            self.net.connect(self.net.get("Vec " + num).getOrigin("X"), self.net.get("Vec NumsN").getTermination("Vec " + num))

        Detector("Dec End"   , detect_vec = [1], **params_det)
        self.net.get("Dec End").addTermination("CNT", [[-2]] * 30, conf.pstc_inhib, False)
        self.net.connect(self.net.get("Vec NumsN").getOrigin("X"), \
                         self.net.get("Dec End").getTermination("Input"))
        self.net.connect(self.net.get("CNT").getOrigin("X"), self.net.get("Dec End").getTermination("CNT"))
        self.net.connect(self.net.get("Dec End").getOrigin("X"), delay.getTermination("dec_end"))

        task_terms    = [self.net.get("DEC").getTermination("Input"), \
                         self.net.get("DECN").getTermination("Input"), \
                         self.net.get("Dec Phase").getTermination("Input")]
        subtask_terms = [self.net.get("CNT").getTermination("Input"), \
                         self.net.get("CNTN").getTermination("Input")]
        task_term     = EnsembleTermination(self.net.network, "task", task_terms)
        subtask_term  = EnsembleTermination(self.net.network, "sbtt", subtask_terms)

        self.exposeTermination(task_term                                           , "task")
        self.exposeTermination(self.net.get("Dec Phase").getTermination("ps_reset"), "ps_reset")
        self.exposeTermination(subtask_term                                        , "subtask")
        self.exposeTermination(cconv_enc_pos.getTermination("A") , "Input") 
        self.exposeTermination(cconv_enc_pos.getTermination("B") , "MB Pos")
        self.exposeTermination(delay.getTermination("Input")     , "motor_done")
        self.exposeTermination(delay.getTermination("motor_busy"), "motor_busy")
        self.exposeOrigin(cleanupmem.getOrigin("X")              , "motor_input")
        self.exposeOrigin(cleanupmem.getOrigin("x0")             , "motor_index")
        self.exposeOrigin(delay.getOrigin("X")                   , "motor_go")
        self.exposeOrigin(self.net.get("Dec End").getOrigin("X") , "stimulus_cont")
        self.exposeOrigin(self.net.get("DEC").getOrigin("X")     , "dec")
        self.exposeOrigin(self.net.get("DECN").getOrigin("X")    , "decn")
        self.exposeOrigin(self.net.get("CNT").getOrigin("X")     , "cnt")
        self.exposeOrigin(self.net.get("CNTN").getOrigin("X")    , "cntn")
        self.exposeOrigin(delay.getOrigin("X")                   , "dec_delay")

        self.net.releaseMemory()
        return

    def connect(self):
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_reset"), \
                             self.spa.getModuleTermination(self.name, "ps_reset"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_tasko"), \
							 self.spa.getModuleTermination(self.name, "task"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_subtasko"), \
							 self.spa.getModuleTermination(self.name, "subtask"))
        self.spa.net.connect(self.spa.getModuleOrigin("enc", "MB Pos"), \
							 self.spa.getModuleTermination(self.name, "MB Pos"))
        self.spa.net.connect(self.spa.getModuleOrigin("motor", "motor_done"), \
                             self.spa.getModuleTermination(self.name, "motor_done"))
        self.spa.net.connect(self.spa.getModuleOrigin("motor", "motor_busy"), \
                             self.spa.getModuleTermination(self.name, "motor_busy"))
        self.spa.net.connect(self.spa.getModuleOrigin(self.name, "motor_input"), \
                             self.spa.getModuleTermination("motor", "motor_input"))
        self.spa.net.connect(self.spa.getModuleOrigin(self.name, "motor_index"), \
                             self.spa.getModuleTermination("motor", "motor_index"))
        self.spa.net.connect(self.spa.getModuleOrigin(self.name, "motor_go"), \
                             self.spa.getModuleTermination("motor", "motor_go"))
        self.spa.net.connect(self.spa.getModuleOrigin(self.name, "stimulus_cont"), \
                             self.spa.getModuleTermination("stimulus", "stimulus_cont"))
        return


def pred_error(x):
    # for each action, prediction error is
    #         a   [        R            +   g   * V(S) -   V(S(t-1))]
    return [conf.learn_alpha * (x[i+len(x)/2] - x[i]) for i in range(len(x)/2)]

def product(x):
    return x[0] * x[1]

def rand_weights(w, range_val = 1e-3):
    for i in range(len(w)):
        for j in range(len(w[0])):
            w[i][j] = random.uniform(-range_val,range_val)
    return w

class RewardEvalModule(spa.module.Module):
    def create(self):
        neur_per_dim = 50

        params_det = dict(net = self.net, num_neurons = 30, tau_in = conf.pstc, tau_inhib = conf.pstc, \
                          sim_mode = conf.det_mode, detect_threshold = 0.35, rand_seed = conf.rand_seed)

        Detector("Learn"   , detect_vec = conf.vocab_data.state_vocab.hrr["LEARN"].v, en_out = True, en_N_out = True, **params_det)
        Detector("Rewarded", detect_vec = conf.vocab_data.vis_vocab.hrr["ONE"].v    , **params_det)
        Detector("NoReward", detect_vec = conf.vocab_data.vis_vocab.hrr["ZER"].v    , **params_det)
        Detector("AnyReward", detect_vec = [conf.vocab_data.vis_vocab.hrr["ONE"].v[n] + conf.vocab_data.vis_vocab.hrr["ZER"].v[n] \
                 for n in range(conf.vocab_data.num_dim)], en_out = False, en_N_out = True, **params_det)

        state_Q_ens = create_ens("StateQ", network = self.net, num_neurons = neur_per_dim * conf.learn_actions, \
                                 num_dim = conf.learn_actions, sim_mode = conf.det_mode, use_array = False, \
                                 en_inhib = False, input_name = None)

        eval_num_neurons = neur_per_dim * conf.learn_actions * 2
        eval_ens  = create_ens("Evaluator", network = self.net, num_neurons = eval_num_neurons, \
                               num_dim = conf.learn_actions * 2, sim_mode = conf.det_mode, use_array = False, \
                               en_inhib = False, input_name = None)
        eval_ens.addTermination("inhib_learn"    , [[-2]] * eval_num_neurons, conf.pstc_inhib, False)
        eval_ens.addTermination("inhib_decn"     , [[-2]] * eval_num_neurons, conf.pstc_inhib, False)
        eval_ens.addTermination("inhib_dec_delay", [[-2]] * eval_num_neurons, conf.pstc_inhib, False)
        eval_ens.addTermination("inhib_anyreward", [[-2]] * eval_num_neurons, conf.pstc_inhib, False)
        self.net.connect(self.net.get("LearnN").getOrigin("X"), eval_ens.getTermination("inhib_learn"))
        self.net.connect(self.net.get("AnyRewardN").getOrigin("X"), eval_ens.getTermination("inhib_anyreward"))
        pred_ori,pred_term = self.net.connect(eval_ens, state_Q_ens, func = pred_error, modulatory = True, \
                                              index_post = range(conf.learn_actions), create_projection = False)
        self.net.connect(pred_ori, pred_term)
        self.net.connect(state_Q_ens.getOrigin("X"), eval_ens, index_post = range(conf.learn_actions))

        out_ens = SimpleNEFEns("ActionOut", conf.vocab_data.num_dim, input_name = "")
        self.net.add(out_ens)

        subtask_terms = []
        for i in range(conf.learn_actions):
            Detector("Action%i" % (i+1), detect_vec = conf.vocab_data.subtask_vocab.hrr["A%i" % (i+1)].v , in_scale = 2.0, **params_det)
            subtask_terms.append(self.net.get("Action%i" % (i+1)).getTermination("Input"))

        for i in range(conf.learn_actions):
            i_str = str(i+1)
            reward_ens = self.net.make("RewardA" + i_str, neur_per_dim * 2, 2, radius = sqrt(2), \
                                       encoders = [[1,1],[1,-1],[-1,1],[-1,-1]], quick = True)
            self.net.connect(self.net.get("Rewarded"), reward_ens, weight = 1 , index_post = [0])
            self.net.connect(self.net.get("NoReward"), reward_ens, weight = -1, index_post = [0])
            for j in range(conf.learn_actions):
                weight = (i == j) - (i != j)
                self.net.connect(self.net.get("Action%i" % (j+1)), reward_ens, weight = weight, index_post = [1])

            pos_action_vec = cconv(conf.vocab_data.vis_vocab.hrr["POS1"].v, conf.vocab_data.vis_vocab.hrr[vocabs.num_strs[i]].v)
            out_ens.addDecodedTermination("Action" + i_str, [[pos_action_vec[n]] for n in range(conf.vocab_data.num_dim)], pstc = conf.pstc)
            self.net.connect(self.net.get("Action" + i_str).getOrigin("X"), \
                             out_ens.getTermination("Action" + i_str))
            self.net.connect(reward_ens, eval_ens, func = product, index_post = [conf.learn_actions + i])


        # vis_terms     = [self.net.get("Rewarded").getTermination("Input"), \
        #                  self.net.get("NoReward").getTermination("Input")]
        vis_terms     = [self.net.get("Rewarded").getTermination("Input"), \
                         self.net.get("NoReward").getTermination("Input"), \
                         self.net.get("AnyRewardN").getTermination("Input")]
        state_terms   = [self.net.get("Learn").getTermination("Input"), \
                         self.net.get("LearnN").getTermination("Input")]
        vis_term      = EnsembleTermination(self.net.network, "vis", vis_terms)
        state_term    = EnsembleTermination(self.net.network, "st" , state_terms)
        subtask_term  = EnsembleTermination(self.net.network, "stt", subtask_terms)

        self.add_source(pred_ori, "pred_error")

        self.exposeTermination(vis_term    , "vis")
        self.exposeTermination(state_term  , "ps_state")
        self.exposeTermination(subtask_term, "ps_subtask")
        self.exposeTermination(eval_ens.getTermination("inhib_dec_delay"), "dec_delay")
        self.exposeTermination(eval_ens.getTermination("inhib_decn")     , "decn")

        self.exposeOrigin(self.net.get("Learn").getOrigin("X"), "learn")
        self.exposeOrigin(out_ens.getOrigin("X"), "action_out")
        return

    def connect(self):
        self.spa.net.connect(self.spa.getModuleOrigin("vis", "vis"), \
                             self.spa.getModuleTermination(self.name, "vis"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_subtasko"), \
                             self.spa.getModuleTermination(self.name, "ps_subtask"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_stateo"), \
                             self.spa.getModuleTermination(self.name, "ps_state"))
        self.spa.net.connect(self.spa.getModuleOrigin(self.name, "action_out"), \
                             self.spa.getModuleTermination("trans", "action_out"))
        self.spa.net.connect(self.spa.getModuleOrigin("dec", "dec_delay"), \
                             self.spa.getModuleTermination(self.name, "dec_delay"))
        self.spa.net.connect(self.spa.getModuleOrigin("dec", "decn"), \
                             self.spa.getModuleTermination(self.name, "decn"))

        statea_origin = self.spa.getModuleOrigin("ps", "ps_statea")
        state_Q_ens = self.net.get("StateQ")
        axon_dim = statea_origin.dimensions
        term_dim = state_Q_ens.getNodeCount()
        state_Q_ens.addTermination("statea", rand_weights(zeros(term_dim, axon_dim), 5e-4), conf.pstc, False)
        self.exposeTermination(state_Q_ens.getTermination("statea"), "ps_statea")
        self.spa.net.connect(statea_origin, self.spa.getModuleTermination(self.name, "ps_statea"))
        self.net.learn(state_Q_ens, state_Q_ens.getTermination("statea"), self.net.get("Evaluator").name, **params_learn)
        return


class MotorModule(spa.module.Module):
    def create(self):
        params_det = dict(net = self.net, num_neurons = 30, tau_in = conf.pstc, tau_inhib = conf.pstc, \
                          sim_mode = conf.det_mode, rand_seed = conf.rand_seed)
        motor_init     = conf.motor_init
        motor_interval = 0
        motor_dim      = conf.vocab_data.motor_vecs_dim

        Detector("State VIS" , detect_vec = conf.vocab_data.state_vocab.hrr["VIS"].v, en_N_out = True, **params_det)
        motor_in   = Selector("Motor In", num_dim = conf.vocab_data.motor_vecs_dim, inhib_scale = 10.0, \
                              ens_per_array = conf.vocab_data.motor_vecs_dim, neurons_per_dim = 50)#, sim_mode = conf.MB_mode)
        self.net.add(motor_in)

        motor_neur_in = self.net.make_array("Motor Neur In", 20, motor_dim, quick = True, \
                                         max_rate = (50,75), mode = SimulationMode.DEFAULT)
        motor_neur_in.addTermination("Inhib", [[[-5]] * 20] * motor_dim, conf.pstc, False)
        self.net.connect(motor_in.getOrigin("X"), motor_neur_in, pstc = conf.pstc)

        motor_neur = self.net.make_array("Motor Neur", 20, motor_dim, quick = True, \
                                         max_rate = (50,75), mode = SimulationMode.DEFAULT)
        self.net.connect(motor_neur_in.getOrigin("X"), motor_neur, pstc = conf.pstc)

        self.net.connect(self.net.get("State VIS").getOrigin("X"), motor_in.getTermination("Suppress 1"))
        self.net.connect(self.net.get("State VISN").getOrigin("X"), motor_in.getTermination("Suppress 2"))

        motor_node = SpaunMotorNode("Motor Node", conf.vocab_data.motor_vecs_dim, len(conf.vocab_data.num_strs), \
                                    out_file = conf.out_file, raw_file = conf.mtr_file, motor_init = motor_init, \
                                    motor_valid = conf.vocab_data.motor_valid, \
                                    unk_vec = conf.vocab_data.motor_vecs[10][:conf.vocab_data.motor_valid[10]], \
                                    pstc = conf.pstc, pstc_raw = conf.pstc * 2)
        self.net.add(motor_node)
        self.net.connect(motor_in.getOrigin("X"), motor_node.getTermination("Input"))

        state_terms       = [self.net.get("State VIS").getTermination("Input"), \
                             self.net.get("State VISN").getTermination("Input")]
        suppress_terms    = [motor_node.getTermination("Suppress"), \
                             motor_neur_in.getTermination("Inhib")]
        state_term        = EnsembleTermination(self.net.network, "stt"     , state_terms)
        suppress_term     = EnsembleTermination(self.net.network, "suppress", suppress_terms)

        self.add_source(motor_node.getOrigin("X"))
        self.add_source(motor_node.getOrigin("Busy"), "busy")
        self.add_source(motor_node.getOrigin("Done"), "done")
        self.add_source(motor_node.getOrigin("Plan"), "plan")
        self.exposeTermination(state_term                           , "state")
        self.exposeTermination(motor_in.getTermination("Input 1")   , "motor_input")
        self.exposeTermination(motor_in.getTermination("Input 2")   , "motor_raw")
        self.exposeTermination(motor_node.getTermination("Index")   , "motor_index")
        self.exposeTermination(motor_node.getTermination("Go")      , "motor_go")
        self.exposeTermination(suppress_term                        , "motor_suppress")
        self.exposeTermination(motor_node.getTermination("Reset")   , "motor_reset")
        return

    def connect(self):
        self.spa.net.connect(self.spa.getModuleOrigin(self.name, "motor_plan"), \
                             self.spa.getModuleTermination("stimulus", "stimulus_motorout"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_stateo"), \
							 self.spa.getModuleTermination(self.name, "state"))
        self.spa.net.connect(self.spa.getModuleOrigin("ps", "ps_reset"), \
							 self.spa.getModuleTermination(self.name, "motor_reset"))
        self.spa.net.connect(self.spa.getModuleOrigin("dec", "cnt"), \
							 self.spa.getModuleTermination(self.name, "motor_suppress"))
        self.spa.net.connect(self.spa.getModuleOrigin("trans", "motor_raw"), \
							 self.spa.getModuleTermination(self.name, "motor_raw"))
        return


def write2file(filepath, write_data, write_mode = 'a'):
    if( not filepath == "" ):
        file_handle = open(filepath, write_mode)
        file_handle.write(write_data)
        file_handle.close()

def write_dot_prod_table(filename, vocab, log_strs):
    if filename == "":
        return
    write2file(filename, "\n#    ,")
    for vec_str in log_strs:
        write2file(filename, "  %s  ," % vec_str[0:3])
    write2file(filename, "\n#----,")
    for vec_str in log_strs:
        write2file(filename, "-------,")
    write2file(filename, "\n")

    for vec_str in log_strs:
        write2file(filename, "#%s ," % vec_str[0:3])
        for vec_str2 in log_strs:
            dot_prod = util_funcs.dot(vocab.hrr[vec_str].v, vocab.hrr[vec_str2].v)
            write2file(filename, " % .2f ," % (dot_prod))
        write2file(filename, "\n")
    write2file(filename, "\n")

def run(world, OS = "WIN", root_path = None, test_type = 0, test_option = None, num_test_run = 5, num_subjects = 1, \
        multi_thread = False, en_logging = True, rand_type = 0,#111023184024,#111010203035,#110926102608, #110921112523, #rand_type = 110518233715, \
        perfect_MB = True, perfect_cconv = True, CUthreshold = 0.4, CUNumsThreshold = 0.3, CUinScale = 1.0, \
        tranf_scale = 0.451, learn_alpha = 1.0, learn_actions = 2, present_int = 0.15, motor_init = 0.15, auto_run = 2, testing = False):

    datetime_str = datetime.datetime.today().strftime("%d/%m/%y - %H:%M:%S")
    println("# START [" + datetime_str + "]")

    if( isinstance(test_type, int) ):
        test_type = [test_type]
    if( test_option is None ):
        test_option = []
        for test_type_val in test_type:
            if( test_type_val == 1 or test_type_val == 0 ):
                test_option.append((None,False))
            if( test_type_val == 2 ):
                test_option.append([[0.21,0.63,5],[0.63,0.21,5],[0.12,0.72,5],[0.72,0.12,5]])
                learn_actions = 2
            if( test_type_val == 3 ):
                test_option.append((2,0))
            if( test_type_val == 4 ):
                test_option.append((1,1))
            if( test_type_val == 5 ):
                test_option.append((5,'K',False))
            if( test_type_val == 6 ):
                test_option.append((3,"AAxB","xB"))
            if( test_type_val == 7 ):
                test_option.append([1,2,3])

    conf.OS = OS;                     conf.test_type = test_type;             conf.test_option = test_option
    conf.num_test_run = num_test_run; conf.num_subjects = num_subjects;       conf.learn_alpha = learn_alpha
    conf.CUthreshold = CUthreshold;   conf.CUNumsThreshold = CUNumsThreshold; conf.CUinScale = CUinScale      
    conf.tranf_scale = tranf_scale;   conf.learn_actions = learn_actions;     conf.present_int = present_int
    conf.motor_init = motor_init

## TODO:
# - Add learn inhib on Out_sel1 (DONE)
# - Add default action on BG    (MAYBE?)
# - decrease learn rate even more? (MAYBE?)

## ------------------------------------- DEFINE RULE SET ------------------------------------------
    class SpaUNRules:
        def task_init(vis = "A"):
            set(ps_task = "X")

        def task_w_init(vis = "ZER", ps_tasko = "X", scale = 0.5):
            set(ps_task = "W", ps_state = "VIS", ps_subtask = "MF")
        def task_w_keep_vis(ps_tasko = "W", ps_stateo = "VIS", scale = 0.5):
            set(ps_state = "VIS")

        def task_r_init(vis = "ONE", ps_tasko = "X", scale = 0.5):
            set(ps_task = "R", ps_state = "TRANS1", ps_subtask = "MF")
        def task_r_keep_tr1(ps_tasko = "R", ps_stateo = "TRANS1", scale = 0.5):
            set(ps_state = "TRANS1")

        def task_l_init(vis = "TWO", ps_tasko = "X", scale = 0.5):
            set(ps_task = "L", ps_state = "LEARN", ps_subtask = "NON")

        for i in range(conf.learn_actions):
            code = """def task_l_a%d(ps_stateo = "LEARN-5*TRANS1-5*TRANS2-5*SKIP-5*VIS-5*CNT", scale = %f, rand_weights = rand_weights):
                          learn(ps_statea = rand_weights, pred_error = vstr_pred_error)
                          set(ps_subtask = "A%d")""" % (i+1,0.35,i+1)
            exec(code)

        def task_m_init(vis = "THR", ps_tasko = "X", scale = 0.5):
            set(ps_task = "M", ps_state = "TRANS1", ps_subtask = "MF")
        def task_m_keep_tr1(ps_tasko = "M", ps_stateo = "TRANS1", scale = 0.5):
            set(ps_state = "TRANS1")
        def task_m_set_fwd(vis = "P", ps_tasko = "M", scale = 0.5):
            set(ps_subtask = "MF")
        def task_m_set_bck(vis = "K", ps_tasko = "M", scale = 0.5):
            set(ps_subtask = "MB")

        def task_c_init(vis = "FOR", ps_tasko = "X", scale = 0.5):
            set(ps_task = "C", ps_state = "SKIP", ps_subtask = "NON")
        def task_c_set_cnt(ps_tasko = "C-DEC", ps_stateo = "TRANS1", scale = 0.5):
            set(ps_state = "CNT")
        def task_c_nomatch(ps_stateo = "CNT", scale = 1.0):
            match(mem_MB2_Norm != mem_MBCnt)
            set(ps_subtask = "CNT")
        def task_c_match(ps_stateo = "CNT", scale = 0.4):
            match(mem_MB2_Norm == mem_MBCnt)
            set(ps_subtask = "MF")

        def task_a_init(vis = "FIV", ps_tasko = "X", scale = 0.5):
            set(ps_task = "A", ps_state = "SKIP", ps_subtask = "NON")
        def task_a_tr1_2_tr2(ps_tasko = "A", ps_stateo = "TRANS1", scale = 0.5):
            set(ps_state = "TRANS2")
        def task_a_keep_tr2(ps_tasko = "A", ps_stateo = "TRANS2", scale = 0.5):
            set(ps_state = "TRANS2")
        def task_a_set_k(ps_tasko = "A", vis = "K", scale = 0.5):
            set(ps_subtask = "AK")
        def task_a_set_p(ps_tasko = "A", vis = "P", scale = 0.5):
            set(ps_subtask = "AP")

        def task_v_init(vis = "SIX", ps_tasko = "X", scale = 0.5):
            set(ps_task = "V", ps_state = "SKIP", ps_subtask = "NON")
        def task_v_tr1_2_skp(ps_tasko = "V", ps_stateo = "TRANS1", scale = 0.5):
            set(ps_state = "SKIP")

        def task_f_init(vis = "SEV", ps_tasko = "X", scale = 0.5):
            set(ps_task = "F", ps_state = "SKIP", ps_subtask = "NON")
        def task_f_tr1_2_tr2(ps_tasko = "F", ps_stateo = "TRANS1", scale = 0.5):
            set(ps_state = "TRANS2")
        def task_f_tr2_2_skp(ps_tasko = "F", ps_stateo = "TRANS2", scale = 0.5):
            set(ps_state = "SKIP")

        def task_qm(vis = "QM"):
            set(ps_task = "DEC")
        def task_skp_2_tr1(ps_stateo = "SKIP"):
            set(ps_state = "TRANS1")

## ------------------------------------- END DEFINE RULE SET ------------------------------------------
## ------------------------------------- DEFINE SPA NETWORK ------------------------------------------

    class SpaUN(spa.core.SPA):
        dimensions = conf.num_dim
        align_hrrs = True

        stimulus = ControlModule()
        vis      = VisionModule()
        ps       = ProdSysBufferModule()
        mem      = MemoryBufferModule()
        trans    = TransformModule()
        enc      = EncodingModule()
        dec      = DecodingModule()
        vstr     = RewardEvalModule()
        motor    = MotorModule()

        BG       = spa.bg.BasalGanglia(SpaUNRules(), pstc_input = 0.01, **params_BG)
        thalamus = spa.thalamus.Thalamus(bg = BG, pstc_route_input = 0.01, pstc_gate = 0.001, route_scale = 1, \
                                         pstc_inhibit = 0.01, pstc_output = 0.011, pstc_route_output = 0.01, \
                                         mutual_inhibit = 2, quick = False)

## ------------------------------------- END DEFINE SPA NETWORK ------------------------------------------

    if( perfect_MB ):
        conf.en_buf = False
        conf.en_reh = False 
    if( not conf.en_reh ):
        conf.fdbk_val = 0

    if( perfect_MB ):
        conf.MB_mode = "ideal"
    if( perfect_cconv ):
        conf.cconv_mode = "direct"

## ------------------------------------ HANDLE OUTPUT LOGGING --------------------------------------

    if( OS == "WIN" ):       default_path = "spaun\\"
    else:                    default_path = "spaun/"

    if( root_path is None or root_path == "" ): conf.root_path = default_path
    else:                                       conf.root_path = root_path

    if( OS == "WIN" ):
        conf.out_filepath = conf.root_path + "out_data\\"
        conf.vis_filepath = conf.root_path + "vis_data\\"
        conf.mtr_filepath = conf.root_path + "motor_data\\"
    else:   # LINUX / UNIX BOXES
        conf.out_filepath = conf.root_path + "out_data/"
        conf.vis_filepath = conf.root_path + "vis_data/"
        conf.mtr_filepath = conf.root_path + "motor_data/"

    sys_hostname = socket.gethostname()
    rand_time_val= int(time.clock() * 1000000) % 1000000
    datetime_str = datetime.datetime.today().strftime("%y%m%d%H%M%S")

    rand_seed = rand_type
    if( not rand_type == 0 ):
        if( rand_type == 1 ):
            rand_seed = eval(datetime.datetime.today().strftime("%y%m%d%H%M"))
    else:
        rand_seed = eval(datetime_str) + rand_time_val

    if( not multi_thread ):
        NodeThreadPool.turnOffMultithreading()
    else:
        NodeThreadPool.setNumThreads(multi_thread)

    exp_desc     = [vocabs.task_strs[test_type[i]] + str(test_option[i]) for i in range(len(test_type))]
    exp_desc_str = ",".join(exp_desc)
    exp_desc_str = exp_desc_str.replace(" ", "")
    filename  = "task_"  + exp_desc_str + "_" + datetime_str + "." + sys_hostname + ".txt"
    logname   = "log_"   + exp_desc_str + "_" + datetime_str + "." + sys_hostname + ".csv"
    motorname = "motor_" + exp_desc_str + "_" + datetime_str + "." + sys_hostname + ".txt"

    if( not OS == "" ):
        conf.out_file = conf.out_filepath + filename
        conf.mtr_file = conf.out_filepath + motorname
    else:
        conf.out_file = ""
        conf.mtr_file = ""

    if( en_logging and not OS == "" ):
        conf.log_file = conf.out_filepath + logname
    else:
        conf.log_file = ""

## ------------------------------------ RUN EXPERIMENT PROPER --------------------------------------

    # Log experiment settings
    log_str = "# OS: %s, multi_thread: %i, rand_seed_base: %d\n" % (conf.OS, multi_thread, rand_seed) + \
              "# test_type: %s, test_option: %s, num_test_run: %i, num_subjects: %i\n" % (str(conf.test_type), str(conf.test_option), conf.num_test_run, conf.num_subjects) + \
              "# present_interval: %0.5f, motor_init: %0.5f\n" % (conf.present_int, conf.motor_init) + \
              "# CUthreshold: %0.5f, CUNumsThreshold: %0.5f, CUinScale: %0.5f\n" % (conf.CUthreshold, conf.CUNumsThreshold, conf.CUinScale) + \
              "# learn_alpha: %0.5f, learn_actions: %i\n" % (conf.learn_alpha, conf.learn_actions) + \
              "# tranf_scale: %0.5f, add_scale: %0.5f\n" % (conf.tranf_scale, conf.add_scale) + \
              "# fdbk_val: %0.5f, decay_val: %0.5f, en_reh: %i, en_buf: %i\n" % (1 + conf.fdbk_val, conf.decay_val, conf.en_reh, conf.en_buf)
    write2file(conf.out_file, log_str)
    write2file(conf.log_file, log_str)
                               

    if( auto_run != 1 ):
        num_subjects = 1

    for i in range(learn_actions):
        vocabs.subtask_strs.append("A" + str(i+1))

    for subject in range(num_subjects):
        write2file(conf.out_file, "\n# ---------------------------- SUBJECT %i ---------------------------- |%i\n" % (subject+1, subject+1))
        write2file(conf.log_file, "\n# ---------------------------- SUBJECT %i ---------------------------- |%i\n" % (subject+1, subject+1))
        write2file(conf.mtr_file, "\n# ---------------------------- SUBJECT %i ---------------------------- |%i\n" % (subject+1, subject+1))

        conf.rand_seed = rand_seed + subject
        conf.vocab_data = vocabs.VocabData(rand_seed)
        conf.motor_dim = conf.vocab_data.motor_vecs_dim

        PDFTools.setSeed(rand_seed)
        random.seed(rand_seed)

        spaun = SpaUN()

        datetime_str = datetime.datetime.today().strftime("%d/%m/%y - %H:%M:%S")
        println("# RUN [" + datetime_str + "]")

        learn_term_name = "ps_statea"
        for node in spaun.net.get("BG").getNode("StrD1").getNodes():
            for term in node.getTerminations():
                if( term.name == learn_term_name ):
                    conf.ctrl_node.add_learned_termination(term, rand_weights)
        for node in spaun.net.get("BG").getNode("StrD2").getNodes():
            for term in node.getTerminations():
                if( term.name == learn_term_name ):
                    conf.ctrl_node.add_learned_termination(term, rand_weights)
        conf.ctrl_node.add_learned_termination(spaun.net.get("vstr").getNode("StateQ").getTermination("statea"), rand_weights)

        ## DATA LOGGING ##
        log_str = "# randseed:%i\n" % (rand_seed)
        write2file(conf.out_file, log_str)
        write2file(conf.log_file, log_str)
        write2file(conf.out_file, "# Neuron Count: %i\n" % spaun.net.getNeuronCount())
        write2file(conf.out_file, "# Est Run Time: %i\n#\n" % conf.est_runtime)
        ## ------------ ##

        println("Est Run Time: " + str(conf.est_runtime))
        println("Neuron Count: " + str(spaun.net.getNeuronCount()))
        write_dot_prod_table(conf.log_file, conf.vocab_data.vis_vocab, conf.vocab_data.num_strs)

        # Set default vocabularies (for interactive mode)
        hrr.Vocabulary.defaults[conf.num_dim]                = conf.vocab_data.vis_vocab
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
            sim_interval_ms = conf.present_int / conf.sim_timestep
            log_interval = int(conf.est_runtime / 2)
            while( sim_interval_ms % log_interval ):
                log_interval -= 1
            log_interval = log_interval / 1000

            logNode = nef.Log(spaun.net, "log", conf.out_filepath, logname, interval = log_interval, tau = 0.01)
            logNode.add("stimulus.stimulus")              # Raw visual input

            logNode.add("vis.Vision.layerNeur.X")         # 50D visual rep (vector)
            logNode.add_spikes("vis.Vision.layerNeur")    # 50D visual rep (spikes)
            logNode.add_vocab("vis.Vision.layerNeur.X", vis_raw_vocab, pairs = False)         # 50D visual rep (vocab)
            
            logNode.add("BG.StrD1.func_str")              # BG utility (vector)
            logNode.add_spikes("BG.StrD1.func_str")       # BG utility (spikes)
            logNode.add("BG.GPi.func_gpi")                # BG output (vector)
            logNode.add_spikes("BG.GPi.func_gpi")         # BG output (spikes)

            logNode.add_vocab("ps.ps_tasko", conf.vocab_data.task_vocab, pairs = False)       # Task buffer (vocab)
            logNode.add_spikes("ps.task.GINT2.buffer")                                        # Task buffer (spikes)
            logNode.add_vocab("ps.ps_stateo", conf.vocab_data.state_vocab, pairs = False)     # State buffer (vocab)
            logNode.add_spikes("ps.state.GINT2.buffer")                                       # State buffer (spikes)
            logNode.add_vocab("ps.ps_subtasko", conf.vocab_data.subtask_vocab, pairs = False) # Subtask buffer (vocab)
            logNode.add_spikes("ps.subtask.buffer")                                           # Subtask buffer (spikes)

            logNode.add_vocab("enc.MB Pos", conf.vocab_data.vis_vocab, terms = conf.vocab_data.pos_strs, pairs = False)
            # logNode.add_spikes("enc.Pos MB.GINT2.buffer") # Pos buffer (spikes)

            logNode.add("motor.Motor In.X")               # Motor output (vector)
            logNode.add_spikes("motor.Motor Neur")        # Motor output (spikes)

            if( 0 in test_type or 1 in test_type ):   # Copy drawing test type
                logNode.add("mem.MB Vis.buffer.X")        # Visual memory (vector)
                logNode.add_spikes("mem.MB Vis.buffer")   # Visual memory (spikes)
                logNode.add_vocab("mem.MB Vis.buffer.X", vis_raw_vocab, pairs = False) # 50D cleaned up visual rep (vocab)
            
            if( 1 in test_type or 3 in test_type ):   # Recognition test type
                logNode.add_vocab("vis.vis", conf.vocab_data.vis_vocab, \
                                  terms = conf.vocab_data.all_vis_strs, pairs = False) # 512D cleaned up visual rep (vocab)
            
            if( 2 in test_type ):   # Learning test type
                logNode.add("vstr.Evaluator.pred_error")  # vStr evaluator (vector)
                logNode.add_spikes("vstr.Evaluator")      # vStr evaluator (spikes)
                logNode.add("vstr.Rewarded")              # vStr rewarded  (vector) 
                logNode.add_spikes("vstr.Rewarded")       # vStr rewarded  (spikes)
                logNode.add("vstr.StateQ")                # vStr state representation (vector)
                for n in range(learn_actions):
                    logNode.add("vstr.RewardA" + str(n+1) + ".product") # vStr action reward
                    logNode.add_spikes("vstr.RewardA" + str(n+1))       # vStr action reward

            if( 1 in test_type or 3 in test_type or 4 in test_type or \
                5 in test_type or 6 in test_type or 7 in test_type):  # Memory test type
                logNode.add_vocab("mem.MB1.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                logNode.add_spikes("mem.MB1.GINT2.buffer")
                logNode.add_vocab("mem.MB2.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                logNode.add_spikes("mem.MB2.GINT2.buffer")
                if( conf.en_buf ):
                    logNode.add_vocab("mem.MB1B.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                    logNode.add_spikes("mem.MB1B.GINT2.buffer")
                    logNode.add_vocab("mem.MB2B.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                    logNode.add_spikes("mem.MB2B.GINT2.buffer")
            
            if( 4 in test_type ):    # Counting test type
                logNode.add_vocab("mem.MB Cnt.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                logNode.add_spikes("mem.MB Cnt.GINT2.buffer")
                logNode.add_vocab("trans.CConv Trans Out.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                logNode.add_vocab("trans.CConv Ans Out.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                logNode.add_vocab("trans.trans_out1", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                logNode.add_vocab("trans.trans_out2", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
            
            if( 5 in test_type ):    # QA test type
                logNode.add_vocab("trans.Output Sel1.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                logNode.add_spikes("trans.CConv Trans")
            
            if( 6 in test_type or 7 in test_type ):    # Induction test types
                logNode.add_vocab("mem.MB Ave.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                logNode.add_spikes("mem.MB Ave.GINT2.buffer")
                logNode.add_vocab("trans.CConv Trans Out.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                logNode.add_vocab("trans.CConv Ans Out.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                logNode.add_vocab("trans.trans_out1", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                logNode.add_vocab("trans.trans_out2", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)

            if( en_logging == 2 ):
                logNode.add_vocab("mem.AM1.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                logNode.add_vocab("mem.AM2.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                logNode.add_vocab("mem.MB2 Out.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                logNode.add_vocab("trans.Out AM K.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                logNode.add_vocab("trans.Out AM P.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
                logNode.add_vocab("trans.Output Sel1.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
#            logNode.add_vocab("mem.mem_MBCnt", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
#            logNode.add_vocab("mem.MBCnt_DebugOut", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
#            logNode.add_vocab("mem.MB Cnt.GINT1.gate.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
#            logNode.add_vocab("mem.MB Cnt.GINT2.gate.X", conf.vocab_data.vis_vocab, terms = conf.vocab_data.spaun_vocab_strs, pairs = False)
#            logNode.add("mem.MBCnt_Cyc")
#            logNode.add("mem.mem_MBVis")
#            logNode.add("trans.motor_raw")
            logNode.add("dec.Cleanup Memory.x0")
            logNode.add("dec.cnt")
            logNode.add("dec.dec_delay")
            logNode.add("dec.stimulus_cont")
            logNode.add("motor.motor_busy")

        if( auto_run > 0 ):
            if( auto_run > 1 ):
                spaun.net.view(play = conf.est_runtime)
            else:
                spaun.net.network.simulator.resetNetwork(False, False)
                spaun.net.network.simulator.run(0, conf.est_runtime, conf.sim_timestep, False)
        
    datetime_str = datetime.datetime.today().strftime("%d/%m/%y - %H:%M:%S")
    write2file(conf.out_file, "\n# END [" + datetime_str + "]\n")
    write2file(conf.mtr_file, "\n# END [" + datetime_str + "]\n")
    println("# END [" + datetime_str + "]")

## for spaun 2: INSTRUCTION breakdown
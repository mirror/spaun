import conf
import spa
import nef
import hrr
import copy

reload(spa)
reload(nef)
reload(hrr)

from util_funcs import *
import random
import datetime
from java.lang.System.err import println

num_strs  = ["ZER", "ONE", "TWO", "THR", "FOR", "FIV", "SIX", "SEV", "EIG", "NIN"]
task_strs = ["W", "R", "L", "M", "C", "A", "V", "F", "X"]
task_vis_strs = ["A"]
## Task list
# W - Drawing (Copying visual input)
# R - Recognition
# L - Learning (Bandit Task)
# M - Memory (forward serial recall)
# C - Counting
# A - Answering
# V - Rapid Variable Creation
# F - Fluid Induction (Ravens)
# X - Task precursor
subtask_strs = ["MF", "MB", "AP", "AK", "CNT", "NON"]
## Subtask list
# MF - Memory recall forward
# MB - Memory recall backward
# AP - Answer position
# AK - Answer kind
# CNT - Counting subtask
# NON - No subtask
subtask_vis_strs = ["K", "P"]

state_strs = ["SKIP", "TRANS1", "TRANS2", "CNT", "VIS", "LEARN"]

all_vis_strs = copy.deepcopy(num_strs)
all_vis_strs.extend(task_strs)
all_vis_strs.extend(subtask_vis_strs)
all_vis_strs.extend(["OPEN", "CLOSE", "QM", "SPACE"])

max_pos_num = 7


class VocabData:
    def __init__(self, rand_seed = 0):       
        random.seed(rand_seed)

        self.num_dim = conf.num_dim

        self.num_strs         = copy.deepcopy(num_strs)
        self.task_strs        = copy.deepcopy(task_strs)
        self.task_vis_strs    = copy.deepcopy(task_vis_strs)
        self.subtask_strs     = copy.deepcopy(subtask_strs)
        self.subtask_vis_strs = copy.deepcopy(subtask_vis_strs)
        self.state_strs       = copy.deepcopy(state_strs)
        self.all_vis_strs     = copy.deepcopy(all_vis_strs)
        
        self.max_pos_num = max_pos_num
        self.pos_strs = ["POS%i" % (n+1) for n in range(self.max_pos_num)]

        self.spaun_vocab_strs = copy.deepcopy(num_strs)
        self.spaun_vocab_strs.extend(self.pos_strs)

        #posxnum_strs = []
        #posxcls_strs = []
        #for pos_str in pos_strs:
        #    for num_str in num_strs:
        #        posxnum_strs.append("%s*%s" % (pos_str, num_str))
        #    posxcls_strs.append("%s*CLOSE" % (pos_str))
        #posx_strs = copy.deepcopy(posxnum_strs)
        #posx_strs.extend(posxcls_strs)

        ### Generate Visual Stimulus Vocabulary (vision -> assoc mem -> prod sys) ###
        self.vis_filepath = conf.vis_filepath

        self.vis_vocab = hrr.Vocabulary(self.num_dim, unitary = ["INC", self.pos_strs[0], "ADD", self.num_strs[0]], 
                                        max_similarity = 0.05, include_pairs = False)

        self.vis_vocab.parse("ADD")
        self.vis_vocab.parse(self.num_strs[0])
        add_vec  = self.vis_vocab.hrr["ADD"].v
        item_vec = self.vis_vocab.hrr[self.num_strs[0]].v
        for i in range(len(self.num_strs) - 1):
            item_vec = cconv(item_vec, add_vec)
            self.vis_vocab.add(self.num_strs[i+1], hrr.HRR(data = item_vec))

        self.vis_vocab.parse("SPACE+OPEN+CLOSE")
        self.vis_vocab.parse("INC")
        #vis_vocab.add("INC*", hrr.HRR(data = invol(vis_vocab.hrr["INC"].v)))
        self.vis_vocab.parse(self.pos_strs[0])

        self.vis_vocab.parse("OPEN+QM")
        for task_str in self.task_vis_strs:
            self.vis_vocab.parse(task_str)           # Task visual variables
        for subtask_vis_str in self.subtask_vis_strs:
            self.vis_vocab.parse(subtask_vis_str)    # SubTask visual variables

        # Add POS2 to POS_Max to vocabulary
        inc_vec = self.vis_vocab.hrr["INC"].v
        self.vis_vocab.add("INC'", hrr.HRR(data = invol(inc_vec)))
        pos_vec = self.vis_vocab.hrr[self.pos_strs[0]].v
        for i in range(self.max_pos_num-1):
            pos_vec = cconv(pos_vec, inc_vec)
            self.vis_vocab.add(self.pos_strs[i+1], hrr.HRR(data = pos_vec))

        # Add POSxITEM pairs to vocabulary
        for i in range(self.max_pos_num):
            for j in range(len(self.num_strs)):
                enc_vec = cconv(self.vis_vocab.hrr[self.pos_strs[i]].v, 
                                self.vis_vocab.hrr[self.num_strs[j]].v)
                str_val = self.pos_strs[i] + "*" + self.num_strs[j]
                self.vis_vocab.add(str_val, hrr.HRR(data = enc_vec))
                self.spaun_vocab_strs.append(str_val)


        #### Generate visual vector sums ###
        self.sum_task_vis_vecs = zeros(1, self.num_dim)
        for task_str in self.task_vis_strs:
            self.sum_task_vis_vecs = [self.sum_task_vis_vecs[d] + self.vis_vocab.hrr[task_str].v[d] 
                                      for d in range(self.num_dim)]

        self.sum_subtask_vis_vecs = zeros(1, self.num_dim)
        for subtask_str in self.subtask_vis_strs:
            self.sum_subtask_vis_vecs = [self.sum_subtask_vis_vecs[d] + self.vis_vocab.hrr[subtask_str].v[d] 
                                         for d in range(self.num_dim)]

        self.sum_num_vecs = zeros(1, self.num_dim)
        for num_str in self.num_strs:
            self.sum_num_vecs = [self.sum_num_vecs[n] + self.vis_vocab.hrr[num_str].v[n] 
                                 for n in range(self.num_dim)]
        
        ## debug
#        for num_str in self.num_strs:
#            print(num_str + ": " + str(dot(self.sum_num_vecs, self.vis_vocab.hrr[num_str].v, False)))
        ## debug
        
        self.ps_reset_vecs       = [self.sum_task_vis_vecs[d]    + self.vis_vocab.hrr["QM"].v[d]    for d in range(self.num_dim)]
        self.ps_state_reset_vecs = [self.sum_subtask_vis_vecs[d] + self.ps_reset_vecs[d]            for d in range(self.num_dim)]
        self.ps_cycle_vecs       = [self.sum_task_vis_vecs[d]    + self.vis_vocab.hrr["CLOSE"].v[d] for d in range(self.num_dim)]

        self.item_reset_vecs     = [self.sum_task_vis_vecs[d] + self.vis_vocab.hrr["OPEN"].v[d] for d in range(self.num_dim)]
        self.pos_reset_vecs      = [self.item_reset_vecs[d]   + self.vis_vocab.hrr["QM"].v[d]   for d in range(self.num_dim)]
        self.pos_cyc_vecs        = [self.sum_num_vecs[d]      + self.vis_vocab.hrr["OPEN"].v[d] + self.vis_vocab.hrr["QM"].v[d] for d in range(self.num_dim)]

        self.posxnum0_vec_list  = [self.vis_vocab.hrr["POS1*" + self.num_strs[n]].v for n in range(len(self.num_strs))]
        self.posxnum1_vec_list  = [self.vis_vocab.hrr["POS1*" + self.num_strs[n+1]].v for n in range(self.max_pos_num)]
#        self.posxnum2_vec_list  = [self.vis_vocab.hrr["POS1*" + self.num_strs[n+1]].v for n in range(self.max_pos_num)]
#        self.posxnum2_vec_list.extend(self.posxnum0_vec_list)
        self.pos_vec_list       = [self.vis_vocab.hrr[self.pos_strs[n]].v for n in range(len(self.pos_strs))]
        self.num_vec_list       = [self.vis_vocab.hrr[self.num_strs[n]].v for n in range(len(self.num_strs))]
#        self.pos_n_num_vec_list = [self.vis_vocab.hrr[self.pos_strs[n]].v for n in range(len(self.pos_strs))]
#        self.pos_n_num_vec_list.extend(self.num_vec_list)

        ### Generate nums vocab (for motor module) ###
        self.nums_dim      = max(len(self.num_strs) + 2, 8)
        self.nums_vocab    = hrr.Vocabulary(self.nums_dim, max_similarity = 0.05, randomize = False)
        self.sum_nums_vecs = zeros(1, self.nums_dim)
        for num_str in self.num_strs:
            self.nums_vocab.parse(num_str)
            self.sum_nums_vecs = [self.sum_nums_vecs[n] + self.nums_vocab.hrr[num_str].v[n] for n in range(self.nums_dim)]

        ### Generate state vocab ###
        self.state_dim   = max(len(self.state_strs) + 2, 8)
        self.state_vocab = hrr.Vocabulary(self.state_dim, max_similarity = 0.05, randomize = False)
        for state_str in self.state_strs:
            self.state_vocab.parse(state_str)

        ### Generate task vocab ###
        self.task_strs.append("DEC")
        self.task_dim = max(len(self.task_strs) + 3, 9)
        self.task_vocab = hrr.Vocabulary(self.task_dim, max_similarity = 0.05, randomize = False)
        for task_str in self.task_strs:
            self.task_vocab.parse(task_str)

        ### Generate subtask vocab ###
        self.subtask_dim = max(len(self.subtask_strs) + 2, 10)
        self.subtask_vocab = hrr.Vocabulary(self.subtask_dim, max_similarity = 0.05, randomize = False)
        for subtask_str in self.subtask_strs:
            self.subtask_vocab.parse(subtask_str)

        ### Generate state vector sums ###
        self.vec_trans1          = [self.state_vocab.hrr["TRANS1"].v[d] for d in range(self.state_dim)]
        self.vec_trans2_skip_cnt = [self.state_vocab.hrr["TRANS2"].v[d] + self.state_vocab.hrr["SKIP"].v[d] + 
                                    self.state_vocab.hrr["CNT"].v[d]    for d in range(self.state_dim)]
        self.vec_skip            = [self.state_vocab.hrr["SKIP"].v[d]   for d in range(self.state_dim)]

        ### Generate subtask vector sums ###
        self.vec_ap_ak           = [self.subtask_vocab.hrr["AP"].v[d] + self.subtask_vocab.hrr["AK"].v[d] 
                                    for d in range(self.subtask_dim)]

        ### Motor Vocabs: ##
        ## Have from scale -1 to 1, but outscale x 15 in motor node?
        self.motor_vecs = []
#        self.motor_vecs.append([ 5,15, 5, 6, 5, 3, 5, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]) # ONE -  8 valid M(10)
#        self.motor_vecs.append([ 4,15, 4,13, 4,12, 4,10, 4, 9, 7, 9, 8, 9, 9, 9, 9,10, 9,11, 9,12, 9,15, 9,12, 9, 5, 9, 2, 9, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]) # FOR - 32 valid M(38) 
#        self.motor_vecs.append([11,15, 9,15, 7,15, 5,15, 5,13, 5,11, 5,10, 7,10, 9,10,10, 9,11, 8,11, 6,11, 4,11, 3,10, 2, 9, 1, 7, 1, 5, 1, 3, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]) # FIV - 38 valid M(40)
#        self.motor_vecs.append([ 5, 1, 6, 2, 7, 3, 8, 4, 9, 5,10, 7,11, 9,11,10,11,11,11,13,10,14, 9,15, 7,15, 6,15, 5,14, 4,13, 4,11, 4,10, 5, 9, 6, 8, 7, 8, 8, 8,10, 9,-1,-1,-1,-1,-1,-1,-1,-1]) # NIN - 46 valid M(42) !!
        self.motor_vecs.append([ 7,15, 4,15, 2,11, 2, 6, 4, 1, 7, 1, 9, 6, 9,11, 8,13, 7,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]) # ZER - 20 valid M(20)
        self.motor_vecs.append([ 5,15, 5, 6, 5, 3, 5, 1, 5, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]) # ONE - 10 valid M(10)
        self.motor_vecs.append([ 3, 8, 3, 9, 4,10, 5,10, 6,10, 7, 9, 7, 8, 7, 7, 6, 5, 5, 4, 4, 3, 3, 2, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]) # TWO - 36 valid M(36)
        self.motor_vecs.append([ 5,15, 6,15, 7,15, 8,15, 9,15,10,14,11,13,11,12,11,11,11,10,10, 9, 9, 8, 8, 8, 7, 8, 8, 8, 9, 8,10, 7,11, 6,11, 5,11, 4,11, 3,10, 2, 9, 1, 8, 1, 7, 1, 6, 1, 5, 1]) # THR - 54 valid M(56)
        self.motor_vecs.append([ 4,15, 4,13, 4,12, 4,10, 4, 9, 7, 9, 8, 9, 9, 9, 9,10, 9,11, 9,12, 9,15, 9,12, 9, 5, 9, 2, 9, 1, 9, 1, 9, 1, 9, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]) # FOR - 32 valid M(38) 
        self.motor_vecs.append([11,15, 9,15, 7,15, 5,15, 5,13, 5,11, 5,10, 7,10, 9,10,10, 9,11, 8,11, 6,11, 4,11, 3,10, 2, 9, 1, 7, 1, 5, 1, 3, 2, 3, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]) # FIV - 40 valid M(40)
        self.motor_vecs.append([ 9,15, 8,14, 7,13, 6,12, 5,11, 4, 9, 4, 7, 4, 6, 4, 5, 4, 3, 5, 2, 6, 1, 7, 1, 8, 1, 9, 1,10, 2,11, 3,11, 4,11, 5,10, 6, 9, 7, 8, 7, 7, 7, 6, 6, 4, 5,-1,-1,-1,-1]) # SIX - 50 valid M(50)
        self.motor_vecs.append([ 5,15, 7,15, 9,15,11,15,12,15,12,14,12,12,11,10,10, 8, 9, 6, 8, 4, 7, 2, 7, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]) # SEV - 26 valid M(26)
        self.motor_vecs.append([ 9,15, 8,15, 7,15, 6,14, 5,13, 5,10, 6, 9, 7, 8, 9, 8,10, 7,11, 6,11, 3,10, 2, 9, 1, 7, 1, 6, 2, 5, 3, 5, 6, 6, 7, 7, 8,10, 9,11,10,11,13,10,14, 9,15,-1,-1,-1,-1]) # EIG - 50 valid M(50)
        self.motor_vecs.append([ 5, 1, 6, 2, 7, 3, 8, 4, 9, 5,10, 7,11, 9,11,10,11,11,11,13,10,14, 9,15, 7,15, 6,15, 5,14, 4,13, 4,11, 4,10, 5, 9, 7, 8,10, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]) # NIN - 42 valid M(42) !!
        self.motor_vecs.append([ 1, 7,10, 7,12, 7,13, 7,14, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]) #  -  - 10 valid

        self.motor_vecs_dim = len(self.motor_vecs[0])
        self.motor_min_val  = 1
        self.motor_max_val  = 15
        motor_val_range = self.motor_max_val - self.motor_min_val
        out_min_val = -1
        out_max_val = 1
        out_val_range = out_max_val - out_min_val
        range_ratio = out_val_range / (motor_val_range * 1.0)

#        self.motor_valid = [20,8,36,54,32,38,50,26,50,46,10]
        self.motor_valid = [20,10,36,54,38,40,50,26,50,46,10]

        ## Process motor vecs
        # - Steps: subtract 1, divide by 14 (so output range is [0,14])
        for i,m_vec in enumerate(self.motor_vecs):
        #    self.motor_vecs[i] = [((self.motor_vecs[i][n] - 1) * 1.0) / (self.motor_max_val - 1) for n in range(self.motor_vecs_dim)]
            self.motor_vecs[i] = [(((self.motor_vecs[i][n] - self.motor_min_val) * range_ratio) + out_min_val) for n in range(self.motor_vecs_dim)]


class ControlInput(nef.SimpleNode):
    def __init__(self, name, interval = 0.5, num_tests = 5, test_type = 0, test_option = 0, \
                 out_file = "", mtr_file = "", max_time_out = 12, vocabs = None, rand_seed = 0, \
                 parent_net = None, ave_motor_digit_time = 1, \
                 sym_list_filename = conf.sym_list_filename, sym_vis_filename = conf.sym_vis_filename, \
                 num_list_filename = conf.num_list_filename, num_vis_filename = conf.num_vis_filename):
        # Control input settings:
        # DRAWING test - 
        #     test_type: 0
        #
        # RECOGNITION test - 
        #     test_type: 1
        #
        # LEARNING test - 
        #     test_type: 2
        #     test_option: [[reward prob 1, reward prob 2, ... , reward prob N, #runs],[...]]
        #     
        # RECALL test - 
        #     test_type: 3
        #     +test_option: maximum number of digits to recall (forward recall)
        #     -test_option: maximum number of digits to recall (backwards recall)
        #
        # COUNTING test - 
        #     test_type: 4
        #     +test_option: starting number
        #     -test_option: randomized start number
        #
        # Q & A test - 
        #     test_type: 5
        #     +test_option: Q number (position question)
        #     -test_option: Q number (kind question)
        #
        # RAPID test -
        #     test_type: 6
        #     test_option: number of training examples
        #
        # FLUID INTELLIGENCE test -
        #     test_type: 7
        #     test_option: maximum number of digits e.g. 4 -> [xx][xxx][xxxx]
        #
        ### Need way to set:
        # - different number outputs for drawing test
        # - reward rates for learning task
        # - [###]K? vs K[###]? for recall
        # - constants & variables length for RVC 
        # - [###] length for FI test

        random.seed(rand_seed)
        self.rand_seed = rand_seed

        self.vocabs = vocabs

        self.vocab = hrr.Vocabulary(28*28, include_pairs = False)

        self.counter        = 0
        self.check_interval = interval
        self.next_check     = 0
        self.num_tests      = num_tests
        self.test_num       = 0
        self.dimension      = len(self.vocab.hrr[self.vocab.hrr.keys()[0]].v)
        
        self.cont_val     = 0
        self.cont         = True
        self.query_string = "QM"

        self.num_vis_filename = conf.vis_filepath + num_vis_filename
        self.num_vis_file_offset = {}
        self.num_vis_vec  = []
        self.use_rand_vis_string = "!"
        self.offset_len = 0

        self.learn_mode          = False
        self.learn_options       = []
        self.learn_start         = []
        self.learn_trials        = 0
        self.learn_trials_done   = 0
        self.learn_wait_interval = 0.5
        self.learnd_terminations = []
        self.learnd_reset_func   = []
        self.motor_out           = -1
        self.been_rewarded = False

        self.time_out     = -1
        self.max_time_out = max_time_out
        
        self.num_tests  = num_tests
        self.cmd_list   = []
        self.write_last = -1
        self.out_file   = out_file
        self.mtr_file   = mtr_file

        num_items = len(vocabs.num_strs)
        num_tasks = len(vocabs.task_strs)
        self.est_digit_ans = []
        self.ave_motor_digit_time = ave_motor_digit_time

        self.parent_net = parent_net

        self.init_vis_vocab(conf.vis_filepath, sym_list_filename, sym_vis_filename, num_list_filename, num_vis_filename)

        if( not isinstance(test_type, list) ):
            test_type = [test_type]
        if( not isinstance(test_option, list) ):
            test_option = [test_option]

        for i,test_type_val in enumerate(test_type):
            test_option_val = test_option[i]
            for test_num in range(num_tests):
                perm_strs = copy.deepcopy(vocabs.num_strs)
                if( not test_type_val == 6 ):
                    random.shuffle(perm_strs)

                self.write_to_file(None, '# ' + str(perm_strs ) + '\n')
                    
                if( test_type_val >= 0 ):
                    self.cmd_list.extend([vocabs.task_strs[5]]) # "A"
                    self.cmd_list.extend([vocabs.num_strs[test_type_val % num_tasks]])
                    if( test_type_val < 2 ):
                        self.cmd_list.extend(["OPEN"])
                
                if( test_type_val == 0 or test_type_val == 1 ): # RECOGNITION or DRAWING test type
                    start_num, use_exemplar = test_option_val
                    if( start_num is None ):
                        test_str = perm_strs[0]
                    else:
                        test_str = num_strs[start_num]
                    if( use_exemplar ):
                        prefix_str = ""
                    else:
                        prefix_str = self.use_rand_vis_string
                    num_test_items = 1
                    self.est_digit_ans.append(num_test_items)
                    self.cmd_list.extend([prefix_str + test_str])

                elif( test_type_val == 2 ): # LEARNING test type
                    num_batch_test = len(test_option_val)

                    self.learn_mode = True
                    self.learn_wait_interval = max(interval, self.learn_wait_interval)
                    self.learn_options.append([])
                    self.est_digit_ans.append(1)
                    
                    for n in range(num_batch_test):
                        for t in range(test_option_val[n][-1]):
                            self.learn_options[test_num].append(test_option_val[n][0:-1])
                    
                    println(str(self.learn_options))
                    
                    self.learn_start.append(len(self.cmd_list))
                    self.learn_trials = len(self.learn_options[test_num])

                    for n in range(self.learn_trials):
                        self.cmd_list.extend([self.query_string])
                        self.cmd_list.extend(["ZER"])

                elif( test_type_val == 3 ): # RECALL test type
                    num_test_items, recall_opt = test_option_val

                    self.est_digit_ans.append(num_test_items)
                    if( recall_opt == 0 ):  # Backward option: use negative numbers for test_option_val
                        self.cmd_list.extend(["K"])
                    self.cmd_list.extend(["OPEN"])
                    for n in range(num_test_items):
                        self.cmd_list.extend([perm_strs[n]])
                    self.cmd_list.extend(["CLOSE"])
                    if( recall_opt == -1 ):  # Backward option: use negative numbers for test_option_val
                        self.cmd_list.extend(["K"])

                elif( test_type_val == 4 ): # COUNTING test type
                    num_start, num_count = test_option_val

                    if( num_start is None ):
                        if( num_count is None ):
                            num_start = int(round(random.random() * 8))
                        else:
                            num_start = int(round(random.random() * (9 - num_count)))
                    else:
                        num_start = max(0, min(num_start, 8))
                    
                    if( num_count is None ):
                        num_count = int(round(random.random() * (9 - num_start)))
        
                    self.est_digit_ans.append(num_count + 2)
                    self.cmd_list.extend(["OPEN"])
                    self.cmd_list.extend([vocabs.num_strs[num_start]])
                    self.cmd_list.extend(["CLOSE"])
                    self.cmd_list.extend(["OPEN"])
                    self.cmd_list.extend([vocabs.num_strs[num_count]])
                    self.cmd_list.extend(["CLOSE"])
                
                elif( test_type_val == 5 ): # Q & A test type
                    num_test_items, QA_type, Ans_index, allow_repeats = test_option_val
                    num_test_items = min(max(abs(num_test_items), 1), max_pos_num)

                    ## TODO: allow repeats
                
                    self.est_digit_ans.append(num_test_items)
                    self.cmd_list.extend(["OPEN"])
                    if( isinstance(num_test_items, list) ):
                        for n in num_test_items:
                            self.cmd_list.extend([vocabs.num_strs[n]])
                    else:
                        for n in range(num_test_items):
                            self.cmd_list.extend([perm_strs[n]])
                    self.cmd_list.extend(["CLOSE"])
                    if( QA_type.lower() == 'k' ):  # K option: use negative numbers for test_option_val
                                            # "Where is X?"
                        self.cmd_list.extend(["K"])
                        if( Ans_index is None ):
                            Ans_index = int(round(random.random() * (num_test_items -1)))
                        qns_num_str = perm_strs[Ans_index]
                    else:                   # P option: use negative numbers for test_option_val
                                            # "What is at position X?"
                        self.cmd_list.extend(["P"])
                        if( Ans_index is None ):
                            Ans_index = int(round(random.random() * (num_test_items -1))) + 1
                        qns_num_str = vocabs.num_strs[Ans_index]
                    self.cmd_list.extend(["OPEN"])
                    self.cmd_list.extend([qns_num_str])
                    self.cmd_list.extend(["CLOSE"])

                elif( test_type_val == 6 ):   # RAPID test type
                    random.shuffle(perm_strs)
                    num_test_items, pattern_Q, pattern_A = test_option_val
                    self.est_digit_ans.append(len(pattern_A))

                    # Pre-process patterns
                    sym_dict = {}
                    num_const = 0
                    num_var   = 0
                    for char in (pattern_Q + pattern_A):
                        if( char.isupper() and not char in sym_dict.keys() ):
                            sym_dict[char] = perm_strs[num_const]
                            num_const += 1
                        if( char.islower() ):
                            sym_dict[char] = num_var
                            num_var += 1
                    rand_strs = perm_strs[num_const:len(perm_strs)]

                    var_index = len(rand_strs)
                    for n in range(num_test_items + 1):
                        # Shuffle variables
                        var_list = [0] * num_var
                        if( var_index >= len(rand_strs) ):
                            random.shuffle(rand_strs)
                            var_index = 0

                        # Question
                        self.cmd_list.extend(["OPEN"])
                        for char in pattern_Q:
                            sym_str = sym_dict[char]
                            if( isinstance(sym_str, int) ):
                                var_value = rand_strs[var_index % len(rand_strs)]
                                self.cmd_list.extend([var_value])
                                var_list[sym_str] = var_value
                                var_index = var_index + 1
                            else:
                                self.cmd_list.extend([sym_str])
                        self.cmd_list.extend(["CLOSE"])
                        
                        # Answer (leave out for last test)
                        if( n < num_test_items ):
                            self.cmd_list.extend(["OPEN"])
                            for char in pattern_A:
                                sym_str = sym_dict[char]
                                if( isinstance(sym_str, int) ):
                                    self.cmd_list.extend([var_list[sym_str]])
                                else:
                                    self.cmd_list.extend([sym_str])
                            self.cmd_list.extend(["CLOSE"])
                                

                elif( test_type_val == 7 ): # RAVENS test type
                    num_test_digits = test_option_val

                    self.est_digit_ans.append(max(num_test_digits))
                    if( len(num_test_digits) != 3 ):
                        for cell in range(len(num_test_digits)):
                            self.cmd_list.extend(["OPEN"])
                            digit_nums = [int(n) for n in str(num_test_digits[cell])]
                            for num in digit_nums:
                                self.cmd_list.extend([vocabs.num_strs[num]])
                            self.cmd_list.extend(["CLOSE"])
                    else:
                        for rows in range(3):
                            for cols in range(3):
                                if( not rows == 2 or not cols == 2):
                                    self.cmd_list.extend(["OPEN"])
                                    for n in range( num_test_digits[cols] ):
                                        self.cmd_list.extend([perm_strs[rows]])
                                    self.cmd_list.extend(["CLOSE"])

                elif( test_type_val == -1 ):   # Debug type
                    for num_str in vocabs.num_strs:
                        self.cmd_list.append(num_str)
                
                elif( test_type_val == -2 ):   # Invalid test type #1: A0A1A2A3>123<?
                    self.cmd_list.extend([vocabs.task_strs[5]])
                    self.cmd_list.extend([vocabs.num_strs[0]])
                    self.cmd_list.extend([vocabs.task_strs[5]])
                    self.cmd_list.extend([vocabs.num_strs[1]])
                    self.cmd_list.extend([vocabs.task_strs[5]])
                    self.cmd_list.extend([vocabs.num_strs[2]])
                    self.cmd_list.extend([vocabs.task_strs[5]])
                    self.cmd_list.extend([vocabs.num_strs[3]])
                    self.est_digit_ans.append(3 + 3)
                    self.cmd_list.extend(["OPEN"])
                    for n in range(3):
                        self.cmd_list.extend([perm_strs[n]])
                    self.cmd_list.extend(["CLOSE"])

                if( not self.learn_mode ):
                    self.cmd_list.extend([self.query_string])
            self.cmd_list.extend(["SPACE"])

        print(self.cmd_list)
        println(self.cmd_list)
        # raise Exception("DEBUG")
        
        nef.SimpleNode.__init__(self, name)
        self.getTermination("Cont").setDimensions(1)

    def init_vis_vocab(self, file_path, sym_list_filename, sym_vis_filename, \
                       num_list_filename, num_vis_filename):
        sym_list = read_csv(file_path + sym_list_filename)
        sym_vecs = read_csv(file_path + sym_vis_filename)

        num_list = open(file_path + num_list_filename, 'r')
        num_vis  = open(file_path + num_vis_filename, 'r')

        num_items = min(len(sym_list), len(sym_vecs))

        for i,vis_str in enumerate(sym_list[0:num_items]):
            self.vocab.add(vis_str[0], hrr.HRR(data = sym_vecs[i]))
            if( vis_str[0] in num_strs ):
                self.num_vis_file_offset[vis_str[0]] = []
        
        # Parse num vis list and store offsets for each line
        offset = 0
        for line in num_vis:
            num_value = int(num_list.readline())
            self.num_vis_file_offset[num_strs[num_value]].append(offset)
            offset += len(line)
            self.offset_len = len(line) ## DEBUG
        num_list.close()
        num_vis.close()
            
    
    def getEstRuntime(self):
        return (self.check_interval * 2) * len(self.cmd_list) + self.ave_motor_digit_time * \
               self.num_tests * (sum(self.est_digit_ans) + self.learn_mode * self.learn_trials) + \
               (sum(self.est_digit_ans) * self.ave_motor_digit_time)

    def reset(self, randomize=False):
        self.next_check = 0
        self.counter    = 0
        self.cont_val   = 0
        self.cont       = True
        self.test_num   = 0
        self.test_write_index = 0
        self.test_write_last  = -1
        self.been_rewarded = False
        self.learn_trials_done = 0
        nef.SimpleNode.reset(self, randomize)
        
        datetime_str = datetime.datetime.today().strftime("%d/%m/%y - %H:%M:%S")
        self.write_str_to_file(self.out_file, "# START [" + datetime_str + "]\n")
        self.write_str_to_file(self.mtr_file, "# START [" + datetime_str + "]\n")

    
    def origin_X(self):
        cmd_index = (self.counter/2) % len(self.cmd_list)
        if( self.t_start >= self.next_check ):
            self.next_check += self.check_interval

            # Check start of next test set
            if( self.counter % 2 == 1 and self.cmd_list[cmd_index] in self.vocabs.task_strs and \
                cmd_index > 0 ):
                # Increment test num
                self.test_num += 1 % self.num_tests
                self.time_out = -1

            if( self.cmd_list[cmd_index] == self.query_string ):
                if( self.learn_mode and self.counter % 2 == 1 ):
                    # Additional wait time for the learning task (to allow model to choose action)
                    self.next_check += max(0, self.learn_wait_interval - self.check_interval)
                self.time_out = self.t_start + self.est_digit_ans[self.test_num] * self.ave_motor_digit_time * 2

            if( self.cont_val ):
                # Set continue flag to true
                self.cont = True
            if( self.cont or (self.t_start > self.time_out and self.time_out > 0) or self.learn_mode ):
                # If timeout occurred, then write a "T" into the output file
                if( self.t_start > self.time_out and self.time_out > 0):
                    self.write_str_to_file(self.out_file, "T")
                self.counter += 1
                if( cmd_index > 0 ): 
                    self.cont = not ((self.cmd_list[cmd_index] == self.query_string) and 
                                    (self.counter % 2 == 0))

            # Handle handwritten digits
            if( self.cmd_list[cmd_index][0] == self.use_rand_vis_string and self.counter % 2 != 0 ):
                cmd_str = self.cmd_list[cmd_index][1:]
                vis_file = open(self.num_vis_filename)
                rand_seed = eval(datetime.datetime.today().strftime("%y%m%d%H%M%S")) + self.rand_seed
                random.seed(rand_seed)
                read_offset = random.choice(self.num_vis_file_offset[cmd_str])
                vis_file.seek(read_offset)
                self.num_vis_vec = eval( "[" + vis_file.readline() + "]" )
                self.write_to_file(None,"<" + str(int(read_offset / self.offset_len)) + ">")
                vis_file.close()
                   
        if( self.counter % 2 == 0 ):
            ret_val = self.vocab.hrr["SPACE"].v
            self.been_rewarded = False
        else:
            if( self.learn_mode and cmd_index > self.learn_start[self.test_num] and 
                self.cmd_list[cmd_index-1] == self.query_string and not self.been_rewarded ):
                reward_options = self.learn_options[self.test_num][self.learn_trials_done]
                rewarded = False
                
                if( self.motor_out < len(reward_options) ):
                    reward_prob = reward_options[self.motor_out]
                    rewarded = (random.random() < reward_prob)
                    self.learn_trials_done += 1
    
                if( rewarded ):
                    self.cmd_list[cmd_index] = "ONE"
                else:
                    self.cmd_list[cmd_index] = "ZER"
                self.been_rewarded = True
            
            if( self.cmd_list[cmd_index][0] == self.use_rand_vis_string ):
                ret_val = self.num_vis_vec
            else:    
                ret_val = self.vocab.hrr[self.cmd_list[cmd_index]].v

            if( not(self.write_last == cmd_index) and self.t_start > 0 ):
                # Reset network 
                if( self.cmd_list[cmd_index] in self.vocabs.task_strs ):
                    self.reset_learned_terminations()
                if( self.learn_mode and self.cmd_list[cmd_index-1] == self.query_string ):
                    self.write_to_file(None,"|")
                self.write_to_file(cmd_index) 
        return ret_val
    
    def origin_Cont(self):
        return [self.cont]       

    def termination_Cont(self, x):
        self.cont_val = (x[0] > 0.9)

    def termination_TimeoutRst(self, x):
        if( x[0] > 0.5 ):
            self.time_out = 0

    def termination_MotorOut(self, x):
        self.motor_out = int(round(x[0]))

    def add_learned_termination(self, termination, reset_func):
        if( termination is None ):
            println("ControlInput.add_learned_termination - WARNING! Null termination given.")
            return
        self.learnd_terminations.append(termination)
        self.learnd_reset_func.append(reset_func)
    
    def reset_learned_terminations(self):
        for i,term in enumerate(self.learnd_terminations):
            orig_r = len(term.transform)
            orig_c = len(term.transform[0])
            term.transform = self.learnd_reset_func[i](zeros(orig_r,orig_c))

    def write_to_file(self, write_index = None, item = ""):
        if( not self.out_file == "" ):
            if( not write_index is None ):
                item = self.cmd_list[write_index]
                if( item[0] == self.use_rand_vis_string ):
                    item = item[1:]
            if( item in self.vocabs.task_strs ):
                if( write_index > 0 and self.counter > 1 ):
                    self.write_str_to_file(self.out_file, "\n")
                    self.write_str_to_file(self.mtr_file, "\n# START TASK\n")
                self.write_str_to_file(self.out_file, item)
            elif( item in self.vocabs.num_strs ):
                self.write_str_to_file(self.out_file, str(self.vocabs.num_strs.index(item)))
                ## Note: If num_strs > 10, need to add comma.
            elif( item == "OPEN" ):
                self.write_str_to_file(self.out_file, "[")
            elif( item == "CLOSE" ):
                self.write_str_to_file(self.out_file, "]")
            elif( item == self.query_string ):
                self.write_str_to_file(self.out_file, "?")
            elif( item == "SPACE" ):
                self.write_str_to_file(self.out_file, "\n")
            else:
                self.write_str_to_file(self.out_file, str(item))
            
            self.write_last = write_index
        return

    def write_str_to_file(self, file_name, str_val):
        if( not file_name == "" ):
            file_handle = open(file_name, 'a')
            file_handle.write(str_val)
            file_handle.close()

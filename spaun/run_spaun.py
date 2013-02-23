##### SPAUN PROPER ######
import spaun_main
import sys

def add_test(test_type, test_option, test_type_val, test_option_val):
    test_type.append(test_type_val)
    test_option.append(test_option_val)

def run(OS = "LIN", root_path = None, multi_thread = False):
    ##### TEST PARAMETERS #####
    # ======================= #
    # Experiment configurations
    num_test_run  = 1               # Number of test runs to run
    num_subjects  = 1               # Number of subjects to run
    test_type_val = 0               # Task type
                                    # - 0: Copy drawing
                                    #   1: Digit recognition
                                    #   2: Learning task (bandit task)
                                    #   3: Serial recall 
                                    #   4: Counting task
                                    #   5: Question & answer task
                                    #   6: Rapid variable creation task
                                    #   7: Raven's matricies induction task
    test_type_val2 = test_type_val  
    test_type_opt = 2               # Task test type (see code below for different tests)

    test_type     = []
    test_option   = []

    if( len(sys.argv) > 1 ):
        test_type_val = int(sys.argv[1]) % 100
        test_type_val2 = int(sys.argv[1])
    if( len(sys.argv) > 2 ):
        test_type_opt = int(sys.argv[2])

    present_int = 0.15

    # Run configurations
    en_logging = True
    if( len(sys.argv) > 3 ):
        en_logging = int(sys.argv[3])
    auto_run = 0

    perfect_MB    = False
    perfect_cconv = False

    # Random number
    if( len(sys.argv) > 4 ):
        rand_type = long(sys.argv[4])
    else:
        rand_type = 0

    # Averager options
    tranf_scale = 0.2
    # tranf_scale = 0.25

    # Learning options
    learn_alpha   = 0.75
    learn_actions = 2

    # Cleanup mem (DEC) parameters
    CUthreshold     = 0.45
    CUNumsThreshold = 0.25
    CUinScale       = 1.0

    # Motor options
    motor_init = 0.15
    motor_step = 0

    ##### Copy drawing #####
    ## (Number, {True: Use exemplar image | False: Use rand image})
    if( test_type_val == 0 ):
        add_test(test_type, test_option, test_type_val, (4,False))
        add_test(test_type, test_option, test_type_val, (4,False))
        add_test(test_type, test_option, test_type_val, (4,False))

    ##### Digit recognition #####
    ## (Number, {True: Use exemplar image | False: Use rand image})
    if( test_type_val == 1 ):
        add_test(test_type, test_option, test_type_val, (9,False))
        add_test(test_type, test_option, test_type_val, (9,False))
        add_test(test_type, test_option, test_type_val, (9,False))

    ##### Learning #####
    ## [[reward0,reward1,num_runs], ... ]
    if( test_type_val == 2 ):
        test_type = [2]
        if( test_type_opt == 0 ):
            learn_alpha = 0.75
            test_option.append([[0.12,0.12,0.72,30],[0.12,0.72,0.12,30],[0.72,0.12,0.12,30]])
            learn_actions = 3
        if( test_type_opt == 1 ):
            learn_alpha = 0.25
            test_option.append([[0.12,0.12,0.72,20],[0.12,0.72,0.12,20],[0.72,0.12,0.12,20],[0.12,0.12,0.72,20]])
            learn_actions = 3
        if( test_type_opt == 2 ):
            test_option.append([[0.21,0.72,40],[0.72,0.21,40],[0.12,0.72,40]])
            learn_alpha = 0.65
            learn_actions = 2

    ##### Memory & recall ######
    ## (Num items, {-1: [x]k | 0: k[x] | 1: Fwd})
    if( test_type_val == 3 ):
        add_test(test_type, test_option, test_type_val, (7,1))
        add_test(test_type, test_option, test_type_val, (6,1))
        add_test(test_type, test_option, test_type_val, (5,1))
        add_test(test_type, test_option, test_type_val, (4,1))

    ##### Counting #####
    ## (Start num, count num)
    if( test_type_val == 4 ):
        add_test(test_type, test_option, test_type_val, (None, 1))
        add_test(test_type, test_option, test_type_val, (None, 2))
        add_test(test_type, test_option, test_type_val, (None, 3))
        add_test(test_type, test_option, test_type_val, (None, 4))
        add_test(test_type, test_option, test_type_val, (None, 5))
        en_logging = True

    ##### Q&A #####\
    ## (Num items - list or integer, {'K': kind qns or 'P': pos qns}, Ans index (0th index), {True: allow repeats | False: no repeats})
    if( test_type_val == 5 ):
        if( test_type_opt == 0 ):
            add_test(test_type, test_option, test_type_val, (7,'K',0,False))
            add_test(test_type, test_option, test_type_val, (7,'K',1,False))
            add_test(test_type, test_option, test_type_val, (7,'K',2,False))
            add_test(test_type, test_option, test_type_val, (7,'K',3,False))
        if( test_type_opt == 1 ):
            add_test(test_type, test_option, test_type_val, (7,'K',4,False))
            add_test(test_type, test_option, test_type_val, (7,'K',5,False))
            add_test(test_type, test_option, test_type_val, (7,'K',6,False))
        if( test_type_opt == 2 ):
            add_test(test_type, test_option, test_type_val, (7,'P',1,False))
            add_test(test_type, test_option, test_type_val, (7,'P',2,False))
            add_test(test_type, test_option, test_type_val, (7,'P',3,False))
            add_test(test_type, test_option, test_type_val, (7,'P',4,False))
        if( test_type_opt == 3 ):
            add_test(test_type, test_option, test_type_val, (7,'P',5,False))
            add_test(test_type, test_option, test_type_val, (7,'P',6,False))
            add_test(test_type, test_option, test_type_val, (7,'P',7,False))
            add_test(test_type, test_option, test_type_val, (7,'P',8,False))
        if( test_type_opt == 4 ):
            add_test(test_type, test_option, test_type_val, (3,'P',0,False))
            add_test(test_type, test_option, test_type_val, (3,'P',2,False))

    ##### RVC #####
    ## {Num of examples, Qns Pattern, Ans Pattern)
    if( test_type_val == 6 ):
        if( test_type_opt == 0 ):
            add_test(test_type, test_option, test_type_val, (3,"AAxB","xB"))
        if( test_type_opt == 1 ):
            add_test(test_type, test_option, test_type_val, (3,"AAxy","xy"))
        if( test_type_opt == 2 ):
            add_test(test_type, test_option, test_type_val, (3,"AAx","x"))
        if( test_type_opt == 3 ):
            add_test(test_type, test_option, test_type_val, (1,"AAxB","xB"))
        if( test_type_opt == 4 ):
            add_test(test_type, test_option, test_type_val, (2,"AAxB","xB"))
        if( test_type_opt == 5 ):
            add_test(test_type, test_option, test_type_val, (4,"AAxB","xB"))

    ##### Ravens #####
    ## [Num item cell_1, num item cell_2, num item cell_3]
    if( test_type_val == 7 ):
        if( test_type_opt == 0 ):
            add_test(test_type, test_option, test_type_val, [1,2,3])
        if( test_type_opt == 1 ):
            add_test(test_type, test_option, test_type_val, [3,2,1])
        if( test_type_opt == 2 ):
            add_test(test_type, test_option, test_type_val, [1,2,3,2,3,4,3,4])
        if( test_type_opt == 3 ):
            add_test(test_type, test_option, test_type_val, [1,3,5])
        if( test_type_opt == 4 ):
            add_test(test_type, test_option, test_type_val, [5,4,3,4,3,2,3,2])


    ##### Combined Type #####
    if( test_type_val == 98 ):
        add_test(test_type, test_option, 7, [1,2,3,2])
    
    ##### Combined Type #####
    if( test_type_val == 99 ):
        tranf_scale = 0.4
        add_test(test_type, test_option, 1, (None,False))
        add_test(test_type, test_option, 0, (2,False))
        add_test(test_type, test_option, 3, (3,1))
        add_test(test_type, test_option, 5, (2,'K',None,False))
        add_test(test_type, test_option, 4, (None,2))
        add_test(test_type, test_option, 6, (3,"Ax","x"))
        add_test(test_type, test_option, 7, [1,2,3,2,3,4,3,4])
        # add_test(test_type, test_option, 2, [[0.12,0.72,10],[0.72,0.12,10]])
        learn_actions = 2

    
    ##### Special Args #####
    if( test_type_val2 == 106 or test_type_val2 == 107 ):
        tranf_scale = 0.1
    if( test_type_val2 == 206 or test_type_val2 == 207 ):
        tranf_scale = 0.2
    if( test_type_val2 == 306 or test_type_val2 == 307 ):
        tranf_scale = 0.3
    if( test_type_val2 == 406 or test_type_val2 == 407 ):
        tranf_scale = 0.4
    if( test_type_val2 == 506 or test_type_val2 == 507 ):
        tranf_scale = 0.5
    if( test_type_val2 == 102 ):
        learn_alpha = 0.25
    if( test_type_val2 == 202 ):
        learn_alpha = 0.5
    if( test_type_val2 == 302 ):
        learn_alpha = 0.65
    if( test_type_val2 == 402 ):
        learn_alpha = 0.75

    print(test_type)
    print(test_option)

    ##### RUN MODEL #####
    spaun_main.run(None, OS, root_path, test_type, test_option, num_test_run, num_subjects, \
                   multi_thread, en_logging, rand_type, \
                   perfect_MB, perfect_cconv, CUthreshold, CUNumsThreshold, CUinScale, \
                   tranf_scale, learn_alpha, learn_actions, present_int, motor_init, auto_run)
    
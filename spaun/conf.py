from ca.nengo.model import SimulationMode

OS = "WIN"
root_path = "spaun\\"
out_file  = ""
log_file  = ""
mtr_file  = ""

num_dim   = 512
vis_dim   = 50
motor_dim = 10

valid_vis_strs = []

out_filepath      = root_path + "out_data\\"
vis_filepath      = root_path + "vis_data\\"
mtr_filepath      = root_path + "motor_data\\"
mu_filename       = "mu_nengo_direct_norm4.csv"
sym_vis_filename  = "sym_vis.csv"
sym_list_filename = "sym_list.csv"
num_vis_filename  = "num_vis.csv"
num_list_filename = "num_list.csv"

vocab_data = None
ctrl_node  = None

present_int  = 0.15
motor_init   = 0.15
sim_timestep = 0.001

test_type    = 0
test_option  = 0
num_test_run = 2
num_subjects = 1

rand_seed = 0

MB_mode    = SimulationMode.DEFAULT
cconv_mode = "default"
det_mode   = SimulationMode.DEFAULT

CUthreshold     = 0.3
CUNumsThreshold = 0.2
CUinScale       = 1.25 #1.5
tranf_scale     = 0.2 #0.451
add_scale       = 1.15

inhib_scale = 5.0

decay_val = 0.975 #0.975 #0.9775
fdbk_val  = 0.3
en_buf    = True
en_reh    = True

learn_alpha   = 1.0
learn_actions = 3

pstc       = 0.01
pstc_inhib = 0.01

neur_cconv = 150

est_runtime = 0
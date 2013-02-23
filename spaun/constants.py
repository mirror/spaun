## Define constants
NUM_DIM = 100

## Global Variables
NUM_ITEMS = 6
HRR_VECS = [[]]

# Order of the HRR vectors in the list (note, 0 indexed)
ITEM_ORDER = []

# Define tolerance for zero
ZEROTOL = 1E-15

# Define the time constant for the relay nodes
PSC_RELAY = 0.0001  # Must be smaller than sim time step
TAU_PSC = 0.005

# Flag indicating whether or not to split the dimensions
# up into their individual components for calculations
SPLIT_DIM = True

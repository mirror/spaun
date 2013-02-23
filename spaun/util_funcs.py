import constants

from ca.nengo.util import VectorGenerator
from ca.nengo.math import PDF
from ca.nengo.math import PDFTools
from ca.nengo.math import LinearApproximator
from ca.nengo.math import Function
from ca.nengo.math.impl import GaussianPDF
from ca.nengo.math.impl import IndicatorPDF
from ca.nengo.math.impl import PostfixFunction
from ca.nengo.math.impl import WeightedCostApproximator
from ca.nengo.math.impl import DefaultFunctionInterpreter
from ca.nengo.util.impl import RandomHypersphereVG
from ca.nengo.model.nef.impl import NEFEnsembleFactoryImpl
from math import floor
from math import sqrt
from math import exp
from math import cos
from math import sin
from math import pi
from random import random

## DEBUG ##
from java.lang.System.err import println


class FixedValVectorGenerator(VectorGenerator):
    def __init__(self, fixed_val):
        self.fixed_val = fixed_val

    def genVectors(self, number, dimensions):
        result = [[self.fixed_val] * dimensions for _ in range(number)]
        return result


class GenericVectorListGenerator(VectorGenerator):
    def __init__(self, vector_list):
        self.vector_list = vector_list

    def genVectors(self, number, dimensions):
        # Pad the vectors with zeros for dimensions > vector length in vector list
        # Cut the vectors short for dimensions < vector length in vector list
        ref_vector_list = [pad(self.vector_list[n], dimensions) for n in \
                           range(len(self.vector_list))]
        ref_list_len = len(ref_vector_list)

        result = [0] * number
        for n in range(number):
            result[n] = ref_vector_list[n % ref_list_len]
        return result


class GenericVectorGenerator(VectorGenerator):
    def __init__(self, ref_vector):
        self.ref_vector = ref_vector

    def genVectors(self, number, dimensions):
        encVec = [self.ref_vector for _ in range(number)]
        return encVec


class DiagVectorGenerator(VectorGenerator):
    vec_comp = sqrt(2.0) / 2.0

    def setVecComponent( self, vec_comp ):
        self.vec_comp = vec_comp

    def getVecComponent( self ):
        return self.vec_comp

    def genVectors(self, number, dimensions):
        ref_vector = [[-self.vec_comp,-self.vec_comp],[-self.vec_comp,self.vec_comp],\
                      [self.vec_comp,-self.vec_comp],[self.vec_comp,self.vec_comp]]
        ref_cnt = -1
        div_2 = floor(dimensions / 2.0)

        result = [[0] * dimensions for _ in range(number)]

        for n in range(number):
            if( dimensions == 1 ):
                if( PDFTools.random() < 0.5 ):
                    result[n][0] = -1
                else:
                    result[n][0] = 1
            else:
                if( n % div_2 == 0 ):
                    ref_cnt = int((ref_cnt + 1) % len(ref_vector))
                result[n][int(n % div_2)] = ref_vector[ref_cnt][0]
                result[n][int(n % div_2 + div_2)] = ref_vector[ref_cnt][1]
        return result


class RandomRangeScaledVectorVG(RandomHypersphereVG):
    scale_low = 0
    scale_high = 1
    ref_vector = [1]

    def setRefVector(self, ref_vector):
        self.ref_vector = ref_vector

    def getRefVector(self):
        return self.ref_vector

    def setScaleRange(self, range_low, range_high):
        self.scale_low = range_low
        self.scale_high = range_high

    def getScaleRange(self):
        return [self.scale_low, self.scale_high]

    def genVectors(self, number, dimensions):
        if( dimensions != len(self.ref_vector) ):
            # If the dimensions does not match the reference vector dimension,
            # default to the random hypersphere vector generator
            RSVG = RandomHypersphereVG(self.onSurface, self.radius, self.axisClusterFactor)
            return RSVG.genVectors(number, dimensions)
        else:
            result = [[0] * len(self.ref_vector) for _ in range(number)]

            for n in range(number):
                scale = self.scale_high + 1

                # Randomly pick a scale from a uniform distribution
                scale = IndicatorPDF(self.scale_low, self.scale_high).sample()[0]

                # Scale the reference vector by the scale
                for i in range(len(self.ref_vector)):
                    result[n][i] = scale * self.ref_vector[i]

        return result


class HRRSpacedVectorGenerator(VectorGenerator):
    def __init__(self, num_steps = 100):
        self.num_steps = num_steps

    def genVectors(self, number, dimensions):
        sampler = HRRSpacedSamples(dimensions, self.num_steps)
        result = [[sampler.sample()] * dimensions for _ in range(number)]
        println(str(result))
        return result



def initHRRVectors(num_dim = constants.NUM_DIM, number = constants.NUM_ITEMS, threshold = 0.05):
    # Store the number of items used
    constants.NUM_ITEMS = number

    # Store the new number of dimensions
    constants.NUM_DIM = num_dim

    # Initialize the HRR vectors
    constants.HRR_VECS = [genHRRVectors(num_dim) for n in range(number)]
    for n in range(number):
        HRR_Vec = constants.HRR_VECS[n]
        constants.HRR_VECS[n] = zeros(1, num_dim)

        while( maxAbsDot(constants.HRR_VECS, HRR_Vec) > threshold ):
            HRR_Vec = genHRRVectors(num_dim)

        constants.HRR_VECS[n] = HRR_Vec

    setItemOrder(range(number))


def setItemOrder(list_order):
    list_len = len(list_order)

    for n in range(list_len):
        if( list_order[n] >= constants.NUM_ITEMS ):
            print('Warning - Set Item Order: Item index exceeds number of items, setting to max acceptable value\n')
            list_order[n] = constants.NUM_ITEMS - 1

    constants.ITEM_ORDER = list_order



def printHRRVectors():
    # Print vectors for debug purposes
    for n in range(len(constants.HRR_VECS)):
        print("Item" + str(n+1) + " = " + str(constants.HRR_VECS[n]) + ";")


def genHRRVectors(numdim):
    # Set up normally distributed vectors (HRR vectors) with a variance of 1/N^0.5
    gPDF = GaussianPDF(0, 1/(numdim ** 0.5))
    HRR_vector = [gPDF.sample()[0] for n in range(numdim)]

    # Normalize the vector
    HRR_vector = normalize(HRR_vector)

    return HRR_vector

def maxAbsDot(vec_list, vec):
    dots = [abs(dot(vec_list[n], vec, False)) for n in range(len(vec_list))]
    return max(dots)


#################### PDF & Custom Functions ######################
class UniformlySpacedSamples(PDF):
    def __init__(self, min_value, max_value, num_samples):
        self.num_samples = num_samples
        self.set(min_value, max_value)

    def set(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        self.step_size = (max_value - min_value) * 1.0 / (self.num_samples - 1)
        self.curr_sample = 0

    def reset(self):
        self.curr_sample = 0

    def sample(self):
        if( self.min_value == self.max_value ):
            return [self.min_value]
        else:
            return_val = self.min_value + self.curr_sample * self.step_size
            self.curr_sample = (self.curr_sample + 1) % self.num_samples
            return [return_val]


class HRRSpacedSamples(PDF):
    def __init__(self, num_dim, num_steps = 100):
        pdf = [sin(i * pi / num_steps) ** (num_dim-2) for i in range(num_steps)]
        pdf_area = sum(pdf)
        self.pdf = [x / pdf_area for x in pdf]
        self.num_steps = num_steps
    
    def convert(self, samp_pt):
        total = 0
        for i,p in enumerate(self.pdf):
            total += p
            if( total > samp_pt ):
                proportion = 1.0 - (total-samp_pt) / p
                angle = (i + proportion) * pi / self.num_steps
                break
            else:
                angle = pi
        return cos(angle)
   
    def sample(self):
        return self.convert(random())


class DotFunction(Function):
    def __init__(self, normalize_opt, dim):
        # normalize_opt - set True if you want to normalize before doing dot product
        # dim - dimension of input vectors
        self.normalize_opt = normalize_opt
        self.dim = dim

    def map(self, input):
        input1 = input[0:self.dim]
        input2 = input[self.dim:2*self.dim]
        return dot(input1, input2, self.normalize_opt)

    def getDimension(self):
        return self.dim*2


class MSEFunction(Function):
    def __init__(self, normalize_opt, dim):
        # normalize_opt - set True if you want to normalize before doing dot product
        # dim - dimension of input vectors
        self.normalize_opt = normalize_opt
        self.dim = dim

    def map(self, input):
        if( self.normalize_opt ):
            input1 = normalize(input[0:self.dim])
            input2 = normalize(input[self.dim:2*self.dim])
        else:
            input1 = input[0:self.dim]
            input2 = input[self.dim:2*self.dim]
        return sum([[input1[n] - input2[n] for n in range(self.dim)][m] ** 2 for m in range(self.dim)])

    def getDimension(self):
        return self.dim*2


class ZeroGateFunction(Function):
    def __init__(self, index, dim, func = None, max_val = 1.5, min_val = -1.5, inhib_threshold = 0.25, in_threshold = -1.5):
        # Note that the dimension is the input dimension
        self.index = index
        self.dim = dim
        self.max_val = max_val
        self.min_val = min_val
        self.func = func
        self.inhib_threshold = inhib_threshold
        self.in_threshold = in_threshold

    def map(self, func_in):
        # func_in layout: [0..dim] - input vector, [end] - control signal
        control = func_in[self.dim]
        ret_val = 0

        if( (func_in[self.index] < self.in_threshold) or (control > self.inhib_threshold )):
            return 0
        else:
            if( self.func is None ):
                ret_val = func_in[self.index]
            elif( isinstance(self.func, Function) ):
                ret_val = self.func.map(func_in[0:self.dim])
            else:
                ret_val = self.func(func_in[0:self.dim])
        return max(min(ret_val, self.max_val), self.min_val)

    def getDimension(self):
        return self.dim + 1


class FilteredStepFunction(Function):
    def __init__(self, index = 0, scale = 20, shift = 0, step_val = 1, mirror = False):
        self.scale = scale
        self.shift = shift
        self.step_val = step_val
        self.index = index
        if( mirror ):
            self.mirror = -1
        else:
            self.mirror = 1
    
    def map(self, func_in):
        return max(-1/exp(self.mirror * (func_in[self.index]-self.shift) * self.scale) + 1, 0) * \
               self.step_val

    def getDimension(self):
        return 1

                

class CConvFunction(Function):
    def __init__(self, index, dim):
        # dim - dimension of input vectors
        self.dim = dim
        self.index = index

    def map(self, input):
        input1 = input[0:self.dim]
        input2 = input[self.dim:2*self.dim]
        return cconv(input1, input2)[self.index]

    def getDimension(self):
        return self.dim*2


#################### Nengo Function Functions ####################
def default_func(num_dim):
    return [PostfixFunction("x" + str(n), num_dim) for n in range(num_dim)]


def interpret(str_exp):
    # Translates the str_exp using the default function intepreter
    # Returns a list of the components of the str_exp
    dfi = DefaultFunctionInterpreter()
    item_list = dfi.getPostfixList(str_exp)
    return item_list


def find_int(item_list):
    # Finds all the integers from item_list and puts them in a dictionary
    # mapping where key = int value, and value = unique item count
    cnt = 0
    int_dict = {}

    for item in item_list:
        if( isinstance(item, int) ):
            if( item not in int_dict ):
                int_dict[item] = cnt
                cnt = cnt + 1

    return int_dict


def compare_list(list1, list2):
    # Compares two lists. Will return true if:
    # - The length of the lists are the same AND
    # - The items in the lists match (order does not matter)
    if( len(list1) != len(list2) ):
        return False

    for n in range(len(list1)):
        if( list1[n] not in list2 or list2[n] not in list1 ):
            return False
    return True


def make_list(int_dict):
    # Converts and the dictionary back into a list where the item is the
    # key and the index is the value corresponding to the key
    int_list = [0] * len(int_dict)

    for item in int_dict.keys():
        int_list[int_dict[item]] = item

    return int_list


def replace_int_str(int_dict, str_exp):
    # Looks for the corresponding item (appends "x" to each item
    # in int_dict, and looks for that) and replaces the numeral
    # following the "x" with the item's corresponding value in
    # int_dict
    # e.g. int_dict = {8:0,3:1,7:2}, str_exp = "(x3 + x7) * x8 + x3"
    #      result = "(x1 + x2) * x0 + x1"
    # Note: if the corresponding value is not found in the int_dict
    #       it is by default replaced with x0
    result = ""
    int_str = ""
    foundX = False

    for ind in range(len(str_exp)):
        char = str_exp[ind]
        if( foundX ):
            if( char.isdigit() ):
                int_str = int_str + char
            if( not char.isdigit() or ind >= len(str_exp) - 1 ):
                foundX = False
                if( int_str.isdigit() ):
                    int_val = int(int_str)
                    if( int_val in int_dict ):
                        result = result + str(int_dict[int_val])
                    else:
                        result = result + str(0)
            if( not char.isdigit() ):
                result = result + char
        else:
            result = result + char
            if( char == "x" ):
                int_str = ""
                foundX = True

    return result


def replace_int_list(int_dict, item_list):
    # Replaces each integer value in item_list with their
    # corresponding value from the dictionary. If the value is
    # not found, the integer is replaced with 0

    for ind in range(len(item_list)):
        if( isinstance(item_list[ind], int) ):
            if( item_list[ind] in int_dict ):
                item_list[ind] = int_dict[item_list[ind]]
            else:
                item_list[ind] = 0

    return item_list


def filter(old_value, new_value, alpha):
    # Note: alpha = timestep / (tau + timestep);
    num_dim = len(new_value)
    result = [alpha * new_value[n] + (1-alpha) * old_value[n] for n in range(num_dim)];
    return result


#################### Vector / Matrix Functions ####################
def eye(num_rows, num_cols = 0, start_row = 0, stop_row = -1, start_col = 0, stop_col = -1):
    return diag(num_rows, num_cols, start_row, stop_row, start_col, stop_col)


def diag(num_rows, num_cols = 0, start_row = 0, stop_row = -1, start_col = 0, stop_col = -1, value = 1.0):
    # Returns an identity matrix of dimension numdim x numdim
    # from row start_ind to row stop_ind (non-inclusive)
    if( num_cols <= 0 ):
        num_cols = num_rows

    # Limit the start rows and cols
    start_row = min(start_row, num_rows)
    start_row = max(start_row, 0)
    start_col = min(start_col, num_cols)
    start_col = max(start_col, 0)

    # Default values for stop row and col
    if( stop_row < 0 ):
        stop_row = num_rows
    if( stop_col < 0 ):
        stop_col = num_cols

    # Limit the stop row and col
    stop_row = min(stop_row, num_rows)
    stop_row = max(stop_row, start_row)
    stop_col = min(stop_col, num_cols)
    stop_col = max(stop_col, start_col)

    row = start_row
    col = start_col

    result = [[0] * num_cols for _ in [0] * num_rows]
    while( row < stop_row and col < stop_col ):
        result[row][col] = value
        row = row + 1
        col = col + 1
    return result


def num_vector(num, rows, cols):
    # Returns an matrix of num of dimension rows x cols
    # Note: if rows == 1, it returns a vector
    if( rows == 1 ):
        return [num] * cols
    else:
        return [[num] * cols for _ in [0] * rows]


def ones(rows, cols):
    # Returns an matrix of ones of dimension rows x cols
    # Note: if rows == 1, it returns a vector
    return num_vector(1.0, rows, cols)


def zeros(rows, cols):
    # Returns an matrix of zeros of dimension rows x cols
    # Note: if rows == 1, it returns a vector
    return num_vector(0.0, rows, cols)


def delta(numdim, n = 0, value = 1.0):
    # Returns a delta vector (one element is 1, the rest is 0)
    # with the 1 at index n
    n = n % numdim
    result = [0.0] * numdim
    result[n] = value
    return result


def transpose(input):
    # Returns the transpose of a vector
    result = [[0] for _ in range(len(input))]
    for n in range(len(input)):
        result[n][0] = input[n]
    return result


def transpose_m(input):
    # Returns the transpose of a matrix
    for n in range(len(input)):
        if( n == 0 ):
            result = transpose(input[n])
        else:
            for m in range(len(input[0])):
                result[m].append(input[n][m])
    return result


def norm(input):
    # Calculates the norm (length) of the input vector
    norm_result = sqrt(sum([input[n] ** 2 for n in range(len(input))]))
    return norm_result


def normalize(input):
    # Normalizes the input vector
    mag = sqrt(sum([input[n] ** 2 for n in range(len(input))]))

    if( mag != 0 ):
        return [input[n] / mag for n in range(len(input))]
    else:
        return input


def pad(input, length):
    # Pads the input vector to make it of length "length"
    # Will truncate the input vector if the length < vector length
    if( length == len(input) ):
        return input
    elif( length < len(input) ):
        return [input[n] for n in range(length)]
    else:
        result = input
        result.extend([0] * (length - len(input)))
        return result


def dot(vec1, vec2, norm_vecs = True):
    # Calculates the dot product of the two inputs, with option to enable normalization
    #   before calculation
    # Check to make sure the vector lengths are the same
    if( len(vec1) != len(vec2) ):
        print("util_funcs.dot - Mismatching vector lengths, returning 0")
        return 0

    # Normalize the two vectors so that we can compare them appropriately
    # if selected
    if( norm_vecs ):
        vec1_norm = normalize(vec1)
        vec2_norm = normalize(vec2)
    else:
        vec1_norm = vec1
        vec2_norm = vec2

    result = 0
    for n in range(len(vec1)):
        result = result + vec1_norm[n] * vec2_norm[n]

    return result


def invol(vec1):
    # Calculates the circular involution of the given vector
    #   i.e. invol([0,1,2,3,4,5]) = [0,5,4,3,2,1]
#    len_vec = len(vec1)
#    return [vec1[-n % len_vec] for n in range(len_vec)]
    return [vec1[-n] for n in range(len(vec1))]


def cconv(item1, item2, invert_first = False, invert_second = False):
    # Calculates the circular convolution of the two inputs
    # Check to make sure the vector lengths are the same
    if( len(item1) != len(item2) ):
        print("util_funcs.cconv - Mismatching vector lengths, returning []")
        return []

    # Get the dimensions of the input vectors
    num_dim = len(item1)

    if( invert_first ):
        item1 = invol(item1)
    if( invert_second ):
        item2 = invol(item2)

    # Initialize the matrix
    DFT_MAT = [[0] * num_dim for _ in [0] * num_dim]
    IDFT_MAT = [[0] * num_dim for _ in [0] * num_dim]

    # Calculate the dft values
    for i in range(num_dim):
        for j in range(num_dim):
            DFT_MAT[i][j] = cos( -2 * pi * i * j / num_dim ) + sin( -2 * pi * i * j / num_dim ) * 1j
            IDFT_MAT[i][j] = 1/(num_dim*1.0) * (cos( 2 * pi * i * j / num_dim ) + sin( 2 * pi * i * j / num_dim ) * 1j)

    # Calculate DFT
    dft_item1 = [sum([item1[n] * DFT_MAT[d][n] for n in range(num_dim)]) for d in range(num_dim)]
    dft_item2 = [sum([item2[n] * DFT_MAT[d][n] for n in range(num_dim)]) for d in range(num_dim)]

    # Do the element wise multiplication
    mult = [dft_item1[n] * dft_item2[n] for n in range(num_dim)]

    # Perform the IDFT
    idft = [(sum([mult[n] * IDFT_MAT[d][n] for n in range(num_dim)])).real for d in range(num_dim)]

    # Return the result
    return idft


def ew_diff(item1, item2):
    # Calculates the element wise difference (item1 - item2) between the two inputs
    # Check to make sure the vector lengths are the same
    if( len(item1) != len(item2) ):
        print("util_funcs.diff - Mismatching vector lengths, returning []")
        return []

    return [item1[n] - item2[n] for n in range(len(item1))]


def ew_sum(item1, item2):
    # Calculates the element wise difference (item1 - item2) between the two inputs
    # Check to make sure the vector lengths are the same
    if( len(item1) != len(item2) ):
        print("util_funcs.diff - Mismatching vector lengths, returning []")
        return []

    return [item1[n] + item2[n] for n in range(len(item1))]


def vec_MSE(item1, item2):
    # Calculates the MSE (mean squared error) of the two vectors
    # MSE is calculated as the mean((item1 - item2) .^ 2)
    if( len(item1) != len(item2) ):
        print("util_funcs.vec_MSE - Mismatching vector lengths, returning 0")
        return 0

    return sum([(item1[n] - item2[n]) ** 2 for n in range(len(item1))]) / (len(item1) * 1.0)


def array_2_list_2D(in_array):
    result = [0] * len(in_array)

    for num in range(len(in_array)):
        result[num] = [in_array[num][d] for d in range(len(in_array[num]))]

    return result


def vector_2_list(in_vec):
    result = [in_vec[d] for d in range(len(in_vec))]
    return result
    

def find_best_match(item, ref_list):
    dots = [dot(item, ref_list[n], False) for n in range(len(ref_list))]
    max_dot = max(dots)

    for n in range(len(ref_list)):
        if( dots[n] == max_dot ):
            return ref_list[n]


def isScalarMatrix(matrix):
    scalar_val = matrix[0][0]
    num_cols = len(matrix[0])
    for row in range(len(matrix)):
        if( not matrix[row][row] == scalar_val or not sum(matrix[row][0:row]) + sum(matrix[row][row+1:num_cols]) == 0):
            return False
    return True            
            


#################### String Output Functions ####################
def list_2_2D_matlab_str(in_list, rjust_amt = 0):
    result = "["

    for n in range(len(in_list)):
        if( n != 0 ):
            result += "; " + "".ljust(rjust_amt + 1)
        result += str(in_list[n])
    result += "]"

    return result



#################### I/O Functions ####################
def read_csv(filename, read_as_str = False):
    data = []
    file_obj = open(filename, 'r')
    for line in file_obj.readlines():
        if( read_as_str ):
            row = [x for x in line.strip().split(',')]
        else:
            try:
                row = [float(x) for x in line.strip().split(',')]
            except:
                row = [x for x in line.strip().split(',')]
        data.append(row)
    file_obj.close()
    return data
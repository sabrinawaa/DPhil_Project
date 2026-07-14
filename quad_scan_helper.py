import numpy as np
from numpy import sin, cos, sinh, cosh
import re
from CLEAR_line import get_ITF

quad_length = 0.226

quad_names = [
    'CA.QFD0350', 
    'CA.QDD0355', 
    'CA.QFD0360',

    'CA.QFD0510', 
    'CA.QDD0515', 
    'CA.QFD0520',

    'CA.QFD0760', 
    'CA.QDD0765', 
    'CA.QFD0770',

    'CA.QDD0870', 
    'CA.QFD0880']

class Drift:
    def __init__(self, L):
        self.L = L
        self.M = np.array([[1, L, 0, 0],
                           [0, 1, 0, 0], 
                           [0, 0, 1, L], 
                           [0, 0, 0, 1]])

    def track(self, B0):
        B = B0 @ self.M.T
        return B
    
class Quad:
    def __init__(self, L, K):
        self.L = L
        self.set_K(K)

    def track(self, B0):
        B = B0 @ self.M.T
        return B
    
    def set_K(self, K):
        self.K = K

        K_sqrt = np.sqrt(np.abs(K))
        L = self.L

        # Just return a drift if K is 0 (avoids divide by zero)
        if K == 0:
            self.M = np.array([[1, L, 0, 0],
                               [0, 1, 0, 0], 
                               [0, 0, 1, L], 
                               [0, 0, 0, 1]])
            return

        # Else
        M_focus = np.array([[cos(L*K_sqrt)        , sin(L*K_sqrt)/K_sqrt],
                            [-sin(L*K_sqrt)*K_sqrt, cos(L*K_sqrt)       ]])
    
        M_defocus = np.array([[cosh(L*K_sqrt)        , sinh(L*K_sqrt)/K_sqrt],
                            [sinh(L*K_sqrt)*K_sqrt, cosh(L*K_sqrt)       ]])
        
        # Reorder focusing and defocusung matrix based on sign
        zeros = np.zeros((2, 2))
        if K > 0:
            self.M = np.block([[M_focus,     zeros],
                               [  zeros, M_defocus]])
        else:
            self.M = np.block([[M_defocus,   zeros],
                               [    zeros, M_focus]])

class Lattice():
    def __init__(self):
        self.elements = []

    def append_element(self, element):
        self.elements.append(element)

    def append_elements(self, elements):
        for element in elements:
            self.elements.append(element)

    def track(self, B0):
        B_list = [B0]
        for element in self.elements:
            B0 = element.track(B0)
            B_list.append(B0)

        return B_list
    
    def get_matrix(self):
        M = np.eye(4)
        for element in self.elements:
            M = element.M @ M

        return M
    
    def get_twiss_matrix(self):
        # x
        M = self.get_matrix()
        C, S, Cp, Sp = M[0:2, 0:2].flatten()
        M_twiss_x = np.array([[ C**2,      -2*C*S,  S**2],
                              [-C*Cp, C*Sp + S*Cp, -S*Sp],
                              [Cp**2,    -2*Cp*Sp, Sp**2]])
        
        # y
        C, S, Cp, Sp = M[2:4, 2:4].flatten()
        M_twiss_y = np.array([[ C**2,      -2*C*S,  S**2],
                              [-C*Cp, C*Sp + S*Cp, -S*Sp],
                              [Cp**2,    -2*Cp*Sp, Sp**2]])
        
        return M_twiss_x, M_twiss_y


# Source (February 2026)
# https://gitlab.cern.ch/acc-models/acc-models-clear/-/tree/master/survey

def get_quad_K(I, P_ref):
    G_0 = I * get_ITF(I) / quad_length
    K = 299.8 * G_0 / P_ref # 1/m2
    return K

# Load survey file
def select_beamline(beamline):

    if beamline == 'First':
        with open('clear_quadrupole_scans/resources/ca.survey0_filtered.tfs') as file:
            lines = file.readlines()
    elif beamline == 'Second':
        with open('clear_quadrupole_scans/resources/cs.survey0_filtered.tfs') as file:
            lines = file.readlines()
    return lines

with open('clear_quadrupole_scans/resources/clearST.survey0_filtered.tfs') as file:
            lines = file.readlines()

# Construct a dict of all elements relevant for the quad scan
element_descriptions = {}
previous_name = None
quad_index = 0

# Loop through the survey line-by-line
for line in lines:
    # Skip preamble
    if line[0:2] != ' "':
        continue

    # Find relevant parts of text
    text = re.findall(r'"([A-Za-z0-9.$_]+)"', line)
    numbers = re.findall('\d+\.\d+', line)

    name = text[0]

    # Skip everything except screens and quads
    if not('BTV' in name or 'QFD' in name or 'QDD' in name or 'QFG' in name or 'QDG' in name):
        continue

    # Skip unused screens
    if name == 'CA.BTV0215' or name == 'CA.BTV0800':
        continue

    # Specify element type
    element_type = name.split('.')[1][0:3]

    # Set length and position
    s_end = float(numbers[0])
    L = float(numbers[1])
    s_start = s_end - L
    s_center = round((s_start + s_end)/2, 5)

    # Adjust length if quad
    if element_type == 'QFD' or element_type == 'QDD' or element_type == 'QFG' or element_type == 'QDG':
        s_end = s_center + quad_length/2
        s_start = s_center - quad_length/2
        L = quad_length

    # Round values to remove float errors
    L = round(L, 4)
    s_start = round(s_start, 4)
    s_center = round(s_center, 5)
    s_end = round(s_end, 4)

    # Add drift from previous element
    if previous_name is not None:
        element_descriptions[previous_name + ' Drift'] = {
            'element_type': 'Drift',
            'L': round(s_start - element_descriptions[previous_name]['s_end'], 4),
            's_start': element_descriptions[previous_name]['s_end'], 
            's_center': round((element_descriptions[previous_name]['s_end'] + s_start)/2, 5), 
            's_end': s_start,
            'quad_index': None,
        }

    # Add current element
    element_descriptions[name] = {
        'element_type': element_type,
        'L': L,
        's_start': s_start, 
        's_center': s_center,
        's_end': s_end,
        'quad_index': quad_index if text[1] == 'QUADRUPOLE' else None,
    }

    if element_type == 'QFD' or element_type == 'QDD' or element_type == 'QFG' or element_type == 'QDG':
        quad_index += 1
    
    previous_name = name

# Return a lattice object from start to end, using a current vector with the currents of each quad in order
def get_lattice(start, end, P_ref, currents, include_end = True):
    # beamline.select_beamline(beamline)
    start_index = list(element_descriptions.keys()).index(start)
    end_index = list(element_descriptions.keys()).index(end)
    if include_end: end_index += 1

    K = get_quad_K(currents, P_ref)
    
    lattice = Lattice()
    for element_description in list(element_descriptions.values())[start_index:end_index]:
        element_type, L, _, _, _, quad_index = element_description.values()
        if element_type == 'Drift':
            element = Drift(L)
            lattice.append_element(element)

        elif element_type == 'QFD':
            element = Quad(L, K[quad_index])
            lattice.append_element(element)
            
        elif element_type == 'QDD':
            element = Quad(L, -K[quad_index])
            lattice.append_element(element)
            
    return lattice
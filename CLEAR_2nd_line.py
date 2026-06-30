#mostly copied from CLEAR files from antonio

import numpy as np
import re
import RF_Track
import math

#RF track does not use positions but build lattices from lengths of elements and drifts

#QUADRUPOLE CALCULATIONS

def get_ITF_2nd(I):
    return 1.29404711e-2 - 2.59458259e-07 * I  # T/A

def get_grad_2nd(I, Lquad):
    return I * get_ITF_2nd(I) / Lquad  # T/m

def get_Quad_K_2nd(G_0, P_ref):
    return 299.8 * G_0 / P_ref  # 1/m^2

def get_Quad_K_from_I_2nd(I, Lquad, P_ref):
    G_0 = get_grad_2nd(I, Lquad)
    return get_Quad_K_2nd(G_0, P_ref)


#DIPOLE CALCULATIONS

def Brho_electron(Ekin_MeV):
    me = 0.511  # MeV
    Etot = Ekin_MeV + me
    p_GeV = math.sqrt(Etot**2 - me**2) / 1000.0
    return p_GeV / 0.299792458  # T*m                   # Magnetic rigidity


def By_from_rbend_p(p_MeV, Lchord_m, angle_deg):
    Brho = (p_MeV / 1000.0) / 0.299792458
    theta = math.radians(angle_deg)
    rho = Lchord_m / (2.0 * math.sin(abs(theta) / 2.0))
    By = Brho / rho
    return -By if angle_deg > 0 else By



def get_2nd_beamline(
    survey_filename="data/CLEAR_2nd_Beamline_Survey.txt",
):
    """
    Build and return an RF_Track lattice from the CLEAR survey file for the 2nd beamline.

    Parameters
    ----------
    survey_filename : str
        Path to CLEAR 2nd Line survey file (TFS filtered version).
    start : str
        Name of start element.
    end : str
        Name of end element.
    P_ref : float
        Reference momentum [MeV/c].
    quad_currents : list of float
        List of quadrupole currents (A), in order.
    include_end : bool
        If True, includes the 'end' element.
    Q : float
        Particle charge (default = 1).

    Returns
    -------
    lattice : RF_Track.Lattice
        The constructed CLEAR lattice.
    """
    with open(survey_filename) as file:
        lines = file.readlines()

    element_descriptions = {}
    previous_name = None
    quad_index = 0
    corr_index = 0
    dip_index = 0 #added to count dipoles for indexing

    for line in lines:
        if not line.startswith(' "'):
            continue

        text = re.findall(r'"([A-Za-z0-9.$_]+)"', line) # extract letters,numbers and .$_ symbols within quotes
        numbers = re.findall(r'\d+\.\d+', line) # extract floating-point numbers
        if not text or len(numbers) < 2:
            continue

        name = text[0]
        
        # Element type
        if 'QFG' in name or 'QDG' in name:
            element_type = 'Quadrupole'
        elif 'BTV' in name:
            element_type = 'Screen'
        elif 'DHG' in name or 'DHJ' in name:
            element_type = 'Corrector'
        elif 'BPC' in name or 'BPM' in name:
            element_type = 'BPM'
        elif len(text) > 1 and text[1] == 'MARKER':
            element_type = 'Marker'
        elif 'BHB' in name:
            element_type = 'Dipole'
        else:
            continue

        s_end = float(numbers[0]) #s defined at end of element
        L = float(numbers[1])
        s_start = s_end - L #extract s_start 
        L, s_start, s_end = round(L, 4), round(s_start, 4), round(s_end, 4)

        if previous_name is not None:
            L_drift = round(s_start - element_descriptions[previous_name]['s_end'], 4)
            if L_drift != 0:
                element_descriptions[previous_name + '_Drift'] = {
                    'element_type': 'Drift',
                    'L': L_drift,
                    's_start': element_descriptions[previous_name]['s_end'],
                    's_end': s_start,
                    'quad_index': None,
                    'corr_index': None,
                }

        element_descriptions[name] = {
            'element_type': element_type,
            'L': L,
            's_start': s_start,
            's_end': s_end,
            'quad_index': quad_index if element_type == 'Quadrupole' else None,
            'corr_index': corr_index if element_type == 'Corrector' else None,
            'dip_index': dip_index if element_type == 'Dipole' else None,
        }

        if element_type == 'Quadrupole':
            quad_index += 1 #counting quads
        if element_type == 'Corrector':
            corr_index += 1
        if element_type == 'Dipole':
            dip_index += 1

        previous_name = name

    return element_descriptions #changed indentation 



def build_rft_lattice(element_descriptions,
        start = 'CS.2NDBL$START',           # tbc where to start, maybe from first quad 350? 
        end = 'CS.2NDLINE$END',
        P_ref = 198,
        quad_currents = [0]*9,
        Q=1,
        include_end=True):
    
    # --- Build lattice ---
    names = list(element_descriptions.keys())
    start_index = names.index(start) if start else 0
    end_index = names.index(end) if end else len(names) - 1
    if include_end:
        end_index += 1

    lattice = RF_Track.Lattice()
    elements = list(element_descriptions.values())

    for name, desc in zip(names[start_index:end_index], elements[start_index:end_index]):
        etype = desc['element_type']
        L = desc['L']
        qidx = desc['quad_index']

        if etype == 'Drift':
            elem = RF_Track.Drift(L)
        elif etype == 'Quadrupole':
            K = get_Quad_K_from_I_2nd(quad_currents[qidx], L, P_ref)
            if 'QDD' in name:
                K = -K
            elem = RF_Track.Quadrupole(L, P_ref / Q, K)
        elif etype == 'Corrector':
            elem = RF_Track.Corrector(L)
        elif etype == 'BPM':
            elem = RF_Track.Bpm(L)
        elif etype in ['Screen', 'Marker']:
            elem = RF_Track.Screen()
        elif etype == 'Dipole':
            elem = RF_Track.Drift(L)
            if '100' in name:
                By = By_from_rbend_p(P_ref, 0.4713, 22.0)
            elif '200' in name:
                By = By_from_rbend_p(P_ref, 0.4713, -22.0)
            else:
                raise ValueError(f"Unknown dipole: {name}")
            elem.set_static_Bfield(0.0, By, 0.0)
        else:
            continue

        lattice.append(elem)

    return lattice

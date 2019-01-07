"""
    Changes 5.7
    ------------
    using perimeter ratio instead of circularity

    Changes 5.6
    -------------
    added random seed indipendent of time.

    Notes: 5.4 vs 5.3, moving back to the model where ROCK1 cells are more stiff.


"""
import sys, os, errno
import argparse
import re
from shutil import copyfile
from xml.dom import minidom

import numpy as np


def main():
    args = read_args()                                # read command line input
    CellLines = configure_cell_lines()                # set/infer cell properties from experiments
    
    model_file = 'AdhesionDriven_CellModel_v5.7.2.latest.xml'
    configure_XML_model(args, model_file, CellLines)  # create folder, and modify XML file
    
    run_model(model_file)                             # runs simulation, saves stderr/stdout to file

def make_sure_path_exists(path):
    """ Function for safely making output folders """
 
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def read_args():

    # Read parameters from the command line
    parser = argparse.ArgumentParser(description=\
    "This is an extended Cellular Potts Model that captures self-sorting in hiPSCs.\n")
    
    #Set command line arguments
    parser.add_argument("cell_line_1",
                        help="Cell line 1.{CDH1-0,CDH1-70,CDH1-75,CDH1-90,ROCK1-20,wildtype}",
                        type=str)
    parser.add_argument('time_1',
                        help='Time cell line 1 recives gene knockout signal[-144,96]',
                        type=str)
    parser.add_argument('cell_line_2',
                        help='Cell line 2.{CDH1-0,CDH1-70,CDH1-75,CDH1-90,ROCK1-20,wildtype}',
                        type=str)
    parser.add_argument('time_2',
                        help='Time cell line 2 recieves gene knockout signal.[-144,96]',
                        type=str)
    parser.add_argument('num_cells_cell_line_2',
                        help='Number of initial cells for cell line 2. 100 cells total [0,99]',
                        type=str)
    parser.add_argument('sim_id',
                        help='Unique identifier for a simulations so replicates can be run.',
                        type=str)
    parser.add_argument('--simulation_time',
                        help='Number of hours the model will simulate. Default is 96 (units are hours)',
                        type=str)
    # Process arguments
    args = parser.parse_args()
    return args



def configure_cell_lines():
    ########################################
    # Create mapping from cell line to repsective parameters.
    # These values were obtained from experimental measuremnts
    # of cell area/perimeter/circularity, and time-depenet gene expression after
    # CRISPR knockout of selected genes. 
    ########################################
    # This will be used to replace region of the XML file
    
    C = {}  # Cell mappings
    cdh1_mutants = [70,75,90] # percent repression of gene

    C['wildtype'] = {
                'adhesion_weak':-100,
                'area_final':137,
                'area_final_sd':34.03,
                'area_edge_final':177,
                'area_edge_final_sd':49.66,
                'membraneElasticity_final':1.12,
                'membraneElasticity_final_edge':1.32,
		        'membrane_str_final':0.5,
                'k_half_hours':48.28,
                'generation_time_final': 20
                }
    C['CDH1-0'] = {
                    'adhesion_weak':-70,
                    'area_final':123,
                    'area_final_sd':51.78,
                    'area_edge_final':223,
                    'area_edge_final_sd':57.96,
                    'membraneElasticity_final':1.10,
                    'membraneElasticity_final_edge':1.23,
                    'membrane_str_final':0.5,
                    'k_half_hours':48.28,
                    'generation_time_final': 18
                    }

    # create mutants for CDH1
    for mutant_strength in cdh1_mutants:    
        s=(100.0-mutant_strength)/100.0  #relative mutatant decrease
        
        # set area (center)
        wt_area = C['wildtype']['area_final']
        cdh1_0_area = C['CDH1-0']['area_final']
        new_area = wt_area + (cdh1_0_area-wt_area)*s

        # set area(edge)
        wt_area_edge = C['wildtype']['area_edge_final']
        cdh1_0_area_edge = C['CDH1-0']['area_edge_final']
        new_area_edge = wt_area_edge + (cdh1_0_area_edge-wt_area_edge)*s
        
        # set membrane (center)
        wt_mem_center = C['wildtype']['membraneElasticity_final']
        cdh1_0_mem_center = C['CDH1-0']['membraneElasticity_final']
        new_membrane = wt_mem_center + (cdh1_0_mem_center-wt_mem_center)*s

        # set membrane (edge)
        wt_mem_edge = C['wildtype']['membraneElasticity_final_edge']
        cdh1_0_mem_edge = C['CDH1-0']['membraneElasticity_final_edge'] 
        new_membrane_edge = wt_mem_edge + (cdh1_0_mem_edge-wt_mem_edge)*s

        #set adhesion
        wt_adhesion = C['wildtype']['adhesion_weak']
        cdh1_0_adhesion = C['CDH1-0']['adhesion_weak'] 
        new_adhesion = wt_adhesion + (cdh1_0_adhesion-wt_adhesion)*s

        #create cell line
        cell_line_name = 'CDH1-%s' % str(mutant_strength)
        C[cell_line_name] = {
                        'adhesion_weak':new_adhesion,
                        'area_final':new_area,
                        'area_final_sd':51.78,
                        'area_edge_final':new_area_edge,
                        'area_edge_final_sd':57.96,
                        'membraneElasticity_final':new_membrane,
                        'membraneElasticity_final_edge':new_membrane_edge,
                        'membrane_str_final':0.5,
                        'k_half_hours':48.28,
                        'generation_time_final': 20
                        }

    # ROCK1 mutant has only one reference.
    # this ref isnt at 0 so rel mutants are not inferred.
    C['ROCK1-20'] = {
                    'adhesion_weak':-85, #new data suggests 50% CDH1 expression
                    'area_final':170,
                    'area_final_sd':53.54,
                    'area_edge_final':228,
                    'area_edge_final_sd':60.06,
                    'membraneElasticity_final':1.17, #was 1.37
                    'membraneElasticity_final_edge':1.41,  #was 1.97
                    'membrane_str_final':1.1, #wt is 0.5
                    'k_half_hours':46.78,
                    'generation_time_final': 20
                    }
    
    return C


##############################################################################
def configure_XML_model(args, model_file,cell_lines):
    output_folder = "%s_%s_%s_%s_%s_%s" % (args.cell_line_1,args.time_1,args.cell_line_2,args.time_2,args.num_cells_cell_line_2,args.sim_id)
    
    output_folder_prefix = "simulations"
    output_folder = os.path.join(output_folder_prefix,output_folder) #prefix folder
    
    
    #create new directory for sim and chdir
    make_sure_path_exists(output_folder)
    copyfile(model_file, os.path.join(output_folder,model_file))
    os.chdir(output_folder)
    
    # Modify new XML file with model parameters
    change_XML(args,model_file,cell_lines)


def change_XML(args,model_file,cell_lines):
    """
        Function: Modifies the XML file used to run simulations.
            1. change cell properties for cell line 1 and cell line 2
            2. change the random seed so simulations are truly stochastic
            3. change the simulation stop time.

    Note. All cell property variables must be prefixed with ct1 or ct2.
          All variables defined in this file must match the corresponding
          variable name in the Morpheus XML file exactly!
    """
    Cell_1 = cell_lines[args.cell_line_1]
    Cell_2 = cell_lines[args.cell_line_2]

    if args.cell_line_1 not in cell_lines:
        print("Error: The cell line %s is not supported by the simulator." % args.cell_line_1)
        print("Try one of these:",list(cell_lines.keys()))
        sys.exit()
    if args.cell_line_2 not in cell_lines:
        print("Error: The cell line %s is not supported by the simulator." % args.cell_line_2)
        print("Try one of these:",list(cell_lines.keys()))
        sys.exit()

    # 1. change ct1 and ct2 global variables
    doc = minidom.parse(model_file)
    variables = doc.getElementsByTagName("Variable")
    
    for var in variables:
        symbol_name = var.attributes["symbol"].value
    
        # set cell line 1
        if symbol_name.startswith('ct1'):
            if 'perturbation' in symbol_name:
                var.setAttribute("value",str(args.time_1))
                #print symbol_name,args.time_1
            else:
                try:
                    #print Cell_1[symbol_name[4:]],symbol_name
                    var.setAttribute("value",str(Cell_1[symbol_name[4:]]))
                except KeyError as exception:
                    pass

        # set cell line 2
        elif symbol_name.startswith('ct2'):
            if 'perturbation' in symbol_name:
                #print symbol_name,args.time_2
                var.setAttribute("value",str(args.time_2))
            else:
                try:
                    #print Cell_2[symbol_name[4:]],symbol_name
                    var.setAttribute("value",str(Cell_2[symbol_name[4:]]))
                except KeyError as exception:
                    pass

        # Set cell ratios
        elif symbol_name == 'num_cells_ct2':
            var.setAttribute("value",str(args.num_cells_cell_line_2))
            #print symbol_name,args.num_cells_cell_line_2
    
    # 2. Change the random seed to a random number not dependent on time.
    random_integer = np.random.randint(0,100000)
    RandomSeed_variable = doc.getElementsByTagName("RandomSeed")[0]
    RandomSeed_variable.setAttribute("value",str(random_integer))

    # 3. Change simulation stop time in XML file
    simulation_stop_time = args.simulation_time
    if simulation_stop_time == None:
        simulation_stop_time = 96
    stop_time = doc.getElementsByTagName("StopTime")[0]
    stop_time.setAttribute("value",str(simulation_stop_time))
    
    # Write XML file 
    file_handle = open(model_file,"w")
    doc.writexml(file_handle)
    file_handle.close()
    #sys.exit()

######################################## running the model ####################
def run_model(model_file):
    status = os.system("morpheus -file %s > model_output.txt 2>&1" % model_file)

    return status

##########################################
if __name__ == "__main__":
    main()

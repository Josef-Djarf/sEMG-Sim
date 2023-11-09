from MU import MotorUnit
from sEMG import SurfaceEMG
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

#########################    #########################    ##########################    #######################

MU1 = MotorUnit()
#MU1_pole_amplitude = MU1.get_tripole_amplitude()
#print(sum(MU1_pole_amplitude))
#MU1_plot_cuurent_AP = MU1.plot_current_distribution_action_potential()
#MU1_pole_distance = MU1.get_tripole_distance()
#print(MU1_pole_distance[0]/MU1_pole_distance[1])
#MU1_single_fibre = MU1.simulate_fibre_action_potential()
#MU1_plot_single_fibre = MU1.plot_single_fibre_action_potential()
#MU1_motor_unit = MU1.simulate_motor_unit()
#MU1_plot_motor_unit = MU1.plot_motor_unit()
#sEMG1 = SurfaceEMG()
#simulate_recruitment = sEMG1.simulate_recruitment_model()
#plot_recruitment = sEMG1.plot_recruitment_model()
#simulate_sEMG = sEMG1.simulate_suface_EMG()

#########################    #########################    ##########################    #######################

MU2 = MotorUnit()
#MU2.A = 10
#MU2.B = -10
#MU2.C = 2
#MU2.plot_length = 3
#MU2.scaling_factor = 3
#MU2_pole_amplitude = MU2.get_tripole_amplitude()
#MU2.y_limit_minimum = -2
#MU2.y_limit_maximum = 2
#MU2_plot_cuurent_AP = MU2.plot_current_distribution_action_potential()
#MU2_pole_distance = MU2.get_tripole_distance()

#MU2.fibre_length = 1
#MU2.conduction_velocity = 1
#MU2.ratio_axial_radial_conductivity = 1
#MU2.radial_conductivity = 1
#MU2.inter_electrode_spacing = 1
#MU2.number_of_electrodes_z = 10
#MU2.inter_electrode_shift = 1
#MU2.initial_neuromuscular_junction = 1
#MU2.fibre_depth = 1
#MU2.extermination_zone_width = 1
#MU2.innervation_zone_width = 1
#MU2.time_length = 10
#MU2_single_fibre = MU2.simulate_fibre_action_potential()
#MU2_plot_single_fibre = MU2.plot_single_fibre_action_potential()

#MU2.motor_unit_radius = 1
#MU2.number_of_fibres = 1
#MU2_motor_unit = MU2.simulate_motor_unit()
#MU2_plot_motor_unit = MU2.plot_motor_unit()

#########################    #########################    ##########################    #######################
MU_test = MotorUnit()
#MU_test.time_length = 35
#MU_test.number_of_electrodes_z = 1
#MU_test.number_of_electrodes_x = 1
#MU_test.plot_single_fibre_action_potential()
#MU_test.plot_motor_unit()

sEMG_test = SurfaceEMG()
#sEMG_test.maximum_excitation_level = 50
#sEMG_test.number_of_motor_units = 1
#sEMG_test.electrodes_in_z = 10
#sEMG_test.electrodes_in_x = 5
#sEMG_test.simulate_recruitment_model()
#sEMG_test.plot_recruitment_model()

sEMG_test.plot_suface_EMG()



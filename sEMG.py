from MU import MotorUnit
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import NullFormatter
plt.rcParams['font.family'] = 'Times New Roman'

class SurfaceEMG:
  """A class that erepresnts a surface EMG (sEMG).

  Methods:
    simulate_recruitment_model: 
    plot_recruitment_model:
    simulate_suface_EMG: 
    plot_suface_EMG:  

  Attributes:
    For simulate_recruitment_model:
      simulation_time: Total simulation time in seconds.
      sampling_rate: Sample rate in Hz.
      ramp: Up-stable-down in seconds.
      maximum_excitation_level: Percentage % of max exc.
      number_of_motor_units: Number of motoneurons in the pool.
      recruitment_range: Range of recruitment threshold values.
      excitatory_gain: Gain of the excitatory drive-firing rate relationship.
      minimum_firing_rate: Minimum firing rate (Hz).
      peak_firing_rate_first_unit: Peak firing rate of the first motoneuron (Hz).
      peak_firing_rate_difference: Desired difference in peak firing rates between the first and last units (Hz):
      inter_spike_interval_coefficient_variation: The inter spike interval variance coefficient.
    
    For simulate_surface_emg:
      twitch_force_range: The range of twitch forces RP (force units).
      motor_unit_density: The motor-unit fibre density (20 unit fibres/mm^2 area of muscle).
      smallest_motor_unit_number_of_fibres: The smallest motor unit innervated 28 fibres.
      largest_motor_unit_number_of_fibres: The largest motor unit innervated 2728 fibres.
      muscle_fibre_diameter: The muscle-fibre diameter (46 µm).
      muscle_cross_sectional_diameter: The muscle cross-sectional diameter (1.5 cm).
      electrodes_in_z: Number of elecrodes in the array, in the direction of the fibre.
      electrodes_in_x: Number of electrodes in the array across the fiber.

    For plotting methods:
      y_limit_minimum: Minimum value of plot y-axis.
      y_limit_maximum: Maximum value of plot y-axis.
  """
  ...

  def __init__(self):
    """Initializes a new sEMG object.

    """
    ...

  ### Sampling parameters
    self.simulation_time = 30 # Total simulation time in seconds.
    self.sampling_rate = 10000 # Sample rate (10 kHz).
    self.ramp = np.array([5, 20, 5]) # Up, stable, and down times of the ramp in seconds.
    self.maximum_excitation_level = 20 # Maximum excitation level as a percentage of maximum.
    
  ### Motorneuron parameters
    self.number_of_motor_units = 200 # Number of motoneurons (units) in the pool.
    self.recruitment_range = 30 # Range of recruitment threshold values.
    self.excitatory_gain = 1 # Gain of the excitatory drive-firing rate relationship.
    self.minimum_firing_rate = 8 # Minumum firing rate.
    self.peak_firing_rate_first_unit = 35 # Peak firing rate of the first motoneuron.
    self.peak_firing_rate_difference = 10 # Desired difference in peak firing rates between the first and last units.
    self.inter_spike_interval_coefficient_variation = 0.15 # The inter spike interval variance coefficient.

  ### Number of fibres parameters
    self.twitch_force_range = 100 # The range of twitch forces RP (force units).
    self.motor_unit_density = 20  # The motor-unit fibre density (20 unit fibres/mm^2 area of muscle).
    self.smallest_motor_unit_number_of_fibres = 28 # The smallest motor unit innervated 28 fibres.
    self.largest_motor_unit_number_of_fibres = 2728 # The largest motor unit innervated 2728 fibres.
    self.muscle_fibre_diameter = 46e-3 # (mm) The muscle-fibre diameter (46 µm).
    self.muscle_cross_sectional_diameter = 15 # (mm) The muscle cross-sectional diameter (1.5 cm).

  ### Plot parameters
    self.electrodes_in_z = 1
    self.electrodes_in_x = 1
    self.y_limit_minimum = -1
    self.y_limit_maximum = 1

  #########################  1  #########################  Simulate Recruitment Model ##########################    #######################
  def simulate_recruitment_model(self):
    """Generates the recruitment and rate coding organization of motor units.

    Arguments:
      According to Models of Recruitment and Rate Coding Organization in Motor-Unit Pools. Fuglevand, et al 1993.
        simulation_time: Entire duration of the simulation (s).
        sampling_rate: Sampling frequency of the simulation (Hz).
        ramp: Excitatory drive function in a trapeziod shape (ramp-up, stable, ramp-down).
        maximum_excitation_level: Maximum excitation level of motor unit in percent (%).
        number_of_motor_units: Total number of motor units in the simulation.
        recruitment_range: The desired maximum for the range of recruitment threshold values.
        excitatory_gain: The gain of the excitatory drive-firing rate relationship.
        minimum_firing_rate: Minimum firing rate (Hz).
        peak_firing_rate_first_unit: Peak firing rate of the first motoneuron (Hz).
        peak_firing_rate_difference: The desired difference in peak firing rates between the first and last units (Hz).
        inter_spike_interval_coefficient_variation: The variance of inter spike interval coefficient.

    Returns:
      A list containing firing time arrays for each motor unit.
    """
    ...
    
  ### Default arguments:
    simulation_time = self.simulation_time
    sampling_rate = self.sampling_rate
    number_of_motor_units = self.number_of_motor_units
    recruitment_range = self.recruitment_range
    peak_firing_rate_first_unit = self.peak_firing_rate_first_unit
    peak_firing_rate_difference = self.peak_firing_rate_difference
    minimum_firing_rate = self.minimum_firing_rate
    excitatory_gain = self.excitatory_gain
    maximum_excitation_level = self.maximum_excitation_level
    ramp = self.ramp
    inter_spike_interval_coefficient_variation = self.inter_spike_interval_coefficient_variation 

  ### Time vector
    time_array = np.linspace(0, simulation_time, simulation_time*sampling_rate)
    self.time_array = time_array
    
  ### Calculate the recruitment threshold excitation. Equation (1) in Fuglevand 1993.
    a = (np.log(recruitment_range) / number_of_motor_units) # Constant related to eq. (1).
    recruitmenexcitatory_drive_thresholdold_excitation = np.exp(a*(np.arange(1, number_of_motor_units + 1, 1)))

  ### Calculate the peak firing rate for each motoneuron. Equation (5) in Fuglevand 1993.
    peak_firing_rate_i = peak_firing_rate_first_unit - peak_firing_rate_difference * (recruitmenexcitatory_drive_thresholdold_excitation / recruitmenexcitatory_drive_thresholdold_excitation[-1])

  ### Calculate the maximum excitation. Equation (8) in Fuglevand 1993.
    maximum_excitation = recruitmenexcitatory_drive_thresholdold_excitation[-1] + (peak_firing_rate_i[-1] - minimum_firing_rate) / excitatory_gain

  ### Define the excitatory drive function.
    excitatory_drive_function = np.concatenate((np.linspace(0, maximum_excitation * (maximum_excitation_level/100), ramp[0] * sampling_rate), np.ones(ramp[1] * sampling_rate) * maximum_excitation * (maximum_excitation_level/100), (np.flip(np.linspace(0, maximum_excitation * (maximum_excitation_level/100),ramp[2] * sampling_rate)))))

  ### Initialize the firing times for each motoneuron.
    firing_times_motor_unit = [[] for i in range(number_of_motor_units)]
    #firing_times_motor_unit = np.array([], dtype=float)

    # iteration_variableate over each motoneuron.
    for i in range(number_of_motor_units):
      # Calculate the thresholded excitatory drive.
      excitatory_drive_threshold = excitatory_drive_function - recruitmenexcitatory_drive_thresholdold_excitation[i]

      # Find the samples that are associated with firing. Above this thresh => fire
      find_excitatory_drive_threshold = np.where(excitatory_drive_threshold >= 0)[0]

      # If there are no samples associated with firing, continue.
      if len(find_excitatory_drive_threshold) == 0:

        continue

      # Calculate the time of the first impulse.
      firing_times_motor_unit[i].append(time_array[find_excitatory_drive_threshold[0]])

      # Calculate points of exceeded threshold 
      excitation_difference = excitatory_drive_threshold[find_excitatory_drive_threshold[0]]
      
      # Time point for the first impulse 
      time_instance = firing_times_motor_unit[i][0]

      # Initialize the firing counter.
      iteration_variable = 0

      ## Iterate until the current time is greater than the last sample point associated with firing.
      while time_instance <= time_array[find_excitatory_drive_threshold[-1]]:
        # Calculate the interspike interval.
        inter_spike_interval = max(1 / (excitatory_gain * excitation_difference + minimum_firing_rate), 1 / peak_firing_rate_i[i])

        firing_times_motor_unit[i].append(firing_times_motor_unit[i][iteration_variable] + (inter_spike_interval_coefficient_variation * inter_spike_interval) * np.random.randn() + inter_spike_interval)
        #firing_times_motor_unit = np.append(firing_times_motor_unit, [firing_times_motor_unit[iteration_variable] + (inter_spike_interval_coefficient_variation * inter_spike_interval) * np.random.randn() + inter_spike_interval])

        # Update the firing counter.
        iteration_variable += 1

        # Find the minimum index of the sample point that is closest to the current firing time.
        minimum_time_index = np.argmin(np.abs(firing_times_motor_unit[i][iteration_variable] - time_array[find_excitatory_drive_threshold]))

        # Update the thresholded excitatory drive.
        excitation_difference = excitatory_drive_threshold[find_excitatory_drive_threshold[minimum_time_index]]

        # Update the current time.
        time_instance = firing_times_motor_unit[i][iteration_variable]

    #print('Firing Times for each Motor Unit', firing_times_motor_unit)
    return firing_times_motor_unit
  
  #########################  2  #########################  Plot Recruitment Model  ##########################    #######################
  def plot_recruitment_model(self):
    """Plots the firing patterns of the motor unit recruitment.

    Arguments:
      firing_times_motor_unit
      time_array

    Returns:
      A plot of the recruitment model for each motor unit.
    """
    ...

  ### Default arguments:
    firing_times_motor_unit = self.simulate_recruitment_model()
    time_array = self.time_array

  ### En lista med olika färger för varje motor enhet
    #colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    def generate_unique_colors(num_colors):
      unique_colors = set()
      colors = []

      while len(unique_colors) < num_colors:
        color = "#{:02X}{:02X}{:02X}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if color not in unique_colors:
            unique_colors.add(color)
            colors.append(color)

      return colors

    num_colors = self.number_of_motor_units  # Ange det önskade antalet unika färger här
    colors = generate_unique_colors(num_colors)

  ### Plot of the firing times motor unit
    fig4 = plt.figure(4)
    for i, time_array in enumerate(firing_times_motor_unit):
      color = colors[i % len(colors)]  # Välj en färg från listan baserat på i
      plt.plot(time_array, [i] * len(time_array), '|', color = color, label = f'Motor Unit {i + 1}')

    plt.xlabel('Time (s)')
    plt.ylabel('Motor Unit Number')
    plt.title('Firing Times of Motoneurons')
    plt.grid(axis = 'x')

    return plt.show()
  
  #########################  3  #########################  Simulate Surface Electromyography  ##########################    #######################
  def simulate_surface_emg(self):
    """Simulates the surface electromyography based on the recruitment model.

    Arguments:
      firing_times_motor_unit
      time_array
      motor_unit_i
      twitch_force_range
      number_of_motor_units
      motor_unit_density
      smallest_motor_unit_number_of_fibres
      largest_motor_unit_number_of_fibres
      muscle_fibre_diameter
      muscle_cross_sectional_diameter
      electrodes_in_z
      electrodes_in_x

    Returns:
      A list containing an array with the simulated surface electromyography based on the recruitment model of each motor unit for all electrodes.
    """
    ...

    firing_times_motor_unit = self.simulate_recruitment_model()
    time_array = self.time_array
    motor_unit_i = MotorUnit()
    twitch_force_range = self.twitch_force_range
    number_of_motor_units = self.number_of_motor_units
    motor_unit_density = self.motor_unit_density
    smallest_motor_unit_number_of_fibres = self.smallest_motor_unit_number_of_fibres
    largest_motor_unit_number_of_fibres = self.largest_motor_unit_number_of_fibres
    muscle_fibre_diameter = self.muscle_fibre_diameter
    muscle_cross_sectional_diameter = self.muscle_cross_sectional_diameter
    electrodes_in_z = self.electrodes_in_z
    electrodes_in_x = self.electrodes_in_x

  ### Calculate the number of fibres innervated by each motor unit according to equation (21) Fuglevand et al 1993.
    # Calculate the peak twitch force for each unit accroding to equation (13) in Fuglevand 1993.
    b = (np.log(twitch_force_range) / number_of_motor_units) # Constant related to eq. (13).
    peak_twitch_force = np.exp(b*(np.arange(1, number_of_motor_units + 1, 1))) # Pi, where i = np.arange(1, number_of_motor_units + 1, 1)
    
    # The numbmer of muscle fibres required to exert one unit of force (1 unit force ≈ twitch force of smallest motor unit)
    total_peak_twitch_forces = np.sum(peak_twitch_force) # P_tot

    ## Calculate the total number of fibres (nf_tot) in a muscle, with a cross-sectional area (Am) and average area of a muscle fiber (Af). 
    # The muscle cross-sectional area (mm^2)
    Am = np.pi * (muscle_cross_sectional_diameter/2)**2 # Am

    # The muscle fibre average area (mm^2)
    Af = np.pi * (muscle_fibre_diameter/2)**2 # Af

    nf_tot = Am/Af # nf_tot 

    ## The number of fibres (nf_i) innervated by each motor unit according to equation (21) Fuglevand et al 1993.
    number_of_fibres = (nf_tot/total_peak_twitch_forces) * peak_twitch_force # nf_i

  ### The area encompassed by each motor-unit territory (Ai), was then calculated from the unit fibre density according to equation (22) Fuglevand et al 1993
    motor_unit_area = number_of_fibres/motor_unit_density # Ai

    motor_unit_radius = (motor_unit_area/np.pi)
  
  ### Calculate simuations of the surface EMG signal
    simulations = []

    for m, element in enumerate(firing_times_motor_unit):
      motor_unit_i.number_of_fibres = int(number_of_fibres[m])
      motor_unit_i.motor_unit_radius = motor_unit_radius[m]
      motor_unit_i.number_of_electrodes_z = electrodes_in_z
      motor_unit_i.number_of_electrodes_x = electrodes_in_x
  
      current_motor_unit = motor_unit_i.simulate_motor_unit()

      simulation = np.full((len(current_motor_unit), len(time_array)), current_motor_unit[0,-1])
      self.simulation = simulation

      for e in range(len(element)):
        # Find the time index where a firing occurs
        time_index = np.argmin(np.abs(time_array - element[e]))
        
        # Add the current motor unit to the simulation at the appropriate time index
        if simulation[:, time_index:time_index + current_motor_unit.shape[1]].shape >= current_motor_unit.shape:
          simulation[:, time_index:time_index + current_motor_unit.shape[1]] += current_motor_unit
    
      simulations.append(simulation)
    
    return simulations
  
  #########################  4  #########################  Plot Suface Electromyography Array  ##########################    #######################
  def plot_suface_emg_array(self):
    """Generates the recruitment and rate coding organization of motor units.

    Arguments:
      firing_times_motor_unit
      time_array

    Returns:
      A plot of the recruitment model for each motor unit.
    """
    ...

  ### Default arguments:
    simulations = self.simulate_surface_emg()
    time_array = self.time_array
    y_limit_minimum = self.y_limit_minimum
    y_limit_maximum = self.y_limit_maximum
    number_of_electrodes_z = self.electrodes_in_z
    number_of_electrodes_x = self.electrodes_in_x

    electrode_one_sum = np.zeros((number_of_electrodes_z * number_of_electrodes_x, len(time_array)))
  
    for m, simulation in enumerate(simulations):
      for ne in range(simulation.shape[0]):
        electrode_one_sum[ne, :] += simulation[ne,:]

  ### Plot the normalized motor unit action potential
    normalized_simulation = - electrode_one_sum
    normalized_simulation = (normalized_simulation - normalized_simulation.mean()) / (normalized_simulation.max() - normalized_simulation.min())

    # The single fibre action potentials recorded by the electrodes positioned along the length of the fibre.
    array_size = np.arange(1, (number_of_electrodes_z*number_of_electrodes_x)+1, 1)
    zeros_array = np.zeros(len(array_size))
    array_size_x = np.arange(0, number_of_electrodes_z*number_of_electrodes_x, number_of_electrodes_x)
    array_size_x = np.append(array_size_x, zeros_array)
    
  ### Plot the simulations for each motor unit as an array 
    fig8 = plt.figure(8)
    for i in range(len(array_size)):
      ax = plt.subplot(number_of_electrodes_z , number_of_electrodes_x, array_size[i])
      plt.subplots_adjust(wspace=0.0, hspace=0.0)
      ax.grid(which = 'both', ls = 'dashed')
      plt.plot(time_array, normalized_simulation[i, :])
      plt.xlim(time_array[0], time_array[-1] - 1)
      if i < len(array_size) - number_of_electrodes_x:
        ax.xaxis.set_major_formatter(NullFormatter())
      plt.ylim(y_limit_minimum,y_limit_maximum)
      ax.yaxis.set_major_formatter(NullFormatter())
      if i == 0:
        ax.set_ylabel(1, rotation = 0, ha = 'center', va = 'center', fontsize = 15)
      for j in range(i):
        if i == array_size_x[j]:
          ax.set_ylabel(array_size[j], rotation = 0, ha = 'center', va = 'center', fontsize = 15)
        elif number_of_electrodes_x == 1:
          ax.set_ylabel(array_size[j]+1, rotation = 0, ha = 'center', va = 'center', fontsize = 15)
    plt.suptitle('The Surface Electromyography Signal Array', fontsize = 20)

    if number_of_electrodes_x > 1:
      fig8.supxlabel('Time (s)\n Electrodes in the x direction, i.e. vertically across the fiber')
    else:
      fig8.supxlabel('Time (s)')
    if number_of_electrodes_z > 1:
      fig8.supylabel('sEMG Signal\n Electrodes in the z direction, i.e. along the fiber', ha = 'center', va = 'center')
    else: 
      fig8.supylabel('sEMG Signal', ha = 'center', va = 'center')

    return plt.show()
 
#  #########################    #########################               ##########################    #######################
                                                          # THE END #
#  #########################    #########################               ##########################    #######################

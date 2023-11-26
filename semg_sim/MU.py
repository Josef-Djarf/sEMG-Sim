import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
plt.rcParams['font.family'] = 'Times New Roman'

class MotorUnit:
  """A class that represents a motor unit (MU).

  Methods:
    get_tripole_amplitude: Calculates the action potential and the current distribution. Returns the tripole amplitudes of the current distribution.
    plot_current_distribution_action_potential: Plots the normalized action potential and current distribution based on the given parameter values.
    get_tripole_distance: Calculates the distance between the poles of the current distribution based on returned value from get_tripole_amplitude.
    simulate_fibre_action_potential: Simulates the action potential recorded by a given number of electrodes, positioned along one fibre, for the given current distriubtion.
    plot_fibre_action_potential: Plots the normalized action potential for one fibre over the electrodes positioned along the fibre. 
    simulate_motor_unit: Simulates the action potential for a given number of fibres, for electrodes positioned along the motor unit.
    plot_motor_unit: Plots the normalized action potential simulated for one motor unit, for a given number of fibres, and electrodes positioned along the motor unit.

  Attributes:
    For get_tripole_amplitude:
      A: Constant to fit the amplitude of the tripole (V).
      B: Resting membrane potential amplitude (V).
      C: Muscle fibre proportionality constant.
      plot_length: Length for plotting the action potential and current distribution expressed in mm (20 mm).
      scaling_factor: A scaling factor expressed in mm^-1, lambda (λ).

    For simulate_fibre_action_potential: 
      fibre_length: Length of fibre (mm).
      neuromuscular_junction: Position of NMJ along the length of the fibre (mm).
      conduction_velocity: Conduction velocity along the muscle fibre (m/s). 
      electrode_shift: Position of center of electrode array along the length of the fibre (mm). [electrode_shift = neuromuscular_junction means the array sits centered above NMJ.]
      number_of_electrodes_z: Number of elecrodes in the array, in the direction of the fibre.
      number_of_electrodes_x: Number of electrodes in the array across the fiber.
      inter_electrode_spacing: Distance between electrodes in the array (mm).
      radial_conductivity: Conductivity in radial direction, across the fibre (m/s).
      ratio_axial_radial_conductivity: Ratio between axial and radial conductivity, (fibre direction conductivity / radial conductivity)
      fibre_depth: Initial depth from the surface EMG down to the fibre (mm).

    For simulate_motor_unit:
      motor_unit_radius: Radius of the motor unit (mm).
      number_of_fibres: The number of fibres in a given motor unit.

    For plotting methods:
      y_limit_minimum: Minimum value of plot y-axis.
      y_limit_maximum: Maximum value of plot y-axis.
  """
  ...

  def __init__(self):
    """Initializes a new Motor Unit object.

    """
    ...

  ### Default values for generating a tripole:
    self.A:float = 96e-3
    self.B:float = -90e-3
    self.C:int = 1
    self.plot_length:int = 20
    self.scaling_factor:int = 1

  ### Default values for a single fibre:
    self.fibre_length:int = 210 
    self.conduction_velocity:int = 4 
    self.ratio_axial_radial_conductivity:int = 6 
    self.radial_conductivity:float = 0.303
    self.inter_electrode_spacing:int = 8
    self.number_of_electrodes_z:int = 10
    self.number_of_electrodes_x:int = 5
    self.electrode_shift:int = 165
    self.initial_neuromuscular_junction:int = 90
    self.fibre_depth:int = 10
    self.extermination_zone_width:int = 10
    self.innervation_zone_width:int = 5
    self.time_length:int = 35 

  ### Default values for a motor unit:
    self.motor_unit_radius:float = 1
    self.number_of_fibres:int = 100

  ### Default values for plots:
    self.y_limit_minimum:int = -1
    self.y_limit_maximum:int = 1

  #########################  1  #########################  Get Tripole Amplitude  ##########################    #######################
  def get_tripole_amplitude(self):
    """Generates the tripole amplitude from the membrane current distribution source.

    Args:
      According to Modeling of Surface Myoelectric Signals, part I. Merletti, 1999.
        A: A suitable constant to fit the amplitude of the action potential expressed in V (96 mV = 96e-3).
        B: The resting membrane poteintial expressed in V (-90 mV = -90e-3).
        C: Muscle fibre (proportionality) constant (1).
        plot_length: Length of plotting the action potential and current distribution expressed in mm (20 mm).
        scaling_factor: A scale factor expressed in mm^-1, which is lambda (λ), (1 mm^-1).

    Returns:
      A numpy array containing the tripole amplitude of the current distribution.
    """
    ...
  
  ### Default arguments:
    A = self.A
    B =  self.B
    C = self.C
    plot_length = self.plot_length
    scaling_factor =  self.scaling_factor
    
  #########################  Action Potential, Current Distribution  #######################
  ### Create the z-axis vector, which is the sampled vector of plotting the action potential and current distribution:
    delta_length = 0.1 # (10 kHz) total distance between samples
    z = np.arange(0, plot_length + delta_length, delta_length)
    self.z = z

  ### A mathematical description of the action potential
    action_potential = A * (scaling_factor * z)**3 * np.exp(-scaling_factor * z) - B #  eq.1.1 Merletti part 1
    self.action_potential = action_potential
    
  ### Calculate the mebrane current distribution which is proportional to the second derivative of the action potential
    current_distribution = C * A * scaling_factor**2 * (scaling_factor * z) * (6 - 6 * scaling_factor * z + scaling_factor**2 * z**2) * np.exp(-scaling_factor * z) # eq. 1.2 Merletti part 1
    self.current_distribution = current_distribution

  #########################  Pole Amplitude  #######################
  ### Calculate the pole amplitude
    ## To calculate pole_one, pole_two, and pole_three
    # Discretize the current distribution to 1 and -1
    current_distribution_discrete = np.where(current_distribution > 0, 1, -1)
 
    # Calculate the absolute difference between each sample of discretized current distribution
    current_distribution_difference = np.abs(np.diff(current_distribution_discrete))

    # Locate the absolute differences that are greater than zero
    pole_location_index = np.where(current_distribution_difference > 0)[0]
    self.pole_location_index = pole_location_index

    # Calculate the poles
    dz = z[1] - z[0]
    self.dz = dz
    
  ### Sum pole magnitude for each part of the tripole.
    pole_one = np.sum(current_distribution[:pole_location_index[1]]) * dz 
    pole_two = np.sum(current_distribution[pole_location_index[1] + 1:pole_location_index[2]]) * dz 
    pole_three = np.sum(current_distribution[pole_location_index[2] + 1:]) * dz 

  ### Use rounding adjustment to set sum of all poles equal to zero.
    pole_rounding_adjustment = np.abs(pole_one + pole_two + pole_three)
    pole_one = pole_one + pole_rounding_adjustment
    pole_two = pole_two - pole_rounding_adjustment
    pole_three = pole_three + pole_rounding_adjustment

  ### Poles amplitude array.
    poles_amplitude = np.array([pole_one, pole_two, pole_three])
    #print('Poles Amplitude =', poles_amplitude)

    return poles_amplitude

  #########################  2  #########################  Plot Current Distribution & Action Potential   ##########################    #######################
  def plot_current_distribution_action_potential(self):
    """Plots the membrane current distribution and action potential.

    Args:
      current_distribution
      action_potential
      z: Sampled vector of plotting

    Returns:
      A plot of the current distribution and action potential.
    """
    ...

  ### default arguments
    self.get_tripole_amplitude()
    current_distribution = self.current_distribution
    action_potential = self.action_potential
    z = self.z

  #########################  Normalization and Plot  #######################
  ### Plot the normalized current distribution and action potential
    # Normalize the signals using min-max feature scaling between points y_limit_minimum and y_limit_maximum.
    y_limit_minimum = self.y_limit_minimum
    y_limit_maximum = self.y_limit_maximum

    normalized_current_distribution = y_limit_minimum + ((current_distribution - current_distribution.min())*(y_limit_maximum-y_limit_minimum)) / (current_distribution.max() - current_distribution.min())
    normalized_action_potential = y_limit_minimum + ((action_potential - action_potential.min())*(y_limit_maximum-y_limit_minimum)) / (action_potential.max() - action_potential.min())

    fig1 = plt.figure(1)

    ax = plt.subplot(2,1,1)
    plt.plot(z, normalized_current_distribution)
    plt.ylabel('Current Distribution')
    plt.yticks([y_limit_minimum, 0 , y_limit_maximum])
    ax.xaxis.set_major_formatter(NullFormatter())
    plt.title('Membrane Current Distribution')

    ax = plt.subplot(2,1,2)
    plt.plot(z, normalized_action_potential)
    plt.ylabel('Action Potential')
    plt.yticks([y_limit_minimum, 0 , y_limit_maximum])
    plt.title('Membrane Action Potential')

    fig1.supxlabel('z, Distance (mm)')

    return plt.show()

  #########################  3  #########################  Get Pole Distance  ##########################    #######################
  def get_tripole_distance(self):
    """Calculates the distance between the poles of the current distribution.

    Args:
      pole_location_index: A list of integers containing the indices of the poles in the current distribution.
      dz: The sampling interval of the current distribution.
      current_distribution

    Returns:
      A list of two floats containing the distances between the poles.
    """
    ...

  ### Default arguments
    self.get_tripole_amplitude()
    pole_location_index = self.pole_location_index
    dz = self.dz
    current_distribution = self.current_distribution

  ### Calculate the distance between the poles
    """a, b represent tripole asymmetry
    a is the distance between pole 1 and pole 2.
    b is the distance between pole 1 and pole 3.
    The following rules of the current sources must hold:
    pole_one + pole_two + pole_three = 0
    pole_two*a + pole_three*b = 0
    """
    
    ## Calculate the cumulative sum of each phase.
    pole_one_sum = np.cumsum(current_distribution[:pole_location_index[1]]) 
    pole_two_sum = np.cumsum(current_distribution[pole_location_index[1]:pole_location_index[2]])
    pole_three_sum = np.cumsum(current_distribution[pole_location_index[2]:])
    
    ## Locate the location index (z-coordinate) at half the cumulative sum of each phase.
    pole_one_location = np.where(pole_one_sum > 0.5 * np.sum(current_distribution[:pole_location_index[1]]))[0]
    pole_two_location = pole_location_index[1] + np.where(pole_two_sum < 0.5 * np.sum(current_distribution[pole_location_index[1]:pole_location_index[2]]))[0]
    pole_three_location = pole_location_index[2] + np.where(pole_three_sum > 0.5 * np.sum(current_distribution[pole_location_index[2]:]))[0]

    ## Calculate the pole positions
    pole_one_position = pole_one_location[0] * dz
    pole_two_position = pole_two_location[0] * dz
    pole_three_position = pole_three_location[0] * dz

    ## Calculate the distance between the poles
    a = pole_two_position - pole_one_position
    b = pole_three_position - pole_one_position
  
    pole_distances = np.array([a, b])
    #print('Pole Distances =', pole_distances)

    return pole_distances
    
  #########################  4  #########################  Simulate Fibre Action Potential  ##########################    #######################
  def simulate_fibre_action_potential(self):
    """Simulates the action potentials recorded by all electrodes for one fibre.

    Args:
      fibre_length
      neuromuscular_junction
      conduction_velocity
      electrode_shift
      number_of_electrodes_z
      inter_electrode_spacing
      radial_conductivity
      ratio_axial_radial_conductivity
      fibre_depth

    Returns:
      A numpy array containing the simulated action potentials for electrodes positioned along the fibre.
    """
    ...

  ### Choose default values for attributes
    fibre_length = self.fibre_length
    neuromuscular_junction = self.initial_neuromuscular_junction
    conduction_velocity = self.conduction_velocity
    electrode_shift = self.electrode_shift
    number_of_electrodes_z = self.number_of_electrodes_z
    number_of_electrodes_x = self.number_of_electrodes_x
    inter_electrode_spacing = self.inter_electrode_spacing
    radial_conductivity = self.radial_conductivity
    ratio_axial_radial_conductivity = self.ratio_axial_radial_conductivity
    extermination_zone_width = self.extermination_zone_width
    innervation_zone_width = self.innervation_zone_width
    fibre_depth = self.fibre_depth
    time_length = self.time_length

  ### Simulation of poles moving along the fibre in time. Create time array.
    delta_time = 0.1 # (10 kHz)
    time_array = np.arange(0, time_length + delta_time, delta_time)  
    self.time_array = time_array

  ### Create the current tripole with initial pole amplitude and positions
    tripole_amplitude = self.get_tripole_amplitude()
    tripole_distance = self.get_tripole_distance()

    a = tripole_distance[0]
    b = tripole_distance[1]
    pole_one = tripole_amplitude[0]
    pole_two = tripole_amplitude[1]
    pole_three = tripole_amplitude[2]
    
    # Array of mirrored tripole amplitudes
    P = np.array([pole_one, pole_two, pole_three, pole_three, pole_two, pole_one]).reshape(-1, 1)
    Pi = np.tile(P,(1,len(time_array)))

  ### Create uniformly distributed tendon ends at the extermination zones of each fibre and uniformly distributed neuromuscular junctions at the innervation zones.
    delta_length = 0.1 # (10 kHz)
    fibre_length_array = np.arange(0, fibre_length + delta_length, delta_length)

  ### Create random variation in the rightmost tendon ends.
    right_tendon_end_variation = fibre_length_array[-1] + (extermination_zone_width/2 * np.random.rand()) * ((np.random.randint(0,2)*2) - 1)

  ### Create variation in the leftmost tendon ends.
    left_tendon_end_variation = fibre_length_array[0] + (extermination_zone_width/2 * np.random.rand()) * ((np.random.randint(0,2)*2) - 1)

  ### Create small variation in the NMJ location.
    neuromuscular_junction = neuromuscular_junction + (innervation_zone_width/2 * np.random.rand()) * ((np.random.randint(0,2)*2) - 1)
  
  ### Force fibrelengths longer than the ordinary defined fibre length to be within the defined max fibre length.
    fibre_length = right_tendon_end_variation - left_tendon_end_variation

  ### Move poles with an initial offset, length of a tripole (b)
    initial_offset = b
    ## Initialise the locations of the poles at action potential initialisation
    location_pole_one = neuromuscular_junction - initial_offset + b # initial location of Pole 1
    location_pole_two = neuromuscular_junction - initial_offset - a + b # initial location of Pole 2
    location_pole_three = neuromuscular_junction - initial_offset # initial location of Pole 3
    location_pole_four = neuromuscular_junction + initial_offset # initial location of Pole 4
    location_pole_five = neuromuscular_junction + initial_offset + a - b # initial location of Pole 5
    location_pole_six = neuromuscular_junction + initial_offset - b # initial location of Pole 6

    ## Array of initial pole locations
    initial_pole_locations = (np.array(np.array([location_pole_one, location_pole_two, location_pole_three, location_pole_four, location_pole_five, location_pole_six]))[np.newaxis]).T

  ### Simulation of poles moving along the fibre in time.
    ## Move poles 1-3 in positive direction, with regards to conduction velocity.
    location_poles_right = np.array(initial_pole_locations[0:3] + conduction_velocity * time_array)
    # Set poles out of bounds to neuromuscular_junction
    location_poles_right[location_poles_right < neuromuscular_junction] = neuromuscular_junction

    ## Move poles 4-6 in negative direction, with regards to conduction velocity.
    location_poles_left = np.array(initial_pole_locations[3:6] - conduction_velocity * time_array)
    # Set poles out of bounds to neuromuscular_junction
    location_poles_left[location_poles_left > neuromuscular_junction] = neuromuscular_junction

    # Combine poles of both directions in one matrix
    location_poles_all = np.vstack((location_poles_right, location_poles_left))

  ### Find and replace out of bounds locations at muscle fibre ends, both postive and negative end. Repalce out of bounds locations with fibre bound.
    location_poles_all[location_poles_all < left_tendon_end_variation] = left_tendon_end_variation
    location_poles_all[location_poles_all > fibre_length] = fibre_length

  ### Defining the detection system
    # Create a vector for the locations of number of electrodes along the fibre.
    electrode_locations_z = np.zeros(number_of_electrodes_z)
    # Create a spacing vector for the electrode locations, with number of electrodes.
    for i in range(number_of_electrodes_z):
        electrode_locations_z[i] = inter_electrode_spacing * i - ((inter_electrode_spacing * (number_of_electrodes_z - 1)) / 2)
    electrode_locations_z = electrode_locations_z + electrode_shift  # Add interelectrode shift.

    electrode_locations_x = np.zeros(number_of_electrodes_x)
    # Create a spacing vector for the electrode locations, with number of electrodes.
    for j in range(number_of_electrodes_x):
        electrode_locations_x[j] = inter_electrode_spacing * j - ((inter_electrode_spacing * (number_of_electrodes_x - 1)) / 2)

  ### Create the single fibre action potential
    single_fibre_action_potential = np.zeros((number_of_electrodes_z, number_of_electrodes_x, len(time_array)))
    # Finding the potentials observed at each electrode.
    #print('self.fibre_depth in single fibre:', self.fibre_depth)
    for z in range(number_of_electrodes_z):
        for x in range(number_of_electrodes_x):
          single_fibre_action_potential[z, x, :] = (1/(2*np.pi * radial_conductivity) * np.sum((Pi / (np.sqrt(((electrode_locations_x[x])**2 + fibre_depth**2) * ratio_axial_radial_conductivity + (electrode_locations_z[z] - location_poles_all)**2))), axis=0))
    single_fibre_action_potential = np.reshape(single_fibre_action_potential, (len(electrode_locations_z)*len(electrode_locations_x), len(time_array)))

    #print('Single Fibre Action Potential', single_fibre_action_potential)
    return single_fibre_action_potential

  #########################  5  #########################  Plot Fibre Action Potential  ##########################    #######################
  def plot_fibre_action_potential(self):
    """Plots the single fibre action potentials computed in simulate_fibre_action_potential().

    Args:

      single_fibre_action_potential
      time_array
      y_limit_minimum
      y_limit_maximum
      number_of_electrodes_z

    Returns:

      A plot of the single fibre action potential for all electrodes.

    """
    ...
  ### Default arguments:
    single_fibre_action_potential = self.simulate_fibre_action_potential()
    time_array = self.time_array
    y_limit_minimum = self.y_limit_minimum
    y_limit_maximum = self.y_limit_maximum
    number_of_electrodes_z = self.number_of_electrodes_z
    number_of_electrodes_x = self.number_of_electrodes_x

  ### Plot the normalized single fibre action potential
    normalized_single_fibre_action_potential = - single_fibre_action_potential
    normalized_single_fibre_action_potential = (normalized_single_fibre_action_potential - normalized_single_fibre_action_potential.mean()) / (normalized_single_fibre_action_potential.max() - normalized_single_fibre_action_potential.min())

    # The single fibre action potentials recorded by the electrodes positioned along the length of the fibre.
    array_size = np.arange(1, (number_of_electrodes_z*number_of_electrodes_x)+1, 1)
    zeros_array = np.zeros(len(array_size))
    array_size_x = np.arange(0, number_of_electrodes_z*number_of_electrodes_x, number_of_electrodes_x)
    array_size_x = np.append(array_size_x,zeros_array)

    fig2 = plt.figure(2)
    for i in range(len(array_size)):
        ax = plt.subplot(number_of_electrodes_z , number_of_electrodes_x, array_size[i])
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        ax.grid(which = 'both', ls = 'dashed')
        plt.plot(time_array, normalized_single_fibre_action_potential[i, :])
        plt.xlim(time_array[0], time_array[-1] - 1)
        if i < len(array_size) - number_of_electrodes_x:
          ax.xaxis.set_major_formatter(NullFormatter())
        plt.ylim(y_limit_minimum,y_limit_maximum)
        ax.yaxis.set_major_formatter(NullFormatter())
        if i == 0:
          ax.set_ylabel(1, rotation = 0, ha = 'center', va = 'center', fontsize=15)
        for j in range(i):
          if i == array_size_x[j]:
            ax.set_ylabel(array_size[j], rotation = 0, ha = 'center', va = 'center', fontsize=15)
          elif number_of_electrodes_x == 1:
            ax.set_ylabel(array_size[j]+1, rotation = 0, ha = 'center', va = 'center', fontsize = 15)

    plt.suptitle('Single Fibre Action Potential', fontsize=20)
    fig2.supxlabel('Time (ms)\n Electrodes in the x direction, i.e. vertically across the fiber')
    fig2.supylabel("Electrodes in the z direction, i.e. along the fiber")

    return plt.show()
  
  #########################  6  #########################  Simulate Motor Unit Action Potential  ##########################    #######################
  def simulate_motor_unit(self):
    """Simulates the summed action potentials from a number of fibres in a motor unit, recorded by all electrodes.

    Args:
      single_fibre_action_potential
      time_array
      number_of_electrodes_z
      number_of_fibres

    Returns:
      A numpy array containing the simulated motor unit action potentials for all electrodes positioned.
    """
    ...

  ### Default arguments:
    self.simulate_fibre_action_potential()
    time_array = self.time_array
    motor_unit_radius = self.motor_unit_radius
    number_of_fibres = self.number_of_fibres
    fibre_depth = self.fibre_depth

  ### Generating the number of fibres within the motor unit cylinder.
    motor_unit_radius_array = (np.array(motor_unit_radius * np.ones(number_of_fibres))[np.newaxis]).T
 
    # Define the number of theta angles around the motor unit.
    theta_angle = np.linspace(0, 2 * np.pi, num=number_of_fibres)

    # Define the numbers between 0 and 1, and create array for single fibres inside the cylinder. 
    fibre_in_cylinder = np.linspace(0, 1, num=number_of_fibres)
    
    # Calculate different distances from surface electrode down through the motor unit with regards to the various angles and positions inside the muscle, using the law of cosines: c = sqrt(a^2 + b^2 -2abcos(theta)).
    fibre_depth_variation = (np.array(np.sqrt(fibre_depth**2 + (np.multiply(motor_unit_radius_array, fibre_in_cylinder))**2 - (2*fibre_depth*motor_unit_radius_array * fibre_in_cylinder) * np.cos(theta_angle)))[np.newaxis]).T
    if number_of_fibres != 1:
      fibre_depth_variation = fibre_depth_variation[:, 1]

  ### Simulate the total action potential of a motor unit for the defined number of single fibres.
    motor_unit_matrix = None
    for i in range(number_of_fibres):
        self.fibre_depth = fibre_depth_variation[i]
        single_fibre = self.simulate_fibre_action_potential()
    
        # Generate matrix for motor unit and add each fibre.
        if i == 0:
            motor_unit_matrix = single_fibre
        else:
            motor_unit_matrix = motor_unit_matrix + single_fibre
    
    motor_unit_matrix = np.vstack((time_array, motor_unit_matrix))
    motor_unit_matrix = motor_unit_matrix[1:]

    #print('Motor Unit Action Potential', motor_unit_matrix)
    return motor_unit_matrix
  
  #########################  7  #########################  Plot Motor Unit Action Potential  ##########################    #######################
  def plot_motor_unit(self):
    """Plots the motor unit action potential from the summed number of single fibres computed in simulate_motor_unit().

    Args:
      motor_unit_matrix
      time_array
      y_limit_minimum
      y_limit_maximum
      number_of_electrodes_z

    Returns:
      A plot of the motor unit action potential for all electrodes.
    """
    ...
  ### Default arguments:
    motor_unit = self.simulate_motor_unit()
    time_array = self.time_array
    y_limit_minimum = self.y_limit_minimum
    y_limit_maximum = self.y_limit_maximum
    number_of_electrodes_z = self.number_of_electrodes_z
    number_of_electrodes_x = self.number_of_electrodes_x

  ### Plot the normalized motor unit action potential
    normalized_motor_unit = - motor_unit
    normalized_motor_unit = (normalized_motor_unit - normalized_motor_unit.mean()) / (normalized_motor_unit.max() - normalized_motor_unit.min())

    # The single fibre action potentials recorded by the electrodes positioned along the length of the fibre.
    array_size = np.arange(1, (number_of_electrodes_z*number_of_electrodes_x)+1, 1)
    zeros_array = np.zeros(len(array_size))
    array_size_x = np.arange(0, number_of_electrodes_z*number_of_electrodes_x, number_of_electrodes_x)
    array_size_x = np.append(array_size_x,zeros_array)

    fig3 = plt.figure(3)
    for i in range(len(array_size)):
      ax = plt.subplot(number_of_electrodes_z , number_of_electrodes_x, array_size[i])
      plt.subplots_adjust(wspace=0.0, hspace=0.0)
      ax.grid(which = 'both', ls = 'dashed')
      plt.plot(time_array, normalized_motor_unit[i, :])
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
    plt.suptitle('Motor Unit Action Potential', fontsize = 20)
    if number_of_electrodes_x > 1:
      fig3.supxlabel('Time (ms)\n Electrodes in the x direction, i.e. vertically across the fiber')
    else:
      fig3.supxlabel('Time (ms)')
    if number_of_electrodes_z > 1:
      fig3.supylabel('Motor Unit Action Potential\n Electrodes in the z direction, i.e. along the fiber',  ha = 'center', va = 'center')
    else:
      fig3.supylabel('Motor Unit Action Potential', ha = 'center', va = 'center')

    return plt.show()
  
#  #########################    #########################               ##########################    #######################
                                                          # THE END #
#  #########################    #########################               ##########################    #######################

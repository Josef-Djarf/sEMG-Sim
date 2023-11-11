import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
plt.rcParams['font.family'] = 'Times New Roman'

class SaveData:
  """A class that saves and plots data.

  Methods:
    save_output_data(): Saves the data of the any signal to a file.
    open_and_load_saved_data(): Opens the already saved data from a file .npy for usage in Python.
    save_single_fibre_action_potential(): Saves the action potential of a single muscle fibre in the motor unit to a file.
    plot_saved_motor_unit(): Plots the saved file for action potential of the motor unit with 10 electrodes in the z-direction and 5 electrodes in the x-direction using the default number of electrodes as in the saved simulated file.
    plot_saved_surface_emg_array():Plots the saved file for surface EMG array by changing the electrodes in the z-direction to 1 and the electrodes in the x-direction to 1 using the default number of electrodes as in the  saved simulated file.

  Attributes:
    For save_output_data:
      filename: The filename to save the output data to.
      output_data: The output data to be saved.

    For open_and_load_saved_data:
      filename: The name of the file containing the saved .npy data.

    For plotting mothods:
      y_limit_minimum: Minimum value of plot y-axis.
      y_limit_minimum: Maximum value of plot y-axis.
      number_of_electrodes_z: Number of elecrodes in the array, in the direction of the fibre.
      number_of_electrodes_x: Number of electrodes in the array across the fiber.
      time_length:
      delta_time:
      simulation_time: Total simulation time in seconds.
      sampling_rate: Sample rate (10 kHz).
        
  """
  ...

  def __init__(self):
    """Initializes a new Motor Unit object.

    """
    ...

    self.y_limit_minimum:int = -1
    self.y_limit_maximum:int = 1
    self.number_of_electrodes_z:int = 10
    self.number_of_electrodes_x:int = 5

    self.time_length:int = 35 
    self.delta_time:float = 0.1 # (10 kHz)
    self.simulation_time:int = 30
    self.sampling_rate:int = 10000

  #########################    #########################  Svae Output Data  ##########################    #######################
  def save_output_data(self, output_data, filename):
    """Saves the output data to a file.

    Args:
        output_data: The output data to be saved.
        filename: The filename to save the output data to.
    """
    ...

    np.save(filename, output_data) # , allow_pickle = True

    # Example usage:

    #output_data = SaveData().save_output_data(output_data, 'filename')

    # Use the code to save data in your folder here.

  #########################    #########################  Open and Load Svaed Data  ##########################    #######################
  def open_and_load_saved_data(self, filename):
    """Opens and loads a saved .npy data from a file.

    Args:
    filename: The name of the file containing the saved .npy data.

    Returns:
    A Python object containing the loaded data.
    """
    ...

    data = np.load(filename) # , allow_pickle = True

    return data

    # Example usage:

    #data = SaveData().open_and_load_saved_data('saved_data.npy')

    # Use the loaded data in your code here.

  #########################    #########################  Plot Saved Motr Unit  ##########################    #######################
  def plot_saved_motor_unit(self, motor_unit_array):
    """Plots the motor unit action potential from the summed number of single fibres computed in simulate_motor_unit().

    Args:
      

    Returns:
      
    """
    ...
  ### Default arguments:
    time_length = self.time_length
    delta_time = self.delta_time
    y_limit_minimum = self.y_limit_minimum
    y_limit_maximum = self.y_limit_maximum
    number_of_electrodes_z = self.number_of_electrodes_z
    number_of_electrodes_x = self.number_of_electrodes_x

    time_array = np.arange(0, time_length + delta_time, delta_time)  

  ### Plot the normalized motor unit action potential
    normalized_motor_unit = - motor_unit_array
    normalized_motor_unit = (normalized_motor_unit - normalized_motor_unit.mean()) / (normalized_motor_unit.max() - normalized_motor_unit.min())

    # The single fibre action potentials recorded by the electrodes positioned along the length of the fibre.
    array_size = np.arange(1, (number_of_electrodes_z*number_of_electrodes_x)+1, 1)
    zeros_array = np.zeros(len(array_size))
    array_size_x = np.arange(0, number_of_electrodes_z*number_of_electrodes_x, number_of_electrodes_x)
    array_size_x = np.append(array_size_x,zeros_array)

    fig1 = plt.figure(1)
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
      fig1.supxlabel('Time (ms)\n Electrodes in the x direction, i.e. vertically across the fiber')
    else:
      fig1.supxlabel('Time (ms)')
    if number_of_electrodes_z > 1:
      fig1.supylabel('Motor Unit Action Potential\n Electrodes in the z direction, i.e. along the fiber',  ha = 'center', va = 'center')
    else:
      fig1.supylabel('Motor Unit Action Potential', ha = 'center', va = 'center')

    return plt.show()
  
    # Example usage:

    #plot_mu = SaveData().plot_saved_motor_unit(data)

    # Use the code to save data in your folder here.

  #########################    #########################  Plot Saved Surface Electromyography Array  ##########################    #######################
  def plot_saved_surface_emg_array(self, surface_emg_array):
    """Generates the recruitment and rate coding organization of motor units.

    Arguments:
      firing_times_motor_unit
      time_array

    Returns:
      A plot of the recruitment model for each motor unit.
    """
    ...

  ### Default arguments:
    simulation_time = self.simulation_time
    sampling_rate = self.sampling_rate
    y_limit_minimum = self.y_limit_minimum
    y_limit_maximum = self.y_limit_maximum
    number_of_electrodes_z = self.number_of_electrodes_z
    number_of_electrodes_x = self.number_of_electrodes_x

    time_array = np.linspace(0, simulation_time, simulation_time*sampling_rate) 

    electrode_one_sum = np.zeros((number_of_electrodes_z * number_of_electrodes_x, len(time_array)))
  
    for m, simulation in enumerate(surface_emg_array):
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
    fig2 = plt.figure(2)
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
    plt.suptitle('The Surface Electromyography Signal', fontsize = 20)

    if number_of_electrodes_x > 1:
      fig2.supxlabel('Time (s)\n Electrodes in the x direction, i.e. vertically across the fiber')
    else:
      fig2.supxlabel('Time (s)')
    if number_of_electrodes_z > 1:
      fig2.supylabel('sEMG Signal\n Electrodes in the z direction, i.e. along the fiber', ha = 'center', va = 'center')
    else: 
      fig2.supylabel('sEMG Signal', ha = 'center', va = 'center')

    return plt.show()
  
    # Example usage:

    #s_emg = SaveData()
    #data = s_emg.open_and_load_saved_data('saved_data.npy')
    #semg.number_of_electrodes_z = 1 or 10 or 13
    #semg.number_of_electrodes_x = 1 or 5
    #plot_semg = s_emg.plot_saved_suface_emg_array(data)

#  #########################    #########################               ##########################    #######################
                                                          # THE END #
#  #########################    #########################               ##########################    #######################

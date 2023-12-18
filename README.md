# sEMG-Sim version: 0.1
  Surface Electromyography Simulator - Multiple parameter modelling of synthetic electromyography (EMG) data.

# install

To install, clone the repository and run:

```
pip install -e .
```
in the root folder.

# General Info:
This code is the result of our Master's thesis project. What follows below is some general information about the project and a discription of the project.
Our objective was to implement an sEMG simulation model in Python. The model is based on mathematical models obtained
from scientific literature. The model contains several adjustable parameters including motor unit properties (size, firing rate, etc),
volume conductor, and setup for the recording electrodes. The work has been completed in association with the Neuroengineering group, Departement of Biomedical Engineering, Faculty of Engineering, Lund Universty, Sweden.

# Authors:  Ahmad Alosta & Josef Djärf

# Abstract:
Surface electromyography (sEMG) measures skeletal muscle function by recording muscle activity from the surface of the skin. The technique can be used to diagnose neuromuscular diseases, and as an aid in rehabilitation, biomedical research, and human-computer interaction. A simulation model for sEMG data can assess decomposition algorithms and help develop new diagnostic tools. Such simulation models have previously not been available. We have written open-source code in Python to generate synthetic sEMG data. The code is publicly accessible via GitHub, an online platform for software development. The implemented model has multiple parameters that influence the artificially generated signal. The model was implemented with a bottom-up design, starting at a single muscle fibre and ending with the sEMG signal generated from up to hundreds of active motor units. The simulated signal can be recorded in potentially dozens of selectively positioned surface electrodes. The model’s foundation is mathematical equations found throughout the scientific literature surrounding motor control and biological signalling, e.g., action potential propagation, membrane current distribution, and motor unit recruitment. We assert that the model incorporates the most significant features for generating sEMG data. The synthetically generated data was decomposed to study the simulated motor unit action potentials. The presented model can be used as ground truth to assess the performance of decomposition algorithms for sEMG. The analysis of sEMG signals can provide valuable insights into muscle activity, contributing to our understanding of motor control and aiding the development of prosthetics and assistive technologies.

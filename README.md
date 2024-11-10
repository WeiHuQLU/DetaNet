# DetaNet
A geometric deep learning model for predicting molecular tensorial properties and selected spectra with high accurately and efficiency
All codes developed in the article 'A deep learning model for predicting selected organic molecular spectra' are shown in the present folder:
'datanet_model/detanet.py' is the main program for DetaNet;
'datanet_model/modules' lists all the submodules of DetaNet;
'trained_param' stores all parameters after DetaNet training.
'datanet_model/constant.py' contains all parameters to simulate the spectra, such as temperature, speed of light, Boltsmann and Planck constant, etc.;
'datanet_model/spectra_simulator.py' is the code to simulate the molecular spectra;
'detanet_model/metrics.py' contains metric functions including the loss function and accuracy function;
'example_calculate_properties_and_spectra.ipynb' is an example to predict the IR, Raman, UV-Vis and NMR spectra for phenol molecule using DetaNet.
'example_load_dataset_and_training_model.ipynb' is an example to training model, load and save parameters

# Required Package:
pytorch > 1.10.0    
torch-geometric > 2.0.2   
torch-scatter > 2.0.9   
torch-cluster > 1.5.9   
torch-sparse > 0.6.12   
torch-spline-conv > 1.2.1   
e3nn == 0.4.4    
numpy > 1.20.1    
matplotlib > 3.3.4    

# qm9s dataset
qm9s dataset can downloaded at https://figshare.com/articles/dataset/QM9S_dataset/24235333

# origin article
https://www.nature.com/articles/s43588-023-00550-y

# cite this work
Zou, Z., Zhang, Y., Liang, L. et al. A deep learning model for predicting selected organic molecular spectra. Nat Comput Sci 3, 957–964 (2023). https://doi.org/10.1038/s43588-023-00550-y

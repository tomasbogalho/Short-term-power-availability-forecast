# Short term power availability forecast
Repository containing the code for :
"Short term power availability forecast for electric vehicle charging hubs using Machine Learning techniques" 
(Master Thesis in Electrical and Computer Engineering - Técnico Lisboa). 

# Author:
Tomás Maria Reis Bogalho - tomas.bogalho@tecnico.ulisboa.pt

# Abstract:
Power consumption forecasting is a field of research that has been showing a great impact on the sustainability of buildings, and is labelled as a time series forecasting problem. In this sort of challenge, data-driven models are commonly applied. The Energias de Portugal (EDP) building located in Lisbon, Portugal, is equipped with a Photovoltaic (PV) system, and comprises an Electric Vehicle Central Charging System (EVCS), capable of adjusting the power used according to the overall available power in the building. It could be possible to optimize the EVCS operation by also feeding it with forecasts of the short term available power. To meet this requirement, a system consisting of Machine Learning (ML) techniques was developed and both meteorological and energetic indicators were used. The solution was accomplished through the implementation and comparison of three distinct architectures: Vanilla Recurrent Neural Network (RNN), Encoder-Decoder and One Dimensional Convolutional Neural Network (1D CNN)-Encoder-Decoder. Additionally, it was introduced Monte Carlo Dropout (MCD), allowing a probabilistic interpretation of the results obtained. It was verified that the Vanilla approach outperformed the remaining architectures, having obtained an average Root Mean Squared Error (RMSE) of 28.95 kW for the deterministic model, and 30.06 kW for the probabilistic model, which showed an improvement over the Naive Standard Approach. The work carried out, not only ensures the optimization of EVCS power usage, but also contributes to the sustainability of the building at a global scale. 

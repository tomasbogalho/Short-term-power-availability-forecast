# Short term power availability forecast
Repository containing the code for :
"Short term power availability forecast for EV charging hubs using Machine Learning techniques" 
(Master Thesis in Electrical and Computer Engineering - Técnico Lisboa). 

# Author:
Tomás Maria Reis Bogalho - tomas.bogalho@tecnico.ulisboa.pt

# Abstract:
Power consumption forecasting is a field of research that has had a great impact on building sustainability. It is labelled as a time series forecasting problem, where data-driven models are typically employed. The \ac{EDP} building located in Lisbon, Portugal, is equipped with a \ac{PV} system, and comprises an \ac{EVCS}, capable of adjusting the power used according to the overall available power in the building. It could be possible to optimize the \ac{EVCS} operation by also feeding it with forecasts of the near future available power. To meet this requirement, a system consisting of \ac{ML} techniques was developed and both meteorological and energetic indicators were used. The solution was accomplished through the implementation and comparison of three distinct architectures: Vanilla \ac{RNN}, Encoder-Decoder and \ac{1D CNN}-Encoder-Decoder. Additionally, it was introduced \ac{MCD}, allowing a probabilistic interpretation of the results obtained. It was verified that the Vanilla approach outperformed the remaining architectures, having obtained an average \ac{RMSE} of 28.95 kW for the deterministic model, and 30.06 kW for the probabilistic model, which showed an improvement over the Naive Standard Approach. The work developed directly contributes to the optimization of energy usage of the \ac{EVCS}, thus improving the sustainability of the building at a global scale.

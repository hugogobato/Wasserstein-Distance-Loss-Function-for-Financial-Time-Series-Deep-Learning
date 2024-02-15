# Wasserstein-Distance-Loss-Function-for-Time-Series
This software automatizes the use of the proposed loss function in https://doi.org/10.1016/j.dajour.2023.100369

# Content
The Python Code named as WD_Loss_Function.py is a PyTorch implementation of the proposed loss function in Souto and Moradi (2024). It contains one class ('MSE_2DWD') for the loss function, one supplementary class ('BasePointLoss'), and two supplementary functions ('_divide_no_nan' and '_weighted_mean'). While the class 'MSE_2DWD' contains the loss function as proposed by Souto and Moradi (2024), the class 'BasePointLoss' transforms the outputs of the neural network models used in the Python library neuralforecast of Nixtla to be used for the proposed loss function. Additionally, the functions '_divide_no_nan' and '_weighted_mean' respectivelly ensure that there is no division by missing values, zeros, or infinity, and that the mean of losses per datapoint is properly estimated.

# Proposed Loss Function Description
The proposed loss function is defined as $L^{*}(y,\hat{y})=\alpha L(y,\hat{y}) + \beta WD_{d}(PH_{y},PH_{\hat{y}})$. where, $L(y,\hat{y})$ is an arbitrary loss function (in the code of this repository, this loss function would be the standard Mean Square Error (MSE), $WD_{d}(PH_{y},PH_{\hat{y}})$ is the $d$-th dimensional Wasserstein Distance (WD) between the Persistent Homology (PH) graphs of the real datapoints ($y$) and the predicted datapoints (̂$\hat{y}$), and $\alpha$ and $\beta$ are importance weights, with $\alpha + \beta = 1$. The introduction of the $d$-th dimensional WD into the loss function theoretically and empirically (as shown in Souto and Moradi (2024)) empowers the neural network model to capture and utilize topological features inherent in the data for forecasting purposes.

# References
Souto, H. G., & Moradi, A. (2024). A novel loss function for neural network models exploring stock realized volatility using Wasserstein Distance. Decision Analytics Journal, 10, 100369. https://doi.org/10.1016/j.dajour.2023.100369

# Authors

@authors: Hugo Gobato Souto* and Amir Moradi**

*International School of Business at HAN University of Applied Sciences, Ruitenberglaan 31, 
6826 CC Arnhem, the Netherlands; hugo.gobatosouto@han.nl; https://orcid.org/0000-0002-7039-0572

Contact author.
**International School of Business at HAN University of Applied Sciences, Ruitenberglaan 31, 
6826 CC Arnhem, the Netherlands; amir.moradi@han.nl; https://orcid.org/0000-0003-1169-7192.

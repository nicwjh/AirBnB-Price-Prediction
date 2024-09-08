### Project Overview

This econometrics project explores the utilization of supervised machine learning methods for price prediction of AirBnB rental listings. The motive of this study is to attempt to build a price prediction model that can prognosticate the price of an AirBnB listing with maximum accuracy. Additionally, we conduct statistical inference on feature significance through a process of robust feature selection.  


### Data
Experiments are run on the AirBnB listings in Major U.S. Cities Dataset sourced from the Kaggle repository [here](https://www.kaggle.com/datasets/rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml). 

### Methods

As a high-level overview of our methodology, the data is first read and preprocessed. After data preprocessing, for models that need to be tuned, we conduct cross-validation on the training data to choose the tuning parameters for our methods. We then use a validation set approach to assess predictive performance on the held-out data for the following regression methods: forward selection, LASSO regression, polynomial regression, and boosting. Finally, results from our experiments are contrasted in a comparative analysis.  

![screenshot](image_1.png)

###  


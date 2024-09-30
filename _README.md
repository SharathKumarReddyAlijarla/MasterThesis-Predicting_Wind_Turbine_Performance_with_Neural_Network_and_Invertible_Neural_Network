# Predicting Wind Turbine Performance with Neural Network and Invertible Neural Network Using Time Series Simulation Data

## Project Overview

This project aims to predict the performance of a wind turbine using two advanced machine learning models: Long Short-Term Memory (LSTM) and Invertible Neural Networks (INN). The project utilizes time series simulation data to forecast critical turbine parameters such as generated power, rotor thrust, and rotor torque. The goal is to enhance the accuracy and reliability of wind turbine performance predictions, which are crucial for optimizing turbine operation and maintenance.

## Project Structure

The project consists of the following main components:

1. **Data Preparation:**
   - Collection and preprocessing of time series data from wind turbine simulations.
   - Data cleaning, filtering, normalization, and splitting into training, validation, and testing sets.

2. **Model Development:**
   - Implementation of LSTM and INN models for performance prediction.
   - Use of the FrEIA library for constructing INN models and implementing bidirectional transformations.

3. **Model Training and Optimization:**
   - Training of the models with various loss functions and optimization techniques.
   - Hyperparameter tuning and performance evaluation using metrics like MSE, MAE, RMSE, and R-squared.

4. **Model Evaluation:**
   - Comparison of the LSTM and INN models' performance across different output parameters and scenarios.
   - Analysis of training and validation loss curves, as well as visual comparisons of actual vs. predicted values.

5. **Results and Discussion:**
   - In-depth analysis of model results, strengths, and limitations.
   - Discussion of the challenges encountered with both LSTM and INN models.

## Files

- `Thesis_Report_Alijarla.pdf`: Final thesis report providing detailed information on the project, methodologies, results, and discussions.
- `INN_Model.py`: Python script implementing the Invertible Neural Network (INN) using the FrEIA library.
- `LSTM_Model.py`: Python script implementing the Long Short-Term Memory (LSTM) network for performance prediction.

## Dependencies

This project relies on the following libraries and packages:

- `numpy`: Numerical computations.
- `pandas`: Data manipulation and analysis.
- `torch`: PyTorch library for constructing neural networks.
- `FrEIA`: Framework for building invertible neural networks.
- `matplotlib`: Visualization of results.
- `scikit-learn`: Additional data processing and evaluation metrics.

To install all required dependencies, use:

```bash
pip install -r requirements.txt

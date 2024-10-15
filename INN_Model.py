import os
import pandas as pd
import concurrent.futures
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Define the subnet constructor for GLOWCouplingBlock
# This function creates a sequential neural network block with two linear layers
# and ReLU activation in between. It serves as the subnet used by the GLOWCouplingBlock module.
def subnet_constructor(c_in, c_out, hidden_size):
    return nn.Sequential(
        nn.Linear(c_in, hidden_size),  # First linear layer with input size and hidden size.
        nn.ReLU(),                     # ReLU activation function.
        nn.Linear(hidden_size, c_out),  # Second linear layer to map hidden size to output size.
    )


# Define the Invertible ResNet architecture using FrEIA framework
# This class defines an invertible neural network using the FrEIA framework,
# with a linear output layer added at the end.
class InvertibleResNetFrEIA(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(InvertibleResNetFrEIA, self).__init__()

        # Create a sequence of nodes to define the invertible network structure
        nodes = [InputNode(input_size, name='input')]  # Start with the input node.

        # Add multiple invertible blocks using GLOWCouplingBlock
        for _ in range(4):  # Create four invertible coupling blocks.
            nodes.append(
                Node(
                    nodes[-1],
                    GLOWCouplingBlock,
                    {'subnet_constructor': lambda c_in, c_out: subnet_constructor(c_in, c_out, hidden_size)}
                )
            )
        nodes.append(OutputNode(nodes[-1], name='output'))  # Define the output node.

        # Create the reversible graph network using the defined nodes
        self.revnet = ReversibleGraphNet(nodes)

        # Define a linear fully connected layer for final mapping to the output size
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Forward pass through the reversible network
        x, _ = self.revnet(x)
        # Linear layer mapping to final outputs
        return self.fc(x)


# Define a custom PyTorch dataset to handle input and output data
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # Input features
        self.y = y  # Corresponding labels

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve input features and labels by index
        return self.X[idx], self.y[idx]


# Function to load data from a single CSV file
# This function reads a CSV file with a specific delimiter and multi-level header
# and returns a DataFrame.
def load_csv(file_path):
    df = pd.read_csv(file_path, delimiter=';', header=[0, 1]).reset_index(drop=True)
    return df


# Function to load data from multiple CSV files using multiprocessing
# This function uses concurrent processing to load multiple CSV files in parallel for efficiency.
def load_data_parallel(csv_files, folder_path):
    # Use ProcessPoolExecutor for parallel file loading
    with concurrent.futures.ProcessPoolExecutor() as executor:
        dfs = list(executor.map(load_csv, [os.path.join(folder_path, file) for file in csv_files]))
    return dfs


# Mean Absolute Percentage Error (MAPE) function
# This function computes MAPE to evaluate model performance in percentage terms.
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


if __name__ == "__main__":
    # Path to the folder containing the CSV files
    folder_path = r'C:\Users\alijarla'

    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Filter out only the CSV files
    csv_files = [file for file in file_list if file.endswith('.csv')]

    # Load data from all CSV files into a single DataFrame using multiprocessing
    dfs = load_data_parallel(csv_files, folder_path)

    # Combine all the DataFrames into one
    data = pd.concat(dfs, axis=0)

    # Combine sensor name and unit in the desired format for the column names
    data.columns = [f"{col[0]}_{col[1]}" for col in data.columns]

    # Data preprocessing: Convert all columns to float values after replacing commas with dots
    for column in data.columns:
        data[column] = data[column].replace({',': '.'}, regex=True).astype(float)

    # Print the first few rows of the DataFrame to verify
    print(data.columns)

    # Normalize input features using StandardScaler
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(
        data[['Time_s', 'PitchAngle[1]_[rad]', 'PitchAngle[2]_[rad]',
              'PitchAngle[3]_[rad]', 'RotorSpeed_[rad/s]',
              'Azimuth_[rad]', 'wind.param.vHub_[m/s]']]
    )

    # Define output variables
    y = data[['GenPower_[W]', 'RotorTorqueAero_[N.m]', 'RotorThrustAero_[N]']]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # Create DataLoader for training
    train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model, loss function, optimizer, and learning rate scheduler
    model = InvertibleResNetFrEIA(input_size=X_train.shape[1], hidden_size=100, output_size=y_train.shape[1])
    criterion = nn.L1Loss()  # Using L1 loss for the regression problem.
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Train the model
    best_loss = float('inf')  # Initialize the best loss value.
    best_mape = float('inf')  # Initialize the best MAPE value.
    early_stopping_counter = 0  # Counter to track early stopping criteria.

    # Loop over epochs
    for epoch in range(100):
        epoch_train_loss = 0.0  # Reset the epoch training loss.

        # Iterate through the training data
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Zero the gradients.
            outputs = model(inputs)  # Forward pass.
            loss = criterion(outputs, targets)  # Compute the loss.
            loss.backward()  # Backpropagation.
            optimizer.step()  # Optimize the parameters.
            epoch_train_loss += loss.item() * inputs.size(0)  # Accumulate loss.

        epoch_train_loss /= len(train_loader.dataset)  # Calculate average training loss.

        # Validation step: Evaluate model on the test set.
        with torch.no_grad():
            model.eval()
            val_loss = criterion(model(X_test_tensor), y_test_tensor).item()  # Compute validation loss.
            val_mape = mean_absolute_percentage_error(y_test_tensor.numpy(), model(X_test_tensor).numpy())  # Compute MAPE.
            scheduler.step(val_loss)  # Adjust learning rate based on validation loss.

            # Save model if validation loss improves.
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')  # Save best model.
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter > 10:  # Check for early stopping condition.
                    print("Early stopping...")
                    break

            model.train()  # Switch back to training mode.

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{100} => Train Loss: {epoch_train_loss}, Val Loss: {val_loss}, Val MAPE: {val_mape}")

        # Track best MAPE value
        if val_mape < best_mape:
            best_mape = val_mape

    # Load the best model for evaluation
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate the model
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)  # Make predictions on the test set.

        # Calculate Mean Squared Error
        mse = mean_squared_error(y_test_tensor.numpy(), y_pred_tensor.numpy())
        print(f"Mean Squared Error: {mse}")

        # Plot predicted vs actual values for the output variables
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test['GenPower_[W]'], y_pred_tensor.numpy()[:, 0])
        plt.xlabel("Actual GenPower")
        plt.ylabel("Predicted GenPower")
        plt.title("Actual vs Predicted GenPower")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.scatter(y_test['RotorTorqueAero_[N.m]'], y_pred_tensor.numpy()[:, 1])
        plt.xlabel("Actual RotorTorqueAero")
        plt.ylabel("Predicted RotorTorqueAero")
        plt.title("Actual vs Predicted RotorTorqueAero")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.scatter(y_test['RotorThrustAero_[N]'], y_pred_tensor.numpy()[:, 2])
        plt.xlabel("Actual RotorThrustAero")
        plt.ylabel("Predicted RotorThrustAero")
        plt.title("Actual vs Predicted RotorThrustAero")
        plt.show()

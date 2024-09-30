import os
import pandas as pd
import concurrent.futures
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo


# Function to load data from a single CSV file.
# This function handles the replacement of commas with dots and converts strings to floats.
def load_csv(file_path):
    df = pd.read_csv(file_path, delimiter=';', header=[0, 1]).reset_index(drop=True)
    df = df.applymap(lambda x: float(str(x).replace(',', '.')) if isinstance(x, str) else x)
    return df


# Function to load data in parallel using ProcessPoolExecutor.
# This allows faster data loading from multiple CSV files concurrently.
def load_data_parallel(csv_files, folder_path):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        dfs = list(executor.map(load_csv, [os.path.join(folder_path, file) for file in csv_files]))
    return dfs


# Custom dataset class to handle input and output data.
# It stores input features (X) and target labels (y) for the dataset.
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define LSTM model architecture with batch normalization.
# The LSTM layer is followed by batch normalization and a fully connected output layer.
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden and cell states with zeros.
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through LSTM layer.
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.batch_norm(out.reshape(-1, self.hidden_size)).reshape(out.size())
        out = self.fc(out)
        return out


if __name__ == "__main__":
    # Define folder paths for data and model saving.
    folder_path = r'C:\Users\alijarla\Desktop\Operational_data'
    save_path = r'C:\Users\alijarla\Desktop\lstm9'
    os.makedirs(save_path, exist_ok=True)
    
    # Get list of all CSV files in the specified folder.
    file_list = os.listdir(folder_path)
    csv_files = [file for file in file_list if file.endswith('.csv')]
    # Load all CSV files in parallel.
    dfs = load_data_parallel(csv_files, folder_path)
    
    # Rename columns to remove multi-level indexing.
    for df in dfs:
        df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]

    # Convert list of DataFrames into a NumPy array for processing.
    data_array = np.array([df.values for df in dfs])

    # Separate input (first 7 columns) and output (remaining columns) features.
    X = np.array([data[:, :7] for data in data_array])
    y = np.array([data[:, 7:] for data in data_array])

    # Split data into training and testing sets without shuffling to preserve time series integrity.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=25)

    # Apply normalization to each input parameter individually using StandardScaler.
    scalers_X = [StandardScaler() for _ in range(X_train.shape[2])]
    X_train_normalized = np.array([scalers_X[i].fit_transform(X_train[:, :, i]) for i in range(X_train.shape[2])]).transpose((1, 2, 0))
    X_test_normalized = np.array([scalers_X[i].transform(X_test[:, :, i]) for i in range(X_test.shape[2])]).transpose((1, 2, 0))

    # Apply normalization to each output parameter individually.
    scalers_y = [StandardScaler() for _ in range(y_train.shape[2])]
    y_train_normalized = np.array([scalers_y[i].fit_transform(y_train[:, :, i]) for i in range(y_train.shape[2])]).transpose((1, 2, 0))
    y_test_normalized = np.array([scalers_y[i].transform(y_test[:, :, i]) for i in range(y_test.shape[2])]).transpose((1, 2, 0))
    
    # Print normalized shapes to verify data.
    print("Normalized shapes")
    print(X_train_normalized.shape, y_train_normalized.shape, X_test_normalized.shape, y_test_normalized.shape)

    # Convert normalized data to PyTorch tensors for model training.
    X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_normalized, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_normalized, dtype=torch.float32)
    
    # Create DataLoader for training data.
    train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Define LSTM model parameters.
    input_size = X_train.shape[2]
    output_size = y_train.shape[2]
    hidden_size = 128  # Size of LSTM hidden layer
    num_layers = 2     # Number of LSTM layers
    dropout = 0.1      # Dropout rate

    # Initialize the LSTM model and move it to the appropriate device (GPU/CPU).
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, dropout=dropout)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Define loss function and optimizer.
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=1000)
    
    # Track best loss and R2 score for early stopping.
    best_loss = float('inf')
    best_r2 = float('-inf')
    early_stopping_counter = 0
    
    # Lists to store training and validation losses for plotting.
    train_losses = []
    val_losses = []
    
    # Training loop for 1000 epochs.
    for epoch in range(1000):
        epoch_train_loss = 0.0
        model.train()  # Set model to training mode.
        
        # Iterate through batches in the DataLoader.
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), targets.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
            # Zero gradients, forward pass, backward pass, and update parameters.
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            # Accumulate batch loss.
            epoch_train_loss += loss.item() * inputs.size(0)
        
        # Calculate average training loss for the epoch.
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation step.
        with torch.no_grad():
            model.eval()  # Set model to evaluation mode.
            val_outputs = model(X_test_tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
            val_loss = criterion(val_outputs, y_test_tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))).item()
            
            # Convert outputs and targets to numpy arrays for evaluation metrics.
            val_outputs_np = val_outputs.cpu().numpy()
            y_test_np = y_test_tensor.cpu().numpy()
            
            # Reshape for R2 score calculation.
            y_test_flat = y_test_np.reshape(-1, y_test_np.shape[-1])
            val_outputs_flat = val_outputs_np.reshape(-1, val_outputs_np.shape[-1])
            
            val_r2 = r2_score(y_test_flat, val_outputs_flat)  # Calculate R2 score.
            val_losses.append(val_loss)  # Append validation loss.
            
            # Save model if validation loss improves.
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                # Early stopping condition: Stop if no improvement for 15 epochs.
                if early_stopping_counter > 15:
                    print("Early stopping...")
                    break
        
        # Print training and validation statistics for the epoch.
        print(f"Epoch {epoch+1}/1000 => Train Loss: {epoch_train_loss:.6f}, Val Loss: {val_loss:.6f}, Val R2: {val_r2:.6f}")
        
        # Track the best R2 score.
        if val_r2 > best_r2:
            best_r2 = val_r2
    # Load the best model for evaluation on test data.
    model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pth')))
    
    with torch.no_grad():
        model.eval()  # Set model to evaluation mode.
        
        # Make predictions on the test data.
        y_pred_tensor = model(X_test_tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        y_pred_np = y_pred_tensor.cpu().numpy()  # Convert predictions to NumPy array.
        y_test_np = y_test_tensor.cpu().numpy()  # Convert test labels to NumPy array.

        # Apply inverse transformation to get predictions and actual values in original scale.
        y_pred_inverse = []
        y_test_inverse = []

        for i in range(y_pred_np.shape[2]):
            y_pred_inverse.append(scalers_y[i].inverse_transform(y_pred_np[:, :, i]))
            y_test_inverse.append(scalers_y[i].inverse_transform(y_test_np[:, :, i]))

        y_pred_inverse = np.array(y_pred_inverse).transpose((1, 2, 0))
        y_test_inverse = np.array(y_test_inverse).transpose((1, 2, 0))

        # Flatten the data for evaluation metrics.
        y_test_flat = y_test_inverse.reshape(-1, y_test_inverse.shape[-1])
        y_pred_flat = y_pred_inverse.reshape(-1, y_pred_inverse.shape[-1])

        # Print first few values for verification.
        print("First actual value:", y_test_flat[0])
        print("First predicted value:", y_pred_flat[0])
        
        # Calculate evaluation metrics: Mean Squared Error, Mean Absolute Error, and R2 Score.
        mse = mean_squared_error(y_test_flat, y_pred_flat)
        mae = mean_absolute_error(y_test_flat, y_pred_flat)
        r2 = r2_score(y_test_flat, y_pred_flat)
        
        print(f"Test MSE: {mse:.6f}, Test MAE: {mae:.6f}, Test R2: {r2:.6f}")

        # Calculate and print Mean Absolute Percentage Error (MAPE).
        mape = np.mean(np.abs((y_test_flat - y_pred_flat) / y_test_flat)) * 100
        print(f"Mean Absolute Percentage Error: {mape:.6f}%")

        # Create interactive Plotly figure for actual vs predicted values of the first output parameter.
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(y_test_flat))), y=y_test_flat[:, 0], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=list(range(len(y_pred_flat))), y=y_pred_flat[:, 0], mode='lines', name='Predicted'))
        fig.update_layout(title='Actual vs Predicted', xaxis_title='Sample Index', yaxis_title='Value')
        pyo.plot(fig)

        # Define output parameter names for visualization and saving purposes.
        output_param_names = [
            'GenPower_[W]', 'RotorTorqueAero_[N.m]', 'RotorThrustAero_[N]', 'Blade1_Mx_root_[N.m]', 'Blade1_My_root_[N.m]',
            'Blade1_Mz_root_[N.m]', 'Blade2_Mx_root_[N.m]', 'Blade2_My_root_[N.m]', 'Blade2_Mz_root_[N.m]', 'Blade3_Mx_root_[N.m]', 
            'Blade3_My_root_[N.m]', 'Blade3_Mz_root_[N.m]', 'Blade1_Fx_root_[N]', 'Blade1_Fy_root_[N]', 'Blade1_Fz_root_[N]', 
            'Blade2_Fx_root_[N]', 'Blade2_Fy_root_[N]', 'Blade2_Fz_root_[N]', 'Blade3_Fx_root_[N]', 'Blade3_Fy_root_[N]', 
            'Blade3_Fz_root_[N]'
        ]

        # Plot actual vs predicted values for each output parameter using Plotly.
        for i, param_name in enumerate(output_param_names):
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_test_flat[:, i], mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(y=y_pred_flat[:, i], mode='lines', name='Predicted'))
            fig.update_layout(
                title=f'Actual vs Predicted for {param_name}',
                xaxis_title='Sample',
                yaxis_title='Value',
                legend=dict(x=0, y=1)
            )
            # Save each figure as an HTML file in the save path.
            pyo.plot(fig, filename=os.path.join(save_path, f'actual_vs_predicted_{param_name}.html'))

        # Plot training loss vs validation loss over epochs.
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=train_losses, mode='lines', name='Training Loss'))
        fig_loss.add_trace(go.Scatter(y=val_losses, mode='lines', name='Validation Loss'))
        fig_loss.update_layout(
            title='Training Loss vs Validation Loss',
            xaxis_title='Epochs',
            yaxis_title='Loss',
            legend=dict(x=0, y=1)
        )
        pyo.plot(fig_loss, filename=os.path.join(save_path, 'training_vs_validation_loss.html'))

    # Load new data for prediction.
    new_data_folder = r'C:\Users\alijarla\Desktop\newdata'
    new_file_list = os.listdir(new_data_folder)
    new_csv_files = [file for file in new_file_list if file.endswith('.csv')]
    
    # Load new data in parallel from CSV files.
    new_dfs = load_data_parallel(new_csv_files, new_data_folder)
    
    # Adjust column names in the new DataFrames.
    for df in new_dfs:
        df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    
    # Convert new DataFrames to NumPy arrays for prediction.
    new_data_array = np.array([df.values for df in new_dfs])
    X_new = np.array([data[:, :7] for data in new_data_array])
    y_new = np.array([data[:, 7:] for data in new_data_array])
    
    # Normalize new input and output data using previously defined scalers.
    X_new_normalized = np.array([scalers_X[i].transform(X_new[:, :, i]) for i in range(X_new.shape[2])]).transpose((1, 2, 0))
    y_new_normalized = np.array([scalers_y[i].transform(y_new[:, :, i]) for i in range(y_new.shape[2])]).transpose((1, 2, 0))
    
    # Convert normalized data to PyTorch tensors for prediction.
    X_new_tensor = torch.tensor(X_new_normalized, dtype=torch.float32)
    y_new_tensor = torch.tensor(y_new_normalized, dtype=torch.float32)
    
    # Make predictions on new data.
    with torch.no_grad():
        model.eval()  # Set model to evaluation mode.
        y_new_pred_tensor = model(X_new_tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        y_new_pred_np = y_new_pred_tensor.cpu().numpy()
        y_new_np = y_new_tensor.cpu().numpy()

        # Apply inverse transformation to get predictions in original scale.
        y_new_pred_inverse = []
        y_new_inverse = []

        for i in range(y_new_pred_np.shape[2]):
            y_new_pred_inverse.append(scalers_y[i].inverse_transform(y_new_pred_np[:, :, i]))
            y_new_inverse.append(scalers_y[i].inverse_transform(y_new_np[:, :, i]))

        y_new_pred_inverse = np.array(y_new_pred_inverse).transpose((1, 2, 0))
        y_new_inverse = np.array(y_new_inverse).transpose((1, 2, 0))

        # Flatten data for saving and plotting.
        y_new_flat = y_new_inverse.reshape(-1, y_new_inverse.shape[-1])
        y_new_pred_flat = y_new_pred_inverse.reshape(-1, y_new_pred_inverse.shape[-1])

    # Save predictions to CSV files and plot actual vs predicted values for new data.
    for i, df in enumerate(new_dfs):
        prediction_df = pd.DataFrame(y_new_pred_inverse[i], columns=output_param_names)
        prediction_df.to_csv(os.path.join(save_path, f'predictions_{new_csv_files[i]}'), index=False, sep=';')
        
        actual_df = pd.DataFrame(y_new_inverse[i], columns=output_param_names)

        # Plot actual vs predicted values for each output parameter in the new data.
        for j, param_name in enumerate(output_param_names):
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=actual_df[param_name], mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(y=prediction_df[param_name], mode='lines', name='Predicted'))
            fig.update_layout(
                title=f'Actual vs Predicted for {param_name} (New Data)',
                xaxis_title='Sample',
                yaxis_title='Value',
                legend=dict(x=0, y=1)
            )
            # Save each figure as an HTML file in the save path.
            pyo.plot(fig, filename=os.path.join(save_path, f'new_data_actual_vs_predicted_{param_name}_{new_csv_files[i]}.html'))

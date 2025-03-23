import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings
import logging

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def cal_SOi(so2):
    """Calculates the individual pollutant index for SO2."""
    si = 0
    if so2 <= 40:
        si = so2 * (50 / 40)
    elif 40 < so2 <= 80:
        si = 50 + (so2 - 40) * (50 / 40)
    elif 80 < so2 <= 380:
        si = 100 + (so2 - 80) * (100 / 300)
    elif 380 < so2 <= 800:
        si = 200 + (so2 - 380) * (100 / 420)
    elif 800 < so2 <= 1600:
        si = 300 + (so2 - 800) * (100 / 800)
    elif so2 > 1600:
        si = 400 + (so2 - 1600) * (100 / 800)
    return si


def cal_Noi(no2):
    """Calculates the individual pollutant index for NO2."""
    ni = 0
    if no2 <= 40:
        ni = no2 * 50 / 40
    elif 40 < no2 <= 80:
        ni = 50 + (no2 - 40) * (50 / 40)
    elif 80 < no2 <= 180:
        ni = 100 + (no2 - 80) * (100 / 100)
    elif 180 < no2 <= 280:
        ni = 200 + (no2 - 180) * (100 / 100)
    elif 280 < no2 <= 400:
        ni = 300 + (no2 - 280) * (100 / 120)
    else:
        ni = 400 + (no2 - 400) * (100 / 120)
    return ni


def cal_RSPMI(rspm):
    """Calculates the individual pollutant index for RSPM."""
    rpi = 0
    if rspm <= 30:
        rpi = rspm * 50 / 30
    elif 30 < rspm <= 60:
        rpi = 50 + (rspm - 30) * 50 / 30
    elif 60 < rspm <= 90:
        rpi = 100 + (rspm - 60) * 100 / 30
    elif 90 < rspm <= 120:
        rpi = 200 + (rspm - 90) * 100 / 30
    elif 120 < rspm <= 250:
        rpi = 300 + (rspm - 120) * (100 / 130)
    else:
        rpi = 400 + (rspm - 250) * (100 / 130)
    return rpi


def cal_SPMi(spm):
    """Calculates the individual pollutant index for SPM."""
    spi = 0
    if spm <= 50:
        spi = spm * 50 / 50
    elif 50 < spm <= 100:
        spi = 50 + (spm - 50) * (50 / 50)
    elif 100 < spm <= 250:
        spi = 100 + (spm - 100) * (100 / 150)
    elif 250 < spm <= 350:
        spi = 200 + (spm - 250) * (100 / 100)
    elif 350 < spm <= 430:
        spi = 300 + (spm - 350) * (100 / 80)
    else:
        spi = 400 + (spm - 430) * (100 / 430)
    return spi


def cal_aqi(si, ni, rspmi, spmi):
    """Calculates the overall Air Quality Index (AQI)."""
    return max(si, ni, rspmi, spmi)


class DataProcessor:
    """Processes the data for AQI calculation and model training."""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        """Loads data from the specified path."""
        logging.info(f"Loading data from {self.data_path}")
        try:
            self.df = pd.read_csv(self.data_path, encoding='unicode_escape')
            logging.info(f"Data loaded successfully with {len(self.df)} rows.")
        except FileNotFoundError as e:
            logging.error(f"Error: {self.data_path} not found. {e}")
            raise

    def prepare_data(self):
        """Preprocesses the data."""
        logging.info("Preparing the data.")
        df = self.df.copy()

        # Drop unnecessary columns
        cols_to_drop = ['agency', 'stn_code', 'sampling_date', 'location_monitoring_station']
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        logging.info(f"Dropped unnecessary columns: {cols_to_drop}")

        # Convert 'date' column to datetime, handle errors
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.set_index('date').sort_index()
            logging.info("Successfully converted 'date' column to datetime and set as index.")
        except KeyError:
            logging.warning("'date' column not found. Time series analysis may not be possible.")
            df['date'] = None  # Setting Date To None

        # Handle missing values
        for col in ['location', 'type']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
                logging.info(f"Filled missing values in column '{col}' with mode.")

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        if df.isnull().any().any():  # Checking for nulls
            logging.warning(f'There are still NULL values in the dataset. Dropping them!')
            df.dropna(inplace=True)

        logging.info("Missing values imputation completed.")

        # Calculate pollutant indices
        df['SOi'] = df['so2'].apply(cal_SOi)
        df['Noi'] = df['no2'].apply(cal_Noi)
        df['Rpi'] = df['rspm'].apply(cal_RSPMI)
        df['SPMi'] = df['spm'].apply(cal_SPMi)
        logging.info("Calculated pollutant indices (SOi, Noi, Rpi, SPMi).")

        # Calculate AQI
        df['AQI'] = df.apply(lambda x: cal_aqi(x['SOi'], x['Noi'], x['Rpi'], x['SPMi']), axis=1)
        logging.info("Calculated Air Quality Index (AQI).")

        self.df = df
        logging.info("Data preparation complete.")
        return df


class BaseModel:
    """Base class for all models."""

    def __init__(self, df, test_size=0.2, random_state=42, seq_length=60):
        self.df = df
        self.test_size = test_size
        self.random_state = random_state
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def prepare_data_for_model(self, features, target):
        """Prepares data for training and testing."""
        logging.info(f"Preparing data for model with features: {features}, target: {target}")

        # Check if 'date' column exists
        if 'date' in self.df.columns and self.df['date'] is not None:  # Changed check date
            logging.info("Using time series data preparation.")
            # Use the resampled data and create sequences
            df_model = self.df[features + [target]].copy()
            df_model = df_model.dropna()
            scaled_data = self.scaler.fit_transform(df_model)

            X, y = self._create_sequences(scaled_data, self.seq_length,
                                          target_index=features.index(target) if target in features else -1)
            # Split into training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size,
                                                                                    shuffle=False)
            logging.info(
                f"Time series data split into training and testing sets. Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")
        else:
            logging.info("Using non-time series data preparation.")
            # If 'date' doesn't exist, assume data isn't time series
            X = self.df[features].values
            y = self.df[target].values
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size,
                                                                                    random_state=self.random_state)
            logging.info(
                f"Non-time series data split into training and testing sets. Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")

    def _create_sequences(self, data, seq_length, target_index):
        """Creates sequences of data for time series analysis."""
        xs = []
        ys = []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            if target_index == -1:
                y = data[i + seq_length, -1]  # if target is not part of the feature
            else:
                y = data[i + seq_length, target_index]  # target feature
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def evaluate(self, predictions):
        """Evaluates the model."""
        logging.info("Evaluating the model.")
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        r2 = r2_score(self.y_test, predictions)
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        logging.info(f"Evaluation results - RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")

        # Plotting the predictions
        plt.figure(figsize=(12, 6))
        plt.plot(self.y_test, label='Actual AQI', linewidth=2)
        plt.plot(predictions, label='Predicted AQI', linewidth=2, linestyle='--')
        plt.title('Actual vs Predicted AQI')
        plt.xlabel('Time')
        plt.ylabel('AQI')
        plt.legend()
        plt.grid(True)
        plt.show()
        logging.info("Plotted results.")


class RandomForestModel(BaseModel):
    """Random Forest model for AQI prediction."""

    def __init__(self, df, test_size=0.2, random_state=42):
        super().__init__(df, test_size, random_state)
        self.model = RandomForestRegressor(n_estimators=100,
                                           random_state=self.random_state)  # Good to have random_state for reproducibility
        self.name = "RandomForest"

    def train(self, features, target):
        """Trains the Random Forest model."""
        logging.info(f"Training {self.name} model.")
        self.prepare_data_for_model(features, target)
        self.model.fit(self.X_train, self.y_train)
        logging.info(f"{self.name} model training complete.")

    def predict(self, X):
        """Predicts using RF"""
        return self.model.predict(X)

    def evaluate_model(self):
        """Evaluates the Random Forest model."""
        logging.info(f"Evaluating {self.name} model.")
        predictions = self.model.predict(self.X_test)
        super().evaluate(predictions)


class LinearRegressionModel(BaseModel):
    """Linear Regression model for AQI prediction."""

    def __init__(self, df, test_size=0.2, random_state=42):
        super().__init__(df, test_size, random_state)
        self.model = LinearRegression()
        self.name = "LinearRegression"

    def train(self, features, target):
        """Trains the Linear Regression model."""
        logging.info(f"Training {self.name} model.")
        self.prepare_data_for_model(features, target)
        self.model.fit(self.X_train, self.y_train)
        logging.info(f"{self.name} model training complete.")

    def predict(self, X):
        """Predicts using RF"""
        return self.model.predict(X)

    def evaluate_model(self):
        """Evaluates the Linear Regression model."""
        logging.info(f"Evaluating {self.name} model.")
        predictions = self.model.predict(self.X_test)
        super().evaluate(predictions)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.name = 'LSTM'  # Added This

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])  # Apply dropout after LSTM
        out = self.fc(out)
        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.name = "GRU"  # Added This

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)

        # GRU forward pass
        out, _ = self.gru(x, h0)

        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])  # Apply dropout after GRU
        out = self.fc(out)
        return out


class TorchBaseModel(BaseModel):
    """Base class for PyTorch models."""

    def __init__(self, df, test_size=0.2, random_state=42, seq_length=60,
                 hidden_size=64, num_layers=2, dropout=0.2, learning_rate=0.001,
                 epochs=50, batch_size=32, device='cpu'):
        super().__init__(df, test_size, random_state, seq_length)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.train_loader, self.test_loader = None, None
        self.history = {'loss': [], 'val_loss': []}
        self.model = None

    def _prepare_torch_data(self):
        """Converts data to PyTorch tensors and creates DataLoaders."""
        logging.info("Preparing data for PyTorch model.")

        # Convert data to PyTorch tensors
        X_train = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(self.y_train, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(self.y_test, dtype=torch.float32).to(self.device)

        # Create TensorDatasets
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)  # No shuffle
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)  # No shuffle
        logging.info("Data prepared and loaded into DataLoaders.")

    def train(self):
        """Trains the model."""
        print(f"Starting training of {self.model.name} model.")
        logging.info(f"Starting training of {self.model.name} model.")
        self.model.to(self.device)  # Move the model to the device

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()  # Set the model to training mode
        for epoch in range(self.epochs):
            total_loss = 0.0
            print(f"Epoch {epoch + 1}/{self.epochs} running")
            logging.info(f"Epoch {epoch + 1}/{self.epochs} running")

            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)  # Move data to the device
                optimizer.zero_grad()  # Clear gradients
                outputs = self.model(batch_X).squeeze()  # Forward pass
                loss = criterion(outputs, batch_y)  # Calculate the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                total_loss += loss.item()
            epoch_loss = total_loss / len(self.train_loader)
            self.history['loss'].append(epoch_loss)

            # Validation (optional)
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Disable gradient calculation for validation
                val_loss = 0.0
                for batch_X, batch_y in self.test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)  # Move data to the device
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            epoch_val_loss = val_loss / len(self.test_loader)
            self.history['val_loss'].append(epoch_val_loss)

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
            logging.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
            self.model.train()  # Return to train mode
        logging.info(f"Training of {self.model.name} complete.")
        print(f"Training of {self.model.name} complete.")

    def predict(self, X):
        """Predicts using the trained model."""
        logging.info("Predicting using the trained model.")
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            X = torch.tensor(X, dtype=torch.float32).to(self.device)  # Move input to the device
            predictions = self.model(X).cpu().numpy()  # Move predictions back to CPU
        return predictions

    def evaluate_model(self):
        """Evaluates the PyTorch model."""
        logging.info("Evaluating the PyTorch model.")
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            predictions = []
            for batch_X, _ in self.test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X).squeeze()
                predictions.extend(outputs.cpu().numpy())

        super().evaluate(np.array(predictions))

        # Plotting the training history
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


class LSTMModel(TorchBaseModel):
    """LSTM model for AQI prediction."""

    def __init__(self, df, input_size, test_size=0.2, random_state=42, seq_length=60,
                 hidden_size=64, num_layers=2, dropout=0.2, learning_rate=0.001,
                 epochs=50, batch_size=32, device='cpu'):
        super().__init__(df, test_size, random_state, seq_length,
                         hidden_size, num_layers, dropout, learning_rate, epochs, batch_size, device)
        self.name = "LSTM"
        output_size = 1

        # Set input size
        self.input_size = input_size
        self.model = LSTM(input_size, hidden_size, num_layers, output_size, dropout)
        self.input_size = input_size

    def train_model(self, features, target):
        """Trains the LSTM model."""
        print(f"Starting the training of {self.name}")
        logging.info(f"Starting the training of {self.name}")
        self.prepare_data_for_model(features, target)
        self._prepare_torch_data()
        super().train()


class GRUModel(TorchBaseModel):
    """GRU model for AQI prediction."""

    def __init__(self, df, input_size, test_size=0.2, random_state=42, seq_length=60,
                 hidden_size=64, num_layers=2, dropout=0.2, learning_rate=0.001,
                 epochs=50, batch_size=32, device='cpu'):
        super().__init__(df, test_size, random_state, seq_length,
                         hidden_size, num_layers, dropout, learning_rate, epochs, batch_size, device)
        self.name = "GRU"
        output_size = 1

        self.model = GRU(input_size, hidden_size, num_layers, output_size, dropout)
        self.input_size = input_size

    def train_model(self, features, target):
        """Trains the GRU model."""
        print(f"Starting the training of {self.name}")
        logging.info(f"Starting the training of {self.name}")
        self.prepare_data_for_model(features, target)
        self._prepare_torch_data()
        super().train()


class AirQualityModel:
    """Main class to manage different models and data processing."""

    def __init__(self, data_path, device='cpu'):
        self.data_path = data_path
        self.processor = DataProcessor(self.data_path)
        self.df = None
        self.device = device

    def load_and_prepare_data(self):
        """Loads and preprocesses the data."""
        logging.info("Loading and preparing data.")
        self.processor.load_data()
        self.df = self.processor.prepare_data()
        logging.info("Loading and preparation of data is done.")

    def train_and_evaluate(self, model_type, features, target, test_size=0.2, random_state=42, seq_length=60, **kwargs):
        """Trains and evaluates the specified model."""
        print(f"Starting train and evaluate on {model_type}.")
        logging.info(f"Starting train and evaluate on {model_type}.")

        if self.df is None:
            logging.error("Data must be loaded and prepared first. Call load_and_prepare_data().")
            print("Error: Data must be loaded and prepared first. Call load_and_prepare_data().")
            return

        # Handle time series models differently if 'date' doesn't exist
        if 'date' not in self.df.columns and (model_type == 'LSTM' or model_type == 'GRU'):
            logging.warning(f" 'date' column missing, skipping {model_type} model.")
            print(f"Warning: 'date' column missing, skipping {model_type} model.")
            return

        # Models using Torch need correct input size for features

        try:
            if model_type == 'RandomForest':
                model = RandomForestModel(self.df, test_size, random_state)
                model.train(features, target)
                model.evaluate_model()

            elif model_type == 'LinearRegression':
                model = LinearRegressionModel(self.df, test_size, random_state)
                model.train(features, target)
                model.evaluate_model()

            elif model_type == 'LSTM':

                input_size = len(features)
                model = LSTMModel(self.df, input_size, test_size, random_state, seq_length=seq_length,
                                  device=self.device, **kwargs)
                model.train_model(features, target)
                model.evaluate_model()

            elif model_type == 'GRU':
                input_size = len(features)
                model = GRUModel(self.df, input_size, test_size, random_state, seq_length=seq_length,
                                 device=self.device, **kwargs)
                model.train_model(features, target)
                model.evaluate_model()

            else:
                logging.error(f"Error: Unknown model type '{model_type}'.")
                print(f"Error: Unknown model type '{model_type}'.")
                return
            print(f"Model {model.name} done")
            logging.info(f"Successfully complete {model.name}.")



        except Exception as e:
            logging.error(f"An error occurred: {e}")
            print(f"An error occurred: {e}")


def main():
    """Main function to execute the air quality prediction."""
    # Use CUDA if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")

    # Parameters
    data_path = 'data/data.csv'  # Changed data_path here
    features = ['SOi', 'Noi', 'Rpi', 'SPMi']
    target = 'AQI'
    test_size = 0.2
    random_state = 42
    seq_length = 60

    # Models hyper parameters
    lstm_hyperparams = {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32
    }

    gru_hyperparams = {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32
    }

    # Create AirQualityModel
    air_model = AirQualityModel(data_path, device=device)
    air_model.load_and_prepare_data()

    # Train and evaluate models
    air_model.train_and_evaluate('RandomForest', features, target, test_size, random_state)
    air_model.train_and_evaluate('LinearRegression', features, target, test_size, random_state)
    air_model.train_and_evaluate('LSTM', features, target, test_size, random_state, seq_length=seq_length,
                                 **lstm_hyperparams)
    air_model.train_and_evaluate('GRU', features, target, test_size, random_state, seq_length=seq_length,
                                 **gru_hyperparams)
    logging.info("The End")


if __name__ == "__main__":
    main()

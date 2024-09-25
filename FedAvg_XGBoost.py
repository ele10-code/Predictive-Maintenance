import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import copy

# Function to split data into local datasets for each client
def split_data_for_clients(X, y, num_clients):
    data_per_client = len(X) // num_clients
    X_clients = []
    y_clients = []
    for i in range(num_clients):
        start = i * data_per_client
        end = (i + 1) * data_per_client if i < num_clients - 1 else len(X)
        X_clients.append(X[start:end])
        y_clients.append(y[start:end])
    return X_clients, y_clients

# Function to perform Federated Averaging (FedAvg)
def fed_avg(local_models):
    global_model = copy.deepcopy(local_models[0])
    global_booster = global_model.get_booster()
    
    # Aggregate the weights
    num_models = len(local_models)
    for i in range(1, num_models):
        booster = local_models[i].get_booster()
        for j in range(len(global_booster.get_dump())):
            # Weighted average
            global_booster.set_score('', 'weight', 
                (global_booster.get_score('', 'weight') + booster.get_score('', 'weight')) / num_models)
    
    global_model._Booster = global_booster
    return global_model

# Dummy data generation (replace with actual data loading)
def generate_dummy_data(num_samples, num_features):
    X = np.random.randn(num_samples, num_features)
    y = np.random.randint(0, 2, num_samples)  # Binary classification
    return X, y

# Example usage of Federated Learning with XGBoost
def federated_learning_with_xgboost(X, y, num_clients, num_rounds, params):
    # Split data among clients
    X_clients, y_clients = split_data_for_clients(X, y, num_clients)
    
    # Initialize global model
    global_model = xgb.XGBClassifier(**params)
    
    for round_num in range(num_rounds):
        local_models = []
        
        for i in range(num_clients):
            # Train local model on each client's data
            local_model = xgb.XGBClassifier(**params)
            local_model.fit(X_clients[i], y_clients[i])
            local_models.append(local_model)
        
        # Aggregate local models to update the global model using FedAvg
        global_model = fed_avg(local_models)
        print(f"Round {round_num + 1}/{num_rounds} - Global model updated")
    
    return global_model

# Generate dummy data (replace with actual data)
num_samples = 10000
num_features = 10
X, y = generate_dummy_data(num_samples, num_features)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Parameters for XGBoost model
params = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Perform Federated Learning
num_clients = 5
num_rounds = 10
global_model = federated_learning_with_xgboost(X_train_scaled, y_train, num_clients, num_rounds, params)

# Evaluate the global model on test data
y_pred_proba = global_model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(f"\nAUC-ROC Score: {roc_auc_score(y_test, y_pred_proba)}")

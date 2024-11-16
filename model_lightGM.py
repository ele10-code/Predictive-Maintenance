import pandas as pd
import numpy as np
import os
import glob
import warnings
import joblib
import logging

# Machine learning imports
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from imblearn.combine import SMOTEENN, SMOTETomek
import lightgbm as lgb

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.getLogger('lightgbm').setLevel(logging.ERROR)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Starting from version 2.2.1")

# Create 'plots' folder if it doesn't exist
plot_folder = 'plots_lightgbm_device_lists'
os.makedirs(plot_folder, exist_ok=True)

def clean_data(df):
    """
    Clean and preprocess the input dataframe.
    """
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna(subset=['value', 'measure_date', 'event_variable'])
    
    # Convert datatypes
    df['measure_date'] = pd.to_datetime(df['measure_date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Remove extreme outliers using IQR method
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['value'] < (Q1 - 3 * IQR)) | (df['value'] > (Q3 + 3 * IQR)))]
    
    return df

def load_and_preprocess_data():
    """
    Load and preprocess all data from CSV files.
    """
    print("Loading and cleaning data...")
    file_list = glob.glob('csv/device_*/measures_device_*.csv')
    if not file_list:
        raise FileNotFoundError("No CSV files found in the specified directory pattern")
        
    data_list = []
    for file in file_list:
        try:
            df = pd.read_csv(file, parse_dates=['measure_date'], low_memory=False)
            df = clean_data(df)
            data_list.append(df)
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    
    if not data_list:
        raise ValueError("No valid data could be loaded from the CSV files")
        
    data = pd.concat(data_list, ignore_index=True)
    
    print("Creating target column...")
    quantiles = data['value'].quantile([0.33, 0.66])
    data['target'] = pd.cut(
        data['value'],
        bins=[-np.inf, quantiles[0.33], quantiles[0.66], np.inf],
        labels=[0, 1, 2]
    )
    data['target'] = data['target'].astype(int)
    
    print("Target variable distribution:")
    print(data['target'].value_counts(normalize=True))
    
    # Feature engineering
    print("Engineering features...")
    data['hour'] = data['measure_date'].dt.hour
    data['day_of_week'] = data['measure_date'].dt.dayofweek
    data['month'] = data['measure_date'].dt.month
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    data['is_business_hours'] = ((data['hour'] >= 9) & 
                                (data['hour'] <= 17) & 
                                ~data['is_weekend']).astype(int)
    
    # Time-based features
    data = data.sort_values(['event_variable', 'measure_date'])
    grouped = data.groupby('event_variable')
    
    # Lag features with error handling
    try:
        data['value_lag_1'] = grouped['value'].shift(1)
        data['value_lag_24'] = grouped['value'].shift(24)
        
        # Rolling statistics
        data['value_rolling_mean'] = grouped['value'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
        data['value_rolling_std'] = grouped['value'].rolling(window=24, min_periods=1).std().reset_index(0, drop=True)
        data['value_rolling_min'] = grouped['value'].rolling(window=24, min_periods=1).min().reset_index(0, drop=True)
        data['value_rolling_max'] = grouped['value'].rolling(window=24, min_periods=1).max().reset_index(0, drop=True)
    except Exception as e:
        print(f"Error in feature engineering: {str(e)}")
        raise
    
    # Drop rows with NaN values
    data = data.dropna()
    
    # Define feature columns
    feature_columns = [
        'event_reference_value', 'hour', 'day_of_week', 'month',
        'is_weekend', 'is_business_hours', 'value_lag_1', 'value_lag_24',
        'value_rolling_mean', 'value_rolling_std', 'value_rolling_min',
        'value_rolling_max'
    ]
    
    return data[feature_columns], data['target'], feature_columns

def detect_outliers(X_train_scaled, outlier_fraction=0.01):
    """
    Detect outliers using multiple methods.
    """
    print("Performing outlier detection...")
    
    outlier_labels = []
    
    # Isolation Forest
    iso_forest = IsolationForest(
        contamination=outlier_fraction, 
        random_state=42, 
        n_jobs=-1,
        max_samples='auto'
    )
    outlier_labels.append(iso_forest.fit_predict(X_train_scaled))

    # Elliptic Envelope
    try:
        ee = EllipticEnvelope(
            contamination=outlier_fraction,
            random_state=42,
            support_fraction=0.7
        )
        outlier_labels.append(ee.fit_predict(X_train_scaled))
    except Exception as e:
        print(f"Elliptic Envelope failed: {str(e)}")
        pass

    # One-Class SVM
    try:
        oc_svm = OneClassSVM(
            nu=outlier_fraction,
            kernel='rbf'
        )
        outlier_labels.append(oc_svm.fit_predict(X_train_scaled))
    except Exception as e:
        print(f"One-Class SVM failed: {str(e)}")
        pass

    # Combine outlier detection results
    outlier_mask = np.any(np.array(outlier_labels) == -1, axis=0)
    
    print(f"Removed {sum(outlier_mask)} outliers from training data.")
    return ~outlier_mask

def plot_results(y_true, y_pred, title_suffix, folder):
    """
    Plot confusion matrix and save it.
    """
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title_suffix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{folder}/confusion_matrix_{title_suffix.replace(" ", "_")}.png')
    plt.close()

def plot_feature_importance(model, feature_columns, title_suffix, folder):
    """
    Plot feature importance and save it.
    """
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Feature Importance - {title_suffix}')
    plt.tight_layout()
    plt.savefig(f'{folder}/feature_importance_{title_suffix.replace(" ", "_")}.png')
    plt.close()
    
    return feature_importance
  
def train_and_evaluate_model(X, y, feature_columns):
    """
    Train and evaluate the LightGBM model with optimized parameters.
    """
    print("Starting model training and evaluation...")
    
    # Split the data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Detect and remove outliers
    outlier_mask = detect_outliers(X_train_scaled)
    X_train_clean = X_train_scaled[outlier_mask]
    y_train_clean = y_train.reset_index(drop=True)[outlier_mask]

    print("Class distribution after outlier removal:")
    print(pd.Series(y_train_clean).value_counts())

    # Initialize resampling methods
    print("\nApplying SMOTEENN for balanced training...")
    try:
        smoteenn = SMOTEENN(random_state=42, n_jobs=-1)
        X_resampled, y_resampled = smoteenn.fit_resample(X_train_clean, y_train_clean)
    except Exception as e:
        print(f"SMOTEENN failed: {str(e)}. Using original data.")
        X_resampled, y_resampled = X_train_clean, y_train_clean

    # Optimized LightGBM parameters
    lgb_param_distributions = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.2],
        'num_leaves': [31, 63],
        'min_child_samples': [50],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'reg_alpha': [0.1, 1],
        'reg_lambda': [0.1, 1],
        'min_split_gain': [0.1],
        'min_gain_to_split': [0.1]
    }

    # Initialize and train LightGBM
    print("\nTraining LightGBM model...")
    lgb_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
        importance_type='gain',
        min_gain_to_split=0.1,
        min_data_in_leaf=50,
        max_cat_threshold=64
    )

    # Hyperparameter tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=lgb_param_distributions,
        n_iter=10,
        cv=cv,
        n_jobs=-1,
        scoring='f1_weighted',
        random_state=42,
        verbose=0
    )

    # Fit model with error handling
    try:
        random_search.fit(X_resampled, y_resampled)
    except Exception as e:
        print(f"Model training failed: {str(e)}")
        raise

    best_model = random_search.best_estimator_
    print("Best parameters:", random_search.best_params_)

    # Evaluate model
    y_val_pred = best_model.predict(X_val_scaled)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    print(f"\nValidation F1 Score: {val_f1:.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred))

    # Plot validation results
    plot_results(y_val, y_val_pred, "LightGBM - Validation", plot_folder)
    feature_importance = plot_feature_importance(best_model, feature_columns, "LightGBM", plot_folder)
    print("\nTop 5 most important features:")
    print(feature_importance.head())

    # Evaluate on test set
    y_test_pred = best_model.predict(X_test_scaled)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"\nTest F1 Score: {test_f1:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # Plot test results
    plot_results(y_test, y_test_pred, "LightGBM - Test", plot_folder)

    return best_model, scaler

  
def analyze_device(device_id, model, scaler, feature_columns, target_class=2):
    """
    Analyze a specific device using the trained model, filtering output for a specific class.
    """
    print(f"\nAnalyzing device {device_id}...")
    device_plot_folder = f'plots_lightgbm_device_{device_id}'
    os.makedirs(device_plot_folder, exist_ok=True)

    try:
        # Load device files
        file_pattern = f'csv/device_{device_id}/measures_device_{device_id}_*.csv'
        device_files = glob.glob(file_pattern)

        if not device_files:
            print(f"No files found for device {device_id}.")
            return None, None

        # Load and preprocess device data
        df_list = []
        for file in device_files:
            df = pd.read_csv(file, parse_dates=['measure_date'], low_memory=False)
            df = clean_data(df)
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True).copy()

        # Feature engineering
        df['hour'] = df['measure_date'].dt.hour
        df['day_of_week'] = df['measure_date'].dt.dayofweek
        df['month'] = df['measure_date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & 
                                  (df['hour'] <= 17) & 
                                  ~df['is_weekend']).astype(int)

        # Time-based features
        df = df.sort_values('measure_date')
        grouped = df.groupby('event_variable')
        
        # Lag features
        df['value_lag_1'] = grouped['value'].shift(1)
        df['value_lag_24'] = grouped['value'].shift(24)
        
        # Rolling statistics
        df['value_rolling_mean'] = grouped['value'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
        df['value_rolling_std'] = grouped['value'].rolling(window=24, min_periods=1).std().reset_index(0, drop=True)
        df['value_rolling_min'] = grouped['value'].rolling(window=24, min_periods=1).min().reset_index(0, drop=True)
        df['value_rolling_max'] = grouped['value'].rolling(window=24, min_periods=1).max().reset_index(0, drop=True)

        # Drop rows with NaN values
        df = df.dropna().copy()

        # Select features and scale
        X_device = df[feature_columns]
        X_device_scaled = scaler.transform(X_device)

        # Make predictions
        y_pred = model.predict(X_device_scaled)
        y_pred_proba = model.predict_proba(X_device_scaled)

        # Add predictions to dataframe
        df.loc[:, 'predicted_class'] = y_pred
        
        # Calculate prediction probabilities
        for i in range(3):
            df.loc[:, f'probability_class_{i}'] = y_pred_proba[:, i]

        # Filtra per la classe specificata
        df_filtered = df[df['predicted_class'] == target_class]

        # Stampa dei risultati predetti per la classe specifica
        class_descriptions = {
            0: "Classe 0: Normale",
            1: "Classe 1: Anomalo",
            2: "Classe 2: Fault"
        }
        description = class_descriptions.get(target_class, "Descrizione non disponibile")

        print(f"\nPredizioni per la classe {target_class} - {description}:")
        # for i, row in df_filtered.iterrows():
        #     print(f"Data: {row['measure_date']}, Valore: {row['value']}, Classe predetta: {row['predicted_class']} - {description}")

        if df_filtered.empty:
            print(f"\nNessuna predizione per la classe {target_class} trovata.")

        # Analysis results
        print("\nPrediction distribution for the filtered class:")
        print(pd.Series(y_pred).value_counts(normalize=True))

        return df_filtered, None
        
    except Exception as e:
        print(f"Error analyzing device {device_id}: {str(e)}")
        return None, None


def analyze_multiple_devices(device_ids, model, scaler, feature_columns):
    """
    Analyze multiple devices and compare their results.
    """
    all_results = {}
    all_dfs = {}
    
    for device_id in device_ids:
        df, results = analyze_device(device_id, model, scaler, feature_columns)
        if df is not None and results is not None:
            all_results[device_id] = results
            all_dfs[device_id] = df
    
    if not all_results:
        print("No devices were successfully analyzed.")
        return None
    
    # Compare devices
    comparison_folder = 'device_comparisons'
    os.makedirs(comparison_folder, exist_ok=True)
    
    try:
        # Compare prediction distributions
        plt.figure(figsize=(12, 6))
        for device_id, results in all_results.items():
            dist = pd.Series(results['prediction_distribution'])
            dist = dist / dist.sum()  # normalize
            plt.bar(dist.index.astype(str) + f'_Device{device_id}', 
                   dist.values, label=f'Device {device_id}')
        plt.title('Prediction Distribution Comparison Across Devices')
        plt.xlabel('Class')
        plt.ylabel('Proportion')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{comparison_folder}/prediction_distribution_comparison.png')
        plt.close()
        
    except Exception as e:
        print(f"Error creating comparison plots: {str(e)}")
    
    return all_results

def main():
    """
    Main execution function.
    """
    try:
        print("Starting predictive maintenance analysis...")
        
        # Load and preprocess data
        X, y, feature_columns = load_and_preprocess_data()
        
        # Train and evaluate model
        best_model, scaler = train_and_evaluate_model(X, y, feature_columns)
        
        # Save the model and scaler
        print("\nSaving model and scaler...")
        model_filename = 'best_lightgbm_model.joblib'
        scaler_filename = 'scaler.joblib'
        joblib.dump(best_model, model_filename)
        joblib.dump(scaler, scaler_filename)
        print(f"Model saved to {model_filename}")
        print(f"Scaler saved to {scaler_filename}")
        
        # Analyze specific devices
        device_ids = [126]  # Add more device IDs as needed
        print("\nAnalyzing specific devices...")
        predicted_class = analyze_device(device_ids, best_model, scaler, feature_columns)
        print(f"The device {device_id} was classified in class: {predicted_class}")
        results = analyze_multiple_devices(device_ids, best_model, scaler, feature_columns)
        print("IL risultato per il device è: ", results)
        
        print("\nAnalysis complete!")
        return results
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        return None

if __name__ == "__main__":
    main()
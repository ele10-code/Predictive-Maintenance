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
  

    plt.tight_layout()
    plt.savefig(f'{plot_folder}/model_performance_iterations.png')
    plt.close()
    
    print("\nModel performance tracking saved.")
    
    # Evaluate on test set after final iteration
    y_test_pred = lgb_model.predict(X_test_scaled)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"\nTest F1 Score: {test_f1:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # Plot test results
    plot_results(y_test, y_test_pred, "LightGBM - Test", plot_folder)

    return lgb_model, scaler

def evaluate_model_iterations(model, X, y, iterations, scaler, feature_columns, folder, device_id):
    """
    Evaluate the model over multiple iterations and plot performance.
    """
    print(f"\nEvaluating model for {iterations} iterations on device {device_id}...")
    f1_scores = []
    
    for i in range(iterations):
        # Split the data into train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i, stratify=y
        )
        
        # Scale the features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Predict and calculate F1 score
        y_pred = model.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)
    
    # Plot the F1 scores over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, iterations + 1), f1_scores, marker='o')
    plt.title(f'F1 Score Over {iterations} Iterations - Device {device_id}')
    plt.xlabel('Iteration')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{folder}/f1_iterations_device_{device_id}.png')
    plt.close()
    
    print(f"Performance plot for device {device_id} saved to {folder}/f1_iterations_device_{device_id}.png")

  
def analyze_device(device_id, model, scaler, feature_columns):
    """
    Analyze a specific device using the trained model and return the most common predicted class.
    """
    print(f"\nAnalyzing device {device_id}...")
    try:
        # Load device files
        file_pattern = f'csv/device_{device_id}/measures_device_{device_id}_*.csv'
        device_files = glob.glob(file_pattern)

        if not device_files:
            print(f"No files found for device {device_id}.")
            return None

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

        # Return the most common predicted class
        most_common_class = pd.Series(y_pred).mode()[0]
        print(f"\nDevice {device_id} was classified in class: {most_common_class}")
        return most_common_class

    except Exception as e:
        print(f"Error analyzing device {device_id}: {str(e)}")
        return None



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
        
        # Analyze specific devices and evaluate iterations
        device_ids = [118, 126, 226, 114]  # Add more device IDs as needed
        for device_id in device_ids:
            predicted_class = analyze_device(device_id, best_model, scaler, feature_columns)
            if predicted_class is not None:
                print(f"The device {device_id} was classified in class: {predicted_class}")
                
                # Evaluate and plot performance over 100 iterations
                evaluate_model_iterations(
                    best_model, X, y, iterations=100, scaler=scaler, 
                    feature_columns=feature_columns, folder=plot_folder, device_id=device_id
                )
            else:
                print(f"Analysis for device {device_id} failed.")
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        return None



if __name__ == "__main__":
    main()
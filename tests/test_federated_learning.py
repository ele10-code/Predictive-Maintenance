import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import xgboost as xgb
import unittest.mock
import json
import sys
import os
import io
from contextlib import redirect_stdout

sys.path.append(os.path.abspath('../'))

from xgboost_model_federated_learning import (
    encrypt_data,
    decrypt_data,
    sanitize_input,
    validate_input,
    apply_data_quality_filters,
    create_base_model,
    train_local_model,
    aggregate_models,
    find_optimal_threshold,
    incremental_train,
    simulate_client_data,
    upload_to_s3,
    download_from_s3,
    lambda_handler,
    local_training_handler,
    global_aggregation_handler,
    S3_BUCKET,
    S3_CLIENT
)

class TestFederatedLearning(unittest.TestCase):
    def setUp(self):
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        self.y = pd.Series(y)
        self.feature_columns = self.X.columns.tolist()

    def run_test_with_output(self, test_method):
        f = io.StringIO()
        with redirect_stdout(f):
            test_method()
        output = f.getvalue()
        print(f"\nOutput for {test_method.__name__}:")
        print(output)

    def test_encrypt_decrypt(self):
        data = {"test": "data"}
        encrypted = encrypt_data(data)
        decrypted = decrypt_data(encrypted)
        self.assertEqual(data, decrypted)
        print(f"Original data: {data}")
        print(f"Encrypted data: {encrypted}")
        print(f"Decrypted data: {decrypted}")

    def test_sanitize_input(self):
        input_data = "test!@#$%^&*()data"
        sanitized = sanitize_input(input_data)
        self.assertEqual(sanitized, "testdata")
        print(f"Original input: {input_data}")
        print(f"Sanitized input: {sanitized}")

    def test_validate_input(self):
        X_valid = self.X.copy()
        X_valid = validate_input(X_valid, self.feature_columns)
        self.assertIsInstance(X_valid, pd.DataFrame)
        self.assertEqual(len(X_valid.columns), len(self.feature_columns))
        print(f"Validated input shape: {X_valid.shape}")

        X_invalid = self.X.drop(columns=['feature_0'])
        with self.assertRaises(ValueError):
            validate_input(X_invalid, self.feature_columns)
        print("ValueError raised for invalid input as expected")

    def test_apply_data_quality_filters(self):
        df = self.X.copy()
        df['outlier'] = 1e10
        df_filtered = apply_data_quality_filters(df, self.feature_columns)
        self.assertLess(len(df_filtered), len(df))
        print(f"Original dataframe shape: {df.shape}")
        print(f"Filtered dataframe shape: {df_filtered.shape}")

    def test_create_base_model(self):
        model = create_base_model()
        self.assertIsInstance(model, xgb.XGBClassifier)
        print(f"Base model created: {model}")

    def test_train_local_model(self):
        base_model = create_base_model()
        local_model = train_local_model(self.X, self.y, base_model)
        self.assertIsInstance(local_model, xgb.XGBClassifier)
        print(f"Local model trained: {local_model}")

    def test_aggregate_models(self):
        models = [create_base_model() for _ in range(3)]
        for model in models:
            model.fit(self.X, self.y)
        aggregated_model = aggregate_models(models)
        self.assertIsInstance(aggregated_model, xgb.XGBClassifier)
        print(f"Aggregated model: {aggregated_model}")

    def test_find_optimal_threshold(self):
        y_true = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.rand(100)
        threshold = find_optimal_threshold(y_true, y_pred_proba)
        self.assertGreater(threshold, 0)
        self.assertLess(threshold, 1)
        print(f"Optimal threshold: {threshold}")

    def test_incremental_train(self):
        model = create_base_model()
        model.fit(self.X[:800], self.y[:800])
        updated_model = incremental_train(model, self.X[800:], self.y[800:])
        self.assertIsInstance(updated_model, xgb.XGBClassifier)
        print(f"Incrementally trained model: {updated_model}")

    def test_simulate_client_data(self):
        X_client, y_client = simulate_client_data(self.X, self.y)
        self.assertLess(len(X_client), len(self.X))
        self.assertEqual(len(X_client), len(y_client))
        print(f"Simulated client data shape: X={X_client.shape}, y={y_client.shape}")

    @unittest.mock.patch('xgboost_model_federated_learning.S3_CLIENT')
    def test_upload_to_s3(self, mock_s3_client):
        mock_s3_client.upload_file.return_value = None
        result = upload_to_s3('test_file.txt', 'test-bucket')
        self.assertTrue(result)
        mock_s3_client.upload_file.assert_called_once_with('test_file.txt', 'test-bucket', 'test_file.txt')
        print(f"Upload to S3 result: {result}")
        print(f"S3 client called with: {mock_s3_client.upload_file.call_args}")

    @unittest.mock.patch('xgboost_model_federated_learning.S3_CLIENT')
    def test_download_from_s3(self, mock_s3_client):
        mock_s3_client.download_file.return_value = None
        result = download_from_s3('test-bucket', 'test_object.txt', 'test_file.txt')
        self.assertTrue(result)
        mock_s3_client.download_file.assert_called_once_with('test-bucket', 'test_object.txt', 'test_file.txt')
        print(f"Download from S3 result: {result}")
        print(f"S3 client called with: {mock_s3_client.download_file.call_args}")

    @unittest.mock.patch('xgboost_model_federated_learning.local_training_handler')
    @unittest.mock.patch('xgboost_model_federated_learning.global_aggregation_handler')
    def test_lambda_handler(self, mock_global, mock_local):
        mock_local.return_value = {'statusCode': 200, 'body': json.dumps('Success')}
        mock_global.return_value = {'statusCode': 200, 'body': json.dumps('Success')}

        event = {'task': 'local_training', 'client_id': 0, 'encrypted_data': ''}
        result = lambda_handler(event, None)
        self.assertEqual(result['statusCode'], 200)
        print(f"Lambda handler result for local training: {result}")

        event = {'task': 'global_aggregation', 'client_ids': [0, 1, 2]}
        result = lambda_handler(event, None)
        self.assertEqual(result['statusCode'], 200)
        print(f"Lambda handler result for global aggregation: {result}")

        event = {'task': 'invalid_task'}
        result = lambda_handler(event, None)
        self.assertEqual(result['statusCode'], 400)
        print(f"Lambda handler result for invalid task: {result}")

    @unittest.mock.patch('xgboost_model_federated_learning.download_from_s3')
    @unittest.mock.patch('xgboost_model_federated_learning.upload_to_s3')
    @unittest.mock.patch('xgboost_model_federated_learning.load')
    @unittest.mock.patch('xgboost_model_federated_learning.dump')
    @unittest.mock.patch('xgboost_model_federated_learning.xgb.XGBClassifier')
    def test_local_training_handler(self, mock_xgb, mock_dump, mock_load, mock_upload, mock_download):
        mock_download.return_value = True
        mock_upload.return_value = True
        mock_load.return_value = create_base_model()
        mock_xgb.return_value.fit.return_value = None
        mock_dump.return_value = None

        event = {
            'client_id': 0,
            'encrypted_data': encrypt_data({'X': self.X.values.tolist(), 'y': self.y.values.tolist()})
        }
        result = local_training_handler(event, None)
        self.assertEqual(result['statusCode'], 200)
        print(f"Local training handler result: {result}")

    @unittest.mock.patch('xgboost_model_federated_learning.download_from_s3')
    @unittest.mock.patch('xgboost_model_federated_learning.upload_to_s3')
    @unittest.mock.patch('xgboost_model_federated_learning.load')
    @unittest.mock.patch('xgboost_model_federated_learning.dump')
    @unittest.mock.patch('xgboost_model_federated_learning.aggregate_models')
    def test_global_aggregation_handler(self, mock_aggregate, mock_dump, mock_load, mock_upload, mock_download):
        mock_download.return_value = True
        mock_upload.return_value = True
        mock_load.return_value = create_base_model()
        mock_aggregate.return_value = create_base_model()
        mock_dump.return_value = None

        event = {'client_ids': [0, 1, 2]}
        result = global_aggregation_handler(event, None)
        self.assertEqual(result['statusCode'], 200)
        print(f"Global aggregation handler result: {result}")

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFederatedLearning)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
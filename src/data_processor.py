import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pyarrow.parquet as pq
from tqdm import tqdm
from config import *
from sklearn.preprocessing import OrdinalEncoder
import logging

class DataProcessor:
    """
    Handles all data loading, preprocessing, and feature engineering for the competition.
    This class is designed to be modular and extensible for future improvements.
    """
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.sensor_stats = None
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
    def load_run_data(self, is_train: bool = True) -> pd.DataFrame:
        """
        Loads and combines all run data files for either training or testing.
        Returns a single DataFrame containing all runs.
        """
        folder = self.data_dir / "train" if is_train else self.data_dir / "test"
        files = list(folder.glob("run_data_*.parquet"))
        if not is_train:
            files = list(folder.glob("run_data.parquet"))
            
        if not files:
            raise FileNotFoundError(f"No run data files found in {folder}")
            
        dfs = []
        for file in tqdm(files, desc="Loading run data"):
            try:
                df = pd.read_parquet(file)
                # Rename columns to match expected format
                df = df.rename(columns={
                    'Tool ID': 'ToolId',
                    'Run Start Time': 'RunStartTime',
                    'Run End Time': 'RunEndTime',
                    'Run ID': 'RunId',
                    'Process Step': 'ProcessStep',
                    'Consumable Life': 'ConsumableLife',
                    'Step ID': 'StepId',
                    'Time Stamp': 'TimeStamp',
                    'Sensor Name': 'SensorName',
                    'Sensor Value': 'SensorValue'
                })
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error loading {file}: {str(e)}")
                continue
                
        if not dfs:
            raise ValueError("No data could be loaded from run data files")
            
        return pd.concat(dfs, ignore_index=True)
    
    def load_incoming_run_data(self, is_train: bool = True) -> pd.DataFrame:
        """
        Loads and combines all incoming run data files for either training or testing.
        Returns a single DataFrame containing all incoming runs.
        """
        folder = self.data_dir / "train" if is_train else self.data_dir / "test"
        files = list(folder.glob("incoming_run_data_*.parquet"))
        if not is_train:
            files = list(folder.glob("incoming_run_data.parquet"))
            
        if not files:
            raise FileNotFoundError(f"No incoming run data files found in {folder}")
            
        dfs = []
        for file in tqdm(files, desc="Loading incoming run data"):
            try:
                df = pd.read_parquet(file)
                # Rename columns to match expected format
                df = df.rename(columns={
                    'Tool ID': 'ToolId',
                    'Run Start Time': 'RunStartTime',
                    'Run End Time': 'RunEndTime',
                    'Run ID': 'RunId',
                    'Process Step': 'ProcessStep',
                    'Consumable Life': 'ConsumableLife',
                    'Step ID': 'StepId',
                    'Time Stamp': 'TimeStamp',
                    'Sensor Name': 'SensorName',
                    'Sensor Value': 'SensorValue'
                })
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error loading {file}: {str(e)}")
                continue
                
        if not dfs:
            raise ValueError("No data could be loaded from incoming run data files")
            
        return pd.concat(dfs, ignore_index=True)
    
    def load_metrology_data(self, is_train: bool = True) -> pd.DataFrame:
        """
        Loads metrology (measurement) data for training or submission.
        Returns a DataFrame with all measurement points.
        """
        try:
            if is_train:
                folder = self.data_dir / "train"
                files = list(folder.glob("metrology_data*.parquet"))
                if not files:
                    raise FileNotFoundError(f"No metrology data files found in {folder}")
                    
                dfs = []
                for file in tqdm(files, desc="Loading metrology data"):
                    df = pd.read_parquet(file)
                    # Rename columns to match expected format
                    df = df.rename(columns={
                        'Tool ID': 'ToolId',
                        'Run ID': 'RunId',
                        'Process Step': 'ProcessStep',
                        'Step ID': 'StepId',
                        'X': 'X',
                        'Y': 'Y',
                        'Measurement': 'Measurement'
                    })
                    dfs.append(df)
                return pd.concat(dfs, ignore_index=True)
            else:
                file = self.data_dir / "submission" / "metrology_data.parquet"
                if not file.exists():
                    raise FileNotFoundError(f"Submission metrology data not found at {file}")
                    
                df = pd.read_parquet(file)
                # Rename columns to match expected format
                df = df.rename(columns={
                    'Tool ID': 'ToolId',
                    'Run ID': 'RunId',
                    'Process Step': 'ProcessStep',
                    'Step ID': 'StepId',
                    'X': 'X',
                    'Y': 'Y',
                    'Measurement': 'Measurement'
                })
                return df
        except Exception as e:
            logging.error(f"Error loading metrology data: {str(e)}")
            raise
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts time-based features from RunStartTime and RunEndTime.
        Adds hour, day, month, quarter, is_weekend, and process duration features.
        """
        df = df.copy()
        
        # Convert to datetime if not already
        for col in ['RunStartTime', 'RunEndTime']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Basic time features
        for col in ['RunStartTime', 'RunEndTime']:
            if col in df.columns:
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_day_of_week'] = df[col].dt.dayofweek
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day_of_month'] = df[col].dt.day
                df[f'{col}_quarter'] = df[col].dt.quarter
                df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Process duration features
        if 'RunStartTime' in df.columns and 'RunEndTime' in df.columns:
            df['process_duration'] = (df['RunEndTime'] - df['RunStartTime']).dt.total_seconds()
            df['process_duration_minutes'] = df['process_duration'] / 60
            df['process_duration_hours'] = df['process_duration'] / 3600
        
        return df
    
    def calculate_sensor_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates global statistics for each sensor for normalization and scaling.
        Adds normalized and min-max scaled sensor values.
        """
        if self.sensor_stats is None:
            # Calculate global sensor statistics for normalization
            self.sensor_stats = df.groupby('SensorName')['SensorValue'].agg([
                'mean', 'std', 'min', 'max'
            ]).reset_index()
            
            # Handle zero std values
            self.sensor_stats['std'] = self.sensor_stats['std'].replace(0, 1)
        
        # Merge with global statistics
        df = df.merge(self.sensor_stats, on='SensorName', how='left')
        
        # Calculate normalized values
        df['SensorValue_normalized'] = (df['SensorValue'] - df['mean']) / df['std']
        df['SensorValue_min_max_scaled'] = (df['SensorValue'] - df['min']) / (df['max'] - df['min'])
        
        return df
    
    def aggregate_sensor_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates sensor data by RunId and SensorName, creating a wide table of features.
        Includes advanced statistics: mean, std, min, max, median, skew, kurtosis, trend, first/last, etc.
        """
        df = self.calculate_sensor_statistics(df)
        
        # Trend (slope) feature
        def calc_trend(x):
            if len(x) < 2:
                return 0
            # Convert datetime to numeric values
            if np.issubdtype(x.dtype, np.datetime64):
                y = pd.to_numeric(x)
            else:
                y = x
            return np.polyfit(np.arange(len(y)), y, 1)[0]
            
        def kurtosis(x):
            return x.kurtosis()
            
        full_agg_dict = {
            'SensorValue': SENSOR_AGG_FEATURES + ['median', 'skew', kurtosis, calc_trend],
            'SensorValue_normalized': SENSOR_AGG_FEATURES,
            'SensorValue_min_max_scaled': SENSOR_AGG_FEATURES,
            'ConsumableLife': ['mean', 'max', 'min', 'std', 'last', 'first'],
            'StepId': ['count', 'nunique'],
            'TimeStamp': ['min', 'max', 'std', calc_trend]
        }
        
        # Only include columns that exist in the DataFrame
        agg_dict = {col: aggs for col, aggs in full_agg_dict.items() if col in df.columns}
        
        try:
            sensor_agg = df.groupby(['RunId', 'SensorName']).agg(agg_dict).reset_index()
            sensor_agg.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in sensor_agg.columns]
            
            # Pivot
            sensor_pivot = sensor_agg.pivot(
                index='RunId', 
                columns='SensorName', 
                values=[col for col in sensor_agg.columns if col != 'RunId' and col != 'SensorName']
            )
            sensor_pivot.columns = [f"{col[0]}_{col[1]}" for col in sensor_pivot.columns]
            return sensor_pivot.reset_index()
        except Exception as e:
            logging.error(f"Error in sensor aggregation: {str(e)}")
            raise

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes all object/categorical columns using ordinal encoding.
        This is simple and robust for tree models.
        """
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            df[cat_cols] = self.encoder.fit_transform(df[cat_cols])
        return df

    def extract_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds spatial features based on X and Y coordinates: distance from center, angle, quadrant.
        """
        df = df.copy()
        
        if 'X' in df.columns and 'Y' in df.columns:
            # Calculate distance from center
            df['distance_from_center'] = np.sqrt(df['X']**2 + df['Y']**2)
            
            # Calculate angle from center
            df['angle_from_center'] = np.arctan2(df['Y'], df['X'])
            
            # Calculate quadrant
            df['quadrant'] = pd.cut(
                df['angle_from_center'], 
                bins=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                labels=[1, 2, 3, 4]
            )
        
        return df
    
    def prepare_features(self, run_data: pd.DataFrame, incoming_data: pd.DataFrame, metrology_data: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline. Merges all features and handles missing values and encoding.
        """
        try:
            # Process each dataset
            run_data = self.extract_time_features(run_data)
            run_agg = self.aggregate_sensor_data(run_data)
            
            incoming_data = self.extract_time_features(incoming_data)
            incoming_agg = self.aggregate_sensor_data(incoming_data)
            
            metrology_data = self.extract_spatial_features(metrology_data)
            
            # Merge features
            features = metrology_data.merge(run_agg, on='RunId', how='left')
            features = features.merge(incoming_agg, on='RunId', how='left', suffixes=('_run', '_incoming'))
            
            # Handle missing values
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].mean())
            
            # Encode categoricals
            features = self.encode_categoricals(features)
            
            return features
        except Exception as e:
            logging.error(f"Error in feature preparation: {str(e)}")
            raise
    
    def prepare_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Loads and prepares the full training set (features and target).
        """
        try:
            run_data = self.load_run_data(is_train=True)
            incoming_data = self.load_incoming_run_data(is_train=True)
            metrology_data = self.load_metrology_data(is_train=True)
            
            features = self.prepare_features(run_data, incoming_data, metrology_data)
            
            if TARGET_COL not in features.columns:
                raise ValueError(f"Target column '{TARGET_COL}' not found in features")
                
            target = features[TARGET_COL]
            features = features.drop(columns=[TARGET_COL])
            
            return features, target
        except Exception as e:
            logging.error(f"Error preparing training data: {str(e)}")
            raise
    
    def prepare_test_data(self) -> pd.DataFrame:
        """
        Loads and prepares the test set (features only).
        """
        try:
            run_data = self.load_run_data(is_train=False)
            incoming_data = self.load_incoming_run_data(is_train=False)
            metrology_data = self.load_metrology_data(is_train=False)
            
            features = self.prepare_features(run_data, incoming_data, metrology_data)
            return features
        except Exception as e:
            logging.error(f"Error preparing test data: {str(e)}")
            raise 
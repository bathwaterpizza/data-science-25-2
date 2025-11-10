#!/usr/bin/env python3
"""
Production Prediction Script for Amazon Sales Forecasting
==========================================================

This script uses the trained and tuned XGBoost model to make predictions
on new product data for the purchased_last_month target variable.

Usage:
    python production_predict.py --input data.csv --output predictions.csv
    python production_predict.py --input data.csv --output predictions.csv --clip

Author: Data Science Team
Date: January 2025
"""

import argparse
import sys
import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ProductionPredictor:
    """
    Production-ready predictor for Amazon product sales forecasting.

    This class handles loading the trained model, preprocessing new data,
    making predictions, and saving results.
    """

    def __init__(
        self,
        model_path: str = "models/xgboost_tuned.pkl",
        scaler_path: str = "scaler.pkl",
        features_path: str = "feature_names.pkl",
    ):
        """
        Initialize the predictor with model and preprocessing artifacts.

        Args:
            model_path: Path to the trained XGBoost model
            scaler_path: Path to the fitted StandardScaler
            features_path: Path to the feature names list
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.features_path = Path(features_path)

        self.model = None
        self.scaler = None
        self.feature_names = None

        logger.info("Initializing ProductionPredictor...")
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model, scaler, and feature names from disk."""
        try:
            # Load model
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"✓ Model loaded from {self.model_path}")

            # Load scaler
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info(f"✓ Scaler loaded from {self.scaler_path}")

            # Load feature names
            if not self.features_path.exists():
                raise FileNotFoundError(
                    f"Feature names file not found: {self.features_path}"
                )

            with open(self.features_path, "rb") as f:
                self.feature_names = pickle.load(f)
            logger.info(f"✓ Feature names loaded ({len(self.feature_names)} features)")

        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data to match the training format.

        This method assumes the input data has already been preprocessed
        through the same pipeline as training (feature engineering, encoding, etc.).

        Args:
            df: Input DataFrame with features

        Returns:
            Preprocessed DataFrame ready for prediction
        """
        logger.info(f"Preprocessing data: {df.shape[0]} rows")

        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}\n"
                f"Expected features: {self.feature_names}"
            )

        # Select and order features to match training
        df_processed = df[self.feature_names].copy()

        # Handle any missing values (fill with 0 as a safe default)
        if df_processed.isnull().any().any():
            logger.warning("Missing values detected, filling with 0")
            df_processed = df_processed.fillna(0)

        logger.info("✓ Data preprocessed successfully")
        return df_processed

    def predict(
        self,
        df: pd.DataFrame,
        clip: bool = False,
        clip_lower_percentile: float = 1.0,
        clip_upper_percentile: float = 99.0,
    ) -> np.ndarray:
        """
        Make predictions on preprocessed data.

        Args:
            df: Preprocessed DataFrame with features
            clip: Whether to clip predictions to reasonable bounds
            clip_lower_percentile: Lower percentile for clipping
            clip_upper_percentile: Upper percentile for clipping

        Returns:
            Array of predictions
        """
        logger.info("Making predictions...")

        # Make predictions using the model (XGBoost doesn't need scaling)
        predictions = self.model.predict(df)

        # Ensure no negative predictions
        predictions = np.maximum(predictions, 0)

        # Apply clipping if requested
        if clip:
            # Use training data bounds (approximated)
            lower_bound = 50  # Approximate 1st percentile from training
            upper_bound = 50000  # Approximate 99th percentile from training

            original_predictions = predictions.copy()
            predictions = np.clip(predictions, lower_bound, upper_bound)

            num_clipped = np.sum(original_predictions != predictions)
            if num_clipped > 0:
                logger.info(
                    f"Clipped {num_clipped} predictions "
                    f"({num_clipped / len(predictions) * 100:.1f}%) "
                    f"to range [{lower_bound}, {upper_bound}]"
                )

        logger.info(f"✓ Generated {len(predictions)} predictions")
        logger.info(
            f"  Prediction stats: min={predictions.min():.2f}, "
            f"max={predictions.max():.2f}, mean={predictions.mean():.2f}, "
            f"median={np.median(predictions):.2f}"
        )

        return predictions

    def predict_from_file(
        self, input_path: str, output_path: str, clip: bool = False
    ) -> pd.DataFrame:
        """
        Load data from file, make predictions, and save results.

        Args:
            input_path: Path to input CSV file
            output_path: Path to output CSV file
            clip: Whether to clip predictions

        Returns:
            DataFrame with predictions
        """
        # Load input data
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(f"Loading input data from {input_path}")
        df_input = pd.read_csv(input_path)
        logger.info(f"✓ Loaded {len(df_input)} rows")

        # Preprocess
        df_processed = self.preprocess_data(df_input)

        # Predict
        predictions = self.predict(df_processed, clip=clip)

        # Create output DataFrame
        df_output = df_input.copy()
        df_output["predicted_purchased_last_month"] = predictions

        # Save results
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_output.to_csv(output_path, index=False)
        logger.info(f"✓ Predictions saved to {output_path}")

        return df_output


def main():
    """Main function to run the prediction script from command line."""
    parser = argparse.ArgumentParser(
        description="Make predictions on Amazon product sales data using trained XGBoost model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python production_predict.py --input new_products.csv --output predictions.csv
  
  # With prediction clipping (recommended)
  python production_predict.py --input data.csv --output results.csv --clip
  
  # Custom model path
  python production_predict.py --input data.csv --output results.csv --model models/custom_model.pkl
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input CSV file with product features",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output CSV file for predictions",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="models/xgboost_tuned.pkl",
        help="Path to trained model file (default: models/xgboost_tuned.pkl)",
    )

    parser.add_argument(
        "--scaler",
        "-s",
        type=str,
        default="scaler.pkl",
        help="Path to fitted scaler file (default: scaler.pkl)",
    )

    parser.add_argument(
        "--features",
        "-f",
        type=str,
        default="feature_names.pkl",
        help="Path to feature names file (default: feature_names.pkl)",
    )

    parser.add_argument(
        "--clip",
        "-c",
        action="store_true",
        help="Apply prediction clipping to prevent extreme outliers",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        # Initialize predictor
        predictor = ProductionPredictor(
            model_path=args.model, scaler_path=args.scaler, features_path=args.features
        )

        # Make predictions
        logger.info("=" * 60)
        logger.info("STARTING PREDICTION PIPELINE")
        logger.info("=" * 60)

        df_results = predictor.predict_from_file(
            input_path=args.input, output_path=args.output, clip=args.clip
        )

        logger.info("=" * 60)
        logger.info("PREDICTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Input file: {args.input}")
        logger.info(f"Output file: {args.output}")
        logger.info(f"Rows processed: {len(df_results)}")
        logger.info(f"Clipping applied: {args.clip}")
        logger.info("=" * 60)

        # Display sample predictions
        logger.info("\nSample predictions (first 5 rows):")
        print(
            df_results[["predicted_purchased_last_month"]].head().to_string(index=False)
        )

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

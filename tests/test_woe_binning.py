import os
import sys
import pandas as pd
import numpy as np
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from woe_binning import (
    calculate_rfms_components,
    normalize_rfms_components,
    woe_binning,
    assign_risk_labels
)  # Import the functions you want to test

class TestWoEFunctions(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.df = pd.DataFrame({
            'CustomerId': [1, 1, 1, 2, 2, 3],
            'TransactionStartTime': pd.to_datetime([
                '2023-01-01', '2023-02-01', '2023-03-01',
                '2023-01-01', '2023-04-01', '2023-01-01']),
            'Amount': [100, 200, 300, 400, 500, 600]
        })

    def test_calculate_rfms_components(self):
        rfms_df = calculate_rfms_components(self.df)
        # Check if RFMS DataFrame has the correct columns
        expected_columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary', 'Stability']
        self.assertTrue(set(expected_columns).issubset(rfms_df.columns))
        self.assertEqual(rfms_df['CustomerId'].nunique(), 3)  # Check unique customers

    def test_normalize_rfms_components(self):
        rfms_df = calculate_rfms_components(self.df)
        normalized_rfms_df = normalize_rfms_components(rfms_df)

        # Check if normalization has been done
        for component in ['Recency', 'Frequency', 'Monetary', 'Stability']:
            self.assertTrue((normalized_rfms_df[component] >= 0).all())
            self.assertTrue((normalized_rfms_df[component] <= 1).all())

    def test_assign_risk_labels(self):
        rfms_df = calculate_rfms_components(self.df)
        normalized_rfms_df = normalize_rfms_components(rfms_df)
        final_df = assign_risk_labels(normalized_rfms_df)

        # Check if Risk_Label column exists
        self.assertIn('Risk_Label', final_df.columns)

        # Check label assignment
        self.assertTrue((final_df['Risk_Label'].isin(['Good', 'Bad'])).all())

    def test_woe_binning(self):
        rfms_df = calculate_rfms_components(self.df)
        normalized_rfms_df = normalize_rfms_components(rfms_df)
        final_df = assign_risk_labels(normalized_rfms_df)

        # Call woe_binning function
        bin_stats, iv = woe_binning(final_df, 'Monetary', 'Risk_Label')

        # Check if binning output is as expected
        self.assertIn('bin', bin_stats.columns)
        self.assertIn('bad_count', bin_stats.columns)
        self.assertIn('good_count', bin_stats.columns)
        self.assertIn('WoE', bin_stats.columns)
        self.assertIn('IV', bin_stats.columns)

        # Check if IV is calculated
        self.assertGreaterEqual(iv, 0)

if __name__ == '__main__':
    unittest.main()

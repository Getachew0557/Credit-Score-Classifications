import os
import sys
# Append the correct src path for custom module imports
sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('../data'))

from load_data import load_data
from eda import (
    dataset_overview,
    summary_statistics,
    plot_numerical_histograms,
    plot_numerical_boxplots,
    plot_pairplots,
    plot_categorical_distributions,
    plot_categorical_vs_target,
    correlation_analysis,
    identify_missing_values,
    plot_outliers,
    plot_boxplots,
)
from feature_engineering import (
    create_aggregate_features,
    extract_transaction_time_features,
    merge_aggregate_and_time_features,
    reorder_columns,
    encode_features,
    handle_missing_values,
    normalize_features
)
from woe_binning import (
    calculate_rfms_components,
    normalize_rfms_components,
    plot_rfms_distribution,
    plot_rfms_components,
    visualize_rfms_space,
    assign_risk_labels,
    woe_binning,
    apply_woe_binning


)

def main():
    # Load the data
    df = load_data('../data/data.csv')

    # Perform EDA
    dataset_overview(df)
    summary_statistics(df)
    plot_numerical_histograms(df)
    plot_numerical_boxplots(df)
    plot_pairplots(df)
    plot_categorical_distributions(df)
    plot_categorical_vs_target(df)
    correlation_analysis(df)
    identify_missing_values(df)
    plot_outliers(df)
    plot_boxplots(df)

    # Create aggregate features
    aggregate_features = create_aggregate_features(df)

    # Extract time-based features
    df = extract_transaction_time_features(df)

    # Merge aggregate features with extracted time-based features
    final_df = merge_aggregate_and_time_features(df, aggregate_features)

    # Reorder columns to place 'FraudResult' at the end
    final_df = reorder_columns(final_df)

    # Handle missing values
    final_df = handle_missing_values(final_df)

    # Encode categorical features
    final_df = encode_features(final_df)

    # Normalize numerical features
    final_df = normalize_features(final_df)

    # Display the final DataFrame
    print("Final DataFrame after feature engineering:\n", final_df.head())

    """Main function to execute the RFMS analysis."""
    rfms_df = calculate_rfms_components(df)
    rfms_df = normalize_rfms_components(rfms_df)
    plot_rfms_distribution(rfms_df)
    plot_rfms_components(rfms_df)
    
    # Visualize RFMS space
    rfms_df = assign_risk_labels(rfms_df)
    visualize_rfms_space(rfms_df)
    
    # Apply WoE binning
    apply_woe_binning(rfms_df)

if __name__ == "__main__":
    main()

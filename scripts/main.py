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

if __name__ == "__main__":
    main()

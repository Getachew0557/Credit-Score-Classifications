import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_rfms_components(df):
    """Calculate RFMS components."""
    df['Recency'] = (df['TransactionStartTime'].max() - df['TransactionStartTime']).dt.days
    frequency_df = df.groupby('CustomerId').size().reset_index(name='Frequency')
    monetary_df = df.groupby('CustomerId')['Amount'].sum().reset_index(name='Monetary')
    stability_df = df.groupby('CustomerId')['Amount'].std().reset_index(name='Stability').fillna(0)

    rfms_df = pd.merge(frequency_df, monetary_df, on='CustomerId')
    rfms_df = pd.merge(rfms_df, stability_df, on='CustomerId')
    rfms_df = pd.merge(rfms_df, df[['CustomerId', 'Recency']].drop_duplicates(), on='CustomerId')
    
    return rfms_df

def normalize_rfms_components(rfms_df):
    """Normalize RFMS components to bring them to the same scale."""
    for component in ['Recency', 'Frequency', 'Monetary', 'Stability']:
        rfms_df[component] = (rfms_df[component] - rfms_df[component].min()) / (rfms_df[component].max() - rfms_df[component].min())
    
    rfms_df['RFMS_Score'] = rfms_df[['Recency', 'Frequency', 'Monetary', 'Stability']].mean(axis=1)
    return rfms_df

def plot_rfms_distribution(rfms_df):
    """Plot the distribution of RFMS Scores."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.histplot(rfms_df['RFMS_Score'], bins=30, kde=True, color='blue', stat='density')
    
    plt.title('Distribution of RFMS Score', fontsize=16)
    plt.xlabel('RFMS Score', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.axvline(rfms_df['RFMS_Score'].mean(), color='red', linestyle='--', label='Mean RFMS Score')
    plt.axvline(rfms_df['RFMS_Score'].median(), color='green', linestyle='--', label='Median RFMS Score')
    plt.legend()
    plt.grid()
    plt.show()

def plot_rfms_components(rfms_df):
    """Visualize RFMS components."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].hist(rfms_df['Recency'], bins=20, color='blue', alpha=0.7)
    axes[0, 0].set_title('Recency Distribution')
    
    axes[0, 1].hist(rfms_df['Frequency'], bins=20, color='green', alpha=0.7)
    axes[0, 1].set_title('Frequency Distribution')
    
    axes[1, 0].hist(rfms_df['Monetary'], bins=20, color='red', alpha=0.7)
    axes[1, 0].set_title('Monetary Distribution')
    
    axes[1, 1].hist(rfms_df['Stability'], bins=20, color='purple', alpha=0.7)
    axes[1, 1].set_title('Stability Distribution')
    
    plt.tight_layout()
    plt.show()

def visualize_rfms_space(final_df):
    """Visualize RFMS space."""
    plt.figure(figsize=(10, 6))
    plt.scatter(final_df['Monetary'], final_df['Frequency'], c=final_df['RFMS_Score'], cmap='viridis')
    plt.colorbar(label='RFMS Score')
    plt.title('RFMS Space: Frequency vs Monetary with RFMS Scores')
    plt.xlabel('Monetary')
    plt.ylabel('Frequency')
    plt.show()

def assign_risk_labels(final_df, threshold=0.5):
    """Assign labels based on RFMS score."""
    final_df['Risk_Label'] = np.where(final_df['RFMS_Score'] > threshold, 'Good', 'Bad')
    return final_df

def woe_binning(df, feature, target):
    """Calculate WoE and IV for binning."""
    df['bin'] = pd.qcut(df[feature], q=10, duplicates='drop')  # Create quantile-based bins
    bin_stats = df.groupby('bin').agg(
        bad_count=(target, lambda x: (x == 'Bad').sum()),
        good_count=(target, lambda x: (x == 'Good').sum()),
        total_count=(target, 'count')
    ).reset_index()

    # Calculate WoE and IV
    bin_stats['bad_rate'] = bin_stats['bad_count'] / bin_stats['bad_count'].sum()
    bin_stats['good_rate'] = bin_stats['good_count'] / bin_stats['good_count'].sum()
    bin_stats['WoE'] = np.log(bin_stats['good_rate'] / bin_stats['bad_rate'])
    bin_stats['IV'] = (bin_stats['good_rate'] - bin_stats['bad_rate']) * bin_stats['WoE']

    return bin_stats[['bin', 'bad_count', 'good_count', 'WoE', 'IV']], bin_stats['IV'].sum()

def apply_woe_binning(final_df):
    """Apply WoE binning to selected features."""
    woe_features = ['Monetary', 'Frequency', 'Recency', 'Stability']
    for feature in woe_features:
        print(f"\nFeature: {feature}")
        bin_stats, iv = woe_binning(final_df, feature, 'Risk_Label')
        print(bin_stats)
        print(f"Information Value (IV) for {feature}: {iv:.4f}")






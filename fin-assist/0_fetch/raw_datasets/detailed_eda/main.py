import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """
    Load dataset from a given file path. Supports CSV, Excel, JSON.

    Parameters:
    ----------
    file_path : str
        The path to the dataset file.

    Returns:
    -------
    df : pandas.DataFrame
        Loaded data as DataFrame.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    elif ext == '.json':
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return df


def summarize_dataframe(df):
    """
    Print general information and summary statistics.
    """
    print("\n=== GLOBAL INFORMATION ===")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nData Types and Non-null Counts:")
    print(df.info())

    print("\n=== MISSING VALUES ===")
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.concat([missing, missing_pct], axis=1, keys=["Missing Count", "% Missing"])  
    print(missing_df[missing_df['Missing Count'] > 0])

    print("\n=== STATISTICAL SUMMARY ===")
    print("\n-- Numeric Columns --")
    print(df.select_dtypes(include=[np.number]).describe().T)
    print("\n-- Categorical Columns --")
    print(df.select_dtypes(include=['object', 'category']).describe().T)


def analyze_numeric(df, output_dir):
    """
    Generate analysis for numeric columns: distributions, outliers, correlations.
    """
    num_df = df.select_dtypes(include=[np.number])
    corr_matrix = num_df.corr()

    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved correlation heatmap to: {heatmap_path}")

    # Histograms and Boxplots
    for col in num_df.columns:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(num_df[col].dropna(), kde=True)
        plt.title(f'Histogram of {col}')

        plt.subplot(1, 2, 2)
        sns.boxplot(x=num_df[col].dropna())
        plt.title(f'Boxplot of {col}')

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f'{col}_dist.png')
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved distribution plots for {col} to: {fig_path}")


def analyze_categorical(df, output_dir):
    """
    Generate analysis for categorical columns: top value counts, missing distribution.
    """
    cat_df = df.select_dtypes(include=['object', 'category'])
    for col in cat_df.columns:
        freq = cat_df[col].value_counts(dropna=False)
        print(f"\n=== Value Counts for {col} ===")
        print(freq.head(10))

        plt.figure(figsize=(8, 6))
        sns.countplot(y=col, data=df, order=freq.index[:10])
        plt.title(f'Top 10 categories in {col}')
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f'{col}_counts.png')
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved count plot for {col} to: {fig_path}")


def detect_outliers(df):
    """
    Identify potential outliers using IQR method and print a summary.
    """
    num_df = df.select_dtypes(include=[np.number])
    print("\n=== POTENTIAL OUTLIERS (IQR METHOD) ===")
    for col in num_df.columns:
        Q1 = num_df[col].quantile(0.25)
        Q3 = num_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = num_df[(num_df[col] < lower_bound) | (num_df[col] > upper_bound)][col]
        print(f"{col}: {len(outliers)} potential outliers ({(len(outliers)/len(num_df)*100):.2f}% of data)")


def correlation_pairs(df, threshold=0.8):
    """
    Find pairs of features with correlation above a certain threshold.
    """
    num_df = df.select_dtypes(include=[np.number])
    corr_matrix = num_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [(column, idx, corr_matrix.loc[idx, column])
                 for column in upper.columns for idx in upper.index
                 if upper.loc[idx, column] > threshold]
    if high_corr:
        print("\n=== HIGHLY CORRELATED PAIRS (>|{threshold}|) ===")
        for col1, col2, corr_val in high_corr:
            print(f"{col1} and {col2}: Correlation = {corr_val:.2f}")
    else:
        print(f"\nNo feature pairs found with correlation above {threshold}")


def main():
    parser = argparse.ArgumentParser(description='Detailed Exploratory Data Analysis Script')
    parser.add_argument('file', type=str, help='Path to the input dataset (CSV, Excel, or JSON)')
    parser.add_argument('--output', type=str, default='eda_output', help='Directory to save output files')
    args = parser.parse_args()

    # Create output directory if not exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    df = load_data(args.file)

    # Summarize DataFrame
    summarize_dataframe(df)

    # Numeric Analysis
    analyze_numeric(df, args.output)

    # Categorical Analysis
    analyze_categorical(df, args.output)

    # Outlier Detection
    detect_outliers(df)

    # Correlation pairs
    correlation_pairs(df)

    print(f"\nEDA completed. All plots saved to directory: {args.output}")

if __name__ == '__main__':
    main()

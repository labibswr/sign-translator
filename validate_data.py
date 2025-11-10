"""
Data validation and analysis script.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.config import *

def analyze_data():
    """Analyze the collected sign language data."""
    data_file = os.path.join(DATA_DIR, 'sign_data.csv')
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return
    
    print("=" * 60)
    print("DATA VALIDATION & ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(data_file)
    
    print(f"Data loaded successfully!")
    print(f"   Total samples: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    # Check for missing values
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("    No missing values found")
    else:
        print("   Missing values detected:")
        print(missing[missing > 0])
    
    # Analyze sign distribution
    print(f"\n Sign Distribution:")
    sign_counts = df['sign'].value_counts().sort_index()
    print(sign_counts)
    
    # Check if all expected signs are present
    expected_signs = set(SIGNS)
    actual_signs = set(df['sign'].unique())
    
    missing_signs = expected_signs - actual_signs
    extra_signs = actual_signs - expected_signs
    
    if missing_signs:
        print(f"\n  Missing signs: {sorted(missing_signs)}")
    if extra_signs:
        print(f"  Extra signs found: {sorted(extra_signs)}")
    if not missing_signs and not extra_signs:
        print(f"\n All {len(SIGNS)} expected signs are present")
    
    # Statistics
    print(f"\n Statistics:")
    print(f"   Samples per sign:")
    print(f"   - Minimum: {sign_counts.min()}")
    print(f"   - Maximum: {sign_counts.max()}")
    print(f"   - Average: {sign_counts.mean():.1f}")
    print(f"   - Median: {sign_counts.median():.1f}")
    print(f"   - Standard deviation: {sign_counts.std():.1f}")
    
    # Check for imbalanced data
    imbalance_ratio = sign_counts.max() / sign_counts.min()
    if imbalance_ratio > 2:
        print(f"\n  Data imbalance detected!")
        print(f"   Ratio (max/min): {imbalance_ratio:.2f}")
        print(f"   Consider collecting more samples for underrepresented signs")
    else:
        print(f"\n Data is relatively balanced (ratio: {imbalance_ratio:.2f})")
    
    # Validate landmark data
    print(f"\n Validating Landmark Data:")
    valid_landmarks = 0
    invalid_landmarks = 0
    
    for idx, row in df.iterrows():
        try:
            landmark_str = row['landmarks']
            clean_str = landmark_str.strip('[]')
            coords = [float(x.strip()) for x in clean_str.split(',')]
            
            if len(coords) == 63:  # 21 landmarks * 3 coordinates
                landmarks = np.array(coords).reshape(21, 3)
                # Check for NaN or infinite values
                if np.isfinite(landmarks).all():
                    valid_landmarks += 1
                else:
                    invalid_landmarks += 1
            else:
                invalid_landmarks += 1
        except:
            invalid_landmarks += 1
    
    print(f"   Valid samples: {valid_landmarks}")
    if invalid_landmarks > 0:
        print(f"   Invalid samples: {invalid_landmarks}")
    
    # Visualizations
    print(f"\n Generating visualizations...")
    
    # Create output directory
    os.makedirs('analysis', exist_ok=True)
    
    # Plot 1: Sign distribution
    plt.figure(figsize=(14, 6))
    sign_counts.plot(kind='bar', color='steelblue')
    plt.title('Samples per Sign', fontsize=14, fontweight='bold')
    plt.xlabel('Sign', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis/sign_distribution.png', dpi=300, bbox_inches='tight')
    print("   Saved: analysis/sign_distribution.png")
    plt.close()
    
    # Plot 2: Distribution statistics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(sign_counts.values, bins=20, color='steelblue', edgecolor='black')
    axes[0].set_title('Distribution of Sample Counts', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Samples', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plot
    axes[1].boxplot(sign_counts.values, vert=True)
    axes[1].set_title('Sample Count Statistics', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Samples', fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/distribution_stats.png', dpi=300, bbox_inches='tight')
    print("   Saved: analysis/distribution_stats.png")
    plt.close()
    
    # Recommendations
    print(f"\n Recommendations:")
    recommendations = []
    
    if sign_counts.min() < SAMPLES_PER_SIGN * 0.8:
        recommendations.append(f"   - Some signs have fewer than {int(SAMPLES_PER_SIGN * 0.8)} samples")
        recommendations.append(f"     Consider collecting more data for: {', '.join(sign_counts[sign_counts < SAMPLES_PER_SIGN * 0.8].index.tolist())}")
    
    if imbalance_ratio > 2:
        recommendations.append(f"   - Data is imbalanced. Aim for similar sample counts per sign")
    
    if invalid_landmarks > 0:
        recommendations.append(f"   - {invalid_landmarks} invalid samples detected. Review data collection")
    
    if len(recommendations) == 0:
        print("  Data looks good! Ready for training.")
    else:
        for rec in recommendations:
            print(rec)
    
    print(f"\n" + "=" * 60)
    print("Analysis complete! Check the 'analysis' folder for visualizations.")
    print("=" * 60)

if __name__ == "__main__":
    analyze_data()


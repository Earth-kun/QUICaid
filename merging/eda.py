import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Set visual styles
plt.style.use('ggplot')
sns.set_palette("viridis")

def load_datasets():
    """Load and preprocess the original and CESNET datasets with special analysis for segment transitions"""
    print("Loading datasets...")
    
    # Load your preprocessed dataset
    try:
        original_df = pd.read_csv("./merged.csv")
        original_df['source'] = 'QUIC.live'
        print(f"Original dataset loaded: {original_df.shape[0]} rows, {original_df.shape[1]} columns")
    except FileNotFoundError:
        print("Warning: Original dataset not found. Please check the path.")
        original_df = None
    
    # Load CESNET dataset with special handling for the alternating traffic patterns
    try:
        cesnet_df = pd.read_csv("./merging/benign_cesnet.csv")
        cesnet_df['source'] = 'CESNET'
        
        # Add segment information based on the notebook's logic
        # The merged dataset alternates between benign and attack traffic in windows of 10,000 records
        window_size = 10000
        
        # Create a segment identifier to track the alternating pattern
        # Even segments (0, 2, 4...) are benign, odd segments (1, 3, 5...) are attacks
        # Moreover, early odd segments are flooding, later odd segments are fuzzing
        cesnet_df['segment'] = (cesnet_df.index // window_size)
        
        # Identify traffic type based on segment pattern
        cesnet_df['traffic_type'] = 'unknown'
        cesnet_df.loc[cesnet_df['segment'] % 2 == 0, 'traffic_type'] = 'benign'
        
        # First half of odd segments are flooding (up to halfway_point)
        halfway_point = 10  # From the notebook
        cesnet_df.loc[(cesnet_df['segment'] % 2 == 1) & 
                      (cesnet_df['segment'] // 2 < halfway_point), 'traffic_type'] = 'flooding'
        
        # Second half of odd segments are fuzzing
        cesnet_df.loc[(cesnet_df['segment'] % 2 == 1) & 
                      (cesnet_df['segment'] // 2 >= halfway_point), 'traffic_type'] = 'fuzzing'
        
        print(f"CESNET dataset loaded: {cesnet_df.shape[0]} rows, {cesnet_df.shape[1]} columns")
        print(f"Traffic type distribution in CESNET dataset:")
        print(cesnet_df['traffic_type'].value_counts())
    except FileNotFoundError:
        print("Warning: CESNET dataset not found. Please check the path.")
        cesnet_df = None
    
    # Check if at least one dataset is loaded
    if original_df is None and cesnet_df is None:
        raise FileNotFoundError("No datasets could be loaded. Please check the file paths.")
        
    # Combine datasets
    try:
        combined_df = pd.read_csv("./merged_cesnet.csv")
        print(f"Merged CESNET dataset loaded directly: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
        
        # Add source column if it doesn't exist
        if 'source' not in combined_df.columns:
            combined_df['source'] = 'Merged'
            print("Added 'Merged' as source identifier")
            
        # Add segment information if it doesn't exist but we have the traffic_type column
        if 'segment' not in combined_df.columns and 'traffic_type' in combined_df.columns:
            window_size = 10000
            combined_df['segment'] = (combined_df.index // window_size)
            print("Added segment information based on index position")
    except FileNotFoundError:
        print("Warning: merged_cesnet.csv not found. Using individual datasets if available.")
        combined_df = None
        # If we couldn't load the merged file but have individual files, combine them
        if original_df is not None or cesnet_df is not None:
            if original_df is not None and cesnet_df is not None:
                combined_df = pd.concat([original_df, cesnet_df], ignore_index=True)
            elif original_df is not None:
                combined_df = original_df.copy()
            else:
                combined_df = cesnet_df.copy()
                
    if combined_df is None:
        raise FileNotFoundError("No datasets could be loaded. Please check the file paths.")

    # Add index-based field for analyzing variance across dataset
    combined_df['index_bin'] = pd.qcut(combined_df.index, 10, labels=False)
    
    # Convert label column to numeric if it's not
    combined_df['label'] = pd.to_numeric(combined_df['label'], errors='coerce')
    
    print(f"Final dataset: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
    print(f"Columns available: {', '.join(combined_df.columns)}")
        
    return combined_df

def analyze_segment_transitions(df):
    """Analyze how variance and correlation patterns change at segment transitions"""
    if 'segment' not in df.columns:
        print("Warning: Segment information not available, skipping transition analysis.")
        return
    
    # Select key features for analysis
    key_features = [
        "flow_pkt_rate", "flow_byte_rate", "total_pkts", "total_bytes",
        "ave_bytes", "std_bytes", "ave_iat", "std_iat"
    ]
    
    # Calculate variance by segment
    print("\nAnalyzing variance changes across segments...")
    segment_variance = df.groupby('segment')[key_features].var()
    
    # Plot variance trends
    plt.figure(figsize=(15, 10))
    for feature in key_features:
        plt.plot(segment_variance.index, segment_variance[feature], 
                 marker='o', label=feature)
    
    # Add vertical lines to mark transitions
    max_segment = df['segment'].max()
    for i in range(1, int(max_segment) + 1):
        if i % 2 == 1:  # Transitions to attack
            plt.axvline(x=i-0.5, color='r', linestyle='--', alpha=0.3)
        else:  # Transitions to benign
            plt.axvline(x=i-0.5, color='g', linestyle='--', alpha=0.3)
    
    # Mark the transition from flooding to fuzzing
    halfway_point = 10
    transition_segment = halfway_point * 2 - 1
    plt.axvline(x=transition_segment+0.5, color='purple', linestyle='-', 
                linewidth=2, label='Flooding to Fuzzing Transition')
    
    plt.title('Feature Variance by Segment')
    plt.xlabel('Segment')
    plt.ylabel('Variance (log scale)')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./merging/eda_results/segment_variance_transition.png")
    plt.close()
    
    # Analyze correlation changes
    print("\nAnalyzing correlation changes across segments...")
    
    # Calculate correlations for different traffic types
    benign_corr = df[df['traffic_type'] == 'benign'][key_features].corr()
    flooding_corr = df[df['traffic_type'] == 'flooding'][key_features].corr()
    fuzzing_corr = df[df['traffic_type'] == 'fuzzing'][key_features].corr()
    
    # Calculate correlation differences
    flood_vs_benign = flooding_corr - benign_corr
    fuzz_vs_benign = fuzzing_corr - benign_corr
    fuzz_vs_flood = fuzzing_corr - flooding_corr
    
    # Plot correlation differences
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    sns.heatmap(flood_vs_benign, cmap='RdBu_r', center=0, annot=True, 
                fmt=".2f", ax=axes[0])
    axes[0].set_title('Correlation Difference: Flooding vs Benign')
    
    sns.heatmap(fuzz_vs_benign, cmap='RdBu_r', center=0, annot=True, 
                fmt=".2f", ax=axes[1])
    axes[1].set_title('Correlation Difference: Fuzzing vs Benign')
    
    sns.heatmap(fuzz_vs_flood, cmap='RdBu_r', center=0, annot=True, 
                fmt=".2f", ax=axes[2])
    axes[2].set_title('Correlation Difference: Fuzzing vs Flooding')
    
    plt.tight_layout()
    plt.savefig("./merging/eda_results/correlation_differences_by_type.png")
    plt.close()
    
    # Create detailed segment-by-segment correlation analysis
    # This helps visualize how correlations change at transition points
    print("\nCreating detailed segment correlation analysis...")
    
    # Select a few key segments around transitions
    # 1. Normal benign (segment 0)
    # 2. First flooding segment (segment 1) 
    # 3. Before fuzzing transition (segment 19)
    # 4. First fuzzing segment (segment 21)
    key_segments = [0, 1, 19, 21]
    segment_names = ['Benign', 'Flooding', 'Late Flooding', 'Fuzzing']
    
    # Create correlation matrices for key segments
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()
    
    for i, (segment, name) in enumerate(zip(key_segments, segment_names)):
        segment_data = df[df['segment'] == segment][key_features]
        corr = segment_data.corr()
        
        sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", 
                    center=0, square=True, linewidths=.5, ax=axes[i])
        axes[i].set_title(f'Correlation Matrix: {name} (Segment {segment})')
    
    plt.tight_layout()
    plt.savefig("./merging/eda_results/segment_correlation_comparison.png")
    plt.close()
    
    # Create statistical summaries for each segment
    print("\nGenerating statistical summaries by segment...")
    segment_stats = df.groupby(['segment', 'traffic_type'])[key_features].agg(['mean', 'std', 'var'])
    segment_stats.to_csv("./merging/eda_results/segment_statistics.csv")
    
    # Create interactive visualization for segment transitions with plotly
    print("\nCreating interactive segment transition visualization...")
    
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       subplot_titles=('Feature Mean by Segment', 
                                       'Feature Variance by Segment'),
                       vertical_spacing=0.1)
    
    # Add mean values
    segment_means = df.groupby('segment')[key_features].mean()
    
    for feature in key_features:
        fig.add_trace(
            go.Scatter(x=segment_means.index, y=segment_means[feature],
                     mode='lines+markers', name=f'{feature} (mean)'),
            row=1, col=1
        )
    
    # Add variance values (log scale)
    for feature in key_features:
        fig.add_trace(
            go.Scatter(x=segment_variance.index, y=segment_variance[feature],
                     mode='lines+markers', name=f'{feature} (variance)'),
            row=2, col=1
        )
    
    # Add transition markers
    for i in range(1, int(max_segment) + 1):
        if i % 2 == 1:  # Transitions to attack
            fig.add_vline(x=i-0.5, line=dict(color='red', width=1, dash='dash'),
                        row='all', col=1)
        else:  # Transitions to benign
            fig.add_vline(x=i-0.5, line=dict(color='green', width=1, dash='dash'),
                        row='all', col=1)
    
    # Mark flooding to fuzzing transition
    fig.add_vline(x=transition_segment+0.5, 
                line=dict(color='purple', width=2),
                row='all', col=1)
    
    fig.update_layout(
        height=800, width=1200,
        title_text='Feature Statistics Across Segments with Transition Points',
        legend=dict(orientation='h', yanchor='bottom', y=-0.3),
        yaxis2=dict(type='log')
    )
    
    fig.write_html("./merging/eda_results/segment_transition_analysis.html")
    
    # Create a special analysis around the flooding-to-fuzzing transition
    print("\nAnalyzing the flooding-to-fuzzing transition...")
    
    # Define segments around the transition
    pre_transition = list(range(transition_segment-2, transition_segment+1))
    post_transition = list(range(transition_segment+1, transition_segment+4))
    
    # Extract data for these segments
    pre_data = df[df['segment'].isin(pre_transition)][key_features]
    post_data = df[df['segment'].isin(post_transition)][key_features]
    
    # Calculate statistics
    pre_mean = pre_data.mean()
    post_mean = post_data.mean()
    pct_change = ((post_mean - pre_mean) / pre_mean) * 100
    
    # Create a comparison table
    comparison_df = pd.DataFrame({
        'Flooding Mean': pre_mean,
        'Fuzzing Mean': post_mean,
        'Percent Change': pct_change
    })
    
    comparison_df.to_csv("./merging/eda_results/flooding_to_fuzzing_comparison.csv")
    
    # Plot the comparison
    plt.figure(figsize=(12, 8))
    comparison_df['Percent Change'].sort_values().plot(kind='barh')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.title('Percent Change in Feature Means: Fuzzing vs Flooding')
    plt.xlabel('Percent Change')
    plt.tight_layout()
    plt.savefig("./merging/eda_results/flooding_to_fuzzing_pct_change.png")
    plt.close()

def data_cleaning(df):
    """Clean the dataset by handling missing values and outliers"""
    print("\nCleaning data...")
    
    # Store original row count
    original_count = len(df)
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    df = df.dropna()
    missing_removed = original_count - len(df)
    
    # Handle potential infinite values
    inf_mask = np.isinf(df.select_dtypes(include=[np.number])).any(axis=1)
    inf_count = inf_mask.sum()
    df = df[~inf_mask]
    
    # Report cleaning results
    print(f"Rows with missing values removed: {missing_removed} ({missing_removed/original_count*100:.2f}%)")
    print(f"Rows with infinite values removed: {inf_count} ({inf_count/original_count*100:.2f}%)")
    print(f"Final dataset size: {len(df)} rows")
    
    return df

def basic_eda(df):
    """Perform basic exploratory data analysis"""
    print("\nPerforming basic EDA...")
    
    # Basic statistics
    print("\nCalculating summary statistics...")
    summary = df.describe().T
    summary['missing'] = df.isnull().sum()
    summary['missing_pct'] = (df.isnull().sum() / len(df)) * 100
    
    # Save statistics to CSV
    summary.to_csv("./merging/eda_results/summary_statistics.csv")
    print("Summary statistics saved to ./merging/eda_results/summary_statistics.csv")
    
    # Check if source column is available
    has_source = 'source' in df.columns
    
    # Distribution of label classes
    plt.figure(figsize=(12, 6))
    
    if has_source and len(df['source'].unique()) > 1:
        # Create cross-tabulation of labels by source
        cross_tab = pd.crosstab(df['label'], df['source'], normalize='columns')
        cross_tab.plot(kind='bar', stacked=False)
        plt.title('Distribution of Traffic Classes by Dataset Source')
        plt.xlabel('Label (0=benign, 1=attack)')
        plt.ylabel('Proportion')
        plt.xticks(rotation=0)
        plt.legend(title='Source')
    else:
        # Simple label distribution
        df['label'].value_counts(normalize=True).plot(kind='bar')
        plt.title('Distribution of Traffic Classes')
        plt.xlabel('Label (0=benign, 1=attack)')
        plt.ylabel('Proportion')
        plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig("./merging/eda_results/label_distribution.png")
    plt.close()
    
    # Create pie charts for label distribution
    plt.figure(figsize=(15, 6))
    
    if has_source and len(df['source'].unique()) > 1:
        # Create individual pie charts for each source
        sources = df['source'].unique()
        num_sources = len(sources)
        
        for i, source in enumerate(sources):
            plt.subplot(1, num_sources + 1, i + 1)
            df[df['source'] == source]['label'].value_counts().plot.pie(
                autopct='%1.1f%%', startangle=90, title=f'{source} Label Distribution')
        
        # Add overall distribution
        plt.subplot(1, num_sources + 1, num_sources + 1)
        df['label'].value_counts().plot.pie(
            autopct='%1.1f%%', startangle=90, title='Combined Label Distribution')
    else:
        # Just the overall distribution
        df['label'].value_counts().plot.pie(
            autopct='%1.1f%%', startangle=90, title='Label Distribution')
    
    plt.tight_layout()
    plt.savefig("./merging/eda_results/label_distribution_pie.png")
    plt.close()
    
    return summary

def feature_distributions(df):
    """Plot distributions of key features"""
    print("\nAnalyzing feature distributions...")
    
    # Select key features to visualize (excluding certain technical features)
    key_features = [
        "flow_pkt_rate", "flow_byte_rate", "total_pkts", "total_bytes",
        "ave_bytes", "std_bytes", "fwd_pkts", "fwd_bytes", 
        "rev_pkts", "rev_bytes", "ave_iat", "std_iat"
    ]
    
    # Create a subplot grid for histograms
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(4, 3)
    
    for i, feature in enumerate(key_features):
        ax = plt.subplot(gs[i])
        
        # Plot by source
        for source, color in zip(['QUIC.live', 'CESNET'], ['blue', 'orange']):
            data = df[df['source'] == source][feature]
            # Trim outliers for better visualization
            data = data[data.between(data.quantile(0.01), data.quantile(0.99))]
            sns.histplot(data, color=color, label=source, kde=True, alpha=0.5, ax=ax)
        
        ax.set_title(f'Distribution of {feature}')
        ax.set_xlabel(feature)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("./merging/eda_results/feature_distributions.png")
    plt.close()
    
    # Create violin plots by label
    plt.figure(figsize=(20, 15))
    
    for i, feature in enumerate(key_features):
        plt.subplot(4, 3, i+1)
        
        # Trim outliers for better visualization
        data = df.copy()
        data[feature] = data[feature][data[feature].between(
            data[feature].quantile(0.01), data[feature].quantile(0.99))]
        
        sns.violinplot(x='label', y=feature, data=data, inner='quartile')
        plt.title(f'{feature} by Label')
        plt.xlabel('Label')
    
    plt.tight_layout()
    plt.savefig("./merging/eda_results/feature_violins_by_label.png")
    plt.close()
    
    # Create interactive distribution comparison if both datasets are available
    print("\nCreating interactive distribution comparisons...")
    
    for feature in key_features:
        # Create boxplots by label and source
        fig = px.box(
            df, x='label', y=feature, color='source',
            title=f'Distribution of {feature} by Label and Source',
            labels={'label': 'Traffic Class'},
            category_orders={"label": sorted(df['label'].unique())},
            template='plotly_white'
        )
        
        fig.update_layout(boxmode='group', height=600, width=900)
        fig.write_html(f"./merging/eda_results/{feature}_boxplot.html")

def correlation_analysis(df):
    """Analyze feature correlations"""
    print("\nAnalyzing feature correlations...")
    
    # Select numeric columns, excluding certain metadata
    exclude_cols = ['index_bin']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, 
                center=0, square=True, linewidths=.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig("./merging/eda_results/correlation_heatmap.png")
    plt.close()
    
    # Create clustered correlation matrix for better visualization
    plt.figure(figsize=(20, 16))
    try:
        # Handle NaN and infinite values in correlation matrix
        corr_matrix_clean = corr_matrix.copy()
        # Replace NaN or infinite values with 0 for clustering purposes
        corr_matrix_clean = corr_matrix_clean.fillna(0)
        mask_inf = np.isinf(corr_matrix_clean)
        if mask_inf.any().any():
            corr_matrix_clean[mask_inf] = 0
            print("Warning: Infinite values in correlation matrix were replaced with zeros for visualization")
        
        # Now create the clustered heatmap
        cluster_grid = sns.clustermap(
            corr_matrix_clean,
            cmap='coolwarm',
            standard_scale=1,
            method='complete',
            figsize=(20, 16)
        )
        plt.suptitle('Clustered Correlation Matrix', y=1.02, fontsize=16)
        plt.savefig("./merging/eda_results/clustered_correlation.png")
    except Exception as e:
        print(f"Warning: Could not create clustered correlation matrix: {str(e)}")
        # Create alternative visualization without clustering
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix (Clustering failed)')
        plt.savefig("./merging/eda_results/correlation_matrix_no_clustering.png")
    plt.close()
    
    # Find highly correlated features
    high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix != 1.0)
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if high_corr.iloc[i, j]:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    # Sort by absolute correlation value
    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)
    
    print("\nHighly correlated feature pairs:")
    for feat1, feat2, corr in high_corr_pairs[:15]:  # Show top 15
        print(f"{feat1} -- {feat2}: {corr:.4f}")
    
    # Save detailed correlations
    pd.DataFrame(high_corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation']).to_csv(
        "./merging/eda_results/high_correlations.csv", index=False)
    
    # Correlation analysis by label
    label_corr = {}
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        label_corr[label] = label_df[numeric_cols].corr()
    
    # Plot correlation heatmaps by label
    for label, corr in label_corr.items():
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, 
                    center=0, square=True, linewidths=.5)
        plt.title(f'Feature Correlation Matrix for Label {label}')
        plt.tight_layout()
        plt.savefig(f"./merging/eda_results/correlation_label_{label}.png")
        plt.close()

def dimensionality_reduction(df):
    """Perform PCA for visualization"""
    print("\nPerforming dimensionality reduction with PCA...")
    
    # Select numeric columns
    exclude_cols = ['label', 'index_bin']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Standardize features
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df[numeric_cols])
    
    # Apply PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_df)
    
    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1], 
        'PC3': pca_result[:, 2],
        'label': df['label']
    })
    
    pca_df['source'] = df['source']
    
    # Print variance explained
    explained_variance = pca.explained_variance_ratio_ * 100
    print(f"Variance explained by PCA components: {explained_variance[0]:.2f}%, "
          f"{explained_variance[1]:.2f}%, {explained_variance[2]:.2f}%")
    
    # Create 3D PCA plot with plotly
    fig = px.scatter_3d(
        pca_df, x='PC1', y='PC2', z='PC3',
        color='label', symbol='source',
        hover_data=['label', 'source'],
        title=f'PCA Visualization (Total variance explained: {sum(explained_variance):.2f}%)',
        labels={'label': 'Traffic Class'},
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title=f'PC1 ({explained_variance[0]:.2f}%)',
            yaxis_title=f'PC2 ({explained_variance[1]:.2f}%)',
            zaxis_title=f'PC3 ({explained_variance[2]:.2f}%)',
        ),
        legend_title_text='Class'
    )
    
    fig.write_html("./merging/eda_results/pca_3d_visualization.html")
    
    # Create 2D PCA plot by label
    plt.figure(figsize=(12, 10))
    
    # Color by label, shape by source
    for source in df['source'].unique():
        source_data = pca_df[pca_df['source'] == source]
        for label in source_data['label'].unique():
            label_data = source_data[source_data['label'] == label]
            plt.scatter(
                label_data['PC1'], label_data['PC2'],
                label=f"{source} - Label {label}",
                alpha=0.7, s=30
            )
    
    plt.title(f'PCA Visualization (PC1 & PC2: {explained_variance[0] + explained_variance[1]:.2f}% variance)')
    plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("./merging/eda_results/pca_2d_visualization.png")
    plt.close()
    
    # Save PCA loadings to understand feature importance
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=['PC1', 'PC2', 'PC3'],
        index=numeric_cols
    )
    
    loadings.to_csv("./merging/eda_results/pca_loadings.csv")
    
    # Plot PCA loadings for the first two components
    plt.figure(figsize=(12, 10))
    
    # Plot arrows for each feature
    for i, feature in enumerate(numeric_cols):
        plt.arrow(
            0, 0,
            loadings.iloc[i, 0]*0.8, loadings.iloc[i, 1]*0.8,
            head_width=0.01, head_length=0.02, fc='blue', ec='blue'
        )
        plt.text(
            loadings.iloc[i, 0]*0.85, loadings.iloc[i, 1]*0.85,
            feature, color='red', ha='center', va='center'
        )
    
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
    plt.title('PCA Loadings (Feature Contributions)')
    plt.tight_layout()
    plt.savefig("./merging/eda_results/pca_loadings_plot.png")
    plt.close()
    
    return pca, explained_variance

def variance_analysis(df):
    """Analyze variance trends across the dataset"""
    print("\nAnalyzing variance trends...")
    
    # Select key features for variance analysis
    variance_features = [
        "flow_pkt_rate", "flow_byte_rate", "total_pkts", "total_bytes",
        "ave_bytes", "std_bytes", "ave_iat", "std_iat"
    ]
    
    # Variance across index bins (dataset position)
    index_groups = df.groupby('index_bin')
    variance_by_index = pd.DataFrame()
    
    for feature in variance_features:
        variance_by_index[feature] = index_groups[feature].var()
    
    # Create a trend plot
    plt.figure(figsize=(14, 8))
    
    for feature in variance_features:
        # Normalize the variance to make comparison easier
        normalized = variance_by_index[feature] / variance_by_index[feature].max()
        plt.plot(variance_by_index.index, normalized, marker='o', label=feature)
    
    plt.title('Normalized Feature Variance Across Dataset')
    plt.xlabel('Index Bins (0=start, 9=end)')
    plt.ylabel('Normalized Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./merging/eda_results/variance_trend_by_index.png")
    plt.close()
    
    # Create heatmap of variance by index
    plt.figure(figsize=(12, 8))
    sns.heatmap(variance_by_index, cmap='viridis', annot=True, fmt='.2e')
    plt.title('Feature Variance Across Dataset Index')
    plt.xlabel('Features')
    plt.ylabel('Index Bins (0=start, 9=end)')
    plt.tight_layout()
    plt.savefig("./merging/eda_results/variance_heatmap_by_index.png")
    plt.close()
    
    # Compare variance by label
    variance_by_label = pd.DataFrame()
    
    for feature in variance_features:
        variance_by_label[feature] = df.groupby('label')[feature].var()
    
    # Plot variance by label
    plt.figure(figsize=(14, 8))
    
    for label in variance_by_label.index:
        values = variance_by_label.loc[label]
        normalized = values / values.max()
        plt.plot(variance_by_label.columns, normalized, marker='o', label=f'Label {label}')
    
    plt.title('Normalized Feature Variance by Label')
    plt.xlabel('Features')
    plt.ylabel('Normalized Variance')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./merging/eda_results/variance_by_label.png")
    plt.close()
    
    # Check if we have both QUIC.live and CESNET sources before comparing them
    sources = df['source'].unique()
    has_quic = 'QUIC.live' in sources
    has_cesnet = 'CESNET' in sources
    
    if has_quic and has_cesnet:
        print("\nComparing variance between QUIC.live and CESNET datasets...")
        
        # Calculate variance by source
        variance_by_source = pd.DataFrame()
        
        for feature in variance_features:
            variance_by_source[feature] = df.groupby('source')[feature].var()
        
        # Plot variance ratio (QUIC.live / CESNET)
        ratio = variance_by_source.loc['QUIC.live'] / variance_by_source.loc['CESNET']
        
        plt.figure(figsize=(12, 8))
        ratio.plot(kind='bar')
        plt.axhline(y=1, color='red', linestyle='--')
        plt.title('Variance Ratio: QUIC.live / CESNET')
        plt.ylabel('Ratio')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("./merging/eda_results/variance_ratio_between_datasets.png")
        plt.close()
        
        # Create detailed variance comparison between datasets
        plt.figure(figsize=(16, 10))
        
        x = np.arange(len(variance_features))
        width = 0.35
        
        plt.bar(x - width/2, variance_by_source.loc['QUIC.live'], width, label='QUIC.live')
        plt.bar(x + width/2, variance_by_source.loc['CESNET'], width, label='CESNET')
        
        plt.title('Feature Variance Comparison Between Datasets')
        plt.xlabel('Features')
        plt.ylabel('Variance (log scale)')
        plt.yscale('log')
        plt.xticks(x, variance_features, rotation=45)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig("./merging/eda_results/variance_comparison_between_datasets.png")
        plt.close()
    else:
        # If we don't have both specific sources, but do have multiple sources
        if len(sources) > 1:
            print(f"\nComparing variance between available sources: {', '.join(sources)}")
            
            # Calculate variance by source
            variance_by_source = pd.DataFrame()
            
            for feature in variance_features:
                variance_by_source[feature] = df.groupby('source')[feature].var()
            
            # Create a heatmap of variance by source
            plt.figure(figsize=(12, 8))
            sns.heatmap(variance_by_source, cmap='viridis', annot=True, fmt='.2e')
            plt.title('Feature Variance by Source')
            plt.tight_layout()
            plt.savefig("./merging/eda_results/variance_by_source_heatmap.png")
            plt.close()
            
            # Create a bar chart comparing sources
            plt.figure(figsize=(16, 10))
            
            # Set up the plot
            x = np.arange(len(variance_features))
            width = 0.8 / len(sources)
            
            # Plot each source
            for i, source in enumerate(sources):
                offset = width * (i - len(sources)/2 + 0.5)
                plt.bar(x + offset, variance_by_source.loc[source], width, label=source)
            
            plt.title('Feature Variance Comparison Between Sources')
            plt.xlabel('Features')
            plt.ylabel('Variance (log scale)')
            plt.yscale('log')
            plt.xticks(x, variance_features, rotation=45)
            plt.legend()
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig("./merging/eda_results/variance_comparison_all_sources.png")
            plt.close()
        else:
            print("\nSkipping source comparison since only one source is available.")

def feature_importance_analysis(df):
    """Analyze feature importance for distinguishing between classes"""
    print("\nAnalyzing feature importance...")
    
    # Select features (excluding certain metadata)
    exclude_cols = ['label', 'source', 'index_bin']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate F-score for each feature (ANOVA)
    f_scores = {}
    p_values = {}
    
    for feature in feature_cols:
        # Group data by label
        groups = [df[df['label'] == label][feature].values for label in df['label'].unique()]
        
        # Remove any NaN or infinite values
        groups = [g[np.isfinite(g)] for g in groups]
        
        # Skip if any group is empty
        if any(len(g) == 0 for g in groups):
            continue
            
        # Perform ANOVA
        try:
            f_stat, p_val = stats.f_oneway(*groups)
            f_scores[feature] = f_stat
            p_values[feature] = p_val
        except:
            print(f"Warning: Could not calculate F-score for {feature}")
    
    # Create DataFrame with results
    importance_df = pd.DataFrame({
        'Feature': list(f_scores.keys()),
        'F-Score': list(f_scores.values()),
        'P-Value': list(p_values.values())
    })
    
    # Sort by F-score (descending)
    importance_df = importance_df.sort_values('F-Score', ascending=False).reset_index(drop=True)
    
    # Save to CSV
    importance_df.to_csv("./merging/eda_results/feature_importance.csv", index=False)
    
    # Plot top 20 features by importance
    plt.figure(figsize=(14, 10))
    sns.barplot(x='F-Score', y='Feature', data=importance_df.head(20))
    plt.title('Top 20 Features by F-Score (ANOVA)')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig("./merging/eda_results/feature_importance_plot.png")
    plt.close()
    
    # Create interactive plot with plotly
    fig = px.bar(
        importance_df.head(30), 
        x='F-Score', 
        y='Feature',
        text='P-Value',
        orientation='h',
        title='Top 30 Features by Importance (F-Score)',
        labels={'F-Score': 'F-Score (log scale)', 'Feature': ''},
        height=800
    )
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    fig.update_traces(texttemplate='p=%{text:.2e}', textposition='outside')
    fig.update_xaxes(type='log')
    
    fig.write_html("./merging/eda_results/feature_importance_interactive.html")

def create_parallel_coordinates(df):
    """Create parallel coordinates plot for multivariate analysis"""
    print("\nCreating parallel coordinates visualization...")
    
    # Select a subset of important features (modify based on your feature_importance_analysis results)
    parallel_features = [
        "flow_pkt_rate", "flow_byte_rate", "total_pkts", "total_bytes",
        "ave_bytes", "ave_iat", "std_bytes", "std_iat",
        "label"
    ]
    
    parallel_features.append('source')
    
    # Sample the data if it's too large
    if len(df) > 5000:
        sample_df = df.sample(5000, random_state=42)
    else:
        sample_df = df
    
    # Create a copy of the data for visualization
    vis_df = sample_df[parallel_features].copy()
    
    # Scale the features for better visualization
    for feature in parallel_features:
        if feature not in ['label', 'source']:
            # Apply logarithmic scaling for skewed features
            if vis_df[feature].max() / (vis_df[feature].median() + 0.001) > 10:
                vis_df[feature] = np.log1p(vis_df[feature])
            
            # Min-max scaling
            min_val = vis_df[feature].min()
            max_val = vis_df[feature].max()
            vis_df[feature] = (vis_df[feature] - min_val) / (max_val - min_val + 0.001)
    
    # Create the parallel coordinates plot
    fig = px.parallel_coordinates(
        vis_df, 
        color='label',
        dimensions=parallel_features,
        color_continuous_scale=px.colors.diverging.Tealrose,
        title='Parallel Coordinates Plot by Traffic Class and Source'
    )
    
    fig.update_layout(height=600, width=1000)
    fig.write_html("./merging/eda_results/parallel_coordinates.html")

def create_radar_charts(df):
    """Create radar charts comparing feature patterns by label"""
    print("\nCreating radar charts...")
    
    # Select features for radar chart
    radar_features = [
        "flow_pkt_rate", "flow_byte_rate", "total_pkts", "total_bytes",
        "ave_bytes", "std_bytes", "ave_iat", "std_iat"
    ]
    
    # Calculate mean values for each label
    radar_data = {}
    
    for label in df['label'].unique():
        label_data = df[df['label'] == label]
        feature_means = {}
        
        for feature in radar_features:
            # Apply logarithmic scaling for better visualization
            values = label_data[feature]
            if values.max() / (values.median() + 0.001) > 10:
                values = np.log1p(values)
            
            # Scale to 0-1 range
            min_val = values.min()
            max_val = values.max()
            scaled_mean = (values.mean() - min_val) / (max_val - min_val + 0.001)
            
            feature_means[feature] = scaled_mean
        
        radar_data[f'Label {label}'] = feature_means
    
    # Create radar chart
    categories = radar_features
    
    fig = go.Figure()
    
    for label_name, values in radar_data.items():
        fig.add_trace(go.Scatterpolar(
            r=[values[f] for f in categories],
            theta=categories,
            fill='toself',
            name=label_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Feature Patterns by Traffic Class",
        showlegend=True
    )
    
    fig.write_html("./merging/eda_results/radar_chart.html")
    
    # If both datasets are available, create radar charts comparing them
    # Calculate mean values for each source
    source_radar_data = {}
    
    for source in df['source'].unique():
        source_data = df[df['source'] == source]
        feature_means = {}
        
        for feature in radar_features:
            # Apply logarithmic scaling for better visualization
            values = source_data[feature]
            if values.max() / (values.median() + 0.001) > 10:
                values = np.log1p(values)
            
            # Scale to 0-1 range
            min_val = values.min()
            max_val = values.max()
            scaled_mean = (values.mean() - min_val) / (max_val - min_val + 0.001)
            
            feature_means[feature] = scaled_mean
        
        source_radar_data[source] = feature_means
    
    # Create radar chart comparing sources
    fig = go.Figure()
    
    for source, values in source_radar_data.items():
        fig.add_trace(go.Scatterpolar(
            r=[values[f] for f in categories],
            theta=categories,
            fill='toself',
            name=source
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Feature Patterns by Dataset Source",
        showlegend=True
    )
    
    fig.write_html("./merging/eda_results/source_radar_chart.html")

def create_feature_ratio_analysis(df):
    """Analyze the ratio of key features between datasets"""
    
    print("\nCreating feature ratio analysis between datasets...")
    
    # Check if we have both QUIC.live and CESNET sources
    sources = df['source'].unique()
    has_quic = 'QUIC.live' in sources
    has_cesnet = 'CESNET' in sources
    
    if not (has_quic and has_cesnet):
        print("Skipping feature ratio analysis: both QUIC.live and CESNET sources are required")
        print(f"Available sources: {', '.join(sources)}")
        return None
    
    # Select key features for comparison
    ratio_features = [
        "flow_pkt_rate", "flow_byte_rate", "total_pkts", "total_bytes",
        "ave_bytes", "std_bytes", "ave_iat", "std_iat",
        "fwd_pkts", "fwd_bytes", "rev_pkts", "rev_bytes"
    ]
    
    # Calculate mean ratios by label
    ratio_by_label = {}
    
    for label in df['label'].unique():
        quic_data = df[(df['source'] == 'QUIC.live') & (df['label'] == label)]
        cesnet_data = df[(df['source'] == 'CESNET') & (df['label'] == label)]
        
        # Skip if either dataset has no data for this label
        if len(quic_data) == 0 or len(cesnet_data) == 0:
            continue
        
        ratios = {}
        
        for feature in ratio_features:
            quic_mean = quic_data[feature].mean()
            cesnet_mean = cesnet_data[feature].mean()
            
            if cesnet_mean == 0:
                ratios[feature] = np.nan
            else:
                ratios[feature] = quic_mean / cesnet_mean
        
        ratio_by_label[f'Label {label}'] = ratios
    
    # Check if we found any valid comparisons
    if not ratio_by_label:
        print("No valid label comparisons found between datasets")
        return None
    
    # Convert to DataFrame
    ratio_df = pd.DataFrame(ratio_by_label)
    
    # Plot ratio comparison
    plt.figure(figsize=(14, 8))
    
    # Transpose for better visualization
    ratio_df_t = ratio_df.T
    
    # Plot as heatmap
    sns.heatmap(ratio_df_t, cmap='RdBu_r', center=1, annot=True, fmt=".2f",
               cbar_kws={'label': 'QUIC.live / CESNET ratio'})
    
    plt.title('Feature Mean Ratio: QUIC.live / CESNET by Label')
    plt.tight_layout()
    plt.savefig("./merging/eda_results/feature_ratio_by_label.png")
    plt.close()
    
    # Create interactive visualization
    fig = px.imshow(
        ratio_df_t,
        labels=dict(x="Feature", y="Label", color="Ratio"),
        x=ratio_df_t.columns,
        y=ratio_df_t.index,
        title='Feature Mean Ratio: QUIC.live / CESNET by Label',
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=1
    )
    
    fig.update_layout(height=600, width=1000)
    fig.write_html("./merging/eda_results/feature_ratio_interactive.html")
    
    return ratio_df

def run_eda_pipeline():
    """Execute the complete EDA pipeline"""
    print("Starting EDA pipeline...")
    
    # Step 1: Load datasets
    df = load_datasets()
    
    # Step 2: Clean data
    df = data_cleaning(df)
    
    # Create a dictionary to track results
    results = {}
    
    # Step 3: Basic EDA
    results['summary'] = basic_eda(df)
    
    # Step 3.5: Always perform segment transition analysis if segment column exists
    if 'segment' in df.columns:
        print("\nPerforming explicit segment transition analysis...")
        analyze_segment_transitions(df)
    
    # Step 4: Analyze feature distributions
    feature_distributions(df)
    
    # Step 5: Correlation analysis
    correlation_analysis(df)
    
    # Step 6: Dimensionality reduction
    pca, variance = dimensionality_reduction(df)
    results['pca'] = pca
    results['explained_variance'] = variance
    
    # Step 7: Variance analysis
    variance_analysis(df)
    
    # Step 8: Feature importance analysis
    feature_importance_analysis(df)
    
    # Step 9: Create parallel coordinates plot
    create_parallel_coordinates(df)
    
    # Step 10: Create radar charts
    create_radar_charts(df)
    
    # Step 11: Feature ratio analysis (only if both datasets are available)
    ratio_results = create_feature_ratio_analysis(df)
    if ratio_results is not None:
        results['feature_ratios'] = ratio_results
    
    print("\nEDA pipeline completed successfully.")
    print(f"Results saved to ./merging/eda_results/ directory")
    
    return results, df

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Run the EDA pipeline
        results, df = run_eda_pipeline()
        
        # Print final dataset shape for reference
        print(f"\nFinal dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Summarize key findings
        print("\nKey Findings:")
        if 'explained_variance' in results:
            print(f"- PCA explained variance: {sum(results['explained_variance']):.2f}%")
        
        label_counts = df['label'].value_counts(normalize=True) * 100
        print("- Class distribution:")
        for label, pct in label_counts.items():
            print(f"  Label {label}: {pct:.1f}%")
            
    except Exception as e:
        import traceback
        print(f"Error in EDA pipeline: {str(e)}")
        traceback.print_exc()
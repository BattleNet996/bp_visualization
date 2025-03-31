import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from jinja2 import Template
import base64
from io import BytesIO

# Set the color palette based on user's PPT colors
color_palette = [
    (112/255, 48/255, 160/255),  # Purple
    (180/255, 85/255, 170/255),  # Pink-Purple
    (160/255, 85/255, 245/255),  # Lavender
    (190/255, 130/255, 225/255), # Light Purple
    (220/255, 175/255, 225/255)  # Very Light Purple
]

# Load the data
df = pd.read_csv('tableau_ready_data.csv')

# Create folder for images if it doesn't exist
os.makedirs('gen_ai_cs_viz', exist_ok=True)

# Basic statistics function
def get_basic_stats(df):
    """Generate basic statistics from the dataset"""
    # Separate Telco and non-Telco data
    telco_df = df[df['Industry'] == 'Telco']
    non_telco_df = df[df['Industry'] != 'Telco']
    
    # Get unique companies for each industry
    industry_company_counts = non_telco_df[['Industry', 'Company']].drop_duplicates()
    industry_counts = industry_company_counts['Industry'].value_counts().reset_index()
    industry_counts.columns = ['Industry', 'Count']
    
    # Add Telco counts separately at the top
    telco_company_count = telco_df[['Industry', 'Company']].drop_duplicates()['Industry'].value_counts().reset_index()
    telco_company_count.columns = ['Industry', 'Count']
    industry_counts = pd.concat([telco_company_count, industry_counts], ignore_index=True)
    
    # Get unique contact parties
    contact_party_company_counts = df[['Contact_Party', 'Company']].drop_duplicates()
    contact_party_counts = contact_party_company_counts['Contact_Party'].value_counts().reset_index()
    contact_party_counts.columns = ['Contact Party', 'Count']
    
    # Get unique contact types
    contact_type_company_counts = df[['Contact_Type', 'Company']].drop_duplicates()
    contact_type_counts = contact_type_company_counts['Contact_Type'].value_counts().reset_index()
    contact_type_counts.columns = ['Contact Type', 'Count']
    
    return {
        'industry_counts': industry_counts,
        'contact_party_counts': contact_party_counts,
        'contact_type_counts': contact_type_counts,
        'telco_count': telco_company_count['Count'].iloc[0] if not telco_company_count.empty else 0
    }

# Convert matplotlib figure to base64 for embedding in HTML
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

# Function to create spider/radar chart for a given dataframe and category
def create_spider_chart(data, category_col, title, filename=None, include_title=False):
    """Create a spider/radar chart for the given category"""
    # Get all possible category values from the data
    all_cat_values = sorted(data[category_col].unique())
    
    # Calculate the frequency of each category value, only counting unique companies
    cat_company_counts = data[['Company', category_col]].drop_duplicates()
    cat_counts_series = cat_company_counts[category_col].value_counts()
    
    # Create a DataFrame with all possible categories, filling missing values with 0
    cat_counts = pd.DataFrame({'Category': all_cat_values})
    cat_counts['Count'] = cat_counts['Category'].map(lambda x: cat_counts_series.get(x, 0))
    
    # Get category labels if available
    if f'{category_col} Label' in data.columns:
        # Get all unique category values and their labels
        label_mapping = data[[category_col, f'{category_col} Label']].drop_duplicates()
        label_mapping = dict(zip(label_mapping[category_col], label_mapping[f'{category_col} Label']))
        cat_counts['Label'] = cat_counts['Category'].map(label_mapping)
    else:
        cat_counts['Label'] = cat_counts['Category']
    
    # Create radar chart
    fig = plt.figure(figsize=(14, 14))  # Further increased figure size
    ax = fig.add_subplot(111, polar=True)
    
    # Calculate angles for each category
    N = len(cat_counts)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Close the polygon
    cat_counts = pd.concat([cat_counts, cat_counts.iloc[0:1]])
    angles = angles + [angles[0]]
    
    # Process labels for line breaks
    processed_labels = []
    for label in cat_counts['Label'].iloc[:-1]:
        # Add newline for labels containing "&" or "and"
        if ' & ' in label:
            label = label.replace(' & ', '\n& ')
        elif ' and ' in label:
            label = label.replace(' and ', '\nand ')
        # Handle long labels (more than 20 chars)
        elif len(label) > 20:
            words = label.split()
            mid_point = len(words) // 2
            first_half = ' '.join(words[:mid_point])
            second_half = ' '.join(words[mid_point:])
            label = f"{first_half}\n{second_half}"
        processed_labels.append(label)
    
    # Plot data
    ax.plot(angles, cat_counts['Count'], 'o-', color=color_palette[0], linewidth=2, label='Count')
    ax.fill(angles, cat_counts['Count'], color=color_palette[0], alpha=0.25)
    
    # Set category labels with significantly larger font size
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(processed_labels, size=22, fontweight='bold')  # Processed labels with line breaks
    
    # Adjust layout to give more space for labels
    plt.gcf().subplots_adjust(bottom=0.2, top=0.8, left=0.1, right=0.9)
    
    # Customize the chart
    if include_title:
        ax.set_title(title, size=22, color=color_palette[0], y=1.1, fontweight='bold')  # Increased title font size
    
    # Enhance the tick labels (y-axis)
    ax.tick_params(axis='y', labelsize=16)  # Increased y-tick font size
    
    # Add a legend with larger font
    ax.legend(loc='upper right', fontsize=18)
    
    # Draw circle at center for better visualization
    ax.grid(True)
    
    # Save if filename provided
    if filename:
        plt.tight_layout()
        plt.savefig(f'gen_ai_cs_viz/{filename}.png', dpi=300, bbox_inches='tight')
        
    return fig

# Function to create ratio-based spider chart comparing Telco and other industries
def create_ratio_spider_chart(df, category_col, title, filename=None):
    """Create a ratio-based spider chart comparing Telco vs. other industries"""
    # Split data into Telco and non-Telco
    telco_df = df[df['Industry'] == 'Telco']
    non_telco_df = df[df['Industry'] != 'Telco']
    
    # Get all possible category values from the data
    all_cat_values = sorted(df[category_col].unique())
    
    # Calculate ratios for Telco
    telco_company_counts = telco_df[['Company', category_col]].drop_duplicates()
    telco_total = len(telco_company_counts['Company'].unique())
    telco_counts_series = telco_company_counts[category_col].value_counts()
    telco_ratios = {cat: telco_counts_series.get(cat, 0) / telco_total if telco_total > 0 else 0 
                   for cat in all_cat_values}
    
    # Calculate ratios for non-Telco
    non_telco_company_counts = non_telco_df[['Company', category_col]].drop_duplicates()
    non_telco_total = len(non_telco_company_counts['Company'].unique())
    non_telco_counts_series = non_telco_company_counts[category_col].value_counts()
    non_telco_ratios = {cat: non_telco_counts_series.get(cat, 0) / non_telco_total if non_telco_total > 0 else 0 
                       for cat in all_cat_values}
    
    # Create DataFrames for plotting
    ratio_df = pd.DataFrame({
        'Category': all_cat_values,
        'Telco Ratio': [telco_ratios[cat] for cat in all_cat_values],
        'Other Industries Ratio': [non_telco_ratios[cat] for cat in all_cat_values]
    })
    
    # Get category labels if available
    if f'{category_col} Label' in df.columns:
        label_mapping = df[[category_col, f'{category_col} Label']].drop_duplicates()
        label_mapping = dict(zip(label_mapping[category_col], label_mapping[f'{category_col} Label']))
        ratio_df['Label'] = ratio_df['Category'].map(label_mapping)
    else:
        ratio_df['Label'] = ratio_df['Category']
    
    # Create radar chart
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, polar=True)
    
    # Calculate angles for each category
    N = len(ratio_df)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Close the polygons
    ratio_df = pd.concat([ratio_df, ratio_df.iloc[0:1]])
    angles = angles + [angles[0]]
    
    # Enhanced colors with higher contrast
    telco_color = (80/255, 10/255, 140/255)  # Darker purple for Telco
    other_color = (220/255, 70/255, 160/255)  # Brighter pink for Other Industries
    
    # Plot data for Telco
    ax.plot(angles, ratio_df['Telco Ratio'], 'o-', color=telco_color, linewidth=2.5, label='Telco')
    ax.fill(angles, ratio_df['Telco Ratio'], color=telco_color, alpha=0.3)
    
    # Plot data for Other Industries
    ax.plot(angles, ratio_df['Other Industries Ratio'], 'o-', color=other_color, linewidth=2.5, label='Other Industries')
    ax.fill(angles, ratio_df['Other Industries Ratio'], color=other_color, alpha=0.3)
    
    # Process labels for line breaks
    processed_labels = []
    for label in ratio_df['Label'].iloc[:-1]:
        # Add newline for labels containing "&" or "and"
        if ' & ' in label:
            label = label.replace(' & ', '\n& ')
        elif ' and ' in label:
            label = label.replace(' and ', '\nand ')
        # Handle long labels (more than 20 chars)
        elif len(label) > 20:
            words = label.split()
            mid_point = len(words) // 2
            first_half = ' '.join(words[:mid_point])
            second_half = ' '.join(words[mid_point:])
            label = f"{first_half}\n{second_half}"
        processed_labels.append(label)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(processed_labels, size=22, fontweight='bold')
    
    # Adjust layout
    plt.gcf().subplots_adjust(bottom=0.2, top=0.8, left=0.1, right=0.9)
    
    # Add a legend with larger font
    ax.legend(loc='upper right', fontsize=18)
    
    # Draw grid
    ax.grid(True)
    
    # Save if filename provided
    if filename:
        plt.tight_layout()
        plt.savefig(f'gen_ai_cs_viz/{filename}.png', dpi=300, bbox_inches='tight')
        
    return fig

# Create spider charts for all industries by each category
def create_all_industry_spider_charts(df):
    """Create spider charts for all industries by each category"""
    spider_charts = {}
    
    # For each category, create a spider chart for non-Telco industries
    non_telco_df = df[df['Industry'] != 'Telco']
    for cat_num in range(1, 5):
        cat_col = f'Cat {cat_num}'
        title = f'Distribution of {cat_col} Across All Industries Except Telco'
        fig = create_spider_chart(non_telco_df, cat_col, title, f'all_industries_{cat_col}_spider', include_title=False)
        spider_charts[f'all_industries_{cat_col}'] = fig_to_base64(fig)
    
    # Create ratio comparison charts for Telco vs. Other Industries
    for cat_num in range(1, 5):
        cat_col = f'Cat {cat_num}'
        title = f'Ratio Comparison of {cat_col}: Telco vs. Other Industries'
        fig = create_ratio_spider_chart(df, cat_col, title, f'telco_vs_others_{cat_col}_ratio')
        spider_charts[f'telco_vs_others_{cat_col}'] = fig_to_base64(fig)
    
    return spider_charts

# Create spider charts for each industry by each category
def create_per_industry_spider_charts(df):
    """Create spider charts for each industry separately by each category"""
    industry_spider_charts = {}
    
    # First add Telco if it exists
    if 'Telco' in df['Industry'].unique():
        telco_df = df[df['Industry'] == 'Telco']
        industry_spider_charts['Telco'] = {}
        
        for cat_num in range(1, 5):
            cat_col = f'Cat {cat_num}'
            title = f'Distribution of {cat_col} in Telco'
            fig = create_spider_chart(telco_df, cat_col, title, f'Telco_{cat_col}_spider', include_title=False)
            industry_spider_charts['Telco'][f'cat_{cat_num}'] = fig_to_base64(fig)
    
    # Get unique industries (excluding Telco)
    industries = [ind for ind in df['Industry'].unique() if ind != 'Telco']
    
    # For each industry and category, create a spider chart
    for industry in industries:
        industry_df = df[df['Industry'] == industry]
        industry_key = industry.replace(' ', '_').replace('&', 'and')
        
        industry_spider_charts[industry] = {}
        
        for cat_num in range(1, 5):
            cat_col = f'Cat {cat_num}'
            title = f'Distribution of {cat_col} in {industry}'
            fig = create_spider_chart(industry_df, cat_col, title, f'{industry_key}_{cat_col}_spider', include_title=False)
            industry_spider_charts[industry][f'cat_{cat_num}'] = fig_to_base64(fig)
    
    return industry_spider_charts

# Create heatmap to identify correlations between categories
def create_heatmap(df):
    """Create heatmaps to identify correlations between categories"""
    heatmaps = {}
    
    # Create pairs of categories for correlation analysis
    category_pairs = [(i, j) for i in range(1, 5) for j in range(i+1, 5)]
    
    # Create heatmaps for all industries
    for cat1, cat2 in category_pairs:
        # Create a crosstab of the two categories, only counting unique companies
        company_cats = df[['Company', f'Cat {cat1}', f'Cat {cat2}']].drop_duplicates()
        
        # Create the crosstab
        cross_tab = pd.crosstab(company_cats[f'Cat {cat1}'], company_cats[f'Cat {cat2}'])
        
        # Create heatmap
        plt.figure(figsize=(14, 12))
        custom_cmap = LinearSegmentedColormap.from_list("custom_purple", 
                                                       [(1, 1, 1)] + color_palette, N=100)
        
        ax = sns.heatmap(cross_tab, cmap=custom_cmap, annot=True, fmt='d',
                   cbar_kws={'label': 'Frequency'}, annot_kws={"size": 14})
        
        # Get category labels
        cat1_labels = df[[f'Cat {cat1}', f'Cat {cat1} Label']].drop_duplicates()
        cat1_label_dict = dict(zip(cat1_labels[f'Cat {cat1}'], cat1_labels[f'Cat {cat1} Label']))
        
        cat2_labels = df[[f'Cat {cat2}', f'Cat {cat2} Label']].drop_duplicates()
        cat2_label_dict = dict(zip(cat2_labels[f'Cat {cat2}'], cat2_labels[f'Cat {cat2} Label']))
        
        # Set new labels with both number and text
        new_x_labels = [f"{idx} - {cat2_label_dict.get(idx, '')}" for idx in cross_tab.columns]
        new_y_labels = [f"{idx} - {cat1_label_dict.get(idx, '')}" for idx in cross_tab.index]
        
        ax.set_xticklabels(new_x_labels, rotation=45, ha='right', fontsize=16)
        ax.set_yticklabels(new_y_labels, rotation=0, fontsize=16)
        
        plt.title(f'Correlation between Cat {cat1} and Cat {cat2}', fontsize=20, color=color_palette[0], fontweight='bold')
        plt.tight_layout()
        
        # Save the figure
        filename = f'heatmap_cat{cat1}_cat{cat2}'
        plt.savefig(f'gen_ai_cs_viz/{filename}.png', dpi=300, bbox_inches='tight')
        
        heatmaps[f'cat{cat1}_cat{cat2}'] = fig_to_base64(plt.gcf())
        plt.close()
    
    # Create heatmaps specifically for Telco
    telco_df = df[df['Industry'] == 'Telco']
    if not telco_df.empty:
        for cat1, cat2 in category_pairs:
            # Create a crosstab of the two categories, only counting unique companies
            company_cats = telco_df[['Company', f'Cat {cat1}', f'Cat {cat2}']].drop_duplicates()
            
            # Create the crosstab
            cross_tab = pd.crosstab(company_cats[f'Cat {cat1}'], company_cats[f'Cat {cat2}'])
            
            # Create heatmap
            plt.figure(figsize=(14, 12))
            custom_cmap = LinearSegmentedColormap.from_list("custom_purple", 
                                                         [(1, 1, 1)] + color_palette, N=100)
            
            ax = sns.heatmap(cross_tab, cmap=custom_cmap, annot=True, fmt='d',
                     cbar_kws={'label': 'Frequency'}, annot_kws={"size": 14})
            
            # Get category labels
            cat1_labels = df[[f'Cat {cat1}', f'Cat {cat1} Label']].drop_duplicates()
            cat1_label_dict = dict(zip(cat1_labels[f'Cat {cat1}'], cat1_labels[f'Cat {cat1} Label']))
            
            cat2_labels = df[[f'Cat {cat2}', f'Cat {cat2} Label']].drop_duplicates()
            cat2_label_dict = dict(zip(cat2_labels[f'Cat {cat2}'], cat2_labels[f'Cat {cat2} Label']))
            
            # Set new labels with both number and text
            new_x_labels = [f"{idx} - {cat2_label_dict.get(idx, '')}" for idx in cross_tab.columns]
            new_y_labels = [f"{idx} - {cat1_label_dict.get(idx, '')}" for idx in cross_tab.index]
            
            ax.set_xticklabels(new_x_labels, rotation=45, ha='right', fontsize=16)
            ax.set_yticklabels(new_y_labels, rotation=0, fontsize=16)
            
            plt.title(f'Correlation between Cat {cat1} and Cat {cat2} in Telco', fontsize=20, color=color_palette[0], fontweight='bold')
            plt.tight_layout()
            
            # Save the figure
            filename = f'telco_heatmap_cat{cat1}_cat{cat2}'
            plt.savefig(f'gen_ai_cs_viz/{filename}.png', dpi=300, bbox_inches='tight')
            
            heatmaps[f'telco_cat{cat1}_cat{cat2}'] = fig_to_base64(plt.gcf())
            plt.close()
    
    return heatmaps

# Function to generate additional insights about Telco vs other industries
def generate_telco_insights(df):
    """Generate specific insights comparing Telco to other industries"""
    telco_df = df[df['Industry'] == 'Telco']
    non_telco_df = df[df['Industry'] != 'Telco']
    
    insights = {}
    
    # Count unique companies for normalization
    telco_companies = telco_df['Company'].nunique()
    non_telco_companies = non_telco_df['Company'].nunique()
    
    # Get unique company-category combinations for accurate counting
    for cat_num in range(1, 5):
        cat_col = f'Cat {cat_num}'
        label_col = f'Cat {cat_num} Label'
        
        # Telco top categories
        telco_cat_company = telco_df[['Company', cat_col, label_col]].drop_duplicates()
        if not telco_cat_company.empty:
            telco_top_cats = telco_cat_company[label_col].value_counts()
            insights[f'telco_cat{cat_num}_top'] = telco_top_cats.index[0] if len(telco_top_cats) > 0 else "N/A"
            
            # Calculate percentage for top category
            top_cat_count = telco_top_cats.iloc[0] if len(telco_top_cats) > 0 else 0
            insights[f'telco_cat{cat_num}_top_percent'] = round((top_cat_count / telco_companies) * 100, 1) if telco_companies > 0 else 0
        else:
            insights[f'telco_cat{cat_num}_top'] = "N/A"
            insights[f'telco_cat{cat_num}_top_percent'] = 0
        
        # Non-Telco top categories
        non_telco_cat_company = non_telco_df[['Company', cat_col, label_col]].drop_duplicates()
        if not non_telco_cat_company.empty:
            non_telco_top_cats = non_telco_cat_company[label_col].value_counts()
            insights[f'non_telco_cat{cat_num}_top'] = non_telco_top_cats.index[0] if len(non_telco_top_cats) > 0 else "N/A"
            
            # Calculate percentage for top category
            top_cat_count = non_telco_top_cats.iloc[0] if len(non_telco_top_cats) > 0 else 0
            insights[f'non_telco_cat{cat_num}_top_percent'] = round((top_cat_count / non_telco_companies) * 100, 1) if non_telco_companies > 0 else 0
        else:
            insights[f'non_telco_cat{cat_num}_top'] = "N/A"
            insights[f'non_telco_cat{cat_num}_top_percent'] = 0
    
    # Find most distinctive categories for Telco compared to other industries
    for cat_num in range(1, 5):
        cat_col = f'Cat {cat_num}'
        label_col = f'Cat {cat_num} Label'
        
        telco_cat_company = telco_df[['Company', cat_col, label_col]].drop_duplicates()
        non_telco_cat_company = non_telco_df[['Company', cat_col, label_col]].drop_duplicates()
        
        # Calculate frequencies
        if not telco_cat_company.empty and not non_telco_cat_company.empty:
            telco_freqs = telco_cat_company[cat_col].value_counts(normalize=True)
            non_telco_freqs = non_telco_cat_company[cat_col].value_counts(normalize=True)
            
            # Find categories with biggest difference (Telco > Other)
            differences = {}
            for cat in telco_freqs.index:
                telco_freq = telco_freqs.get(cat, 0)
                non_telco_freq = non_telco_freqs.get(cat, 0)
                differences[cat] = telco_freq - non_telco_freq
            
            # Get the category with biggest positive difference (more in Telco)
            if differences:
                most_distinctive_cat = max(differences.items(), key=lambda x: x[1])
                if most_distinctive_cat[1] > 0:  # Only if Telco has more
                    cat_label = telco_df[telco_df[cat_col] == most_distinctive_cat[0]][label_col].iloc[0] if not telco_df[telco_df[cat_col] == most_distinctive_cat[0]].empty else "N/A"
                    insights[f'telco_distinctive_cat{cat_num}'] = cat_label
                    insights[f'telco_distinctive_cat{cat_num}_diff'] = round(most_distinctive_cat[1] * 100, 1)  # Convert to percentage
                else:
                    insights[f'telco_distinctive_cat{cat_num}'] = "None"
                    insights[f'telco_distinctive_cat{cat_num}_diff'] = 0
            else:
                insights[f'telco_distinctive_cat{cat_num}'] = "None"
                insights[f'telco_distinctive_cat{cat_num}_diff'] = 0
    
    return insights

# Generate all visualizations
basic_stats = get_basic_stats(df)
all_industry_spider_charts = create_all_industry_spider_charts(df)
per_industry_spider_charts = create_per_industry_spider_charts(df)
heatmaps = create_heatmap(df)
telco_insights = generate_telco_insights(df)

# Save the results for the HTML generator
np.save('gen_ai_cs_viz/basic_stats.npy', basic_stats)
np.save('gen_ai_cs_viz/all_industry_spider_charts.npy', all_industry_spider_charts)
np.save('gen_ai_cs_viz/per_industry_spider_charts.npy', per_industry_spider_charts)
np.save('gen_ai_cs_viz/heatmaps.npy', heatmaps)
np.save('gen_ai_cs_viz/telco_insights.npy', telco_insights)

print("Data analysis and visualization complete. Now generating HTML...")

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
    # Get unique companies for each industry
    industry_company_counts = df[['Industry', 'Company']].drop_duplicates()
    industry_counts = industry_company_counts['Industry'].value_counts().reset_index()
    industry_counts.columns = ['Industry', 'Count']
    
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
        'contact_type_counts': contact_type_counts
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
def create_spider_chart(data, category_col, title, filename=None):
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
    
    # Plot data
    ax.plot(angles, cat_counts['Count'], 'o-', color=color_palette[0], linewidth=2, label='Count')
    ax.fill(angles, cat_counts['Count'], color=color_palette[0], alpha=0.25)
    
    # Set category labels with significantly larger font size
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_counts['Label'].iloc[:-1], size=22, fontweight='bold')  # Increased font size and made bold
    
    # Adjust layout to give more space for labels
    plt.gcf().subplots_adjust(bottom=0.2, top=0.8, left=0.1, right=0.9)
    
    # Customize the chart
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

# Create spider charts for all industries by each category
def create_all_industry_spider_charts(df):
    """Create spider charts for all industries by each category"""
    spider_charts = {}
    
    # For each category, create a spider chart
    for cat_num in range(1, 5):
        cat_col = f'Cat {cat_num}'
        title = f'Distribution of {cat_col} Across All Industries'
        fig = create_spider_chart(df, cat_col, title, f'all_industries_{cat_col}_spider')
        spider_charts[f'all_industries_{cat_col}'] = fig_to_base64(fig)
    
    return spider_charts

# Create spider charts for each industry by each category
def create_per_industry_spider_charts(df):
    """Create spider charts for each industry separately by each category"""
    industry_spider_charts = {}
    
    # Get unique industries
    industries = df['Industry'].unique()
    
    # For each industry and category, create a spider chart
    for industry in industries:
        industry_df = df[df['Industry'] == industry]
        industry_key = industry.replace(' ', '_').replace('&', 'and')
        
        industry_spider_charts[industry] = {}
        
        for cat_num in range(1, 5):
            cat_col = f'Cat {cat_num}'
            title = f'Distribution of {cat_col} in {industry}'
            fig = create_spider_chart(industry_df, cat_col, title, f'{industry_key}_{cat_col}_spider')
            industry_spider_charts[industry][f'cat_{cat_num}'] = fig_to_base64(fig)
    
    return industry_spider_charts

# Create heatmap to identify correlations between categories
def create_heatmap(df):
    """Create heatmaps to identify correlations between categories"""
    heatmaps = {}
    
    # Create pairs of categories for correlation analysis
    category_pairs = [(i, j) for i in range(1, 5) for j in range(i+1, 5)]
    
    for cat1, cat2 in category_pairs:
        # Create a crosstab of the two categories, only counting unique companies
        # Get unique company-category combinations
        company_cats = df[['Company', f'Cat {cat1}', f'Cat {cat2}']].drop_duplicates()
        
        # Create the crosstab
        cross_tab = pd.crosstab(company_cats[f'Cat {cat1}'], company_cats[f'Cat {cat2}'])
        
        # Create heatmap
        plt.figure(figsize=(14, 12))  # Increased figure size
        custom_cmap = LinearSegmentedColormap.from_list("custom_purple", 
                                                       [(1, 1, 1)] + color_palette, N=100)
        
        ax = sns.heatmap(cross_tab, cmap=custom_cmap, annot=True, fmt='d',
                   cbar_kws={'label': 'Frequency'}, annot_kws={"size": 14})  # Increased annotation size
        
        # Get category labels
        cat1_labels = df[[f'Cat {cat1}', f'Cat {cat1} Label']].drop_duplicates()
        cat1_label_dict = dict(zip(cat1_labels[f'Cat {cat1}'], cat1_labels[f'Cat {cat1} Label']))
        
        cat2_labels = df[[f'Cat {cat2}', f'Cat {cat2} Label']].drop_duplicates()
        cat2_label_dict = dict(zip(cat2_labels[f'Cat {cat2}'], cat2_labels[f'Cat {cat2} Label']))
        
        # Set new labels with both number and text
        new_x_labels = [f"{idx} - {cat2_label_dict.get(idx, '')}" for idx in cross_tab.columns]
        new_y_labels = [f"{idx} - {cat1_label_dict.get(idx, '')}" for idx in cross_tab.index]
        
        ax.set_xticklabels(new_x_labels, rotation=45, ha='right', fontsize=16)  # Increased font size
        ax.set_yticklabels(new_y_labels, rotation=0, fontsize=16)  # Increased font size
        
        plt.title(f'Correlation between Cat {cat1} and Cat {cat2}', fontsize=20, color=color_palette[0], fontweight='bold')  # Increased title font size
        plt.tight_layout()
        
        # Save the figure
        filename = f'heatmap_cat{cat1}_cat{cat2}'
        plt.savefig(f'gen_ai_cs_viz/{filename}.png', dpi=300, bbox_inches='tight')
        
        heatmaps[f'cat{cat1}_cat{cat2}'] = fig_to_base64(plt.gcf())
        plt.close()
    
    return heatmaps

# Generate all visualizations
basic_stats = get_basic_stats(df)
all_industry_spider_charts = create_all_industry_spider_charts(df)
per_industry_spider_charts = create_per_industry_spider_charts(df)
heatmaps = create_heatmap(df)

# Save the results for the HTML generator
np.save('gen_ai_cs_viz/basic_stats.npy', basic_stats)
np.save('gen_ai_cs_viz/all_industry_spider_charts.npy', all_industry_spider_charts)
np.save('gen_ai_cs_viz/per_industry_spider_charts.npy', per_industry_spider_charts)
np.save('gen_ai_cs_viz/heatmaps.npy', heatmaps)

print("Data analysis and visualization complete. Now generating HTML...")

# Now let's generate the HTML template

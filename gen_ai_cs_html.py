import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from jinja2 import Template

# Load the visualization data
basic_stats = np.load('gen_ai_cs_viz/basic_stats.npy', allow_pickle=True).item()
all_industry_spider_charts = np.load('gen_ai_cs_viz/all_industry_spider_charts.npy', allow_pickle=True).item()
per_industry_spider_charts = np.load('gen_ai_cs_viz/per_industry_spider_charts.npy', allow_pickle=True).item()
heatmaps = np.load('gen_ai_cs_viz/heatmaps.npy', allow_pickle=True).item()
telco_insights = np.load('gen_ai_cs_viz/telco_insights.npy', allow_pickle=True).item()

# Load the original data for additional insights
df = pd.read_csv('tableau_ready_data.csv')

# Generate insights about the data
def generate_insights():
    insights = {}
    
    # Separate Telco and non-Telco data
    telco_df = df[df['Industry'] == 'Telco']
    non_telco_df = df[df['Industry'] != 'Telco']
    
    # Get unique company-category combinations for accurate counting
    # For all industries except Telco
    cat1_company = non_telco_df[['Company', 'Cat 1', 'Cat 1 Label']].drop_duplicates()
    cat2_company = non_telco_df[['Company', 'Cat 2', 'Cat 2 Label']].drop_duplicates()
    cat3_company = non_telco_df[['Company', 'Cat 3', 'Cat 3 Label']].drop_duplicates()
    cat4_company = non_telco_df[['Company', 'Cat 4', 'Cat 4 Label']].drop_duplicates()
    
    # For Telco industry
    telco_cat1_company = telco_df[['Company', 'Cat 1', 'Cat 1 Label']].drop_duplicates()
    telco_cat2_company = telco_df[['Company', 'Cat 2', 'Cat 2 Label']].drop_duplicates()
    telco_cat3_company = telco_df[['Company', 'Cat 3', 'Cat 3 Label']].drop_duplicates()
    telco_cat4_company = telco_df[['Company', 'Cat 4', 'Cat 4 Label']].drop_duplicates()
    
    # Categories frequency insights for all industries except Telco
    insights['cat1_top'] = cat1_company['Cat 1 Label'].value_counts().index[0] if not cat1_company.empty else "N/A"
    insights['cat2_top'] = cat2_company['Cat 2 Label'].value_counts().index[0] if not cat2_company.empty else "N/A"
    insights['cat3_top'] = cat3_company['Cat 3 Label'].value_counts().index[0] if not cat3_company.empty else "N/A"
    insights['cat4_top'] = cat4_company['Cat 4 Label'].value_counts().index[0] if not cat4_company.empty else "N/A"
    
    # Categories frequency insights for Telco
    insights['telco_cat1_top'] = telco_cat1_company['Cat 1 Label'].value_counts().index[0] if not telco_cat1_company.empty else "N/A"
    insights['telco_cat2_top'] = telco_cat2_company['Cat 2 Label'].value_counts().index[0] if not telco_cat2_company.empty else "N/A"
    insights['telco_cat3_top'] = telco_cat3_company['Cat 3 Label'].value_counts().index[0] if not telco_cat3_company.empty else "N/A"
    insights['telco_cat4_top'] = telco_cat4_company['Cat 4 Label'].value_counts().index[0] if not telco_cat4_company.empty else "N/A"
    
    # Industry specific insights
    industry_insights = {}
    for industry in df['Industry'].unique():
        industry_df = df[df['Industry'] == industry]
        
        # Get unique companies for this industry with each category
        ind_cat1_company = industry_df[['Company', 'Cat 1', 'Cat 1 Label']].drop_duplicates()
        ind_cat2_company = industry_df[['Company', 'Cat 2', 'Cat 2 Label']].drop_duplicates()
        ind_cat3_company = industry_df[['Company', 'Cat 3', 'Cat 3 Label']].drop_duplicates()
        ind_cat4_company = industry_df[['Company', 'Cat 4', 'Cat 4 Label']].drop_duplicates()
        
        # Only add to insights if there's data for this industry
        if not ind_cat1_company.empty and not ind_cat2_company.empty and not ind_cat3_company.empty and not ind_cat4_company.empty:
            industry_insights[industry] = {
                'cat1_top': ind_cat1_company['Cat 1 Label'].value_counts().index[0],
                'cat2_top': ind_cat2_company['Cat 2 Label'].value_counts().index[0],
                'cat3_top': ind_cat3_company['Cat 3 Label'].value_counts().index[0],
                'cat4_top': ind_cat4_company['Cat 4 Label'].value_counts().index[0],
            }
    
    insights['industry_insights'] = industry_insights
    
    # Correlation insights
    cat_pairs = [(i, j) for i in range(1, 5) for j in range(i+1, 5)]
    correlation_insights = {}
    
    for cat1, cat2 in cat_pairs:
        # Get unique company combinations for these two categories
        company_cats = df[['Company', f'Cat {cat1}', f'Cat {cat2}']].drop_duplicates()
        
        # Create crosstab with unique company-category combinations
        cross_tab = pd.crosstab(company_cats[f'Cat {cat1}'], company_cats[f'Cat {cat2}'])
        
        if not cross_tab.empty:
            max_idx = np.unravel_index(cross_tab.values.argmax(), cross_tab.shape)
            
            # Get the labels for the maximum correlation
            cat1_val = cross_tab.index[max_idx[0]]
            cat2_val = cross_tab.columns[max_idx[1]]
            
            # Get the corresponding labels
            cat1_labels = df[df[f'Cat {cat1}'] == cat1_val][f'Cat {cat1} Label'].drop_duplicates()
            cat2_labels = df[df[f'Cat {cat2}'] == cat2_val][f'Cat {cat2} Label'].drop_duplicates()
            
            if not cat1_labels.empty and not cat2_labels.empty:
                cat1_label = cat1_labels.iloc[0]
                cat2_label = cat2_labels.iloc[0]
                
                correlation_insights[f'cat{cat1}_cat{cat2}'] = {
                    'cat1_val': cat1_val,
                    'cat2_val': cat2_val,
                    'cat1_label': cat1_label,
                    'cat2_label': cat2_label,
                    'frequency': cross_tab.values.max()
                }
    
    insights['correlation_insights'] = correlation_insights
    
    # Add Telco vs Other industries insights
    insights.update(telco_insights)
    
    return insights

# Generate insights
insights = generate_insights()

# Create the HTML template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GenAI in Customer Service Analysis</title>
    <style>
        :root {
            --primary-color: rgb(112, 48, 160);
            --secondary-color: rgb(180, 85, 170);
            --tertiary-color: rgb(160, 85, 245);
            --quaternary-color: rgb(190, 130, 225);
            --quinary-color: rgb(220, 175, 225);
            --bg-color: #f9f6fd;
            --text-color: #333;
            --section-bg: white;
            --card-shadow: 0 4px 6px rgba(112, 48, 160, 0.1);
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--bg-color);
            margin: 0;
            padding: 0;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: linear-gradient(135deg, var(--primary-color), var(--tertiary-color));
            color: white;
            padding: 30px 0;
            text-align: center;
            border-radius: 0 0 20px 20px;
            margin-bottom: 30px;
        }
        
        h1, h2, h3, h4 {
            color: var(--primary-color);
        }
        
        header h1 {
            margin: 0;
            color: white;
            font-size: 2.5em;
        }
        
        header p {
            margin: 10px 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        section {
            background: var(--section-bg);
            margin-bottom: 30px;
            padding: 25px;
            border-radius: 10px;
            box-shadow: var(--card-shadow);
        }
        
        .section-title {
            border-bottom: 2px solid var(--quaternary-color);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        .viz-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
            gap: 20px;
            margin-top: 20px;
        }
        
        .viz-card {
            flex: 1 1 45%;
            min-width: 300px;
            margin-bottom: 20px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: var(--card-shadow);
            transition: transform 0.3s ease;
        }
        
        .viz-card:hover {
            transform: translateY(-5px);
        }
        
        .viz-card img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .viz-card-content {
            padding: 15px;
        }
        
        .insight-box {
            background-color: rgba(190, 130, 225, 0.1);
            border-left: 4px solid var(--tertiary-color);
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }
        
        .basic-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: space-between;
        }
        
        .stat-card {
            flex: 1 1 30%;
            min-width: 250px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: var(--card-shadow);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        
        table, th, td {
            border: 1px solid #eee;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: var(--quinary-color);
            color: var(--primary-color);
        }
        
        tr:nth-child(even) {
            background-color: #f9f6fd;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            margin-top: 30px;
            background: linear-gradient(135deg, var(--tertiary-color), var(--quaternary-color));
            color: white;
            border-radius: 10px;
        }
        
        @media (max-width: 768px) {
            .viz-card {
                flex: 1 1 100%;
            }
            
            .stat-card {
                flex: 1 1 100%;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>GenAI in Customer Service Analysis</h1>
            <p>Visualization and Insights for AI Tools in Customer Service Across Industries</p>
        </div>
    </header>
    
    <div class="container">
        <section>
            <h2 class="section-title">Project Overview</h2>
            <p>This analysis explores the application of GenAI tools in customer service across different industries. The data has been categorized along four dimensions:</p>
            <ol>
                <li><strong>Problem-Solution:</strong> What business challenge does AI solve?</li>
                <li><strong>AI Technology:</strong> What key technology is the use case built on?</li>
                <li><strong>Customer Journey:</strong> Where does AI enhance customer interactions?</li>
                <li><strong>Data Modality:</strong> What types of data does the solution work with?</li>
            </ol>
            
            <div class="insight-box">
                <h3>Key Findings</h3>
                <p>The most common use case across all industries is <strong>{{ insights.cat1_top }}</strong> for Problem-Solution, utilizing <strong>{{ insights.cat2_top }}</strong> technology, primarily enhancing the <strong>{{ insights.cat3_top }}</strong> stage of the customer journey, with <strong>{{ insights.cat4_top }}</strong> being the dominant data modality.</p>
            </div>
        </section>
        
        <section>
            <h2 class="section-title">Basic Statistics</h2>
            <div class="basic-stats">
                <div class="stat-card">
                    <h3>Industry Distribution</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Industry</th>
                                <th>Count</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for _, row in basic_stats.industry_counts.iterrows() %}
                            <tr>
                                <td>{{ row['Industry'] }}</td>
                                <td>{{ row['Count'] }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                
                <div class="stat-card">
                    <h3>Contact Party</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Contact Party</th>
                                <th>Count</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for _, row in basic_stats.contact_party_counts.iterrows() %}
                            <tr>
                                <td>{{ row['Contact Party'] }}</td>
                                <td>{{ row['Count'] }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                
                <div class="stat-card">
                    <h3>Contact Type</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Contact Type</th>
                                <th>Count</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for _, row in basic_stats.contact_type_counts.iterrows() %}
                            <tr>
                                <td>{{ row['Contact Type'] }}</td>
                                <td>{{ row['Count'] }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                
                <div class="stat-card">
                    <h3>Telco Industry Statistics</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Category</th>
                                <th>Top Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Problem-Solution</td>
                                <td>{{ insights.telco_cat1_top }}</td>
                            </tr>
                            <tr>
                                <td>AI Technology</td>
                                <td>{{ insights.telco_cat2_top }}</td>
                            </tr>
                            <tr>
                                <td>Customer Journey</td>
                                <td>{{ insights.telco_cat3_top }}</td>
                            </tr>
                            <tr>
                                <td>Data Modality</td>
                                <td>{{ insights.telco_cat4_top }}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>
        
        <section>
            <h2 class="section-title">1. Overall Summary - Spider Web Charts by Category</h2>
            <p>These spider web charts show the distribution of each category across all industries except Telco in the dataset.</p>
            
            <div class="viz-container">
                {% for cat_num in range(1, 5) %}
                <div class="viz-card">
                    <img src="data:image/png;base64,{{ all_industry_spider_charts['all_industries_Cat ' + cat_num|string] }}" alt="Spider Chart for Category {{ cat_num }}">
                    <div class="viz-card-content">
                        <p>Frequency distribution of different values in Category {{ cat_num }} across all industries except Telco.</p>
                    </div>
                </div>
                {% endfor %}
            </div>
            
            <div class="insight-box">
                <h3>Overall Category Distribution Insights</h3>
                <p>Across all industries (excluding Telco), we observe that "{{ insights.cat1_top }}" dominates in Problem-Solution (Category 1), while "{{ insights.cat2_top }}" is the most prevalent AI Technology (Category 2). For Customer Journey stages (Category 3), "{{ insights.cat3_top }}" shows the highest presence, and "{{ insights.cat4_top }}" is the most common Data Modality (Category 4).</p>
            </div>
            
            <h3>Telco vs. Other Industries Comparison</h3>
            <p>These ratio-based spider charts compare the distribution of categories between Telco and other industries, normalized by the number of use cases.</p>
            
            <div class="viz-container">
                {% for cat_num in range(1, 5) %}
                <div class="viz-card">
                    <img src="data:image/png;base64,{{ all_industry_spider_charts['telco_vs_others_Cat ' + cat_num|string] }}" alt="Telco vs Others Ratio Chart for Category {{ cat_num }}">
                    <div class="viz-card-content">
                        <p>Ratio comparison of Category {{ cat_num }} values between Telco (purple) and other industries (pink).</p>
                    </div>
                </div>
                {% endfor %}
            </div>
            
            <div class="insight-box">
                <h3>Telco vs. Other Industries Insights</h3>
                <p>When comparing Telco to other industries, we observe that "{{ insights.telco_cat1_top }}" is the most prevalent in Problem-Solution for Telco (vs. "{{ insights.non_telco_cat1_top }}" for other industries). In AI Technology, Telco predominantly uses "{{ insights.telco_cat2_top }}" (vs. "{{ insights.non_telco_cat2_top }}" elsewhere).</p>
                <p>For Customer Journey stages, Telco focuses on "{{ insights.telco_cat3_top }}" (vs. "{{ insights.non_telco_cat3_top }}" in other industries), while "{{ insights.telco_cat4_top }}" is the dominant Data Modality in Telco (vs. "{{ insights.non_telco_cat4_top }}" in other sectors).</p>
                
                <h4>Most Distinctive Aspects of Telco</h4>
                <ul>
                    {% for cat_num in range(1, 5) %}
                    {% if insights['telco_distinctive_cat' + cat_num|string] != "None" and insights['telco_distinctive_cat' + cat_num|string + '_diff'] > 0 %}
                    <li><strong>Category {{ cat_num }}:</strong> "{{ insights['telco_distinctive_cat' + cat_num|string] }}" is {{ insights['telco_distinctive_cat' + cat_num|string + '_diff'] }}% more common in Telco than in other industries.</li>
                    {% endif %}
                    {% endfor %}
                </ul>
            </div>
        </section>
        
        <section>
            <h2 class="section-title">2. Industry-Specific Spider Web Charts</h2>
            
            {% if 'Telco' in per_industry_spider_charts %}
            <h3>Telco</h3>
            <div class="viz-container">
                {% for cat_num in range(1, 5) %}
                <div class="viz-card">
                    <img src="data:image/png;base64,{{ per_industry_spider_charts['Telco']['cat_' + cat_num|string] }}" alt="Spider Chart for Telco - Category {{ cat_num }}">
                    <div class="viz-card-content">
                        <p>Distribution of Category {{ cat_num }} values specific to the Telco industry.</p>
                    </div>
                </div>
                {% endfor %}
            </div>
            
            <div class="insight-box">
                <h3>Telco Industry Insights</h3>
                <p>In the Telco industry, the dominant Problem-Solution (Category 1) is "{{ insights.industry_insights['Telco'].cat1_top }}", utilizing "{{ insights.industry_insights['Telco'].cat2_top }}" technology (Category 2). This industry primarily focuses on the "{{ insights.industry_insights['Telco'].cat3_top }}" stage of the customer journey (Category 3), with "{{ insights.industry_insights['Telco'].cat4_top }}" as the primary data modality (Category 4).</p>
            </div>
            {% endif %}
            
            {% for industry, charts in per_industry_spider_charts.items() %}
            {% if industry != 'Telco' %}
            <h3>{{ industry }}</h3>
            <div class="viz-container">
                {% for cat_num in range(1, 5) %}
                <div class="viz-card">
                    <img src="data:image/png;base64,{{ charts['cat_' + cat_num|string] }}" alt="Spider Chart for {{ industry }} - Category {{ cat_num }}">
                    <div class="viz-card-content">
                        <p>Distribution of Category {{ cat_num }} values specific to the {{ industry }} industry.</p>
                    </div>
                </div>
                {% endfor %}
            </div>
            
            <div class="insight-box">
                <h3>{{ industry }} Industry Insights</h3>
                <p>In the {{ industry }} industry, the dominant Problem-Solution (Category 1) is "{{ insights.industry_insights[industry].cat1_top }}", utilizing "{{ insights.industry_insights[industry].cat2_top }}" technology (Category 2). This industry primarily focuses on the "{{ insights.industry_insights[industry].cat3_top }}" stage of the customer journey (Category 3), with "{{ insights.industry_insights[industry].cat4_top }}" as the primary data modality (Category 4).</p>
            </div>
            {% endif %}
            {% endfor %}
        </section>
        
        <section>
            <h2 class="section-title">3. Correlation Analysis - Heatmaps</h2>
            <p>These heatmaps show the correlations between different categories, helping to identify patterns across the dataset.</p>
            
            <h3>All Industries</h3>
            <div class="viz-container">
                {% for key, img in heatmaps.items() %}
                {% if not key.startswith('telco_') %}
                <div class="viz-card">
                    <img src="data:image/png;base64,{{ img }}" alt="Heatmap for {{ key }}">
                    <div class="viz-card-content">
                        {% set cats = key.split('_') %}
                        <p>Correlation between {{ cats[0]|replace('cat', 'Category ') }} and {{ cats[1]|replace('cat', 'Category ') }} across all industries.</p>
                    </div>
                </div>
                {% endif %}
                {% endfor %}
            </div>
            
            <h3>Telco Industry</h3>
            <div class="viz-container">
                {% for key, img in heatmaps.items() %}
                {% if key.startswith('telco_') %}
                <div class="viz-card">
                    <img src="data:image/png;base64,{{ img }}" alt="Heatmap for {{ key }}">
                    <div class="viz-card-content">
                        {% set cats = key.replace('telco_', '').split('_') %}
                        <p>Correlation between {{ cats[0]|replace('cat', 'Category ') }} and {{ cats[1]|replace('cat', 'Category ') }} in the Telco industry.</p>
                    </div>
                </div>
                {% endif %}
                {% endfor %}
            </div>
            
            <div class="insight-box">
                <h3>Correlation Insights</h3>
                {% for key, insight in insights.correlation_insights.items() %}
                {% set cats = key.split('_') %}
                <p><strong>{{ cats[0]|replace('cat', 'Category ') }} and {{ cats[1]|replace('cat', 'Category ') }}:</strong> The strongest correlation is between "{{ insight.cat1_label }}" ({{ cats[0]|replace('cat', 'Cat ') }}) and "{{ insight.cat2_label }}" ({{ cats[1]|replace('cat', 'Cat ') }}), appearing {{ insight.frequency }} times in the dataset.</p>
                {% endfor %}
            </div>
        </section>
        
        <section>
            <h2 class="section-title">Conclusions and Recommendations</h2>
            <p>Based on the analysis of GenAI tools in customer service across different industries, with a special focus on the Telco industry, several key patterns and opportunities emerge:</p>
            
            <div class="insight-box">
                <h3>Key Findings for Telco Industry</h3>
                <ol>
                    <li><strong>Telco's Distinctive Use Cases:</strong> Within the telecommunications industry, "{{ insights.telco_cat1_top }}" is the predominant business challenge being addressed by GenAI, compared to "{{ insights.non_telco_cat1_top }}" in other industries.</li>
                    
                    <li><strong>Technology Adoption in Telco:</strong> "{{ insights.telco_cat2_top }}" shows the highest adoption rate in the Telco sector, indicating its particular effectiveness for telecommunications customer service challenges.</li>
                    
                    <li><strong>Customer Journey Focus in Telco:</strong> Telco GenAI applications mainly target the "{{ insights.telco_cat3_top }}" stage, which differs from the "{{ insights.non_telco_cat3_top }}" focus in other industries, reflecting the unique customer interaction patterns in telecommunications.</li>
                    
                    <li><strong>Data Modality in Telco:</strong> "{{ insights.telco_cat4_top }}" is the primary modality used in Telco GenAI applications, compared to "{{ insights.non_telco_cat4_top }}" elsewhere, highlighting the specific data types that telecommunication customer service relies on.</li>
                    
                    {% for cat_num in range(1, 5) %}
                    {% if insights['telco_distinctive_cat' + cat_num|string] != "None" and insights['telco_distinctive_cat' + cat_num|string + '_diff'] > 0 %}
                    <li><strong>Distinctive Category {{ cat_num }} Feature:</strong> "{{ insights['telco_distinctive_cat' + cat_num|string] }}" is {{ insights['telco_distinctive_cat' + cat_num|string + '_diff'] }}% more prevalent in Telco than in other industries, representing a unique characteristic of AI applications in telecommunications.</li>
                    {% endif %}
                    {% endfor %}
                </ol>
                
                <h3>Comparison Between Telco and Other Industries</h3>
                <p>When comparing telecommunications to other sectors, we observe several important distinctions in how GenAI is deployed for customer service:</p>
                <ul>
                    <li>Telco's ratio of "{{ insights.telco_cat1_top }}" use cases is {% if insights.telco_cat1_top == insights.non_telco_cat1_top %}similar to{% elif insights.telco_cat1_top_percent > insights.non_telco_cat1_top_percent %}higher than{% else %}lower than{% endif %} other industries ({{ insights.telco_cat1_top_percent }}% vs. {{ insights.non_telco_cat1_top_percent }}%).</li>
                    
                    <li>For AI technology, Telco's preference for "{{ insights.telco_cat2_top }}" is {% if insights.telco_cat2_top == insights.non_telco_cat2_top %}aligned with{% elif insights.telco_cat2_top_percent > insights.non_telco_cat2_top_percent %}stronger than{% else %}weaker than{% endif %} other sectors ({{ insights.telco_cat2_top_percent }}% vs. {{ insights.non_telco_cat2_top_percent }}%).</li>
                    
                    <li>In the customer journey, Telco's focus on "{{ insights.telco_cat3_top }}" stages is {% if insights.telco_cat3_top == insights.non_telco_cat3_top %}consistent with{% elif insights.telco_cat3_top_percent > insights.non_telco_cat3_top_percent %}more intense than{% else %}less pronounced than{% endif %} other industries ({{ insights.telco_cat3_top_percent }}% vs. {{ insights.non_telco_cat3_top_percent }}%).</li>
                    
                    <li>For data modalities, Telco's use of "{{ insights.telco_cat4_top }}" is {% if insights.telco_cat4_top == insights.non_telco_cat4_top %}comparable to{% elif insights.telco_cat4_top_percent > insights.non_telco_cat4_top_percent %}higher than{% else %}lower than{% endif %} other sectors ({{ insights.telco_cat4_top_percent }}% vs. {{ insights.non_telco_cat4_top_percent }}%).</li>
                </ul>
                
                <h3>Recommendations for Telco Customer Service</h3>
                <ol>
                    <li><strong>Prioritize AI Solutions:</strong> Focus on implementing AI technologies for "{{ insights.telco_cat1_top }}" and "Customer Support & Query Resolution" as these represent the highest-value opportunities in telecommunications.</li>
                    
                    <li><strong>Technology Investment:</strong> Continue investing in "{{ insights.telco_cat2_top }}" capabilities while exploring complementary technologies that can enhance customer service operations.</li>
                    
                    <li><strong>Customer Journey Enhancement:</strong> Strengthen AI applications in the "{{ insights.telco_cat3_top }}" stage, which is crucial for telecommunications customer experiences.</li>
                    
                    <li><strong>Multimodal Capabilities:</strong> While "{{ insights.telco_cat4_top }}" remains dominant, consider expanding AI capabilities to handle multiple data types simultaneously, especially for complex customer interactions.</li>
                    
                    <li><strong>Industry Benchmarking:</strong> Look to other industries with advanced GenAI implementations, particularly in areas where Telco may be lagging, to identify transferable best practices.</li>
                </ol>
            </div>
        </section>
        
        <div class="footer">
            <p>GenAI in Customer Service Analysis | Created for Telco Industry Research</p>
        </div>
    </div>
</body>
</html>
"""

# Render the template
template = Template(html_template)
html_output = template.render(
    basic_stats=basic_stats,
    all_industry_spider_charts=all_industry_spider_charts,
    per_industry_spider_charts=per_industry_spider_charts,
    heatmaps=heatmaps,
    insights=insights
)

# Write the HTML to file
with open('gen_ai_customer_service_analysis.html', 'w', encoding='utf-8') as f:
    f.write(html_output)

print("HTML report generated successfully: gen_ai_customer_service_analysis.html")

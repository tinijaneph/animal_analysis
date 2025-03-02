import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Animal Health Patterns Explorer",
    page_icon="🐾",
    layout="wide"
)

# Load all the data
@st.cache_data
def load_data():
    try:
        # Load all the derived datasets
        df = pd.read_csv('animal_data.csv')
        symptom_by_species = pd.read_csv('symptom_by_species.csv')
        danger_correlation = pd.read_csv('symptom_danger_correlation.csv')
        species_danger = pd.read_csv('species_danger_rates.csv')
        symptom_count_stats = pd.read_csv('symptom_count_by_species.csv')
        symptom_danger_corr = pd.read_csv('symptom_count_danger_correlation.csv')
        combination_stats = pd.read_csv('symptom_combination_analysis.csv')
        species_specific_symptoms = pd.read_csv('species_specific_symptoms.csv')
        species_danger_patterns = pd.read_csv('species_danger_patterns.csv')
        
        # Create unpivoted version of the original data
        symptom_columns = ['symptoms1', 'symptoms2', 'symptoms3', 'symptoms4', 'symptoms5']
        df['SymptomCount'] = df[symptom_columns].apply(
            lambda row: row[row != "None reported"].count(), axis=1
        )
        unpivoted_data = []
        
        for idx, row in df.iterrows():
            for col in symptom_columns:
                if pd.notna(row[col]) and row[col] != "None reported":
                    unpivoted_data.append({
                        'AnimalName': row['AnimalName'],
                        'Symptom': row[col],
                        'Dangerous': row['Dangerous']
                    })
        
        unpivoted_df = pd.DataFrame(unpivoted_data)
        
        return {
            'df': df,
            'unpivoted_df': unpivoted_df,
            'symptom_by_species': symptom_by_species,
            'danger_correlation': danger_correlation,
            'species_danger': species_danger,
            'symptom_count_stats': symptom_count_stats,
            'symptom_danger_corr': symptom_danger_corr,
            'combination_stats': combination_stats,
            'species_specific_symptoms': species_specific_symptoms,
            'species_danger_patterns': species_danger_patterns
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return empty DataFrames as fallback
        return {
            'df': pd.DataFrame(),
            'unpivoted_df': pd.DataFrame(),
            'symptom_by_species': pd.DataFrame(),
            'danger_correlation': pd.DataFrame(),
            'species_danger': pd.DataFrame(),
            'symptom_count_stats': pd.DataFrame(),
            'symptom_danger_corr': pd.DataFrame(),
            'combination_stats': pd.DataFrame(),
            'species_specific_symptoms': pd.DataFrame(),
            'species_danger_patterns': pd.DataFrame()
        }

data = load_data()

# Define colors
danger_color = "#FF5A5F"
safe_color = "#5FBCD3"
neutral_color = "#5F9EA0"

# App title and introduction
st.title("🐾 Animal Health Patterns Explorer")

st.markdown("""
This interactive dashboard explores patterns and trends in animal health conditions 
across different species. Analyze common symptoms, dangerous conditions, and species-specific 
health patterns through various visualizations.
""")

# Create tabs for different research questions
tab1, tab2, tab3, tab4 = st.tabs([
    "Symptom Frequency & Danger", 
    "Species Risk Analysis", 
    "Symptom Combinations", 
    "Species-Specific Patterns"
])

# Tab 1: Symptom Frequency and Danger Correlation
with tab1:
    st.header("Symptom Frequency and Danger Correlation")
    st.markdown("""
    This section explores the frequency of symptoms across different species and 
    their correlation with dangerous conditions.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Most Common Symptoms by Species")
        
        # Check if data is available
        if not data['symptom_by_species'].empty:
            # Filter options
            species_options = sorted(data['symptom_by_species']['AnimalName'].unique())
            selected_species = st.multiselect(
                "Select species to compare:", 
                options=species_options,
                default=species_options[:5] if len(species_options) > 5 else species_options
            )
            
            top_n = st.slider("Number of top symptoms to show:", 5, 20, 10)
            
            # Filter data based on selection
            if selected_species:
                filtered_data = data['symptom_by_species'][data['symptom_by_species']['AnimalName'].isin(selected_species)]
            else:
                filtered_data = data['symptom_by_species']
            
            # Aggregate to get top symptoms
            top_symptoms = filtered_data.groupby('Symptom')['Count'].sum().reset_index().sort_values('Count', ascending=False).head(top_n)
            
            # Heatmap data
            if not top_symptoms.empty and not filtered_data.empty:
                # Create heatmap data
                heatmap_data = filtered_data[filtered_data['Symptom'].isin(top_symptoms['Symptom'])]
                
                # Pivot for heatmap
                pivot_data = heatmap_data.pivot_table(
                    index='Symptom', 
                    columns='AnimalName', 
                    values='Percentage', 
                    fill_value=0
                )
                
                # Sort by overall frequency
                pivot_data = pivot_data.reindex(top_symptoms['Symptom'])
                
                # Create heatmap
                fig = px.imshow(
                    pivot_data,
                    color_continuous_scale='Blues',
                    labels=dict(x="Species", y="Symptom", color="Frequency (%)"),
                    aspect="auto",
                    title="Symptom Frequency by Species (%)"
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data to create the heatmap visualization.")
        else:
            st.warning("No symptom by species data available.")
    
    with col2:
        st.subheader("Symptoms Most Associated with Danger")
        
        # Check if data is available
        if not data['danger_correlation'].empty:
            # Filter and sort danger correlation data
            danger_data = data['danger_correlation'].sort_values('DangerCoefficient', ascending=False)
            
            if not danger_data.empty:
                danger_data = danger_data.head(10)  # Default to top 10 if slider not available
                
                # Create bar chart
                fig = px.bar(
                    danger_data,
                    x='DangerCoefficient',
                    y='Symptom',
                    orientation='h',
                    color='DangerCoefficient',
                    color_continuous_scale=['#FFEDA0', '#FD8D3C', '#E31A1C'],
                    labels={'DangerCoefficient': 'Danger Coefficient', 'Symptom': ''},
                    title="Symptoms Most Associated with Dangerous Conditions"
                )
                
                fig.update_layout(
                    height=600,
                    yaxis={'categoryorder':'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No danger correlation data available.")
        else:
            st.warning("No danger correlation data available.")
    
    # Analysis insights
    st.subheader("Key Insights:")
    
    col1, col2 = st.columns(2)
    with col1:
        # Calculate some key insights if data is available
        if 'top_symptoms' in locals() and not top_symptoms.empty:
            most_common_symptom = top_symptoms.iloc[0]['Symptom']
            st.metric("Most Common Symptom", most_common_symptom)
        
        if 'danger_data' in locals() and not danger_data.empty:
            most_dangerous_symptom = danger_data.iloc[0]['Symptom']
            danger_coefficient = danger_data.iloc[0]['DangerCoefficient']
            st.metric("Most Dangerous Symptom", most_dangerous_symptom, 
                      f"{danger_coefficient:.2f}x more likely to be dangerous")
    
    with col2:
        if 'top_symptoms' in locals() and not top_symptoms.empty and 'danger_data' in locals() and not danger_data.empty:
        # Get specific insights from the data
            most_common = top_symptoms.iloc[0]['Symptom']
            second_common = top_symptoms.iloc[1]['Symptom'] if len(top_symptoms) > 1 else ""
            most_dangerous = danger_data.iloc[0]['Symptom']
            danger_coefficient = danger_data.iloc[0]['DangerCoefficient']
        
            # Get species-specific insights if species were selected
            species_insights = ""
            if selected_species and len(selected_species) <= 3:  # Only for focused comparisons
                # Extract species-specific top symptoms
                species_specific_insights = []
                for species in selected_species:
                    species_data = filtered_data[filtered_data['AnimalName'] == species]
                    if not species_data.empty:
                        top_for_species = species_data.sort_values('Percentage', ascending=False).iloc[0]['Symptom']
                        species_specific_insights.append(f"- **{species}**: Most common symptom is '{top_for_species}'")
                
                if species_specific_insights:
                    species_insights = "\n\n**Species-specific patterns:**\n" + "\n".join(species_specific_insights)
            
            st.markdown(f"""
            #### Observations:
            - '{most_common}' is the most prevalent symptom across {len(selected_species) if selected_species else "all"} species
            - '{most_dangerous}' shows the strongest association with dangerous conditions ({danger_coefficient:.2f}x higher risk)
            - {len(top_symptoms)} symptoms account for the majority of all reported cases
            - The data suggests monitoring for '{most_dangerous}' should be prioritized given its strong danger association
            {species_insights}
            """)
        else:
            st.markdown("""
            #### Observations:
            - Select species and parameters above to generate specific insights
            - Common symptoms vary significantly across species
            - Some symptoms are strongly associated with dangerous conditions
            """)

# Tab 2: Species Risk Analysis
with tab2:
    st.header("Species Risk Analysis")
    st.markdown("""
    This section examines which animal species are more prone to dangerous health conditions
    and how symptom patterns relate to risk.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Species Danger Rates")
        
        # Check if data is available
        if not data['species_danger'].empty:
            # Sort species by danger rate
            species_danger_sorted = data['species_danger'].sort_values('DangerRate', ascending=False)
            
            # Create bar chart with error bars
            fig = go.Figure()
            
            # Add bar chart
            fig.add_trace(go.Bar(
                x=species_danger_sorted['AnimalName'],
                y=species_danger_sorted['DangerRate'],
                marker_color=px.colors.sequential.Reds[::-1],  # Use a red color scale
                name='Danger Rate',
                error_y=dict(
                    type='data',
                    array=species_danger_sorted['CI_Upper'] - species_danger_sorted['DangerRate'],
                    arrayminus=species_danger_sorted['DangerRate'] - species_danger_sorted['CI_Lower'],
                    visible=True
                )
            ))
            
            # Add sample size as text
            for i, row in species_danger_sorted.iterrows():
                fig.add_annotation(
                    x=row['AnimalName'],
                    y=row['DangerRate'] + 0.05,
                    text=f"n={row['TotalCases']}",
                    showarrow=False,
                    font=dict(size=10, color="black")
                )
            
            fig.update_layout(
                title="Species Ordered by Danger Rate (with 95% Confidence Intervals)",
                xaxis_title="Animal Species",
                yaxis_title="Proportion of Dangerous Cases",
                yaxis=dict(range=[0, 1]),
                height=500,
                template="plotly_white",  # Use a clean template
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No species danger rate data available.")
    
    with col2:
        st.subheader("Symptom Count Distribution by Species")
        
        # Check if data is available and has required columns
        if not data['df'].empty and 'AnimalName' in data['df'].columns and 'SymptomCount' in data['df'].columns and 'Dangerous' in data['df'].columns:
            try:
                # Create box plot of symptom counts by species
                fig = px.box(
                    data['df'],
                    x='AnimalName',
                    y='SymptomCount',
                    color='Dangerous',
                    color_discrete_map={'Yes': danger_color, 'No': safe_color},
                    notched=True,
                    points="all",
                    title="Distribution of Symptom Counts by Species"
                )
                
                fig.update_layout(
                    xaxis_title="Animal Species",
                    yaxis_title="Number of Symptoms",
                    height=500,
                    template="plotly_white",  # Use a clean template
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating box plot: {str(e)}")
                
                # Fallback to a simpler visualization
                st.write("Showing a simpler alternative visualization:")
                
                # Create a simple bar chart instead
                symptom_avg = data['df'].groupby(['AnimalName', 'Dangerous'])['SymptomCount'].mean().reset_index()
                
                fig = px.bar(
                    symptom_avg,
                    x='AnimalName',
                    y='SymptomCount',
                    color='Dangerous',
                    color_discrete_map={'Yes': danger_color, 'No': safe_color},
                    barmode='group',
                    title="Average Symptom Count by Species"
                )
                
                fig.update_layout(
                    xaxis_title="Animal Species",
                    yaxis_title="Average Number of Symptoms",
                    height=500,
                    template="plotly_white",  # Use a clean template
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Required data for symptom count distribution is not available.")
    
    # Correlation between symptom count and danger
    st.subheader("Correlation: Symptom Count vs. Danger")
    
    # Check if data is available
    if not data['symptom_danger_corr'].empty:
        # Sort correlations
        corr_data = data['symptom_danger_corr'].sort_values('Correlation', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            corr_data,
            x='AnimalName',
            y='Correlation',
            color='Correlation',
            color_continuous_scale=px.colors.sequential.Viridis,  # Use a better color scale
            title="Correlation Between Symptom Count and Dangerous Condition by Species"
        )
        
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=0,
            x1=len(corr_data) - 0.5,
            y1=0,
            line=dict(color="black", width=1, dash="dash")
        )
        
        fig.update_layout(
            xaxis_title="Animal Species",
            yaxis_title="Correlation Coefficient",
            height=400,
            template="plotly_white",  # Use a clean template
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        #### Interpretation:
        - The correlation coefficient indicates the relationship between symptom count and danger
        - A value near zero suggests no linear relationship between symptom count and danger
        - This means that having more symptoms does not necessarily increase the likelihood of a dangerous condition
        - Further analysis should focus on specific symptoms or combinations rather than just symptom count
        """)
    else:
        st.warning("No data available for symptom-danger correlation.")
    
    # Key insights
    st.subheader("Key Insights:")
    
    col1, col2 = st.columns(2)
    with col1:
        if 'species_danger_sorted' in locals() and not species_danger_sorted.empty:
            highest_risk_species = species_danger_sorted.iloc[0]['AnimalName']
            highest_risk_rate = species_danger_sorted.iloc[0]['DangerRate']
            
            st.metric("Highest Risk Species", highest_risk_species, 
                     f"{highest_risk_rate:.1%} danger rate")
        
        if 'corr_data' in locals() and not corr_data.empty:
            highest_corr_species = corr_data.iloc[0]['AnimalName']
            highest_corr = corr_data.iloc[0]['Correlation']
            
            st.metric("Strongest Symptom-Danger Correlation", highest_corr_species,
                     f"{highest_corr:.2f} correlation")
    
    with col2:
        if 'species_danger_sorted' in locals() and not species_danger_sorted.empty and 'corr_data' in locals() and not corr_data.empty:
            # Extract specific insights
            highest_risk = species_danger_sorted.iloc[0]['AnimalName']
            highest_risk_rate = species_danger_sorted.iloc[0]['DangerRate']
            lowest_risk = species_danger_sorted.iloc[-1]['AnimalName']
            lowest_risk_rate = species_danger_sorted.iloc[-1]['DangerRate']
            
            # Correlation insights
            positive_corr = corr_data[corr_data['Correlation'] > 0.3].shape[0]
            negative_corr = corr_data[corr_data['Correlation'] < -0.3].shape[0]
            strong_corr_species = corr_data.iloc[0]['AnimalName'] if abs(corr_data.iloc[0]['Correlation']) > 0.3 else None
            
            # Risk difference calculation
            risk_ratio = highest_risk_rate / max(lowest_risk_rate, 0.001)  # Avoid division by zero
            
            st.markdown(f"""
            #### Observations:
            - **{highest_risk}** shows the highest danger rate at **{highest_risk_rate:.1%}**, which is **{risk_ratio:.1f}x** higher than **{lowest_risk}** ({lowest_risk_rate:.1%})
            - {positive_corr} species show a positive correlation between symptom count and danger (more symptoms = higher risk)
            - {negative_corr} species show a negative correlation (more symptoms = lower risk)
            - {"For **" + strong_corr_species + "**, symptom count is a strong predictor of danger" if strong_corr_species else "No species shows a strong correlation between symptom count and danger"}
            - The data suggests different risk assessment models may be needed for different species
            """)
        else:
            st.markdown("""
            #### Observations:
            - Some species have significantly higher rates of dangerous conditions
            - The relationship between symptom count and danger varies by species
            - For some species, more symptoms correlate with dangerous conditions
            - For others, the number of symptoms is less predictive of danger
            """)

# Tab 3: Symptom Combinations
with tab3:
    st.header("Symptom Combinations and Co-occurrence")
    st.markdown("""
    This section analyzes how symptoms appear together and which combinations 
    are most associated with dangerous conditions.
    """)
    
    # Check if data is available
    if not data['combination_stats'].empty:
        # Filter options
        combination_type = st.selectbox(
            "Select combination type:",
            options=['All', 'Single', 'Pair', 'Triplet'],
            index=0
        )
        
        min_occurrences = st.slider("Minimum number of occurrences:", 3, 20, 5)
        
        # Filter data
        if combination_type != 'All':
            filtered_combinations = data['combination_stats'][
                (data['combination_stats']['CombinationType'] == combination_type) & 
                (data['combination_stats']['Count'] >= min_occurrences)
            ]
        else:
            filtered_combinations = data['combination_stats'][
                data['combination_stats']['Count'] >= min_occurrences
            ]
        
        # Sort by danger rate
        if not filtered_combinations.empty:
            top_combinations = filtered_combinations.sort_values(
                ['DangerRate', 'Count'], 
                ascending=[False, False]
            ).head(15)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("Most Dangerous Symptom Combinations")
                
                if not top_combinations.empty:
                    # Create bar chart
                    fig = px.bar(
                        top_combinations,
                        y='Combination',
                        x='DangerRate',
                        color='DangerRate',
                        color_continuous_scale=['#FFEDA0', '#FD8D3C', '#E31A1C'],
                        hover_data=['Count', 'CombinationType'],
                        labels={'DangerRate': 'Danger Rate', 'Combination': ''},
                        title=f"Top Dangerous Symptom Combinations (min. {min_occurrences} occurrences)"
                    )
                    
                    fig.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        xaxis={'range': [0, 1]},
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No combinations found with at least {min_occurrences} occurrences.")
            
            with col2:
                st.subheader("Symptom Co-occurrence Network")
                
                try:
                    # Create network graph of symptom co-occurrence
                    # Get top symptoms for clarity
                    if not data['danger_correlation'].empty:
                        top_n_symptoms = 15
                        top_symptoms = data['danger_correlation'].sort_values('DangerCorrelation', ascending=False).head(top_n_symptoms)['Symptom'].tolist()
                        
                        # Filter to pairs containing these symptoms
                        network_data = data['combination_stats'][
                            (data['combination_stats']['CombinationType'] == 'Pair') & 
                            (data['combination_stats']['Count'] >= min_occurrences)
                        ]
                        
                        # Create network
                        G = nx.Graph()
                        
                        # Add edges from pair combinations
                        for _, row in network_data.iterrows():
                            pair = row['Combination'].split(' + ')
                            if len(pair) == 2:
                                # Only include if both symptoms are in our top list
                                if pair[0] in top_symptoms and pair[1] in top_symptoms:
                                    G.add_edge(
                                        pair[0], 
                                        pair[1], 
                                        weight=row['Count'],
                                        danger=row['DangerRate']
                                    )
                        
                        # Add nodes with danger correlation
                        for symptom in top_symptoms:
                            if symptom in G.nodes:
                                danger_val = data['danger_correlation'][data['danger_correlation']['Symptom'] == symptom]['DangerCorrelation'].values
                                if len(danger_val) > 0:
                                    G.nodes[symptom]['danger'] = float(danger_val[0])
                                else:
                                    G.nodes[symptom]['danger'] = 0.5
                        
                        if len(G.nodes) > 0:
                            # Create positions with spring layout
                            pos = nx.spring_layout(G, seed=42)
                            
                            # Create edge traces
                            edge_traces = []
                            for edge in G.edges(data=True):
                                x0, y0 = pos[edge[0]]
                                x1, y1 = pos[edge[1]]
                                weight = edge[2]['weight']
                                danger = edge[2]['danger']
                                
                                # Normalize edge width based on weight
                                width = 1 + (weight / network_data['Count'].max()) * 5
                                
                                edge_trace = go.Scatter(
                                    x=[x0, x1, None],
                                    y=[y0, y1, None],
                                    line=dict(width=width, color=f'rgba(150,150,150,0.5)'),
                                    hoverinfo='text',
                                    text=f"{edge[0]} + {edge[1]}<br>Count: {weight}<br>Danger: {danger:.2f}",
                                    mode='lines'
                                )
                                edge_traces.append(edge_trace)
                            
                            # Create node trace
                            node_trace = go.Scatter(
                                x=[pos[node][0] for node in G.nodes()],
                                y=[pos[node][1] for node in G.nodes()],
                                mode='markers+text',
                                text=[node for node in G.nodes()],
                                textfont=dict(size=10),
                                textposition="top center",
                                marker=dict(
                                    showscale=True,
                                    colorscale='Reds',
                                    size=[10 + len(list(G.neighbors(node))) for node in G.nodes()],
                                    color=[G.nodes[node].get('danger', 0.5) for node in G.nodes()],
                                    colorbar=dict(
                                        title='Danger<br>Correlation',
                                        thickness=15,
                                        tickvals=[0.3, 0.6, 0.9],
                                        ticktext=['Low', 'Medium', 'High']
                                    ),
                                    line=dict(width=2, color='DarkSlateGrey')
                                ),
                                hovertemplate='%{text}<br>Connections: %{marker.size}<br>Danger: %{marker.color:.2f}<extra></extra>'
                            )
                            
                            # Create figure
                            fig = go.Figure(data=edge_traces + [node_trace])
                            
                            fig.update_layout(
                                title="Symptom Co-occurrence Network",
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Not enough connected symptoms for network visualization.")
                    else:
                        st.warning("Danger correlation data not available for network visualization.")
                except Exception as e:
                    st.error(f"Error creating network visualization: {str(e)}")
                    st.warning("Could not create the network visualization. Try adjusting the parameters.")
            
            # Key insights
            st.subheader("Key Insights:")
            
            col1, col2 = st.columns(2)
            with col1:
                if 'top_combinations' in locals() and not top_combinations.empty:
                    top_danger_combo = top_combinations.iloc[0]['Combination']
                    top_danger_rate = top_combinations.iloc[0]['DangerRate']
                    
                    st.metric("Most Dangerous Combination", top_danger_combo,
                             f"{top_danger_rate:.1%} danger rate")
                
                # Find most connected symptom if network exists
                if 'G' in locals() and len(G.nodes) > 0:
                    most_connected = None
                    max_connections = 0
                    for node in G.nodes():
                        connections = len(list(G.neighbors(node)))
                        if connections > max_connections:
                            max_connections = connections
                            most_connected = node
                    
                    if most_connected:
                        st.metric("Most Connected Symptom", most_connected,
                                 f"{max_connections} connections")
            
            with col2:
                if 'top_combinations' in locals() and not top_combinations.empty:
                    # Extract specific insights
                    top_combo = top_combinations.iloc[0]['Combination']
                    top_combo_rate = top_combinations.iloc[0]['DangerRate']
                    top_combo_count = top_combinations.iloc[0]['Count']
                    
                    # Get combination type stats
                    combo_type_counts = filtered_combinations['CombinationType'].value_counts()
                    most_common_type = combo_type_counts.idxmax() if not combo_type_counts.empty else "N/A"
                    
                    # Network insights
                    network_insight = ""
                    if 'G' in locals() and len(G.nodes) > 0:
                        most_connected = None
                        max_connections = 0
                        for node in G.nodes():
                            connections = len(list(G.neighbors(node)))
                            if connections > max_connections:
                                max_connections = connections
                                most_connected = node
                        if most_connected:
                            network_insight = f"\n- The network analysis shows **'{most_connected}'** is the most connected symptom ({max_connections} connections), suggesting it's a central indicator"
                    
                    # Calculate danger threshold
                    high_danger_combos = filtered_combinations[filtered_combinations['DangerRate'] > 0.7].shape[0]
                    
                    st.markdown(f"#### Observations for {selected_species_tab4}:")
                    st.markdown(f"- '{most_distinctive}' is **{distinctiveness_score:.1f}x** more common in {selected_species_tab4} than in other species and appears in {proportion:.1%} of cases")
                    st.markdown(f"- '{top_danger}' is the strongest indicator of danger with {relative_risk:.1f}x higher risk and a {danger_rate:.1%} danger rate when present")
                    st.markdown(f"- {high_risk_symptoms} symptoms show at least 2x higher risk of danger when present")
                    st.markdown(f"- This species shows a unique symptom profile that requires targeted monitoring{heatmap_insight}")
                else:
                    st.markdown("""
                    #### Observations:
                    - Certain symptom combinations are much more likely to indicate dangerous conditions
                    - The network visualization reveals clusters of symptoms that frequently co-occur
                    - Central symptoms in the network often serve as important indicators
                    - Some symptoms are more "connected" and frequently appear with many other symptoms
                    """)
        else:
            st.warning(f"No combinations found with at least {min_occurrences} occurrences for the selected type.")
    else:
        st.warning("No symptom combination data available.")


with tab4:
    st.header("Species-Specific Health Patterns")
    st.markdown("""
    This section explores which symptoms are distinctly common in certain species and 
    which symptoms are most predictive of danger for specific species.
    """)
    
    # Species selection
    species_list = sorted(data['species_specific_symptoms']['AnimalName'].unique())
    selected_species_tab4 = st.selectbox(
        "Select species to analyze:",
        options=species_list
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distinctive Symptoms for Selected Species")
        
        # Filter data for selected species
        distinctive_symptoms = data['species_specific_symptoms'][
            data['species_specific_symptoms']['AnimalName'] == selected_species_tab4
        ].sort_values('Distinctiveness', ascending=False).head(10)
        
        # Create bar chart
        fig = px.bar(
            distinctive_symptoms,
            y='Symptom',
            x='Distinctiveness',
            color='Distinctiveness',
            color_continuous_scale='Viridis',
            hover_data=['Count', 'PropWithinSpecies'],
            labels={'Distinctiveness': 'Distinctiveness Score', 'Symptom': ''},
            title=f"Symptoms Distinctively Common in {selected_species_tab4}"
        )
        
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Danger Indicators for Selected Species")
        
        # Filter danger patterns for selected species
        species_danger_indicators = data['species_danger_patterns'][
            data['species_danger_patterns']['AnimalName'] == selected_species_tab4
        ].sort_values('RelativeRisk', ascending=False).head(10)
        
        # Create bar chart
        fig = px.bar(
            species_danger_indicators,
            y='Symptom',
            x='RelativeRisk',
            color='RelativeRisk',
            color_continuous_scale=['#FFEDA0', '#FD8D3C', '#E31A1C'],
            hover_data=['DangerRate', 'SymptomCount'],
            labels={'RelativeRisk': 'Relative Risk', 'Symptom': ''},
            title=f"Symptoms Most Predictive of Danger in {selected_species_tab4}"
        )
        
        # Add reference line at 1.0
        fig.add_shape(
            type="line",
            x0=1,
            y0=-0.5,
            x1=1,
            y1=len(species_danger_indicators) - 0.5,
            line=dict(color="black", width=1, dash="dash")
        )
        
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Comparison across species
    st.subheader("Compare Symptoms Across Species")
    
    # Select symptoms to compare
    all_symptoms = sorted(data['unpivoted_df']['Symptom'].unique())
    selected_symptoms = st.multiselect(
        "Select symptoms to compare across species:",
        options=all_symptoms,
        default=all_symptoms[:5] if len(all_symptoms) >= 5 else all_symptoms
    )
    
    if selected_symptoms:
        # Filter data
        symptom_comparison = data['unpivoted_df'][
            data['unpivoted_df']['Symptom'].isin(selected_symptoms)
        ]
        
        # Group by species and symptom, calculate danger rate
        symptom_species_danger = symptom_comparison.groupby(['AnimalName', 'Symptom'])['Dangerous'].apply(
            lambda x: (x == 'Yes').mean()
        ).reset_index()
        symptom_species_danger.columns = ['AnimalName', 'Symptom', 'DangerRate']
        
        # Create heatmap
        fig = px.density_heatmap(
            symptom_species_danger,
            x='AnimalName',
            y='Symptom',
            z='DangerRate',
            color_continuous_scale='Reds',
            title="Danger Rate by Species and Symptom"
        )
        
        fig.update_layout(
            xaxis_title="Species",
            yaxis_title="Symptom",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("Key Insights:")
    
    col1, col2 = st.columns(2)
    with col1:
        if not distinctive_symptoms.empty:
            most_distinctive = distinctive_symptoms.iloc[0]['Symptom']
            distinctiveness_score = distinctive_symptoms.iloc[0]['Distinctiveness']
            
            st.metric(f"Most Distinctive Symptom for {selected_species_tab4}", 
                     most_distinctive,
                     f"{distinctiveness_score:.1f}x more common")
        
        if not species_danger_indicators.empty:
            top_danger_indicator = species_danger_indicators.iloc[0]['Symptom']
            relative_risk = species_danger_indicators.iloc[0]['RelativeRisk']
            
            st.metric(f"Top Danger Indicator for {selected_species_tab4}",
                     top_danger_indicator,
                     f"{relative_risk:.1f}x higher risk")
    
    with col2:
        if not distinctive_symptoms.empty and not species_danger_indicators.empty:
            # Extract specific insights for the selected species
            most_distinctive = distinctive_symptoms.iloc[0]['Symptom']
            distinctiveness_score = distinctive_symptoms.iloc[0]['Distinctiveness']
            proportion = distinctive_symptoms.iloc[0]['PropWithinSpecies']
            
            # Danger indicators
            top_danger = species_danger_indicators.iloc[0]['Symptom']
            relative_risk = species_danger_indicators.iloc[0]['RelativeRisk']
            danger_rate = species_danger_indicators.iloc[0]['DangerRate']
            
            # Count symptoms with high relative risk
            high_risk_symptoms = species_danger_indicators[species_danger_indicators['RelativeRisk'] > 2].shape[0]
            
            # Additional insights from heatmap if available
            heatmap_insight = ""
            if 'symptom_species_danger' in locals() and not symptom_species_danger.empty:
                species_data = symptom_species_danger[symptom_species_danger['AnimalName'] == selected_species_tab4]
                if not species_data.empty:
                    max_danger_symptom = species_data.sort_values('DangerRate', ascending=False).iloc[0]['Symptom']
                    max_danger_rate = species_data.sort_values('DangerRate', ascending=False).iloc[0]['DangerRate']
                    heatmap_insight = f"\n- Among the selected symptoms, '{max_danger_symptom}' shows the highest danger rate ({max_danger_rate:.1%}) for {selected_species_tab4}"
            
            st.markdown(f"""
            #### Observations for {selected_species_tab4}:
            - '{most_distinctive}' is **{distinctiveness_score:.1f}x** more common in {selected_species_tab4} than in other species and appears in {proportion:.1%} of cases
            - '{top_danger}' is the strongest indicator of danger with {relative_risk:.1f}x higher risk and a {danger_rate:.1%} danger rate when present
            - {high_risk_symptoms} symptoms show at least 2x higher risk of danger when present
            - This species shows a unique symptom profile that requires targeted monitoring{heatmap_insight}
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            #### Observations for {selected_species_tab4}:
            - Each species has a unique symptom profile
            - Some symptoms occur at much higher rates in certain species
            - The symptoms that predict danger vary significantly by species
            - Understanding species-specific patterns helps with targeted monitoring
            """)

# Footer with additional information
st.markdown("---")
st.markdown("""
### About this Project
This interactive dashboard was created to explore animal health patterns and trends across different species. 
It analyzes a comprehensive dataset of animal symptoms and conditions to identify key risk factors and species-specific patterns.
""")

# report_type = st.sidebar.selectbox(
#     "Select report type:",
#     ["Overall Summary", "Species-Specific Analysis", "Symptom Risk Analysis"]
# )
    

# # About section
# st.sidebar.markdown("---")
# st.sidebar.info("""
# **Project: Exploring Animal Health Patterns**

# This project analyzes the "Animal Condition Classification Dataset" 
# to identify patterns and trends in animal health conditions across different species.
# """)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from nilearn import plotting
import scipy.stats as stats

# Set page configuration
st.set_page_config(
    page_title="Brain Connectivity Visualization",
    page_icon="🧠",
    layout="wide"
)

# App title and description
st.title("Brain Connectivity: Neurotypical vs Autism Spectrum Disorder")
st.markdown("""
This application visualizes differences in brain connectivity between neurotypical individuals and those with Autism Spectrum Disorder (ASD).
The visualizations are based on functional connectivity data and highlight differences in connection strength and network organization.
""")

# Sidebar for navigation and options
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Overview", "Connectivity Matrices", "Network Visualization", "Regional Differences", "About"])
# Generate synthetic data for demonstration purposes
@st.cache_data
def generate_sample_data(seed=42):
    np.random.seed(seed)
    # Define brain regions (simplified for demonstration)
    regions = [
        'Frontal_L', 'Frontal_R', 'Parietal_L', 'Parietal_R',
        'Temporal_L', 'Temporal_R', 'Occipital_L', 'Occipital_R',
        'Cingulate_L', 'Cingulate_R', 'Amygdala_L', 'Amygdala_R',
        'Hippocampus_L', 'Hippocampus_R', 'Thalamus_L', 'Thalamus_R'
    ]
    
    # Create connectivity matrices
    # Neurotypical - stronger local connectivity, balanced long-range
    nt_matrix = np.zeros((len(regions), len(regions)))
    for i in range(len(regions)):
        for j in range(len(regions)):
            # Local connectivity (within same brain area left/right)
            if i // 2 == j // 2:
                nt_matrix[i, j] = 0.7 + 0.2 * np.random.random()
            # Long-range connectivity
            else:
                nt_matrix[i, j] = 0.3 * np.random.random()
    
    # ASD - weaker local connectivity in some regions, potentially stronger in others, less balanced
    asd_matrix = np.zeros((len(regions), len(regions)))
    for i in range(len(regions)):
        for j in range(len(regions)):
            # Simulate underconnectivity in frontal-temporal connections
            if (i < 4 and 4 <= j < 8) or (4 <= i < 8 and j < 4):
                asd_matrix[i, j] = 0.15 * np.random.random()
            # Local connectivity
            elif i // 2 == j // 2:
                # Some local connections are weaker
                if i < 8:  # Cortical regions
                    asd_matrix[i, j] = 0.5 + 0.2 * np.random.random()
                else:  # Subcortical regions - potentially stronger
                    asd_matrix[i, j] = 0.8 + 0.2 * np.random.random()
            # Other long-range connections
            else:
                # More variability in long-range connectivity
                asd_matrix[i, j] = 0.3 * np.random.random() + 0.1 * (i % 2)
    
    # Ensure matrices are symmetric (undirected connectivity)
    nt_matrix = (nt_matrix + nt_matrix.T) / 2
    asd_matrix = (asd_matrix + asd_matrix.T) / 2
    
    # Set diagonal to zero (no self-connections)
    np.fill_diagonal(nt_matrix, 0)
    np.fill_diagonal(asd_matrix, 0)
    
    # Calculate difference matrix
    diff_matrix = asd_matrix - nt_matrix
    
    return regions, nt_matrix, asd_matrix, diff_matrix

# Load data
regions, nt_matrix, asd_matrix, diff_matrix = generate_sample_data()

# Overview page
if page == "Overview":
    st.header("Brain Connectivity Overview")
    
    st.subheader("What is Brain Connectivity?")
    st.markdown("""
    Brain connectivity refers to the patterns of anatomical links (structural connectivity), statistical dependencies (functional connectivity), 
    or causal interactions (effective connectivity) between distinct regions of the brain. These patterns can be analyzed at different spatial scales.
    
    In this application, we focus on functional connectivity differences between neurotypical brains and those with Autism Spectrum Disorder.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Neurotypical Brain Connectivity")
        st.markdown("""
        Typical characteristics:
        - Balanced local and long-range connectivity
        - Efficient information transfer between brain regions
        - Well-organized functional networks
        - Appropriate integration and segregation of information
        """)
        
        # Simple visualization for neurotypical connectivity
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(nt_matrix, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        ax.set_title('Neurotypical Brain Connectivity Matrix')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Autism Spectrum Disorder Brain Connectivity")
        st.markdown("""
        Common differences observed in research:
        - Local over-connectivity in some regions
        - Long-range under-connectivity, especially between frontal and posterior regions
        - Altered functional network organization
        - Less efficient information integration across brain regions
        - Greater variability in connectivity patterns
        """)
        
        # Simple visualization for ASD connectivity
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(asd_matrix, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        ax.set_title('ASD Brain Connectivity Matrix')
        st.pyplot(fig)
    
    st.subheader("Key Research Findings")
    st.markdown("""
    Research on brain connectivity in autism has found:
    
    1. **Underconnectivity Theory**: Many studies suggest reduced connectivity between frontal and posterior brain regions in autism.
    
    2. **Local Over-connectivity**: Some research indicates increased connectivity within specific brain regions.
    
    3. **Network Organization**: Differences in the organization of functional networks, including default mode network, social brain network, and language networks.
    
    4. **Developmental Trajectory**: Connectivity differences that change over the lifespan, with potentially different patterns in children vs. adults with autism.
    
    5. **Heterogeneity**: Significant variation in connectivity patterns across individuals with autism, reflecting the heterogeneous nature of the condition.
    
    It's important to note that brain connectivity in autism is complex and findings can vary based on methodology, age of participants, and specific autism presentations.
    """)

# Connectivity Matrices page
elif page == "Connectivity Matrices":
    st.header("Connectivity Matrices Visualization")
    
    st.markdown("""
    Connectivity matrices show the strength of connections between brain regions. 
    Warmer colors (red) indicate stronger connectivity, while cooler colors (blue) indicate weaker connectivity.
    """)
    
    view_option = st.radio("Select View", ["Side by Side", "Difference Map"])
    
    if view_option == "Side by Side":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Neurotypical Connectivity")
            fig, ax = plt.subplots(figsize=(10, 8))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(nt_matrix, cmap=cmap, center=0, square=True, 
                        xticklabels=regions, yticklabels=regions,
                        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            ax.set_title('Neurotypical Brain Connectivity Matrix')
            st.pyplot(fig)
            
            st.markdown("""
            **Key Observations**:
            - Balanced connectivity across brain regions
            - Strong connections between corresponding left and right hemispheric regions
            - Efficient network organization with appropriate functional integration
            """)
        
        with col2:
            st.subheader("ASD Connectivity")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(asd_matrix, cmap=cmap, center=0, square=True, 
                        xticklabels=regions, yticklabels=regions,
                        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            ax.set_title('ASD Brain Connectivity Matrix')
            st.pyplot(fig)
            
            st.markdown("""
            **Key Observations**:
            - Reduced connectivity between frontal and temporal regions
            - Potentially increased local connectivity in some subcortical areas
            - More variable pattern of connectivity across the brain
            """)
    
    else:  # Difference Map
        st.subheader("Connectivity Difference Map (ASD - Neurotypical)")
        fig, ax = plt.subplots(figsize=(12, 10))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(diff_matrix, cmap=cmap, center=0, square=True, 
                    xticklabels=regions, yticklabels=regions,
                    linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        ax.set_title('Difference in Brain Connectivity (ASD - Neurotypical)')
        st.pyplot(fig)
        
        st.markdown("""
        **Interpretation**:
        - **Red areas** indicate stronger connectivity in ASD compared to neurotypical brains
        - **Blue areas** indicate weaker connectivity in ASD compared to neurotypical brains
        - **White/neutral areas** indicate similar connectivity strength
        
        This difference map highlights the underconnectivity between frontal and temporal regions (blue),
        and potential overconnectivity in some local subcortical networks (red) in ASD brains.
        """)
    
    st.subheader("Region Connectivity Strength Comparison")
    
    # Calculate average connectivity for each region
    nt_conn_strength = np.mean(nt_matrix, axis=1)
    asd_conn_strength = np.mean(asd_matrix, axis=1)
    
    # Create a DataFrame for plotting
    conn_df = pd.DataFrame({
        'Region': regions,
        'Neurotypical': nt_conn_strength,
        'ASD': asd_conn_strength
    })
    
    # Plot region connectivity strength comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    conn_df.plot(x='Region', y=['Neurotypical', 'ASD'], kind='bar', ax=ax)
    plt.ylabel('Average Connectivity Strength')
    plt.title('Average Connectivity Strength by Brain Region')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    This bar chart shows the average connectivity strength for each brain region,
    comparing neurotypical and ASD brains. Regions with substantial differences
    may be particularly important in understanding the neural basis of autism.
    """)

# Network Visualization page
elif page == "Network Visualization":
    st.header("Brain Network Visualization")
    
    st.markdown("""
    These visualizations represent the brain as a network, where nodes are brain regions
    and edges represent the strength of connections between them. Only connections above
    a certain threshold are shown to highlight the network structure.
    """)
    
    # Threshold slider
    threshold = st.slider("Connection Strength Threshold", 0.0, 1.0, 0.4, 0.05)
    
    col1, col2 = st.columns(2)
    
    # Function to create network graph
    def create_brain_network(conn_matrix, threshold, title):
        G = nx.Graph()
        
        # Add nodes
        for i, region in enumerate(regions):
            G.add_node(i, name=region)
        
        # Add edges above threshold
        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                if conn_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=conn_matrix[i, j])
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Position nodes in a circle
        pos = nx.circular_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', alpha=0.8, ax=ax)
        
        # Draw edges with width proportional to weight
        for (u, v, d) in G.edges(data=True):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=d['weight'] * 5, alpha=0.7, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels={i: regions[i] for i in G.nodes()}, font_size=10, ax=ax)
        
        plt.title(title)
        plt.axis('off')
        return fig
    
    with col1:
        nt_network = create_brain_network(nt_matrix, threshold, "Neurotypical Brain Network")
        st.pyplot(nt_network)
        
        st.markdown("""
        **Observations**:
        - More balanced and distributed connectivity
        - Better integration between different brain regions
        - More connections between frontal and posterior regions
        """)
    
    with col2:
        asd_network = create_brain_network(asd_matrix, threshold, "ASD Brain Network")
        st.pyplot(asd_network)
        
        st.markdown("""
        **Observations**:
        - Fewer long-range connections, particularly between frontal and temporal regions
        - More variable connection patterns
        - Some regions showing relatively isolated connectivity
        """)
    
    st.subheader("Network Metrics Comparison")
    
    # Calculate network metrics
    def calculate_network_metrics(conn_matrix, threshold):
        G = nx.Graph()
        
        # Add nodes
        for i in range(len(regions)):
            G.add_node(i)
        
        # Add edges above threshold
        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                if conn_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=conn_matrix[i, j])
        
        # Calculate metrics
        avg_path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else np.nan
        avg_clustering = nx.average_clustering(G)
        density = nx.density(G)
        
        return {
            'Average Path Length': avg_path_length,
            'Average Clustering Coefficient': avg_clustering,
            'Network Density': density
        }
    
    try:
        nt_metrics = calculate_network_metrics(nt_matrix, threshold)
        asd_metrics = calculate_network_metrics(asd_matrix, threshold)
        
        # Create DataFrame for metrics
        metrics_df = pd.DataFrame({
            'Metric': list(nt_metrics.keys()),
            'Neurotypical': list(nt_metrics.values()),
            'ASD': list(asd_metrics.values())
        })
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df.plot(x='Metric', y=['Neurotypical', 'ASD'], kind='bar', ax=ax)
        plt.ylabel('Value')
        plt.title('Network Metrics Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Network Metrics Interpretation**:
        
        - **Average Path Length**: Average number of steps needed to go from one node to another. Higher values in ASD may indicate less efficient information transfer.
        
        - **Average Clustering Coefficient**: Measure of node clustering or segregation. Differences suggest altered local processing.
        
        - **Network Density**: Ratio of existing connections to all possible connections. Lower density in ASD suggests fewer connections overall.
        
        These metrics quantify differences in brain network organization between neurotypical and ASD brains.
        """)
    
    except Exception as e:
        st.warning(f"Could not calculate network metrics at the current threshold. Try lowering the threshold to ensure connected networks. Error: {str(e)}")

# Regional Differences page
elif page == "Regional Differences":
    st.header("Regional Connectivity Differences")
    
    st.markdown("""
    This page explores specific connectivity differences in key brain regions that have been
    implicated in autism research. Research suggests particular importance of connectivity
    involving frontal, temporal, and subcortical regions.
    """)
    
    # Region selection
    region_options = [f"{i+1}. {region}" for i, region in enumerate(regions)]
    selected_region_index = st.selectbox("Select Region to Analyze", range(len(region_options)), format_func=lambda x: region_options[x])
    
    # Extract connectivity for selected region
    nt_conn = nt_matrix[selected_region_index]
    asd_conn = asd_matrix[selected_region_index]
    diff_conn = asd_conn - nt_conn
    
    # Create DataFrame for plotting
    region_df = pd.DataFrame({
        'Connected Region': regions,
        'Neurotypical Connectivity': nt_conn,
        'ASD Connectivity': asd_conn,
        'Difference (ASD - NT)': diff_conn
    })
    
    # Display the selected region's connectivity profile
    st.subheader(f"Connectivity Profile: {regions[selected_region_index]}")
    
    # Plot connectivity comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(regions))
    ax.bar(x, nt_conn, width=0.4, label='Neurotypical', align='edge', alpha=0.7)
    ax.bar([i+0.4 for i in x], asd_conn, width=0.4, label='ASD', align='edge', alpha=0.7)
    ax.set_xticks([i+0.2 for i in x])
    ax.set_xticklabels(regions, rotation=90)
    ax.set_ylabel('Connectivity Strength')
    ax.set_title(f'Connectivity Profile for {regions[selected_region_index]}')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    # Plot difference
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['red' if x > 0 else 'blue' for x in diff_conn]
    ax.bar(range(len(regions)), diff_conn, color=colors)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=90)
    ax.set_ylabel('Difference in Connectivity (ASD - NT)')
    ax.set_title(f'Connectivity Differences for {regions[selected_region_index]}')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    **Interpretation**:
    - **Red bars** indicate stronger connectivity in ASD compared to neurotypical brains
    - **Blue bars** indicate weaker connectivity in ASD compared to neurotypical brains
    
    This analysis allows you to see how a selected brain region's connectivity pattern
    differs between neurotypical and ASD brains. Research suggests that certain connection
    patterns, especially those involving frontal-posterior connectivity, are particularly
    affected in autism.
    """)
    
    # Statistical analysis
    st.subheader("Statistical Analysis")
    
    # Perform t-test
    t_stat, p_val = stats.ttest_ind(nt_conn, asd_conn)
    
    st.write(f"T-statistic: {t_stat:.3f}")
    st.write(f"P-value: {p_val:.3f}")
    
    if p_val < 0.05:
        st.write("There is a statistically significant difference in the connectivity pattern of this region between neurotypical and ASD brains.")
    else:
        st.write("No statistically significant difference in the connectivity pattern of this region was detected.")
    
    # Additional insights based on the region
    if selected_region_index < 4:  # Frontal regions
        st.markdown("""
        **Frontal Lobe Connectivity in Autism**:
        Research has shown altered frontal lobe connectivity in autism, particularly:
        - Reduced long-range connectivity with posterior brain regions
        - Altered connectivity with language networks
        - Differences in connectivity with the default mode network
        
        These alterations may relate to challenges in executive function, social cognition, and behavioral flexibility.
        """)
    elif 4 <= selected_region_index < 8:  # Temporal regions
        st.markdown("""
        **Temporal Lobe Connectivity in Autism**:
        Temporal lobe connectivity differences in autism include:
        - Altered connectivity with social brain networks
        - Differences in auditory and language processing networks
        - Changes in connectivity with emotion processing regions
        
        These differences may relate to language development, social perception, and sensory processing in autism.
        """)
    elif 8 <= selected_region_index < 10:  # Cingulate
        st.markdown("""
        **Cingulate Connectivity in Autism**:
        The cingulate cortex shows connectivity differences in autism, including:
        - Altered connectivity with the default mode network
        - Differences in salience network integration
        - Changes in emotion regulation networks
        
        The cingulate plays important roles in attention, emotion, and cognitive control.
        """)
    else:  # Subcortical
        st.markdown("""
        **Subcortical Connectivity in Autism**:
        Subcortical structures show connectivity differences in autism:
        - Altered amygdala connectivity related to emotion processing
        - Differences in thalamic connectivity affecting sensory integration
        - Hippocampal connectivity variations potentially affecting memory systems
        
        These differences may contribute to sensory hypersensitivity, emotional regulation differences, and information processing patterns seen in autism.
        """)

elif page == "About":
    st.header("About This Application")
    
    st.markdown("""
    ### Purpose
    
    This application was created to visualize and explain differences in brain connectivity between neurotypical individuals and those with Autism Spectrum Disorder (ASD). 
    It serves as an educational tool to help understand the neurobiological aspects of autism.
    
    ### Data Information
    
    **Important Note**: The data used in this application is synthetic and created for demonstration purposes. 
    It is based on patterns reported in scientific literature but does not represent actual brain scans or individual data.
    
    Real neuroimaging studies use various techniques to measure brain connectivity:
    - Functional MRI (fMRI)
    - Diffusion Tensor Imaging (DTI)
    - Electroencephalography (EEG)
    - Magnetoencephalography (MEG)
    
    ### Further Reading
    
    For more information about brain connectivity in autism, consider exploring the following resources:
    
    1. Hull, J. V., et al. (2017). Resting-state functional connectivity in autism spectrum disorders: A review. Frontiers in Psychiatry, 7, 205.
    
    2. Just, M. A., et al. (2012). Autism as a neural systems disorder: A theory of frontal-posterior underconnectivity. Neuroscience & Biobehavioral Reviews, 36(4), 1292-1313.
    
    3. Picci, G., et al. (2016). Functional brain connectivity for fMRI in autism spectrum disorders: Progress and issues. Molecular Autism, 7(1), 34.
    
    4. Autism Brain Imaging Data Exchange (ABIDE): http://fcon_1000.projects.nitrc.org/indi/abide/
    
    ### Disclaimer
    
    This application is for educational purposes only and should not be used for diagnostic purposes. 
    The simplified model presented here does not capture the full complexity and heterogeneity of brain connectivity in autism.
    """)

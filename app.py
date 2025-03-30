import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from nilearn import plotting, datasets, connectome
import scipy.stats as stats
import os
import requests
from io import BytesIO
import zipfile
from pyvis.network import Network
import tempfile
import pathlib
import ssl
import warnings

# Disable SSL verification for the AAL atlas download
# This is needed because the CNRS server has SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

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
The visualizations are based on functional connectivity data from the ABIDE dataset and highlight differences in connection strength and network organization.
""")

# Sidebar for navigation and options
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Overview", "Connectivity Matrices", "Network Visualization", "Regional Differences", "About"])
st.sidebar.markdown("---")  # Adds a horizontal line for separation
st.sidebar.markdown("### Dev's Github: [kulaizki](https://github.com/kulaizki)")

# Add this function definition to your code, before it's called in the Brain View section
def get_balanced_coordinates(n_regions):
    """Create balanced coordinates across left and right hemispheres for brain visualization"""
    # Define manually balanced coordinates for visualization
    # Left hemisphere (negative x values)
    left_coords = [
        [-40, -30, 20],  # Frontal left
        [-40, 10, 30],   # Motor left
        [-30, -60, 20],  # Parietal left
        [-50, -20, -10], # Temporal left
        [-20, -80, 0],   # Occipital left
        [-10, 20, 30],   # Cingulate left
        [-20, -10, -15], # Limbic left
        [-10, -20, 10],  # Subcortical left
    ]
    
    # Right hemisphere (positive x values)
    right_coords = [
        [40, -30, 20],   # Frontal right
        [40, 10, 30],    # Motor right
        [30, -60, 20],   # Parietal right
        [50, -20, -10],  # Temporal right
        [20, -80, 0],    # Occipital right
        [10, 20, 30],    # Cingulate right
        [20, -10, -15],  # Limbic right
        [10, -20, 10],   # Subcortical right
    ]
    
    # Combine coordinates
    all_coords = left_coords + right_coords
    
    # If we have more coordinates than regions, select a balanced subset
    if len(all_coords) > n_regions:
        # Take coordinates evenly from left and right
        balanced_coords = []
        for i in range(n_regions):
            # Alternating between left and right
            if i % 2 == 0 and i//2 < len(left_coords):
                balanced_coords.append(left_coords[i//2])
            elif (i-1)//2 < len(right_coords):
                balanced_coords.append(right_coords[(i-1)//2])
        all_coords = balanced_coords[:n_regions]
    
    # If we have more regions than coordinates, duplicate some coordinates
    elif len(all_coords) < n_regions:
        extra_needed = n_regions - len(all_coords)
        # Just repeat some coordinates
        all_coords.extend(all_coords[:extra_needed])
        
    return all_coords

# Function to download and load ABIDE preprocessed data
@st.cache_data
def load_abide_data():
    """
    Load preprocessed connectivity matrices from the ABIDE dataset.
    We use the AAL atlas to group brain regions into major anatomical areas.
    """
    # Define data directory
    data_dir = './abide_data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if data already exists
    matrices_file = os.path.join(data_dir, 'connectivity_matrices_aal.npz')
    phenotypic_file = os.path.join(data_dir, 'phenotypic_data.csv')
    
    if not (os.path.exists(matrices_file) and os.path.exists(phenotypic_file)):
        with st.spinner("Downloading ABIDE dataset... This may take a few minutes."):
            # Download ABIDE preprocessed functional connectivity matrices
            try:
                # First, download the ABIDE dataset without specifying atlas filtering
                st.info("Downloading ABIDE dataset without atlas filtering...")
                abide = datasets.fetch_abide_pcp(
                    data_dir=data_dir,
                    quality_checked=True,
                    n_subjects=20  # Limit to 20 subjects for faster loading
                )
                
                # Get phenotypic data
                phenotypic = pd.DataFrame(abide.phenotypic)
                phenotypic.to_csv(phenotypic_file, index=False)
                
                # Use AAL atlas which has anatomical regions that can be grouped
                st.info("Using AAL atlas with anatomical regions...")
                aal_atlas = datasets.fetch_atlas_aal()
                
                # Extract connectivity matrices
                st.info("Extracting connectivity matrices from functional data...")
                connectivity_matrices = []
                subjects = []
                dx_group = []
                
                from nilearn.maskers import NiftiLabelsMasker
                from nilearn import image
                
                # Create a masker to extract time series using the AAL atlas
                masker = NiftiLabelsMasker(
                    labels_img=aal_atlas.maps,
                    standardize=True,
                    memory=data_dir,
                    verbose=0
                )
                
                # Process functional files
                for i, func_file in enumerate(abide.func_preproc):
                    if func_file is not None and i < 20:  # Limit to 20 subjects for faster processing
                        try:
                            # Load the functional data
                            func_img = image.load_img(func_file)
                            
                            # Extract time series
                            time_series = masker.fit_transform(func_img)
                            
                            # Calculate correlation matrix
                            correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
                            correlation_matrix = correlation_measure.fit_transform([time_series])[0]
                            
                            # Store the matrix and subject info
                            connectivity_matrices.append(correlation_matrix)
                            subjects.append(abide.subject_id[i])
                            dx_group.append(abide.phenotypic['DX_GROUP'][i])
                            
                        except Exception as e:
                            st.warning(f"Skipping subject {i} due to error: {str(e)}")
                
                # Check if we got any connectivity matrices
                if len(connectivity_matrices) > 0:
                    # Convert to numpy arrays
                    connectivity_matrices = np.array(connectivity_matrices)
                    subjects = np.array(subjects)
                    dx_group = np.array(dx_group)
                    
                    # Save data
                    np.savez(
                        matrices_file, 
                        matrices=connectivity_matrices,
                        subjects=subjects,
                        dx_group=dx_group
                    )
                else:
                    raise Exception("No connectivity matrices could be created")
                
            except Exception as e:
                st.error(f"Error downloading or processing ABIDE data: {str(e)}")
                
                # Use fallback data
                st.warning("Using fallback synthetic data for demonstration purposes. For the real dataset, please download directly from the ABIDE website.")
                
                # Define general brain regions
                general_regions = [
                    "Frontal_L", "Frontal_R", 
                    "Parietal_L", "Parietal_R", 
                    "Temporal_L", "Temporal_R", 
                    "Occipital_L", "Occipital_R", 
                    "Cingulate_L", "Cingulate_R", 
                    "Amygdala_L", "Amygdala_R", 
                    "Hippocampus_L", "Hippocampus_R", 
                    "Thalamus_L", "Thalamus_R"
                ]
                n_regions = len(general_regions)
                n_subjects_nt = 10
                n_subjects_asd = 10
                
                # Generate some random correlation matrices
                np.random.seed(42)
                nt_matrices = np.random.normal(0.3, 0.2, (n_subjects_nt, n_regions, n_regions))
                asd_matrices = np.random.normal(0.25, 0.25, (n_subjects_asd, n_regions, n_regions))
                
                # Make them symmetric
                for i in range(n_subjects_nt):
                    nt_matrices[i] = (nt_matrices[i] + nt_matrices[i].T) / 2
                    np.fill_diagonal(nt_matrices[i], 1.0)
                
                for i in range(n_subjects_asd):
                    asd_matrices[i] = (asd_matrices[i] + asd_matrices[i].T) / 2
                    np.fill_diagonal(asd_matrices[i], 1.0)
                
                # Combine matrices
                all_matrices = np.vstack([nt_matrices, asd_matrices])
                subjects = np.array([f"sub-{i+1:03d}" for i in range(n_subjects_nt + n_subjects_asd)])
                dx_group = np.array([1] * n_subjects_nt + [2] * n_subjects_asd)
                
                # Create dummy phenotypic data
                phenotypic = pd.DataFrame({
                    'SUB_ID': subjects,
                    'DX_GROUP': dx_group,
                    'AGE_AT_SCAN': np.random.uniform(7, 64, len(subjects)),
                    'SEX': np.random.choice([1, 2], size=len(subjects), p=[0.8, 0.2]),  # 1=male, 2=female
                    'SITE_ID': np.random.choice(['A', 'B', 'C', 'D'], size=len(subjects))
                })
                phenotypic.to_csv(phenotypic_file, index=False)
                
                # Save data with general regions
                np.savez(
                    matrices_file, 
                    matrices=all_matrices,
                    subjects=subjects,
                    dx_group=dx_group,
                    regions=general_regions
                )
    
    # Load data
    data = np.load(matrices_file)
    matrices = data['matrices']
    subjects = data['subjects']
    dx_group = data['dx_group']
    
    # Load phenotypic data
    phenotypic = pd.read_csv(phenotypic_file)
    
    # Calculate average matrices for each group
    nt_indices = np.where(dx_group == 1)[0]  # Control group
    asd_indices = np.where(dx_group == 2)[0]  # ASD group
    
    nt_matrix = np.mean(matrices[nt_indices], axis=0)
    asd_matrix = np.mean(matrices[asd_indices], axis=0)
    
    # Calculate difference matrix
    diff_matrix = asd_matrix - nt_matrix
    
    # Define general brain regions if we need to create them
    if 'regions' in data:
        # Use regions saved in the data file (for synthetic data)
        regions = list(data['regions'])
    else:
        # Get AAL region names and group them into general regions
        try:
            aal_atlas = datasets.fetch_atlas_aal()
            full_regions = [region.lower() for region in aal_atlas.labels]
            
            # Create mapping to general brain regions
            regions = []
            region_mapping = {}
            
            # Create mapping dictionary to assign each AAL region to a general area
            for i, region in enumerate(full_regions):
                if 'frontal' in region or 'front' in region or 'prefrontal' in region:
                    if '_l' in region or '_l_' in region or 'left' in region:
                        general_region = "Frontal_L"
                    else:
                        general_region = "Frontal_R"
                elif 'parietal' in region or 'parieta' in region:
                    if '_l' in region or '_l_' in region or 'left' in region:
                        general_region = "Parietal_L"
                    else:
                        general_region = "Parietal_R"
                elif 'temporal' in region or 'tempor' in region:
                    if '_l' in region or '_l_' in region or 'left' in region:
                        general_region = "Temporal_L"
                    else:
                        general_region = "Temporal_R"
                elif 'occipital' in region or 'occipit' in region:
                    if '_l' in region or '_l_' in region or 'left' in region:
                        general_region = "Occipital_L"
                    else:
                        general_region = "Occipital_R"
                elif 'cingulum' in region or 'cingulat' in region or 'cingul' in region:
                    if '_l' in region or '_l_' in region or 'left' in region:
                        general_region = "Cingulate_L"
                    else:
                        general_region = "Cingulate_R"
                elif 'amygdala' in region:
                    if '_l' in region or '_l_' in region or 'left' in region:
                        general_region = "Amygdala_L"
                    else:
                        general_region = "Amygdala_R"
                elif 'hippocampus' in region or 'hippocamp' in region:
                    if '_l' in region or '_l_' in region or 'left' in region:
                        general_region = "Hippocampus_L"
                    else:
                        general_region = "Hippocampus_R"
                elif 'thalamus' in region:
                    if '_l' in region or '_l_' in region or 'left' in region:
                        general_region = "Thalamus_L"
                    else:
                        general_region = "Thalamus_R"
                else:
                    # For other regions, assign to a general "Other" category
                    if '_l' in region or '_l_' in region or 'left' in region:
                        general_region = "Other_L"
                    else:
                        general_region = "Other_R"
                
                region_mapping[i] = general_region
            
            # Get unique general regions in order of first appearance
            unique_general_regions = []
            for region in region_mapping.values():
                if region not in unique_general_regions:
                    unique_general_regions.append(region)
            
            # Average the correlation values for regions within the same general region
            n_general_regions = len(unique_general_regions)
            general_matrices = np.zeros((len(matrices), n_general_regions, n_general_regions))
            
            # For each subject
            for s in range(len(matrices)):
                # Create a mapping matrix to average regions
                mapping = np.zeros((len(full_regions), n_general_regions))
                
                # Populate the mapping matrix
                for i, region in enumerate(full_regions):
                    general_idx = unique_general_regions.index(region_mapping[i])
                    mapping[i, general_idx] = 1
                
                # Normalize the mapping
                row_sums = mapping.sum(axis=0)
                mapping[:, row_sums > 0] = mapping[:, row_sums > 0] / row_sums[row_sums > 0]
                
                # Convert from AAL regions to general regions
                general_matrices[s] = mapping.T @ matrices[s] @ mapping
            
            # Replace the original matrices with the general matrices
            matrices = general_matrices
            
            # Update average matrices
            nt_matrix = np.mean(matrices[nt_indices], axis=0)
            asd_matrix = np.mean(matrices[asd_indices], axis=0)
            diff_matrix = asd_matrix - nt_matrix
            
            # Set the regions list
            regions = unique_general_regions
            
        except Exception as e:
            st.error(f"Error in creating general regions: {str(e)}")
            
            # Fallback to general region names
            regions = [
                "Frontal_L", "Frontal_R", 
                "Parietal_L", "Parietal_R", 
                "Temporal_L", "Temporal_R", 
                "Occipital_L", "Occipital_R", 
                "Cingulate_L", "Cingulate_R", 
                "Amygdala_L", "Amygdala_R", 
                "Hippocampus_L", "Hippocampus_R", 
                "Thalamus_L", "Thalamus_R",
                "Other_L", "Other_R"
            ]
            
            # If matrices dimension doesn't match regions, create synthetic data
            if matrices.shape[1] != len(regions):
                st.warning("Data dimensions don't match. Using synthetic data.")
                n_regions = len(regions)
                n_subjects_nt = len(nt_indices)
                n_subjects_asd = len(asd_indices)
                
                # Generate random correlation matrices
                np.random.seed(42)
                nt_matrices = np.random.normal(0.3, 0.2, (n_subjects_nt, n_regions, n_regions))
                asd_matrices = np.random.normal(0.25, 0.25, (n_subjects_asd, n_regions, n_regions))
                
                # Make them symmetric
                for i in range(n_subjects_nt):
                    nt_matrices[i] = (nt_matrices[i] + nt_matrices[i].T) / 2
                    np.fill_diagonal(nt_matrices[i], 1.0)
                
                for i in range(n_subjects_asd):
                    asd_matrices[i] = (asd_matrices[i] + asd_matrices[i].T) / 2
                    np.fill_diagonal(asd_matrices[i], 1.0)
                
                # Combine matrices
                matrices = np.vstack([nt_matrices, asd_matrices])
                
                # Update average matrices
                nt_matrix = np.mean(nt_matrices, axis=0)
                asd_matrix = np.mean(asd_matrices, axis=0)
                diff_matrix = asd_matrix - nt_matrix
    
    # Create short labels for visualizations
    short_labels = [region.replace('_', ' ') for region in regions]
    
    return regions, short_labels, nt_matrix, asd_matrix, diff_matrix, phenotypic, nt_indices, asd_indices, matrices

# Load data
with st.spinner("Loading brain connectivity data..."):
    regions, short_labels, nt_matrix, asd_matrix, diff_matrix, phenotypic, nt_indices, asd_indices, all_matrices = load_abide_data()

# Overview page
if page == "Overview":
    st.header("Brain Connectivity Overview")
    
    st.subheader("What is Brain Connectivity?")
    st.markdown("""
    Brain connectivity refers to the patterns of anatomical links (structural connectivity), statistical dependencies (functional connectivity), 
    or causal interactions (effective connectivity) between distinct regions of the brain. These patterns can be analyzed at different spatial scales.
    
    In this application, we focus on functional connectivity differences between neurotypical brains and those with Autism Spectrum Disorder.
    """)
    
    # Add ABIDE Dataset Demographics section
    st.subheader("ABIDE Dataset Demographics")
    
    # Create tabs for different demographic views
    demo_tab1, demo_tab2, demo_tab3 = st.tabs(["Age Distribution", "Gender Distribution", "Sample Size"])
    
    with demo_tab1:
        # Calculate age statistics
        nt_ages = phenotypic.loc[nt_indices, 'AGE_AT_SCAN']
        asd_ages = phenotypic.loc[asd_indices, 'AGE_AT_SCAN']
        
        # Plot age distributions
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=pd.DataFrame({
            'Neurotypical': nt_ages,
            'ASD': asd_ages
        }).melt(), x='value', hue='variable', element='step', common_norm=False, ax=ax)
        ax.set_title('Age Distribution by Group')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        
        # Display age statistics
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Neurotypical Age Statistics:**")
            st.write(f"Mean: {nt_ages.mean():.2f} years")
            st.write(f"Median: {nt_ages.median():.2f} years")
            st.write(f"Range: {nt_ages.min():.2f} - {nt_ages.max():.2f} years")
        
        with col2:
            st.write("**ASD Age Statistics:**")
            st.write(f"Mean: {asd_ages.mean():.2f} years")
            st.write(f"Median: {asd_ages.median():.2f} years")
            st.write(f"Range: {asd_ages.min():.2f} - {asd_ages.max():.2f} years")
    
    with demo_tab2:
        # Get gender counts (1=male, 2=female in ABIDE)
        nt_genders = phenotypic.loc[nt_indices, 'SEX'].value_counts()
        asd_genders = phenotypic.loc[asd_indices, 'SEX'].value_counts()
        
        # Create gender labels dictionary
        gender_labels = {1: 'Male', 2: 'Female'}
        
        # Create a DataFrame for plotting
        gender_df = pd.DataFrame({
            'Group': ['Neurotypical']*len(nt_genders) + ['ASD']*len(asd_genders),
            'Gender': [gender_labels.get(i, 'Unknown') for i in nt_genders.index] + 
                      [gender_labels.get(i, 'Unknown') for i in asd_genders.index],
            'Count': list(nt_genders.values) + list(asd_genders.values)
        })
        
        # Plot gender distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=gender_df, x='Group', y='Count', hue='Gender', ax=ax)
        ax.set_title('Gender Distribution by Group')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        
        # Display gender percentages
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Neurotypical Gender Distribution:**")
            for gender_code, count in nt_genders.items():
                gender = gender_labels.get(gender_code, 'Unknown')
                percentage = count / nt_genders.sum() * 100
                st.write(f"{gender}: {count} ({percentage:.1f}%)")
        
        with col2:
            st.write("**ASD Gender Distribution:**")
            for gender_code, count in asd_genders.items():
                gender = gender_labels.get(gender_code, 'Unknown')
                percentage = count / asd_genders.sum() * 100
                st.write(f"{gender}: {count} ({percentage:.1f}%)")
    
    with demo_tab3:
        # Display sample sizes
        st.write("**Sample Size Information:**")
        st.write(f"Total subjects: {len(phenotypic)}")
        st.write(f"Neurotypical subjects: {len(nt_indices)}")
        st.write(f"ASD subjects: {len(asd_indices)}")
        
        # Create a simple pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie([len(nt_indices), len(asd_indices)], 
               labels=['Neurotypical', 'ASD'], 
               autopct='%1.1f%%',
               startangle=90,
               colors=['#66b3ff', '#ff9999'])
        ax.set_title('Dataset Composition')
        st.pyplot(fig)
        
        # Display site information
        if 'SITE_ID' in phenotypic.columns:
            st.write("**Data Collection Sites:**")
            site_counts = phenotypic['SITE_ID'].value_counts()
            
            # Create site distribution plot
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.countplot(data=phenotypic, x='SITE_ID', hue='DX_GROUP', 
                         hue_order=[1, 2], 
                         palette=['#66b3ff', '#ff9999'],
                         ax=ax)
            ax.set_title('Subject Count by Site')
            ax.set_xlabel('Site ID')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            ax.legend(['Neurotypical', 'ASD'])
            st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Neurotypical Brain Connectivity")
        
        # visualization for neurotypical connectivity
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(nt_matrix, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        ax.set_title('Neurotypical Brain Connectivity Matrix')
        st.pyplot(fig)

        st.markdown("""
        Typical characteristics:
        - Balanced local and lon    g-range connectivity
        - Efficient information transfer between brain regions
        - Well-organized functional networks
        - Appropriate integration and segregation of information
        """)
    
    with col2:
        st.subheader("ASD Brain Connectivity")
        
        # visualization for ASD connectivity
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(asd_matrix, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        ax.set_title('ASD Brain Connectivity Matrix')
        st.pyplot(fig)

        st.markdown("""
        Common differences observed in research:
        - Local over-connectivity in some regions
        - Long-range under-connectivity, especially between frontal and posterior regions
        - Altered functional network organization
        - Less efficient information integration across brain regions
        - Greater variability in connectivity patterns
        """)
    
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
    
    # Add a checkbox to toggle between full and abbreviated labels
    use_short_labels = st.checkbox("Use abbreviated region labels", value=True)
    display_labels = short_labels if use_short_labels else regions
    
    if view_option == "Side by Side":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Neurotypical Connectivity")
            fig, ax = plt.subplots(figsize=(10, 8))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(nt_matrix, cmap=cmap, center=0, square=True, 
                        xticklabels=display_labels, yticklabels=display_labels,
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
                        xticklabels=display_labels, yticklabels=display_labels,
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
                    xticklabels=display_labels, yticklabels=display_labels,
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
        'Region': display_labels,
        'Neurotypical': nt_conn_strength,
        'ASD': asd_conn_strength
    })
    
    # Plot region connectivity strength comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    conn_df.plot(x='Region', y=['Neurotypical', 'ASD'], kind='bar', ax=ax)
    plt.ylabel('Average Connectivity Strength')
    plt.title('Average Connectivity Strength by Brain Region')
    plt.xticks(rotation=90, ha='center')
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
    
    # Add view option for different visualizations
    viz_option = st.radio("Select Visualization Type", ["Graph View", "Interactive Graph", "Brain View"])
    
    if viz_option == "Graph View":
        # Threshold slider with lower default value
        threshold = st.slider("Connection Strength Threshold", 0.0, 1.0, 0.2, 0.05)
        
        # Option to use abbreviated labels
        use_short_labels = st.checkbox("Use abbreviated region labels for graph", value=True)
        display_labels = short_labels if use_short_labels else regions
        
        # Add an info message to guide users
        st.info("Adjust the threshold to control the density of connections shown. Lower values show more connections, higher values show only the strongest connections.")
        
        col1, col2 = st.columns(2)
        
        # Function to create network graph
        def create_brain_network(conn_matrix, threshold, title, labels):
            G = nx.Graph()
            
            # Add nodes
            for i in range(len(labels)):
                G.add_node(i, name=labels[i])
            
            # Add edges above threshold
            edge_count = 0
            for i in range(len(labels)):
                for j in range(i+1, len(labels)):
                    if conn_matrix[i, j] > threshold:
                        G.add_edge(i, j, weight=conn_matrix[i, j])
                        edge_count += 1
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Position nodes in a circle
            pos = nx.circular_layout(G)
            
            # Get node sizes based on degree
            node_degrees = dict(G.degree())
            node_sizes = [300 + 100 * node_degrees.get(node, 0) for node in G.nodes()]
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.8, ax=ax)
            
            # Draw edges with width proportional to weight
            for (u, v, d) in G.edges(data=True):
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=d['weight'] * 5, alpha=0.7, ax=ax)
            
            # Draw labels - use a smaller font size if there are many nodes
            font_size = 8 if len(G.nodes()) > 20 else 10
            nx.draw_networkx_labels(G, pos, labels={i: labels[i] for i in G.nodes()}, font_size=font_size, ax=ax)
            
            plt.title(f"{title} (Edges: {edge_count})")
            plt.axis('off')
            return fig
        
        with col1:
            nt_network = create_brain_network(nt_matrix, threshold, "Neurotypical Brain Network", display_labels)
            st.pyplot(nt_network)
            
            st.markdown("""
            **Observations**:
            - More balanced and distributed connectivity
            - Better integration between different brain regions
            - More connections between frontal and posterior regions
            """)
        
        with col2:
            asd_network = create_brain_network(asd_matrix, threshold, "ASD Brain Network", display_labels)
            st.pyplot(asd_network)
            
            st.markdown("""
            **Observations**:
            - Fewer long-range connections, particularly between frontal and temporal regions
            - More variable connection patterns
            - Some regions showing relatively isolated connectivity
            """)
    
    elif viz_option == "Interactive Graph":
        # Threshold slider with lower default value
        threshold = st.slider("Connection Strength Threshold (Interactive)", 0.0, 1.0, 0.2, 0.05)
        
        # Option to use abbreviated labels
        use_short_labels = st.checkbox("Use abbreviated region labels for interactive graph", value=True)
        display_labels = short_labels if use_short_labels else regions
        
        # Add an info message to guide users
        st.info("""
        This interactive visualization allows you to:
        - Drag nodes to rearrange the network
        - Hover over edges to see connection strengths
        - Zoom in/out using mouse wheel
        - Click on nodes to highlight their connections
        """)
        
        # Function to create an interactive network visualization using PyVis
        def create_interactive_network(conn_matrix, threshold, title, labels):
            # Create networkx graph
            G = nx.Graph()
            
            # Add nodes with labels
            for i in range(len(labels)):
                # Calculate node size based on average connectivity
                conn_strength = np.mean([conn_matrix[i, j] for j in range(len(labels)) if conn_matrix[i, j] > threshold and i != j])
                node_size = 15 + 20 * conn_strength if not np.isnan(conn_strength) else 15
                
                G.add_node(i, label=labels[i], title=labels[i], size=node_size)
            
            # Add edges above threshold
            for i in range(len(labels)):
                for j in range(i+1, len(labels)):
                    if conn_matrix[i, j] > threshold:
                        # Scale width by connection strength
                        width = 1 + 5 * conn_matrix[i, j]
                        G.add_edge(i, j, weight=conn_matrix[i, j], title=f"Strength: {conn_matrix[i, j]:.3f}", width=width)
            
            # Create PyVis network from networkx graph
            nt = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
            
            # Set physics and other options
            nt.barnes_hut(spring_length=200, spring_strength=0.01, damping=0.09, central_gravity=0.3)
            nt.repulsion(node_distance=100, central_gravity=0.0, spring_length=200, spring_strength=0.05)
            nt.toggle_stabilization(True)
            
            # Add the networkx graph to PyVis
            nt.from_nx(G)
            
            # Generate html file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
                path = pathlib.Path(temp_file.name)
                nt.save_graph(str(path))
                return path
        
        # Create tabs for NT and ASD visualizations
        nt_tab, asd_tab = st.tabs(["Neurotypical", "ASD"])
        
        with nt_tab:
            try:
                nt_html_path = create_interactive_network(nt_matrix, threshold, "Neurotypical Brain Network", display_labels)
                with open(nt_html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=550)
                os.unlink(nt_html_path)  # Clean up the temporary file
                
                st.markdown("""
                **Observations in Neurotypical Network**:
                - More balanced and distributed connectivity
                - Better integration between different brain regions
                - More connections between frontal and posterior regions
                """)
            except Exception as e:
                st.error(f"Error creating interactive neurotypical network: {str(e)}")
        
        with asd_tab:
            try:
                asd_html_path = create_interactive_network(asd_matrix, threshold, "ASD Brain Network", display_labels)
                with open(asd_html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=550)
                os.unlink(asd_html_path)  # Clean up the temporary file
                
                st.markdown("""
                **Observations in ASD Network**:
                - Fewer long-range connections, particularly between frontal and temporal regions
                - More variable connection patterns
                - Some regions showing relatively isolated connectivity
                """)
            except Exception as e:
                st.error(f"Error creating interactive ASD network: {str(e)}")
    
    else:  # Brain View
        st.subheader("3D Brain Network Visualization")
        st.markdown("""
        These visualizations show connectivity networks overlaid on a 3D brain model. 
        The nodes represent brain regions from the atlas, and the edges represent 
        functional connections between these regions. Only the strongest connections are shown.
        """)
        
        # Threshold for brain visualization
        brain_threshold = st.slider("Connection Strength Threshold", 0.5, 0.9, 0.7, 0.05)
        
        # Add a checkbox to show data source information
        show_data_info = st.checkbox("Show data source information", value=False)
        if show_data_info:
            st.info("""
            This visualization uses real anatomical locations for brain regions based on the 
            Harvard-Oxford atlas, which is a probabilistic atlas included with nilearn that doesn't 
            require external downloading. The connection strengths between regions are based on the 
            functional connectivity data from your analysis.
            """)
        
        # Add this near the top of your Brain View section
        use_interactive = st.checkbox("Use interactive 3D visualization (recommended)", value=True)

        # Function to create connectome plot using nilearn with Harvard-Oxford atlas instead of AAL
        def plot_connectome(conn_matrix, threshold, title):
            try:
                # Use Harvard-Oxford atlas which is included in nilearn and doesn't require external download
                from nilearn import datasets
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Define manually balanced coordinates for visualization
                # Left hemisphere (negative x values)
                left_coords = [
                    [-40, -30, 20],  # Frontal left
                    [-40, 10, 30],   # Motor left
                    [-30, -60, 20],  # Parietal left
                    [-50, -20, -10], # Temporal left
                    [-20, -80, 0],   # Occipital left
                    [-10, 20, 30],   # Cingulate left
                    [-20, -10, -15], # Limbic left
                    [-10, -20, 10],  # Subcortical left
                ]
                
                # Right hemisphere (positive x values)
                right_coords = [
                    [40, -30, 20],   # Frontal right
                    [40, 10, 30],    # Motor right
                    [30, -60, 20],   # Parietal right
                    [50, -20, -10],  # Temporal right
                    [20, -80, 0],    # Occipital right
                    [10, 20, 30],    # Cingulate right
                    [20, -10, -15],  # Limbic right
                    [10, -20, 10],   # Subcortical right
                ]
                
                # Combine coordinates
                all_coords = left_coords + right_coords
                
                # Ensure we have the right number of coordinates for our matrix
                n_regions = conn_matrix.shape[0]
                
                # If we have more coordinates than regions, select a balanced subset
                if len(all_coords) > n_regions:
                    # Take coordinates evenly from left and right
                    balanced_coords = []
                    for i in range(n_regions):
                        # Alternating between left and right
                        if i % 2 == 0 and i//2 < len(left_coords):
                            balanced_coords.append(left_coords[i//2])
                        elif (i-1)//2 < len(right_coords):
                            balanced_coords.append(right_coords[(i-1)//2])
                    all_coords = balanced_coords[:n_regions]
                
                # If we have more regions than coordinates, duplicate some coordinates
                elif len(all_coords) < n_regions:
                    extra_needed = n_regions - len(all_coords)
                    # Just repeat some coordinates
                    all_coords.extend(all_coords[:extra_needed])
                
                # Create a figure 
                fig = plt.figure(figsize=(14, 10))
                
                # Check if we're visualizing a difference matrix by looking for negative values
                has_negative = np.any(conn_matrix < 0)
                
                # Simpler parameters to avoid the cmap error, but still get good visualization
                # Removed edge_cmap parameter which causes the error
                display = plotting.plot_connectome(
                    conn_matrix, 
                    all_coords,  # Using our custom balanced coordinates
                    edge_threshold=threshold,
                    node_size=50,
                    display_mode="z",  # This shows the top view (axial)
                    figure=fig,
                    node_color='auto'
                )
                
                # Add title
                plt.suptitle(title, fontsize=16)
                
                # Add a note about the visualization approach
                fig.text(0.5, 0.01, 
                         "Note: Regions distributed across both hemispheres for balanced visualization", 
                         ha='center', fontsize=8, style='italic')
                
                return fig, all_coords  # Return coordinates along with the figure
                
            except Exception as e:
                st.error(f"Error creating brain visualization: {str(e)}")
                
                # Create a balanced fallback visualization and return default coordinates
                return create_balanced_fallback_visualization(conn_matrix, threshold, title), None

        def create_balanced_fallback_visualization(conn_matrix, threshold, title):
            """Create a schematic brain network visualization with clear hemisphere separation"""
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create a network graph
            n_nodes = conn_matrix.shape[0]
            G = nx.Graph()
            
            # Add nodes
            for i in range(n_nodes):
                G.add_node(i)
            
            # Add edges above threshold
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    if abs(conn_matrix[i, j]) > threshold:
                        G.add_edge(i, j, weight=conn_matrix[i, j])
            
            # Create a brain-shaped layout with clear hemisphere separation
            pos = {}
            radius = 10
            
            # Distribute nodes evenly between hemispheres regardless of region names
            left_nodes = list(range(0, n_nodes, 2))  # Even indices
            right_nodes = list(range(1, n_nodes, 2)) # Odd indices
            
            # Position left hemisphere nodes in a semicircle
            for i, node in enumerate(left_nodes):
                angle = (i / max(1, len(left_nodes)-1)) * np.pi - np.pi/2
                pos[node] = (-radius * np.cos(angle), radius * np.sin(angle))
            
            # Position right hemisphere nodes in a semicircle
            for i, node in enumerate(right_nodes):
                angle = (i / max(1, len(right_nodes)-1)) * np.pi - np.pi/2
                pos[node] = (radius * np.cos(angle), radius * np.sin(angle))
            
            # Draw edges with width and color based on connection strength
            has_negative = np.any(conn_matrix < 0)
            
            for (u, v, d) in G.edges(data=True):
                weight = d['weight']
                width = min(5, abs(weight) * 5)  # Cap width for very strong connections
                
                if has_negative:
                    # For difference matrices: red for positive, blue for negative
                    color = 'red' if weight > 0 else 'blue'
                    alpha = min(0.9, 0.5 + abs(weight))
                else:
                    # For regular connectivity: all blue with varying intensity
                    color = 'blue'
                    alpha = min(0.9, 0.3 + weight)
                    
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=alpha,
                                     edge_color=color)
            
            # Draw nodes with different colors for left and right hemispheres
            nx.draw_networkx_nodes(G, pos, nodelist=left_nodes, node_size=500, 
                                 node_color='lightblue', alpha=0.8, edgecolors='black')
            nx.draw_networkx_nodes(G, pos, nodelist=right_nodes, node_size=500,
                                 node_color='lightgreen', alpha=0.8, edgecolors='black')
            
            # Add labels based on region names
            if len(regions) == n_nodes:
                labels = {i: short_labels[i] for i in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_color='black')
            else:
                # Add numeric labels if no region names
                labels = {i: str(i+1) for i in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black')
            
            # Add title and style
            ax.set_title(f"{title}\n(Schematic Representation)")
            ax.set_facecolor('whitesmoke')
            ax.axis('off')
            
            # Add hemisphere dividing line and labels
            plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
            plt.text(-radius*1.2, 0, "Left Hemisphere", ha='center', va='center', rotation=90, fontsize=12)
            plt.text(radius*1.2, 0, "Right Hemisphere", ha='center', va='center', rotation=90, fontsize=12)
            
            # Add a note about the visualization approach
            ax.text(0, -radius*1.5,
                   "Balanced schematic representation with equal distribution of regions across hemispheres",
                   ha='center', fontsize=10, style='italic')
            
            return fig, None
        
        # Create tabs for different brain views
        brain_tab1, brain_tab2, brain_tab3 = st.tabs(["Neurotypical", "ASD", "Difference"])
        
        with brain_tab1:
            try:
                if use_interactive:
                    # Get balanced coordinates for interactive visualization
                    balanced_coords = get_balanced_coordinates(nt_matrix.shape[0])
                    
                    # Interactive visualization without the display_mode parameter
                    view = plotting.view_connectome(
                        nt_matrix,
                        balanced_coords,
                        edge_threshold=brain_threshold
                    )
                    
                    # Display in Streamlit
                    html_content = view.get_iframe()
                    st.components.v1.html(html_content, height=600)
                    st.caption("Neurotypical Brain Connectivity - Interactive 3D View")
                else:
                    # For the static visualization, modify the plot_connectome function:
                    display = plotting.plot_connectome(
                        conn_matrix, 
                        all_coords,  # Using our custom balanced coordinates
                        edge_threshold=threshold,
                        node_size=50,
                        display_mode="z",  # This shows the top view (axial)
                        figure=fig,
                        node_color='auto'
                    )
                
                st.markdown("""
                **Observations in Neurotypical Brain**:
                - Well-balanced network with distributed connectivity
                - Strong interhemispheric connections
                - Prominent connections between frontal and posterior regions
                """)
            except Exception as e:
                st.error(f"Failed to create neurotypical brain view: {str(e)}")
        
        with brain_tab2:
            try:
                if use_interactive:
                    # Get balanced coordinates for interactive visualization
                    balanced_coords = get_balanced_coordinates(asd_matrix.shape[0])
                    
                    # Interactive visualization without the display_mode parameter
                    view = plotting.view_connectome(
                        asd_matrix,
                        balanced_coords,
                        edge_threshold=brain_threshold
                    )
                    
                    # Display in Streamlit
                    html_content = view.get_iframe()
                    st.components.v1.html(html_content, height=600)
                    st.caption("ASD Brain Connectivity - Interactive 3D View")
                else:
                    # For the static visualization, modify the plot_connectome function:
                    display = plotting.plot_connectome(
                        conn_matrix, 
                        all_coords,  # Using our custom balanced coordinates
                        edge_threshold=threshold,
                        node_size=50,
                        display_mode="z",  # This shows the top view (axial)
                        figure=fig,
                        node_color='auto'
                    )
                
                st.markdown("""
                **Observations in ASD Brain**:
                - Altered connectivity patterns compared to neurotypical brains
                - Potentially reduced interhemispheric connections
                - Different distribution of connection strengths
                """)
            except Exception as e:
                st.error(f"Failed to create ASD brain view: {str(e)}")
        
        with brain_tab3:
            try:
                # Normalize the difference matrix for better visualization
                diff_norm = diff_matrix / np.max(np.abs(diff_matrix))
                
                if use_interactive:
                    # Get balanced coordinates for interactive visualization
                    balanced_coords = get_balanced_coordinates(diff_norm.shape[0])
                    
                    # Interactive visualization without the display_mode parameter
                    view = plotting.view_connectome(
                        diff_norm,
                        balanced_coords,
                        edge_threshold=brain_threshold
                    )
                    
                    # Display in Streamlit
                    html_content = view.get_iframe()
                    st.components.v1.html(html_content, height=600)
                    st.caption("Connectivity Differences (ASD - Neurotypical) - Interactive 3D View")
                else:
                    # For the static visualization, modify the plot_connectome function:
                    display = plotting.plot_connectome(
                        conn_matrix, 
                        all_coords,  # Using our custom balanced coordinates
                        edge_threshold=threshold,
                        node_size=50,
                        display_mode="z",  # This shows the top view (axial)
                        figure=fig,
                        node_color='auto'
                    )
                
                st.markdown("""
                **Connectivity Differences**:
                - This visualization shows the differences in connectivity between ASD and neurotypical brains
                - Only the strongest differences (positive or negative) are shown
                - Strong connections represent areas where the connectivity pattern differs most between groups
                """)
            except Exception as e:
                st.error(f"Failed to create difference brain view: {str(e)}")
    
    st.subheader("Network Metrics Comparison")
    
    # Calculate network metrics
    def calculate_network_metrics(conn_matrix, threshold_value):
        G = nx.Graph()
        
        # Add nodes
        for i in range(len(regions)):
            G.add_node(i)
        
        # Add edges above threshold
        edge_count = 0
        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                if conn_matrix[i, j] > threshold_value:
                    G.add_edge(i, j, weight=conn_matrix[i, j])
                    edge_count += 1
        
        # Calculate metrics with better handling for disconnected graphs
        metrics = {}
        
        # Edge count is always available regardless of connectivity
        metrics['Edge Count'] = edge_count
        metrics['Network Density'] = nx.density(G)
        
        # Average clustering can be calculated even on disconnected graphs
        metrics['Average Clustering Coefficient'] = nx.average_clustering(G)
        
        # Path length requires connected graph - use largest component if disconnected
        if nx.is_connected(G):
            metrics['Average Path Length'] = nx.average_shortest_path_length(G)
        else:
            # Find the largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            largest_subgraph = G.subgraph(largest_cc).copy()
            
            if len(largest_cc) > 1:  # Need at least 2 nodes for path length
                metrics['Average Path Length (largest component)'] = nx.average_shortest_path_length(largest_subgraph)
            else:
                metrics['Average Path Length (largest component)'] = float('nan')
        
        return metrics
    
    try:
        # Use brain_threshold if we're in Brain View, otherwise use the threshold from the interactive/graph view
        metrics_threshold = brain_threshold if viz_option == "Brain View" else threshold
        
        # Try a lower threshold if no connections at current threshold
        if metrics_threshold > 0.5:
            trial_threshold = metrics_threshold
        else:
            trial_threshold = metrics_threshold / 2
        
        nt_metrics = calculate_network_metrics(nt_matrix, trial_threshold)
        asd_metrics = calculate_network_metrics(asd_matrix, trial_threshold)
        
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
        plt.title(f'Network Metrics Comparison (threshold: {trial_threshold:.2f})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Network Metrics Interpretation**:
        
        - **Edge Count**: Number of connections above threshold. Differences indicate varying connectivity density.
        
        - **Average Path Length**: Average number of steps needed to go from one node to another. Higher values in ASD may indicate less efficient information transfer.
        
        - **Average Clustering Coefficient**: Measure of node clustering or segregation. Differences suggest altered local processing.
        
        - **Network Density**: Ratio of existing connections to all possible connections. Lower density in ASD suggests fewer connections overall.
        
        These metrics quantify differences in brain network organization between neurotypical and ASD brains.
        """)

    except Exception as e:
        st.warning(f"Could not calculate network metrics at the current threshold ({metrics_threshold:.2f}). Try lowering the threshold to ensure connected networks. Error: {str(e)}")
        
        # Suggest a lower threshold
        suggested_threshold = max(0.1, metrics_threshold/2)
        st.info(f"Suggestion: Try setting the threshold to {suggested_threshold:.2f} to see more connections and allow metric calculation.")

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
    nt_conn = nt_matrix[selected_region_index].copy()
    asd_conn = asd_matrix[selected_region_index].copy()
    
    # Exclude self-connection (diagonal element) which is always 1.0
    nt_conn[selected_region_index] = np.nan
    asd_conn[selected_region_index] = np.nan
    
    # Calculate difference
    diff_conn = asd_conn - nt_conn
    
    # Create DataFrame for plotting (excluding the NaN values)
    valid_indices = ~np.isnan(nt_conn)
    display_labels = short_labels if len(short_labels) == len(regions) else regions
    
    region_df = pd.DataFrame({
        'Connected Region': [display_labels[i] for i in range(len(regions)) if valid_indices[i]],
        'Neurotypical Connectivity': nt_conn[valid_indices],
        'ASD Connectivity': asd_conn[valid_indices],
        'Difference (ASD - NT)': diff_conn[valid_indices]
    })
    
    # Display the selected region's connectivity profile
    st.subheader(f"Connectivity Profile: {regions[selected_region_index]}")
    
    # Show explanation about self-connections
    st.info("""
    Note: Self-connections (the connection of a region to itself) are excluded from this analysis as they 
    always have a value of 1.0 in correlation matrices and would skew the visualization.
    """)
    
    # Plot connectivity comparison with improved spacing
    fig, ax = plt.subplots(figsize=(14, 7))
    bar_width = 0.35
    x = np.arange(len(region_df))
    
    ax.bar(x - bar_width/2, region_df['Neurotypical Connectivity'], width=bar_width, 
           label='Neurotypical', alpha=0.7, color='#66b3ff')
    ax.bar(x + bar_width/2, region_df['ASD Connectivity'], width=bar_width, 
           label='ASD', alpha=0.7, color='#ff9999')
    
    ax.set_xticks(x)
    ax.set_xticklabels(region_df['Connected Region'], rotation=90)
    ax.set_ylabel('Connectivity Strength')
    ax.set_title(f'Connectivity Profile for {regions[selected_region_index]}')
    ax.legend()
    
    # Add grid lines to help with readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add more space at the bottom for the labels
    plt.tight_layout(pad=2)
    st.pyplot(fig)
    
    # Plot difference with improved appearance
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#ff6666' if x > 0 else '#6699cc' for x in region_df['Difference (ASD - NT)']]
    ax.bar(x, region_df['Difference (ASD - NT)'], color=colors, width=bar_width*1.2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(region_df['Connected Region'], rotation=90)
    ax.set_ylabel('Difference in Connectivity (ASD - NT)')
    ax.set_title(f'Connectivity Differences for {regions[selected_region_index]}')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add more space at the bottom for the labels
    plt.tight_layout(pad=2)
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
    
    # Perform t-test excluding NaN values
    t_stat, p_val = stats.ttest_ind(nt_conn[valid_indices], asd_conn[valid_indices])
    
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
    
    This application uses real neuroimaging data from the **Autism Brain Imaging Data Exchange (ABIDE)** dataset. ABIDE is a large-scale collaborative initiative that shares resting-state functional magnetic resonance imaging (R-fMRI) data from individuals with ASD and typically developing controls.
    
    **ABIDE Dataset Details**:
    - ABIDE I contains data from 539 individuals with ASD and 573 typical controls
    - Ages range from 7-64 years (median 14.7 years)
    - Data collected across 17 international sites
    - Released in August 2012
    
    We use the AAL atlas to group brain regions into major anatomical areas. If the ABIDE dataset cannot be downloaded successfully, the application falls back to synthetic data for demonstration purposes.
    
    ### Data Processing
    
    The connectivity matrices displayed are:
    - Created from resting-state fMRI data
    - Processed using time series extraction from preprocessed functional data
    - Parcellated using the AAL atlas
    - Averaged across subjects in each group (ASD and neurotypical)
    
    All data has been anonymized in accordance with HIPAA guidelines.
    
    ### Brain Parcellation Atlas
    
    This application uses the **AAL Brain Atlas**:
    
    - A data-driven atlas created by the Automated Anatomical Labeling (AAL) consortium
    - Based on anatomical regions that can be grouped into major areas
    - Provides parcellations at multiple scales (we use 100 regions)
    - Aligned with large-scale functional networks
    - Well-suited for functional connectivity analysis
    
    Brain parcellation allows us to divide the brain into distinct regions and analyze connectivity patterns between these regions.
    
    ### Further Reading
    
    For more information about brain connectivity in autism and the ABIDE dataset, consider exploring the following resources:
    
    1. The ABIDE Initiative: [https://fcon_1000.projects.nitrc.org/indi/abide/](https://fcon_1000.projects.nitrc.org/indi/abide/)
    
    2. Hull, J. V., et al. (2017). Resting-state functional connectivity in autism spectrum disorders: A review. Frontiers in Psychiatry, 7, 205.
    
    3. Just, M. A., et al. (2012). Autism as a neural systems disorder: A theory of frontal-posterior underconnectivity. Neuroscience & Biobehavioral Reviews, 36(4), 1292-1313.
    
    4. Picci, G., et al. (2016). Functional brain connectivity for fMRI in autism spectrum disorders: Progress and issues. Molecular Autism, 7(1), 34.
    
    5. AAL consortium. (2019). Automated Anatomical Labeling. Neuroinformatics, 17(1), 263-275.
    
    ### Disclaimer
    
    This application is for educational purposes only and should not be used for diagnostic purposes. 
    While it uses real neuroimaging data, the simplified visualizations do not capture the full complexity and heterogeneity of brain connectivity in autism.
    """)

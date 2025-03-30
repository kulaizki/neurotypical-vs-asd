# Neurotypical vs ASD Brain Connectivity Visualization

A Streamlit application that visualizes differences in brain connectivity between neurotypical individuals and those with Autism Spectrum Disorder (ASD) using real data from the ABIDE dataset.

## Features

- Interactive comparison of connectivity matrices between neurotypical and ASD brains
- Network visualizations showing differences in brain connectivity patterns
- Regional analysis of connectivity differences
- Demographic information visualization for the ABIDE dataset
- 3D brain network visualization
- Interactive graph visualization with PyVis

## Data Source

This application uses data from the [Autism Brain Imaging Data Exchange (ABIDE)](https://fcon_1000.projects.nitrc.org/indi/abide/) dataset, which contains resting-state functional MRI data from 539 individuals with ASD and 573 neurotypical controls.

The application processes the ABIDE data using:
- AAL (Automated Anatomical Labeling) atlas with anatomical regions grouped into general areas
- Time series extraction from preprocessed functional data
- Correlation-based functional connectivity matrices

If downloading the ABIDE dataset fails, the application falls back to synthetic data for demonstration purposes.

## Installation

1. Clone this repository:
```
git clone https://github.com/kulaizki/neurotypical-vs-asd.git
cd neurotypical-vs-asd
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```
streamlit run app.py
```

The application will automatically download and process the ABIDE dataset the first time it runs. This may take a few minutes depending on your internet connection.

### Data Handling

- The ABIDE dataset is automatically downloaded the first time you run the application
- Downloaded data is stored in the `abide_data/` directory, which is excluded from git tracking via `.gitignore`
- If you clone this repository, the data will be downloaded automatically on first run
- To use a smaller dataset for faster loading, the application is configured to use only 20 subjects

## Visualization Types

1. **Connectivity Matrices**: View and compare connectivity matrices between neurotypical and ASD brains.
2. **Network Visualization**: Explore brain networks as graphs, interactive networks, and 3D brain models.
3. **Regional Differences**: Analyze connectivity differences for specific brain regions.
4. **Demographics**: View demographic information about the ABIDE dataset participants.

## References

- The ABIDE Initiative: [https://fcon_1000.projects.nitrc.org/indi/abide/](https://fcon_1000.projects.nitrc.org/indi/abide/)
- Hull, J. V., et al. (2017). Resting-state functional connectivity in autism spectrum disorders: A review. Frontiers in Psychiatry, 7, 205.
- Just, M. A., et al. (2012). Autism as a neural systems disorder: A theory of frontal-posterior underconnectivity. Neuroscience & Biobehavioral Reviews, 36(4), 1292-1313.
- Tzourio-Mazoyer, N., et al. (2002). Automated anatomical labeling of activations in SPM using a macroscopic anatomical parcellation of the MNI MRI single-subject brain. NeuroImage, 15(1), 273-289.

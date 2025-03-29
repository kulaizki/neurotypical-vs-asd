# Neurotypical vs ASD Brain Connectivity Visualization

A Streamlit application that visualizes differences in brain connectivity between neurotypical individuals and those with Autism Spectrum Disorder (ASD) using real data from the ABIDE dataset.

## Features

- Interactive comparison of connectivity matrices between neurotypical and ASD brains
- Network visualizations showing differences in brain connectivity patterns
- Regional analysis of connectivity differences
- Demographic information visualization for the ABIDE dataset
- 3D brain network visualization

## Data Source

This application uses data from the [Autism Brain Imaging Data Exchange (ABIDE)](https://fcon_1000.projects.nitrc.org/indi/abide/) dataset, which contains resting-state functional MRI data from 539 individuals with ASD and 573 neurotypical controls.

The application processes the ABIDE data using:
- Schaefer 2018 brain atlas with 100 regions of interest
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

## Visualization Types

1. **Connectivity Matrices**: View and compare connectivity matrices between neurotypical and ASD brains.
2. **Network Visualization**: Explore brain networks as graphs and 3D brain models.
3. **Regional Differences**: Analyze connectivity differences for specific brain regions.
4. **Demographics**: View demographic information about the ABIDE dataset participants.

## References

- The ABIDE Initiative: [https://fcon_1000.projects.nitrc.org/indi/abide/](https://fcon_1000.projects.nitrc.org/indi/abide/)
- Hull, J. V., et al. (2017). Resting-state functional connectivity in autism spectrum disorders: A review. Frontiers in Psychiatry, 7, 205.
- Just, M. A., et al. (2012). Autism as a neural systems disorder: A theory of frontal-posterior underconnectivity. Neuroscience & Biobehavioral Reviews, 36(4), 1292-1313.
- Schaefer, A., et al. (2018). Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI. Cerebral Cortex, 28(9), 3095-3114.

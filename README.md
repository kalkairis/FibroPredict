# FibroPredict
Code accompanying the manuscript "Fibro Predict a machine learning risk score for Advanced Liver Fibrosis in the General Population using Israeli Electronic Health Records" by Kalka __et al.__

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Project structure

The code is organized into three main stages:

- **cohort/** – utilities for constructing the study cohort.
- **model/** – training scripts for building predictive models.
- **analysis/** – analyses and visualizations used in the manuscript,
  including the original survival analysis code and figure generation.

## Missing cutils implementation

The project uses several helpers from a proprietary package named `cutils` that
interfaces with the Clalit Health Services database. These utilities are not
available publicly. A file named `cutils_placeholders.py` located in the root of
this repository lists the expected functions and loader classes together with
notes on what each should return. To run the code you must implement these
methods for your own data source.

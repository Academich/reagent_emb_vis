# Visualization of the reagent space

A web-app for the exploration of the embedding space of reagents used in reaction data.

## Environment installation
Run the following commands to install the environment for the app:

```python
conda create -n reagent_emb_vis_app python=3.10 -y
conda activate reagent_emb_vis_app
pip install -r requirements.txt
```

## Usage

### Introduction
The app is a visual way of exploring the co-occurrence statistics of reagents in reactions.
The app displays UMAP projections of reagent embeddings derived by decomposing the _PMI matrix_ of reagents with _singular value decomposition._

A _PMI matrix_ contains _pointwise mutual information scores_. For two reagents _a_ and _b_, their _PMI score_ is derived from reagent occurrence counts.  
Factorising this matrix using SVD yields dense embeddings for reagents, which tend to be similar for two reagents if these reagents are encountered  
in similar contexts, i.e. together with the same other reagents.  
For example, two different palladium catalysts for Suzuki coupling will not be used together in a reaction, but they may be used with the same bases and solvents.  
Therefor, those two catalysts will get similar embeddings and will lie close together. Those embeddings are then projected on the 2D plane and the surface of the unit sphere  
by the UMAP algorithm. It's a dimensionality reduction algorithm that tries to preserve distance relations between original points when projecting them to a lower-dimensional space.  
The map of UMAP projections of reagent embeddings is displayed in the app.

### Data preparation
The embeddings for reagents are calculated based on a file with reagents that are used in their respective reagents.  
The input file must contain reagent SMILES sets for some reaction in every row, and those SMILES must be separated by some separator. e.g. `;`.  
Example `reagents.txt`:

```python
CCO;c1ccccc1
[H-].[Na+];C1CCOC1
NN
```
Every row in this file contains reagents for some reaction in the dataset of interest. The reactions themselves are not relevant.

The app uses coordinates in a CSV file, which is prepared using the `build_embeddings.py` script.

Run the following command:
```python
python3 build_embeddings.py -i <PATH TO THE TEXT FILE WITH REAGENT SMILES> --min_count <MINIMAL OCCURENCE COUNT FOR REAGENTS TO BE CONSIDERED> -o <PATH TO THE OUTPUT CSV FILE> -d <DIMENSONALITY OF REAGENT EMBEDDINGS>
```
For more information, run `python3 build_embeddings.py --help`.

### App usage
Run the app with the following command
```python
python3 app.py
```
The app will be running on http://localhost:8050 by default.  
Upload a CSV file build by the `build_embeddings.py` script.

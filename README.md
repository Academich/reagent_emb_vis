# Visualization of the reagent space

A web-app for the exploration of the embedding space of reagents used in reaction data.

The app is a visual way of exploring the co-occurrence statistics of reagents in reactions.
The app displays UMAP projections of reagent embeddings derived by decomposing the _PMI matrix_ of reagents with _singular value decomposition._

A _PMI matrix_ contains _pointwise mutual information scores_. For two reagents _a_ and _b_, their _PMI score_ is derived from reagent occurrence counts.  
Factorising this matrix using SVD yields dense embeddings for reagents, which tend to be similar for two reagents if these reagents are encountered  
in similar contexts, i.e. together with the same other reagents. For example, two different palladium catalysts for Suzuki coupling will not be used together in a reaction, but they may be used with the same bases and solvents.  
Therefore, those two catalysts will get similar embeddings and will lie close together. Those embeddings are then projected on the 2D plane and the surface of the unit sphere  
by the UMAP algorithm. It's a dimensionality reduction algorithm that tries to preserve distance relations between original points when projecting them to a lower-dimensional space.  
The map of UMAP projections of reagent embeddings is displayed in the app.

## Environment installation
Run the following commands to install the environment for the app:

```python
conda create -n reagent_emb_vis_app python=3.10 -y
conda activate reagent_emb_vis_app
pip install -r requirements.txt
pip install -e .
```

## App usage
Run the app with the following command
```python
python3 app.py
```
The app will be running on http://localhost:8050. By default, it shows the map of USPTO reagent embeddings determined by AAM
reading the infomation from `data/default/uspto_aam_rgs_min_count_100_d_50.csv`.
Users can also upload their own reagent data, prepared with the appropriate scripts in the way described below.

## Standard USPTO reagents
The file `data/standard_reagents.csv` contains the information about ~600 reagents that occur in USPTO, with their roles and names. 
The entries in the file are ordered by occurrence frequency in the descending order. 

## Dataset
We download the USPTO dataset using `rxnutils` by executing the following commands from the `data` directory:

```python
python -m rxnutils.data.uspto.download
```
```python
python -m rxnutils.data.uspto.combine
```

It downloads the file `data/uspto_data.csv`. Then, we do the initial filtering of this dataset with the following command executed from the project directory:
```python
python3 -m rxnutils.pipeline.runner --pipeline uspto/pipeline.yml --data data/uspto_data.csv --output data/uspto_filtered.csv
```

Finally, we extract the reagents from the filtered dataset:

```python
python3 scripts/prepare_reagents.py -i data/uspto_filtered.csv --output_dir uspto_aam_reagents -c ReactionSmiles --reagents aam --fragment_grouping cxsmiles --canonicalization remove_aam --n_jobs 9 --min_reagent_occurrences 1 --verbose
```
The script `prepare_reagents.py` as various options. For example, it can determine reagents either by atom mapping or by fingerprints.


## Reagent embeddings preparation
The embeddings for reagents are calculated using the script `build_embeddings.py` based on a file with reagents that are used in their respective reagents.  
The input file must contain reagent SMILES sets for some reaction in every row, and those SMILES must be separated by some separator. e.g. `;`.  
Example:

```python
CCO;c1ccccc1
[H-].[Na+];C1CCOC1
NN
```
Every row in this file contains reagents for some reaction in the dataset of interest. The reactions themselves are not relevant. 
The script `prepare_reagents.py` prepares a suitable input for `build_embeddings.py`.

The app uses coordinates in a CSV file, which is prepared using the `build_embeddings.py` script.

Run the following command:
```python
python3 scripts/build_embeddings.py -i <PATH TO THE TEXT FILE WITH REAGENT SMILES> --standard data/standard_reagents.csv --min_count <MINIMAL OCCURENCE COUNT FOR REAGENTS TO BE CONSIDERED> -o <PATH TO THE OUTPUT CSV FILE> -d <DIMENSONALITY OF REAGENT EMBEDDINGS>
```
For more information, run `python3 build_embeddings.py --help`.

The default reagent embeddings were built with the following command:

```python
python3 scripts/build_embeddings.py -i uspto_aam_reagents/reagents-1128297.txt --standard data/standard_reagents.csv -d 50 -o data/uspto_aam_rgs_min_count_100_d_50.csv --min_count 100
```

Upload a CSV file build by the `build_embeddings.py` script.

## Reports

For the insights about reagents in USPTO please follow the notebook `notebooks/results.ipynb`.

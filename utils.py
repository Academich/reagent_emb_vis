import io
import base64

import pandas as pd
import numpy as np

from dash.html import Div

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdDepictor


def parse_uploaded_content(contents) -> pd.DataFrame | Div:
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        if "class" in df.columns:
            df["numerical_label"] = df["class"].map({v: i for i, v in enumerate(sorted(df["class"].unique()))})
    except Exception as e:
        return Div(["There was an error processing this file."])

    return df


def smi2svg(smi):
    mol = Chem.MolFromSmiles(smi)
    rdDepictor.Compute2DCoords(mol)
    mc = Chem.Mol(mol.ToBinary())
    Chem.Kekulize(mc)
    drawer = Draw.MolDraw2DSVG(400, 400)
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')
    return svg


def to_spherical_coordinates(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def uniform_sphere_points(n_points=100):
    theta = np.linspace(0, 2 * np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from utils import uniform_sphere_points, to_cartesian_coordinates


def update_flat_chart(df: pd.DataFrame):
    color = [px.colors.qualitative.Light24[i] for i in df["numerical_label"]]
    return {'data': [go.Scatter(
        x=df["x"],
        y=df["y"],
        hovertext=df.index,
        hoverinfo='text',
        text=df["smiles"] + "|" + df["name"],
        mode='markers',
        marker={
            'size': 12,
            'opacity': 0.6,
            'color': color
        }
    )],
        'layout': go.Layout(
            xaxis={'title': "axis 1"},
            yaxis={'title': "axis 2"},
            width=800,
            height=800,
            title=f"Points displayed: {len(df)}"
        )}


def generate_sphere():
    surface_points = uniform_sphere_points()
    surface_color = "#e4eff9"
    shrink = 0.98
    return go.Surface(
        x=surface_points[0] * shrink,
        y=surface_points[1] * shrink,
        z=surface_points[2] * shrink,
        hoverinfo='skip',
        contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
        colorscale=[[0, surface_color], [1, surface_color]],  # Single color, no gradient
        showscale=False,
        lighting=dict(ambient=1.0, diffuse=1.0, fresnel=0.0, specular=0.0, roughness=0.0),  # Remove lighting
        opacity=1,
        cmin=0,  # Set cmin and cmax to the same value
        cmax=0,
    )


def update_sphere_chart(df: pd.DataFrame):
    color = [px.colors.qualitative.Light24[i] for i in df["numerical_label"]]
    x, y, z = to_cartesian_coordinates(df["theta"], df["phi"])
    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        text=df["smiles"] + "|" + df["name"],
        hovertext=df.index,
        hoverinfo='text',
        mode='markers',
        marker=dict(size=8, color=color, opacity=0.7),
    )
    sphere_surface = generate_sphere()

    layout = go.Layout(
        scene=dict(
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.2)
            ),
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
        ),
        width=800,
        height=800,
        title=f"Points displayed: {len(df)}"
    )
    return {'data': [scatter, sphere_surface],
            'layout': layout}


def update_class_selector(classes: dict[str, list]):
    active_classes_state = classes["active"]
    class_names = classes["names"]

    inactive_color = "white"
    colors = [px.colors.qualitative.Light24[i] if v else inactive_color for i, v in enumerate(active_classes_state)]
    linewidth = [1 if v else 3 for i, v in enumerate(active_classes_state)]
    hovertext = [f"{class_names[i]} (on)" if v else f"{class_names[i]} (off)" for i, v in
                 enumerate(active_classes_state)]
    figure = dict(
        data=[
            go.Pie(
                values=[1] * len(active_classes_state),
                textinfo='none',
                labels=class_names,
                hovertext=hovertext,
                hoverinfo='text',
                marker=dict(colors=colors, line=dict(color='grey', width=linewidth))
            )
        ],
        layout=go.Layout(title="Click on sections to hide/show reagent classes",
                         legend=dict(itemclick=False, itemdoubleclick=False)))
    return figure

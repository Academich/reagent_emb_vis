import pandas as pd

import dash
import dash_dangerously_set_inner_html as dhtml
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go

from figures import update_flat_chart, update_sphere_chart, update_class_selector
from utils import parse_uploaded_content, smi2svg

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.config.suppress_callback_exceptions = True

app.layout = html.Div(children=[
    html.Div([
        html.H1('Embedding space', style={'textAlign': 'center'}),
        dcc.Upload(
            id='csv-uploader',
            children=html.Button('Upload File')
        ),
        dcc.Store(id='uploaded-csv'),
        # Button to toggle between graphs
        html.Button('Toggle Graph', id='toggle-button', style={'display': 'none'}),

    ], className='row'),

    html.Div([
        dcc.Store(id='selected-class', data=None),

        # Pie chart as clickable buttons
        dcc.Graph(
            id='pie-chart',
            config={'staticPlot': False},  # Enable interactivity
            style={'display': 'none'}
        ),
    ]),

    html.Div([
        dcc.RangeSlider(min=0, max=2, step=1, persistence=True, id="range-slider")
    ], className='row', style={'display': 'none'}, id="range-slider-div"),

    html.Div([
        html.Div([html.Div(id='molimg')
                  ]),
        html.Div(id="smiles")
    ], style={'width': '50%', 'display': 'inline-block', 'text-align': 'center', 'vertical-align': 'top'}),

    html.Div([
        dcc.Graph(id='graph', className='six columns', style={'display': 'none'}),
    ], style={'width': '50%', 'display': 'inline-block'})
])


@app.callback(
    [Output('uploaded-csv', 'data')],
    [Input('csv-uploader', 'contents')],
    [State('csv-uploader', 'filename')]
)
def store_uploaded_data(contents, filename):
    if contents is not None:
        # Parse the uploaded content and store it in a variable
        # Replace this with your actual data processing logic
        parsed_data = parse_uploaded_content(contents)
        if parsed_data is not None:
            return [parsed_data.to_json(date_format='iso', orient='split')]

    return [None]


@app.callback(
    [Output('graph', 'style'),
     Output('toggle-button', 'style'),
     Output('pie-chart', 'style'),
     Output('range-slider-div', 'style'),

     Output('range-slider', 'min'),
     Output('range-slider', 'max'),
     Output('range-slider', 'step'),
     Output('range-slider', 'marks'),
     Output('range-slider', 'value'),

     Output('graph', 'figure'),
     Output('pie-chart', 'figure'),

     Output('selected-class', 'data')],
    [Input('uploaded-csv', 'data')],
    prevent_initial_call=True
)
def display_hidden_elements(contents):
    if contents is None:
        raise PreventUpdate

    data = parse_contents(contents)
    active_classes = [True] * len(data["numerical_label"].unique())

    class_names = data.set_index("numerical_label")["class"].to_dict()
    class_names = [class_names[k] for k in sorted(class_names.keys())]
    classes = {"names": class_names, "active": active_classes}

    range_min = 0
    range_max = data.shape[0]
    range_step = 1
    range_slider_params = [
        range_min,
        range_max,
        range_step,
        {i: str(i) for i in range(range_min, range_max, range_max // 10)},
        [range_min, range_max]
    ]
    style = [{'display': 'block'}] * 4

    return style + range_slider_params + [update_flat_chart(data), update_class_selector(classes),
                                          classes]


@app.callback(
    [Output('selected-class', 'data', allow_duplicate=True),
     Output('pie-chart', 'figure', allow_duplicate=True)],
    [Input('pie-chart', 'clickData')],
    [State('selected-class', 'data')],
    prevent_initial_call=True
)
def update_pie_chart(click_data, active_classes):
    if click_data is None:
        raise PreventUpdate

    selected_class = click_data['points'][0]['i']
    active_classes["active"][selected_class] = not active_classes["active"][selected_class]

    pie_figure = update_class_selector(active_classes)

    return [active_classes, pie_figure]


@app.callback(
    Output('graph', 'figure', allow_duplicate=True),
    Input('toggle-button', 'n_clicks'),
    Input('selected-class', 'data'),
    Input('range-slider', 'value'),
    State('uploaded-csv', 'data'),
    prevent_initial_call=True  # Prevent the callback from running on page load
)
def update_graph(n_clicks, selected_class, slider_value, contents):
    # Determine which graph to show based on the number of clicks
    if contents is None:
        return {}

    data = parse_contents(contents)
    from_idx, to_idx = slider_value
    data = data.iloc[from_idx:to_idx]
    data = data[data["numerical_label"].isin([i for i, v in enumerate(selected_class["active"]) if v])]

    if n_clicks is None or n_clicks % 2 == 0:
        return update_flat_chart(data)
    else:
        return update_sphere_chart(data)


def parse_contents(contents: str) -> pd.DataFrame:
    data = eval(contents)
    return pd.DataFrame(data["data"], index=data["index"], columns=data["columns"])


@app.callback(
    [Output('molimg', 'children'),
     Output('smiles', 'children')],
    [Input('graph', 'hoverData')],
)
def update_img_from_flat_map(hoverData):
    try:
        smiles = hoverData['points'][0]['text']
        svg = smi2svg(smiles)
        text = html.P(smiles,
                      style={'font-size': '20px', 'font-family': 'Courier'})
    except:
        raise PreventUpdate
    return [dhtml.DangerouslySetInnerHTML(svg), text]


if __name__ == "__main__":
    app.run_server(debug=True)

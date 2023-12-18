from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

import dash
import dash_dangerously_set_inner_html as dhtml
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from rendering.figures import update_flat_chart, update_sphere_chart, update_class_selector
import rendering.utils as ut

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.config.suppress_callback_exceptions = True

app.layout = html.Div(children=[
    html.Div([
        html.H1('The space of reagent embeddings', style={'textAlign': 'center'}),
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
        html.Div([
            html.H6('Selector button: click on sections to hide/show reagent roles.',
                    style={'text-align': 'center'}),
            dcc.Graph(
                id='pie-chart',
                config={'staticPlot': False},  # Enable interactivity
                style={'display': 'none'}
            )]),
        html.Div([
            html.H6('Range slider: drag both ends to display only the points with indices in the selected range.',
                    style={'text-align': 'center'}),
            dcc.RangeSlider(min=0, max=2, step=1, persistence=True, id="range-slider")
        ], className='row', style={'display': 'none'}, id="range-slider-div"),
        html.Div([dcc.Input(value='', id='filter-input', placeholder='Filter: SMARTS pattern',
                            debounce=True, style={'display': 'none'})])
    ]),

    html.Div([
        html.Div([html.Div(id='molimg')
                  ]),
        html.Div(id="smiles"),
        html.Div(id="mol_name")
    ], style={'width': '50%', 'display': 'inline-block', 'text-align': 'center', 'vertical-align': 'top'}),

    html.Div([
        dcc.Graph(id='graph', className='six columns', style={'display': 'none'}),
    ], style={'width': '50%', 'display': 'inline-block'}),

])

DEFAULT_PATH = Path("data").resolve() / "default" / "uspto_aam_rgs_min_count_100_d_50.csv"


@app.callback(
    [Output('uploaded-csv', 'data')],
    [Input('csv-uploader', 'contents')],
    [State('csv-uploader', 'filename')]
)
def store_uploaded_data(contents, filename):
    if contents is not None:
        # Parse the uploaded content and store it in a variable
        # Replace this with your actual data processing logic
        parsed_data = ut.parse_uploaded_content(contents)
        if parsed_data is not None:
            return [parsed_data.to_json(date_format='iso', orient='split')]
    try:
        parsed_data = pd.read_csv(DEFAULT_PATH)
        parsed_data = ut.add_numerical_labels_for_classes(parsed_data)
        return [parsed_data.to_json(date_format='iso', orient='split')]
    except Exception as e:
        return [None]


@app.callback(
    [Output('graph', 'style'),
     Output('toggle-button', 'style'),
     Output('pie-chart', 'style'),
     Output('range-slider-div', 'style'),
     Output('filter-input', 'style'),

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

    data = ut.parse_contents(contents)
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
    style = [{'display': 'block'}] * 5

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
    Input('filter-input', 'value'),
    State('uploaded-csv', 'data'),
    prevent_initial_call=True  # Prevent the callback from running on page load
)
def update_graph(n_clicks, selected_class, slider_value, pattern_string, contents):
    # Determine which graph to show based on the number of clicks
    if contents is None:
        return {}

    data = ut.parse_contents(contents)

    # Filter with the slider
    from_idx, to_idx = slider_value
    data = data.iloc[from_idx:to_idx]

    # Filter with the pie-chart button
    data = data[data["numerical_label"].isin([i for i, v in enumerate(selected_class["active"]) if v])]

    # Filter with the regex filter
    if pattern_string:
        sma_ptn = ut.smarts_pattern(pattern_string)
        data = data[data["smiles"].apply(lambda x: ut.match_smiles_to_smarts(x, sma_ptn))]

    if n_clicks is None or n_clicks % 2 == 0:
        return update_flat_chart(data)
    else:
        return update_sphere_chart(data)


@app.callback(
    [Output('molimg', 'children'),
     Output('smiles', 'children'),
     Output('mol_name', 'children')],
    [Input('graph', 'hoverData')],
)
def update_img_from_flat_map(hoverData):
    try:
        smiles, name = hoverData['points'][0]['text'].split("|")
        svg = ut.smi2svg(smiles.replace("\\", "").replace("/", ""))
        text_smiles = html.P(smiles,
                             style={'font-size': '20px', 'font-family': 'Courier'})
        text_name = html.P(name,
                           style={'font-size': '20px', 'font-family': 'Courier'})
    except:
        raise PreventUpdate
    return [dhtml.DangerouslySetInnerHTML(svg), text_smiles, text_name]


if __name__ == "__main__":
    RDLogger.DisableLog('rdApp.*')

    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="IP address for the app.")
    parser.add_argument("--port", type=int, default=8050, help="Port for the app.")
    parser.add_argument("--debug", action="store_true", help="Flag for the debug mode.")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)

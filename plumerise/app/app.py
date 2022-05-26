from datetime import date, datetime
from faulthandler import disable
from math import floor
from turtle import color

import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import diskcache
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import callback_context  # type: ignore
from dash import dcc  # import dash_core_components as dcc
from dash import html  # import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.long_callback import DiskcacheLongCallbackManager
from flask_caching import Cache
from geopy import distance
from importlib_metadata import distributions
from numpy import abs, log10
from pandas import Categorical, DataFrame
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from plumerise.model.ensemble_solve import solve_ensemble

from ..data import StationList
from ..model import MetData, Params, solve_ensemble
from ..utilities.number_tools import (float_string, lin_levels, lin_steps,
                                      log_levels, log_steps, nice_round_down,
                                      nice_round_up, sci_string)
from ..utilities.string_tools import string_html, units_html

# from dash_extensions.enrich import (DashProxy, Input, MultiplexerTransform,
                                    # Output, ServersideOutput,
                                    # ServersideOutputTransform, State, dcc,
                                    # html)



def GreatCircleDistance(lat, lon, lat0, lon0):

    if np.abs(lat) > 90:
        return np.nan
    else:
        return distance.distance((lat0, lon0), (lat, lon)).km



distributions = {
    "Uniform": "Uniform",
    "Truncated normal": "TruncNorm",
    "Beta": "Beta",
}

uniform = {"dist-param-0": "Minimum", "dist-param-1": "Maximum", "dist-param-2": None, "dist-param-3": None}
truncnorm = {"dist-param-0": "Minimum", "dist-param-1": "Maximum", "dist-param-2": "mean", "dist-param-3": "std. dev."}
beta = {"dist-param-0": "Minimum", "dist-param-1": "Maximum", "dist-param-2": "alpha", "dist-param-3": "beta"}


def build_parameters_table_row(param):
    if param['unit']:
        title = f"{param['name']} ({param['unit']})"
    else:
        title = f"{param['name']}"
    content = [
        html.Td(title),
        html.Td(
            daq.ToggleSwitch(
                id=param['label']+"-choice",
                label=["Fixed", "Uncertain"],
                style={"width": "150px"},
                value=False,
            )
        ),
        html.Td(
            dbc.RadioItems(
                id=param['label']+"-dist",
                className="btn-group invisible",
                inputClassName="btn-check",
                labelClassName='btn btn-outline-primary',
                labelCheckedClassName="active",
                options=[
                    {"label": label, "value": value}
                    for value, label in distributions.items()
                ],
                value="Uniform",
                # optionHeight=20,
            )
        ),
        html.Td(
            dcc.Input(
                id=param['label']+"-dist-param-0",
                name=param['label']+"-dist-param-0",
                className="InputBox sml",
                type="number",
                min=param['min'],
                max=param['max'],
                value=param['default'],
                placeholder="Minimum",
                pattern="^(?=.)([+-]?([0-9]*)(\.(?[0-9]+))?)$",
            )
        ),
        html.Td(
            dcc.Input(
                id=param['label']+"-dist-param-1",
                name=param['label']+"-dist-param-1",
                className="InputBox sml invisible",
                type="number",
                min=param['min'],
                max=param['max'],
                placeholder="Maximum",
                pattern="^(?=.)([+-]?([0-9]*)(\.(?[0-9]+))?)$",
            )
        ),
        html.Td(
            dcc.Input(
                id=param['label']+"-dist-param-2",
                name=param['label']+"-dist-param-2",
                className="InputBox sml invisible",
                type="number",
                min=0,
                placeholder="mean",
                # pattern="^(?=.)([+-]?([0-9]*)(\.(?[0-9]+))?)$",
            )
        ),
        html.Td(
            dcc.Input(
                id=param['label']+"-dist-param-3",
                name=param['label']+"-dist-param-3",
                className="InputBox sml invisible",
                type="number",
                min=0,
                placeholder="std. dev.",
                # pattern="^[+]?([.]\d+|\d+[.]?\d*)$",
            )
        ),
    ]
    return content


def build_params_table(Params):
    table_rows = []
    for param in Params:
        table_rows.append(
            html.Tr(
                children=build_parameters_table_row(param)
            )
        )
    return table_rows


def make_dashboard():

    cache = diskcache.Cache("./cache")
    long_callback_manager = DiskcacheLongCallbackManager(cache)

    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            dbc.icons.FONT_AWESOME,
        ],
        suppress_callback_exceptions=True,
    )
    
    # app = DashProxy(
        # __name__,
        # external_stylesheets=[
            # dbc.themes.BOOTSTRAP,
            # dbc.icons.FONT_AWESOME,
        # ],
        # suppress_callback_exceptions=True,
        # transforms=[ServersideOutputTransform(), MultiplexerTransform()],
    # )

    available_met = {
        "Wyoming": "Wyoming weather web",
        # "GFS": "Global forecast system",
    }

    observables = {
        "Htop": "Plume top centreline height",
        "HtopLg": "Plume top Gaussian edge height",
        "Hn": "Neutral height",
    }

    met_tab = html.Div(
        className="pane",
        children=[
            html.Div(
                className="flex-container",
                style={"float": "inline-start"},
                children=[
                    html.Div(
                        id="location-input",
                        className="column column-one",
                        children=[
                            html.Div(
                                className="formfield fflrg",
                                children=[
                                    html.Div(
                                        children=[
                                            html.Label(
                                                children=["Latitude:"],
                                                className="Dropdown-label fl",
                                            ),
                                            dcc.Input(
                                                id="latitude-input",
                                                name="latitude-input",
                                                className="InputBox",
                                                type="number",
                                                min=-90.,
                                                max=90.,
                                                placeholder="Latitude",
                                            ),
                                        ]
                                    ),
                                ],
                            ),
                            html.Div(
                                className="formfield fflrg",
                                children=[
                                    html.Div(
                                        children=[
                                            html.Label(
                                                children=["Longitude:"],
                                                className="Dropdown-label fl",
                                            ),
                                            dcc.Input(
                                                id="longitude-input",
                                                name="longitude-input",
                                                className="InputBox",
                                                type="number",
                                                min=-180.,
                                                max=360.,
                                                placeholder="Longitude",
                                            ),
                                        ]
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        id="met-selector",
                        className="column column-two",
                        children=[
                            html.Div(
                                className="flex-input-column",
                                children=[
                                    html.Label(
                                        "Meteorological data source:",
                                        className="Dropdown-label",
                                    ),
                                    dcc.Dropdown(
                                        id="met-select",
                                        options=[
                                            {"label": label, "value": value}
                                            for value, label in available_met.items()
                                        ],
                                        value="Wyoming",
                                        clearable=False,
                                        placeholder="Select a source",
                                        # optionHeight=20,
                                    ),
                                ],
                            ),
                            html.Div(
                                className="formfield fflrg",
                                children=[
                                    html.Div(
                                        children=[
                                            html.Span(
                                                id="met-extra-info-label",
                                                children=["test"],
                                                className="OutputBox invisible",
                                            ),
                                        ]
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        id="datetime-selector",
                        className="column column-three",
                        children=[
                            html.Div(
                                className="formfield fflrg",
                                children=[
                                    html.Label(
                                        "Date:",
                                        className="Dropdown-label fl",
                                    ),
                                    dcc.DatePickerSingle(
                                        id="met-date-picker",
                                        min_date_allowed=date(2000, 1, 1),
                                        max_date_allowed=date.today(),
                                        initial_visible_month=date.today(),
                                        date=date.today(),
                                        display_format='DD/MM/YYYY',
                                        with_portal=True,
                                    ),
                                ],
                            ),
                            html.Div(
                                className="formfield fflrg",
                                children=[
                                    html.Label(
                                        "Time:",
                                        className="Dropdown-label fl",
                                    ),
                                    dcc.Input(
                                        id="met-time-input",
                                        name="met-time-input",
                                        className="InputBox",
                                        type="text",
                                        placeholder="HH:MM",
                                        pattern="^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        id="met-get-button",
                        className="column column-four",
                        children=[
                            html.Div(
                                className="row",
                                children=[
                                    html.Div(
                                        className="formfield",
                                        style={"width": "240px"},
                                        children=[
                                            html.Button(
                                                className="button",
                                                id="met-submit-button",
                                                style={"width": "150px"},
                                                disabled="disabled",
                                                n_clicks=0,
                                                children=[
                                                    html.I(
                                                        className="fas fa-cloud-sun-rain",
                                                        style={"margin-right": "10px"},
                                                    ),
                                                    html.Span(
                                                        "Get data",
                                                        style={"font-size": "10pt"},
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ],
    )

    parameters_modal = [
        dbc.ModalHeader(dbc.ModalTitle("Parameters"), className="modal-header"),
        dbc.ModalBody(className="modal-body", children=[
            html.Div(
                className="pane",
                children=[
                    html.Table(id="parameters-table",
                        children=[
                            html.Tbody(
                                build_params_table(Params)
                            )
                        ]
                    )
                ]
            ),
            html.Div(
                className="pane",
                children=[
                    html.Div(
                        className="flex-container",
                        style={"float": "inline-start", "margin-left":"auto"},
                        children=[
                            html.Div(
                                id="param-cancel",
                                className="column column-one",
                                children=[
                                    html.Button(
                                        className="button",
                                        id="params-reset-button",
                                        style={"width": "150px"},
                                        n_clicks=0,
                                        children=[
                                            html.I(
                                                className="fas fa-solid fa-check",
                                                style={"margin-right": "10px"},
                                            ),
                                            html.Span(
                                                "Reset parameters",
                                                style={"font-size": "10pt"},
                                            ),
                                        ],
                                    )
                                ]
                            ),
                            html.Div(
                                id="param-confirm",
                                className="column column-two",
                                children=[
                                    html.Button(
                                        className="button",
                                        id="params-submit-button",
                                        style={"width": "150px"},
                                        disabled="disabled",
                                        n_clicks=0,
                                        children=[
                                            html.I(
                                                className="fas fa-solid fa-check",
                                                style={"margin-right": "10px"},
                                            ),
                                            html.Span(
                                                "Confirm parameters",
                                                style={"font-size": "10pt"},
                                            ),
                                        ],
                                    )
                                ]
                            )
                        ]
                    ),
                ]
            )
        ])
    ]

    parameters_tab = html.Div(
        className="pane",
        children=[
            html.Div(
                className="flex-container",
                style={"float": "inline-start"},
                children=[
                    html.Div(
                        id="param-settings",
                        className="column column-one",
                        children=[
                            html.Button(
                                className="button",
                                id="set-parameters-button",
                                style={"width": "150px"},
                                n_clicks=0,
                                children=[
                                    html.I(
                                        className="fas fa-solid fa-list-check",
                                        style={"margin-right": "10px"},
                                    ),
                                    html.Span(
                                        "Set Parameters",
                                        style={"font-size": "10pt"},
                                    ),
                                ],
                            ),
                        ]
                    ),
                    html.Div(
                        id="ensemble-settings",
                        className="column column-two",
                        children=[
                            html.Div(
                                className="formfield fflrg",
                                children=[
                                    html.Div(
                                        children=[
                                            html.Label(
                                                children=["Ensemble size:"],
                                                className="Dropdown-label fl",
                                            ),
                                            dcc.Input(
                                                id="ensemble-size",
                                                name="ensemble-size",
                                                className="InputBox",
                                                type="number",
                                                value=1000,
                                                min=1,
                                                max=1_000_000,
                                            ),
                                        ]
                                    ),
                                ],
                            ),
                        ]
                    ),
                    html.Div(
                        id="run-ensemble",
                        className="column column-three",
                        children=[
                            html.Div(
                                className="row",
                                children=[
                                    html.Div(
                                        className="formfield",
                                        style={"width": "240px"},
                                        children=[
                                            html.Button(
                                                className="button",
                                                id="run-ensemble-button",
                                                style={"width": "150px"},
                                                disabled=True,
                                                n_clicks=0,
                                                children=[
                                                    html.I(
                                                        className="fas fa-solid fa-cogs",
                                                        style={"margin-right": "10px"},
                                                    ),
                                                    html.Span(
                                                        "Run simulations",
                                                        style={"font-size": "10pt"},
                                                    ),
                                                ],
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        className="formfield",
                                        style={"width": "240px"},
                                        children=[
                                            html.Button(
                                                className="button",
                                                id="cancel-ensemble-button",
                                                style={"width": "150px"},
                                                disabled=True,
                                                n_clicks=0,
                                                children=[
                                                    html.I(
                                                        className="fas fa-solid fa-ban",
                                                        style={"margin-right": "10px"},
                                                    ),
                                                    html.Span(
                                                        "Cancel simulations",
                                                        style={"font-size": "10pt"},
                                                    ),
                                                ],
                                            ),
                                        ]
                                    ),
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        id="obs-ensemble",
                        className="column column-four invisible",
                        children=[
                            html.Div(
                                className="formfield fflrg",
                                children=[
                                    html.H4("Set observation")
                                ]
                            ),
                            html.Div(
                                className="flex-input-column",
                                children=[
                                    html.Label(
                                        "Observed quantity:",
                                        className="Dropdown-label",
                                    ),
                                    dcc.Dropdown(
                                        id="obs-select",
                                        options=[
                                            {"label": label, "value": value}
                                            for value, label in observables.items()
                                        ],
                                        value="Htop",
                                        clearable=False,
                                        placeholder="Select observable",
                                        optionHeight=25,
                                    ),
                                ],
                            ),
                            html.Div(
                                className="formfield",
                                children=[
                                    html.Span("Range:")
                                ]
                            ),
                            html.Div(
                                className="formfield fflrg",
                                children=[
                                    html.Div(
                                        children=[
                                            html.Label(
                                                children=["Minimum height (km):"],
                                                className="Dropdown-label fl",
                                            ),
                                            dcc.Input(
                                                id="obs-min-input",
                                                name="obs-min-input",
                                                className="InputBox sml",
                                                type="number",
                                                min=0,
                                                placeholder="Minimum",
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            html.Div(
                                className="formfield fflrg",
                                children=[
                                    html.Div(
                                        children=[
                                            html.Label(
                                                children=["Maximum height (km):"],
                                                className="Dropdown-label fl",
                                            ),
                                            dcc.Input(
                                                id="obs-max-input",
                                                name="obs-max-input",
                                                className="InputBox sml",
                                                type="number",
                                                min=0,
                                                placeholder="Maximum",
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            html.Div(
                                className="flex-input-column",
                                # style={"width": "240px"},
                                children=[
                                    html.Button(
                                        className="button",
                                        id="obs-button",
                                        style={"width": "150px", "margin":"auto"},
                                        disabled=True,
                                        n_clicks=0,
                                        children=[
                                            html.I(
                                                className="fas fa-solid fa-bullseye",
                                                style={"margin-right": "10px"},
                                            ),
                                            html.Span(
                                                "Set observation",
                                                style={"font-size": "10pt"},
                                            ),
                                        ],
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            ),

            dbc.Modal(parameters_modal, id="set-parameters-modal", className="modal-content lrg", is_open=False, backdrop=False),
        ]
    )

    app.layout = html.Div(
        className="body",
        children=[
            dcc.Store(id="met-source-store", storage_type="session"),
            dcc.Store(id="met-latitude-store", storage_type="session"),
            dcc.Store(id="met-longitude-store", storage_type="session"),
            dcc.Store(id="met-date-store", storage_type="session"),
            dcc.Store(id="met-time-store", storage_type="session"),
            dcc.Store(id="params-valid-flag", storage_type="session"),
            dcc.Store(id='modal-open', storage_type="session"),
            dcc.Store(id="met", storage_type="session"),
            dcc.Store(id="params", storage_type="session"),
            dcc.Store(id="ensemble", storage_type="session"),
            dcc.Store(id="ensemble-run-complete", storage_type="session"),
            dcc.Store(id="obs-quantity", storage_type="session"),
            dcc.Store(id="obs-min-store", storage_type="session"),
            dcc.Store(id="obs-max-store", storage_type="session"),
            html.Div(
                className="header",
                children=[
                    html.Img(
                        src=app.get_asset_url("uob_logo.png"),
                        style={"float": "left"},
                        alt="University of Bristol",
                        height=49,
                    ),
                    html.Div(
                        className="modelTitle",
                        children=[html.Span("PlumeRise Ensemble")],
                    ),
                ],
            ),
            html.Div(style={"clear": "both"}),
            html.Div(
                className="guipane",
                children=[
                    html.Div(
                        id="image-container",
                        className="imagepane",
                        children=[
                            html.Div(
                                id='ensemble-image',
                                className="imagehp", children=[

                                ]
                            ),
                            html.Div(
                                id='obs-image',
                                className="imagehp", children=[

                                ]
                            ),
                        ],
                    ),
                    html.Div(style={"clear": "both", "margin-top": 10}),
                    html.Div(
                        dcc.Tabs(
                            id="outertabs",
                            className="tabs",
                            value="metTab",
                            children=[
                                dcc.Tab(
                                    label="Meteorological data",
                                    value="metTab",
                                    className="outertab",
                                    selected_className="outertab--selected",
                                    children=met_tab,
                                ),
                                dcc.Tab(
                                    label="PlumeRise ensemble",
                                    value="ensembleTab",
                                    className="outertab",
                                    selected_className="outertab--selected",
                                    children=parameters_tab,
                                ),
                            ],
                        )
                    ),
                    html.Div(style={"clear": "both"}),
                ],
            ),
            html.Div(
                id="footer",
                children=[
                    html.Div(
                        id="footertext",
                        children=[
                            "Copyright © University of Bristol 2022\u0020|\u0020",
                            html.A(
                                href="http://www.bristol.ac.uk/web/policies/terms-conditions.html",
                                target="_blank",
                                children=["Terms and conditions"],
                            ),
                            "\u0020|\u0020",
                            html.A(
                                href="http://www.bristol.ac.uk/web/policies/privacy-policy.html",
                                target="_blank",
                                children=["Privacy policy"],
                            ),
                        ],
                    )
                ],
            ),
        ],
    )

    @app.callback(
        Output("met-source-store", "data"),
        Input("met-select", "value"),
    )
    def store_dataset(selected_dataset):
        return {"dataset": selected_dataset}  # type:ignore

    @app.callback(
        Output("met-latitude-store", "data"),
        Input("latitude-input", "value"),
    )
    def store_latitude(latitude):
        return {"value": latitude}  # type:ignore

    @app.callback(
        Output("met-longitude-store", "data"),
        Input("longitude-input", "value"),
    )
    def store_longitude(longitude):
        return {"value": longitude}  # type:ignore

    @app.callback(
        Output("met-date-store", "data"),
        Input("met-date-picker", "date"),
    )
    def store_date(date):
        return {"value": date}  # type:ignore

    @app.callback(
        Output("met-time-store", "data"),
        Input("met-time-input", "value"),
    )
    def store_time(time):
        return {"value": time}  # type:ignore
    
    @app.callback(
        Output("obs-min-store", "data"),
        Input("obs-min-input", "value"),
        prevent_initial_call=True,
    )
    def store_obs_min(val):
        return {"value": val}

    @app.callback(
        Output("obs-max-store", "data"),
        Input("obs-max-input", "value"),
        prevent_initial_call=True,
    )
    def store_obs_max(val):
        return {"value": val}
    
    @app.callback(
        Output("obs-button", "disabled"),
        Input("obs-min-input", "value"),
        Input("obs-max-input", "value"),
        prevent_initial_call=True,
    )
    def activate_obs_button(minval, maxval):
        if minval is None:
            return True
        if maxval is None:
            return True
        if minval>0 and maxval>=minval:
            return False
        return True


    # @app.callback(
    #     Output("set-parameters-modal", "is_open"),
    #     Input("modal-open", "data"),
    #     State("set-parameters-modal", "is_open")
    # )
    # def toggle_modal(n1, is_open):
    #     if n1:
    #         return not is_open
    #     return is_open
    
    # @app.callback(
    #     Output("model-open", "value"),
    #     Input("set-parameters-button", "n_clicks"),
    #     State("set-parameters-modal", "is_open")
    # )
    # def toggle_modal_store(n1, is_open):
    #     if n1:
    #         return not is_open
    #     return is_open

    @app.callback(
        Output("met-submit-button", "disabled"),
        Input("met-source-store", "data"),
        Input("met-latitude-store", "data"),
        Input("met-longitude-store", "data"),
        Input("met-date-store", "data"),
        Input("met-time-store", "data"),
    )
    def activate_met_button(
        met_source,
        met_latitude,
        met_longitude,
        met_date,
        met_time,
    ):

        if met_source["dataset"] == "":
            disabled = True
            return disabled

        if met_latitude["value"] is None:
            disabled = True
            return disabled

        if met_longitude["value"] is None:
            disabled = True
            return disabled

        if met_date["value"] is None:
            disabled = True
            return disabled

        if met_time["value"] is None:
            disabled = True
            return disabled

        disabled = False
        return disabled

    def param_settings(var, dist, dist_className, param1_className, param2_className, param3_className):
        param0_placeholder = 'N/A'
        param1_placeholder = 'N/A'
        param2_placeholder = 'N/A'
        param3_placeholder = 'N/A'
        if var:
            param0_placeholder = "Minimum"
            # range_1_className = "InputBox sml"
            dist_className = dist_className.replace(" invisible","")
            match dist:
                case "Uniform":
                    param1_className = param1_className.replace("invisible","")
                    if "invisible" not in param2_className:
                        param2_className = param2_className + " invisible"
                    if "invisible" not in param3_className:
                        param3_className = param3_className + " invisible"
                    param0_placeholder = uniform['dist-param-0']
                    param1_placeholder = uniform['dist-param-1']
                    param2_placeholder = uniform['dist-param-2']
                    param3_placeholder = uniform['dist-param-3']
                case "Truncated normal":
                    param1_className = param1_className.replace("invisible","")
                    param2_className = param2_className.replace(" invisible","")
                    param3_className = param3_className.replace(" invisible","")
                    param0_placeholder = truncnorm['dist-param-0']
                    param1_placeholder = truncnorm['dist-param-1']
                    param2_placeholder = truncnorm['dist-param-2']
                    param3_placeholder = truncnorm['dist-param-3']
                case "Beta":
                    param1_className = param1_className.replace("invisible","")
                    param2_className = param2_className.replace(" invisible","")
                    param3_className = param3_className.replace(" invisible","")
                    param0_placeholder = beta['dist-param-0']
                    param1_placeholder = beta['dist-param-1']
                    param2_placeholder = beta['dist-param-2']
                    param3_placeholder = beta['dist-param-3']
        else:
            param0_placeholder = "Elevation (m)"
            param1_className = "InputBox sml invisible"
            dist_className = dist_className + " invisible"
            if "invisible" not in param2_className:
                param2_className = param2_className + " invisible"
            if "invisible" not in param3_className:
                param3_className = param2_className + " invisible"
        
        return dist_className, param1_className, param2_className, param3_className, param0_placeholder, param1_placeholder, param2_placeholder, param3_placeholder

    for param in Params:
        app.callback(
            Output(param['label']+"-dist", "className"),
            Output(param['label']+"-dist-param-1", "className"),
            Output(param['label']+"-dist-param-2", "className"),
            Output(param['label']+"-dist-param-3", "className"),
            Output(param['label']+"-dist-param-0", "placeholder"),
            Output(param['label']+'-dist-param-1', 'placeholder'),
            Output(param['label']+'-dist-param-2', 'placeholder'),
            Output(param['label']+'-dist-param-3', 'placeholder'),
            Input(param['label']+"-choice", "value"),
            Input(param['label']+"-dist", "value"),
            Input(param['label']+"-dist", "className"),
            Input(param['label']+"-dist-param-1", "className"),
            Input(param['label']+"-dist-param-2", "className"),
            Input(param['label']+"-dist-param-3", "className"),
            prevent_initial_call=True
        )(param_settings)


    def activate_param_confirm(*param_choice):
        if not any(param_choice):
            return True
        else:
            return False

    app.callback(
        Output("params-submit-button", "disabled"),
        [Input(param['name'].lower().replace(' ','-') + "-choice", "value") for param in Params]
    )(activate_param_confirm)


    def activate_run_button(met_data, params_data):
        if (met_data is not None) and (params_data is not None):
            return False
        else:
            return True
    app.callback(
        Output("run-ensemble-button", "disabled"),
        [Input("met", "data"), Input("params", "data")],
        prevent_initial_call=True
    )(activate_run_button)

    @app.callback(
        [Output("set-parameters-modal", "is_open"),
         Output("params", "data")
        ],
        [Input("params-submit-button", "n_clicks"),
         Input("set-parameters-button", "n_clicks")],
        [(State(param['label']+"-choice", "value"), 
          State(param['label']+"-dist", "value"), 
          State(param['label']+"-dist-param-0", "value"), 
          State(param['label']+"-dist-param-1", "value"), 
          State(param['label']+"-dist-param-2", "value"), 
          State(param['label']+"-dist-param-3", "value")) for param in Params],
        prevent_initial_call=True
    )
    def submit_params(submit_clicked, set_clicked, *param_state):
        changed_id = [p["prop_id"] for p in callback_context.triggered][0]
        modal_open = False
        if changed_id == "set-parameters-button.n_clicks":
            modal_open = True
        if changed_id == "params-submit-button.n_clicks":
            modal_open = False
        
        param_store = dict()
        param_set = tuple(param_state[n:n+6] for n in range(0, len(param_state), 6))
        param_set = param_set[0]
        for j, param in enumerate(Params):
            label = param['label']
            param_store[label+'-fixed'] = not param_state[j][0]
            param_store[label+'-dist'] = param_state[j][1]
            param_store[label+'-dist-param-0'] = param_state[j][2]
            param_store[label+'-dist-param-1'] = param_state[j][3]
            param_store[label+'-dist-param-2'] = param_state[j][4]
            param_store[label+'-dist-param-3'] = param_state[j][5]
        return modal_open, param_store

    def reset_param(clicked):
        changed_id = [p["prop_id"] for p in callback_context.triggered][0]
        if changed_id == "params-reset-button.n_clicks":
            out = []
            for param in Params:
                out.append(False)
            for param in Params:
                out.append("Uniform")
            for param in Params:
                out.append(param['default'])
            for param in Params:
                out.append("")
            for param in Params:
                out.append("")
            for param in Params:
                out.append("")
            return tuple(out)
        

    app.callback(
        [Output(param['name'].lower().replace(' ','-')+"-choice", "value") for param in Params] +
        [Output(param['name'].lower().replace(' ','-')+"-dist", "value") for param in Params] +
        [Output(param['name'].lower().replace(' ','-')+"-dist-param-0", "value") for param in Params] +
        [Output(param['name'].lower().replace(' ','-')+"-dist-param-1", "value") for param in Params] +
        [Output(param['name'].lower().replace(' ','-')+"-dist-param-2", "value") for param in Params] +
        [Output(param['name'].lower().replace(' ','-')+"-dist-param-3", "value") for param in Params],
        [Input("params-reset-button", "n_clicks")],
        prevent_initial_call=True
    )(reset_param)

    # @app.callback(
    #     Output("vent-elevation-dist-param-0", "placeholder"),
    #     Output("vent-elevation-dist-param-1", "className"),
    #     Input("vent-elevation-choice", "value"),
    #     prevent_initial_call=True
    # )
    # def elevation_settings(elevation_choice):

    #     if elevation_choice:
    #         elevation_range_0_placeholder = "Minimum"
    #         elevation_range_1_className = "InputBox sml"
    #     else:
    #         elevation_range_0_placeholder = "Elevation (m)"
    #         elevation_range_1_className = "InputBox sml invisible"
        
    #     return elevation_range_0_placeholder, elevation_range_1_className

    @app.callback(
        Output("met", "data"),
        Output("met-extra-info-label", "children"),
        Output("met-extra-info-label", "className"),
        Input("met-submit-button", "n_clicks"),
        Input("met-source-store", "data"),
        Input("met-latitude-store", "data"),
        Input("met-longitude-store", "data"),
        Input("met-date-store", "data"),
        Input("met-time-store", "data"),
        prevent_initial_call=True
    )
    def update_met(
        clicked, met_source, met_latitude, met_longitude, met_date, met_time
    ):

        changed_id = [p["prop_id"] for p in callback_context.triggered][0]
        if changed_id == "met-submit-button.n_clicks":

            met_datetime = datetime.fromisoformat(met_date['value']+' '+met_time['value'])

            met_data = None
            msg = ""
            match met_source['dataset']:

                case 'Wyoming':

                    if met_datetime > datetime.utcnow():
                        msg = 'Cannot get forecasts from Wyoming Weather Web.'
                    else:

                        if met_datetime.hour>12:
                            hours_to_try = ['12:00','00:00']
                        else:
                            hours_to_try = ['00:00']

                        met_stations = StationList.copy()
                        met_stations = met_stations[(met_stations.date_start <= met_datetime.year) & (met_stations.date_end >= met_datetime.year)]
                        met_stations["distance"] = met_stations.apply(
                            lambda row: GreatCircleDistance(
                                row["lat"],
                                row["lon"],
                                met_latitude["value"],
                                met_longitude["value"],
                            ),
                            axis=1,
                        )

                        met_station = met_stations[
                            met_stations.distance == met_stations.distance.min()
                        ]

                        met_station_id = met_station.iloc[0].code[-5:]

                        for hr in hours_to_try:
                            try:
                                MD = MetData(
                                    met_date['value'], hr, StationID=str(met_station_id), MetType="Radiosonde", MetSource="Wyoming"
                                )
                                msg = f"{met_source['dataset']}, {met_station_id}, {met_date['value']}, {hr}"
                                met_data = MD.Data
                                break
                            except:
                                msg = f"Could not get data"
            
            return  met_data, msg, "OutputBox"
    
        else:
            return  None, None, "OutputBox invisible"

    @app.long_callback(
        output=dict(df_out=Output("ensemble", "data"),
                    fig_out=Output("ensemble-image", "children"),
                    obs_ensemble_class=Output("obs-ensemble", "className")),
        inputs=dict(clicked=Input("run-ensemble-button", "n_clicks")),
        state=dict(met_data=State("met", "data"),
                   params_data=State("params", "data"),
                   ensemble_size=State("ensemble-size", "value")
        ),
        running=[
            # (Output("run-ensemble-button", "disabled"), True, False),
            (Output("cancel-ensemble-button", "disabled"), False, True),
        ],
        cancel=[Input("cancel-ensemble-button", "n_clicks")],
        prevent_initial_call=True,
        manager=long_callback_manager,
    )
    def ensemble_run(clicked, met_data, params_data, ensemble_size):
        
        changed_id = [p["prop_id"] for p in callback_context.triggered][0]
        if changed_id == "run-ensemble-button.n_clicks":

            df = solve_ensemble(params_data, ensemble_size, met_data)

            # fig = px.scatter(df, x='Q0', y='Htop', log_x=True)
            fig = make_ensemble_figure(df, color_column="source-temperature")

            images = html.Div(
                        id="result-plot",
                        className="imagehp",
                        children=[dcc.Graph(id="ensemble-plot", figure=fig)],
                    )

            obs_ensemble_class = "column column-four"

            return dict(
                df_out=df.to_dict('records'),
                fig_out=images,
                obs_ensemble_class=obs_ensemble_class
                )


    def make_ensemble_figure(df, color_column="source-temperature"):
        
        df['H_label'] = df["Htop"].apply(lambda x: sci_string(x/1e3, precision=2, start_sci=2))
        df['Hn_label'] = df["Hn"].apply(lambda x: 0 if x<0 else sci_string(x/1e3, precision=2, start_sci=2))
        df["Q_label"] = df["Q0"].apply(lambda x: sci_string(x, precision=2, start_sci=2))
        df["Qs_label"] = df["Qs0"].apply(lambda x: sci_string(x, precision=2, start_sci=2))
        df["Collapse_label"] = df["Collapse"].apply(lambda x: 'Collapsing column' if x==1.0 else 'Buoyant column')
        df['Htop'] = df['Htop']/1e3

        fig = go.Figure(
            data=go.Scattergl(
                x=df['Qs0'],
                y=df['Htop'],
                mode='markers',
                marker=dict(
                    color=df[color_column],
                    colorscale='thermal',
                    line_width=0,
                    showscale=True,
                ),
                customdata=np.stack((df['Q_label'],
                                     df['Qs_label'],
                                     df['H_label'],
                                     df['Hn_label'],
                                     df['Collapse_label']), axis=-1
                ),
                hovertemplate="Total mass eruption rate = %{customdata[0]} kg/s<br>"
                    + "Tephra mass eruption rate = %{customdata[1]} kg/s<br>"
                    + "Plume top height = %{customdata[2]} km a.s.l.<br>"
                    + "Neutral buoyancy height = %{customdata[3]} km a.s.l.<br>"
                    + "%{customdata[4]}<br>"
                    + "<extra></extra>",
            )
        )

        Qmin = nice_round_down(df['Qs0'].min())
        Qmax = nice_round_up(df['Qs0'].max())
        Hmin = nice_round_down(df['Htop'].min())
        Hmax = nice_round_up(df['Htop'].max())

        fig.update_layout(
            plot_bgcolor="#f8f6f4",
            paper_bgcolor="#f8f6f4",
            height=500,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
            hoverdistance=10,
            hoverlabel=dict(
                bgcolor="#efebe7",
                font_size=12,
                font_family="Source Sans",
            ),
            coloraxis_colorbar=dict(
                title=color_column,
            ),
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor="LightGrey",
            zerolinecolor="DarkGrey",
            exponentformat="power",
            type="log",
            range=[np.minimum(np.log10(Qmin), np.log10(Qmax)-1), np.maximum(np.log10(Qmax), np.log10(Qmin)+1)],
            dtick=1,
            title="Tephra mass eruption rate (kg/s)",
            minor=dict(ticklen=4, tickcolor="LightGrey"),
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="LightGrey",
            zerolinecolor="DarkGrey",
            # exponentformat="power",
            range=[0, Hmax],
            title="Plume top height (km a.s.l.)",
            minor=dict(ticklen=4, tickcolor="LightGrey"),
        )
        return fig

    @app.callback(
        output=dict(fig_out=Output("obs-image", "children")),
        inputs=dict(clicked=Input("obs-button", "n_clicks")),
        state=dict(
                   df_in=State("ensemble", "data"),
                   params_data=State("params", "data"),
                   obs=State("obs-select", "value"),
                   obs_min=State("obs-min-store","data"),
                   obs_max=State("obs-max-store","data"),
        ),
        prevent_initial_call=True,
    )
    def infer_from_ensemble(clicked, df_in, params_data, obs, obs_min, obs_max):
        changed_id = [p["prop_id"] for p in callback_context.triggered][0]
        if changed_id == "obs-button.n_clicks":

            hmin = float(obs_min['value'])
            hmax = float(obs_max['value'])

            df = DataFrame(df_in)
            
            df = df.loc[(df[obs]>= hmin) & (df[obs]<=hmax)]

            df['log_Qs0'] = np.log10(df["Qs0"])

            # Qmin = nice_round_down(df['log_Qs0'].min())
            # Qmax = nice_round_up(df['log_Qs0'].min())

            hist, bin_edges = np.histogram(df['log_Qs0'], density=False)
            hist = hist/df.shape[0]
            bin_mid = 0.5*(bin_edges[1:]+bin_edges[:-1])
            bin_width = bin_edges[1:]-bin_edges[:-1]
            Qbins = np.power(10, bin_edges)
            Qbin_mid = 0.5*(Qbins[1:]+Qbins[:-1])
            Qbin_width = Qbins[1:]-Qbins[:-1]
            Qbins = [sci_string(b, precision=2, start_sci=2) for b in Qbins]

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x = Qbin_mid,
                    y = hist,
                    width = Qbin_width,
                    marker =dict(color='#1d91c0',),
                    opacity=0.75,
                )
            )
            
            # fig.add_trace(
            #     go.Histogram(
            #         x=df['Qs0'],
            #         histnorm='probability',
            #         marker_color='#1d91c0',
            #         opacity=0.75,
            #         # hovertemplate="Tephra mass eruption rate (kg/s): %{customdata[1]} kg/s<br>"
            #             # + "Plume top height = %{customdata[2]} km a.s.l.<br>"
            #             # + "Neutral buoyancy height = %{customdata[3]} km a.s.l.<br>"
            #             # + "%{customdata[4]}<br>"
            #             # + "<extra></extra>",
            #         # )
            # )

            fig.update_layout(
                plot_bgcolor="#f8f6f4",
                paper_bgcolor="#f8f6f4",
                height=500,
                margin={"l": 20, "r": 20, "t": 40, "b": 20},
                hoverdistance=10,
                hoverlabel=dict(
                    bgcolor="#efebe7",
                    font_size=12,
                    font_family="Source Sans",
                ),
            )

            fig.update_xaxes(
                showgrid=True,
                gridcolor="LightGrey",
                zerolinecolor="DarkGrey",
                # dtick=1,
                exponentformat="power",
                type="log",
                title="Tephra mass eruption rate (kg/s)",
                minor=dict(ticklen=4, tickcolor="LightGrey"),
            )
            fig.update_yaxes(
                showgrid=True,
                gridcolor="LightGrey",
                zerolinecolor="DarkGrey",
                title="Proportion in ensemble",
                minor=dict(ticklen=4, tickcolor="LightGrey"),
            )

            images = html.Div(
                id="obs-result-plot",
                className="imagehp",
                children=[dcc.Graph(id="obs-plot", figure=fig)],
            )

            return dict(fig_out=images)


    app.run_server(host="127.0.0.1", debug=True)

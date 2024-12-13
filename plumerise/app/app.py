from datetime import date, datetime, timezone
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
from dash import ctx  # type: ignore
from dash import dcc  # import dash_core_components as dcc
from dash import html  # import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.long_callback import DiskcacheLongCallbackManager
from flask_caching import Cache
from geopy import distance
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


def build_parameters_table_row(label, param):
    if param['unit']:
        title = f"{param['name']} ({param['unit']})"
    else:
        title = f"{param['name']}"
    content = [
        html.Td(title),
        html.Td(
            daq.ToggleSwitch(
                id=label+"-choice",
                label=["Fixed", "Uncertain"],
                style={"width": "150px"},
                value=False,
            )
        ),
        html.Td(
            dbc.RadioItems(
                id=label+"-dist",
                className="btn-group invisible",
                inputClassName="btn-check",
                labelClassName='btn btn-outline-primary',
                labelCheckedClassName="active",
                options=[
                    {"label": dist_label, "value": dist_value}
                    for dist_value, dist_label in distributions.items()
                ],
                value="Uniform",
                # optionHeight=20,
            )
        ),
        html.Td(
            dcc.Input(
                id=label+"-dist-param-0",
                name=label+"-dist-param-0",
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
                id=label+"-dist-param-1",
                name=label+"-dist-param-1",
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
                id=label+"-dist-param-2",
                name=label+"-dist-param-2",
                className="InputBox sml invisible",
                type="number",
                min=0,
                placeholder="mean",
                # pattern="^(?=.)([+-]?([0-9]*)(\.(?[0-9]+))?)$",
            )
        ),
        html.Td(
            dcc.Input(
                id=label+"-dist-param-3",
                name=label+"-dist-param-3",
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
    for label, param in Params.items():
        table_rows.append(
            html.Tr(
                children=build_parameters_table_row(label, param)
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
                                ],
                            )
                        ]
                    ),
                    html.Div(
                        id="obs-ensemble",
                        className="column column-four invisible",
                        children=[
                            html.Div(
                                id="output-choice",
                                className="flex-input-column",
                                # className="formfield",
                                style={"padding-bottom": "5px"},
                                children=[
                                    html.Label(
                                        "Quantity of interest:",
                                        className="Dropdown-label",
                                    ),
                                    dcc.Dropdown(
                                        id="output-select",
                                        options=(
                                            [
                                                {"label": "Tephra mass eruption rate", "value":"Qs0"},
                                                {"label": "Mass eruption rate", "value":"Q0"},
                                            ]
                                            + [{"label": label, "value": value} for value, label in observables.items()]
                                            + [{"label": p['name'], "value": label} for label, p in Params.items()]
                                        ),
                                        value="Qs0",
                                        clearable=False,
                                        placeholder="Select output",
                                        optionHeight=35,
                                    ),
                                ],
                            ),
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
                                        optionHeight=35,
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

    export_tab = html.Div(
        className="pane",
        children=[
            html.Div(
                id="export-choice",
                className="flex-input-column",
                # className="formfield",
                style={"padding-bottom": "5px"},
                children=[
                    html.Label(
                        "Export:",
                        className="Dropdown-label",
                    ),
                    dcc.Dropdown(
                        id="export-select",
                        options=(
                            [
                                {"label": "Full ensemble", "value":"full"},
                                {"label": "Matching runs", "value":"match"},
                            ]
                        ),
                        value="full",
                        clearable=False,
                        placeholder="Select data",
                        optionHeight=35,
                    ),
                ],
            ),
            html.Div(
                className="formfield",
                style={"width": "240px"},
                children=[
                    html.Button(
                        className="button",
                        id="data-download-button",
                        disabled="disabled",
                        style={"width": "150px"},
                        n_clicks=0,
                        children=[
                            html.I(
                                className="fas fa-download",
                                style={"margin-right": "10px"},
                            ),
                            html.Span(
                                "Download Ensemble",
                                style={"font-size": "10pt"},
                            ),
                        ],
                    ),
                    dcc.Download(id="ensemble-download"),
                ],
            ),
        ],
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
            dcc.Store(id="out-quantity", storage_type="session"),
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
                                dcc.Tab(
                                    label="Data export",
                                    value="exportTab",
                                    className="outertab right",
                                    selected_className="outertab--selected",
                                    children=export_tab,
                                )
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
        prevent_initial_call=True,
    )
    def activate_met_button(
        met_source,
        met_latitude,
        met_longitude,
        met_date,
        met_time,
    ):

        # if met_source is None:
        #     disabled = True
        #     return disabled

        # if met_source["dataset"] == "":
        if met_source["dataset"] is None:
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

    for label in Params.keys():
        app.callback(
            Output(label+"-dist", "className"),
            Output(label+"-dist-param-1", "className"),
            Output(label+"-dist-param-2", "className"),
            Output(label+"-dist-param-3", "className"),
            Output(label+"-dist-param-0", "placeholder"),
            Output(label+'-dist-param-1', 'placeholder'),
            Output(label+'-dist-param-2', 'placeholder'),
            Output(label+'-dist-param-3', 'placeholder'),
            Input(label+"-choice", "value"),
            Input(label+"-dist", "value"),
            Input(label+"-dist", "className"),
            Input(label+"-dist-param-1", "className"),
            Input(label+"-dist-param-2", "className"),
            Input(label+"-dist-param-3", "className"),
            prevent_initial_call=True
        )(param_settings)


    def activate_param_confirm(*param_choice):
        if not any(param_choice):
            return True
        else:
            return False

    app.callback(
        Output("params-submit-button", "disabled"),
        [Input(label + "-choice", "value") for label in Params.keys()],
        prevent_initial_call=True
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
        [(State(label+"-choice", "value"), 
          State(label+"-dist", "value"), 
          State(label+"-dist-param-0", "value"), 
          State(label+"-dist-param-1", "value"), 
          State(label+"-dist-param-2", "value"), 
          State(label+"-dist-param-3", "value")) for label in Params.keys()],
        prevent_initial_call=True
    )
    def submit_params(submit_clicked, set_clicked, *param_state):
        # changed_id = [p["prop_id"] for p in ctx.triggered][0]
        changed_id = ctx.triggered_id
        modal_open = False
        if changed_id == "set-parameters-button":
            modal_open = True
        if changed_id == "params-submit-button":
            modal_open = False
        
        param_store = dict()
        # param_set = tuple(param_state[n:n+6] for n in range(0, len(param_state), 6))
        # param_set = param_set[0]
        for j, label in enumerate(Params.keys()):
            # label = param['label']
            # param_store[label+'-fixed'] = not param_state[j][0]
            # param_store[label+'-dist'] = param_state[j][1]
            # param_store[label+'-dist-param-0'] = param_state[j][2]
            # param_store[label+'-dist-param-1'] = param_state[j][3]
            # param_store[label+'-dist-param-2'] = param_state[j][4]
            # param_store[label+'-dist-param-3'] = param_state[j][5]

            param_store[label] = {
                'fixed': not param_state[j][0],
                'dist': param_state[j][1],
                'dist-param-0': param_state[j][2],
                'dist-param-1': param_state[j][3],
                'dist-param-2': param_state[j][4],
                'dist-param-3': param_state[j][5],
            }

        return modal_open, param_store

    def reset_param(clicked):
        # changed_id = [p["prop_id"] for p in ctx.triggered][0]
        changed_id = ctx.triggered_id
        if changed_id == "params-reset-button":
            out = []
            for param in Params.items():
                out.append(False)
            for param in Params.items():
                out.append("Uniform")
            for param in Params.items():
                out.append(param['default'])
            for param in Params.items():
                out.append("")
            for param in Params.items():
                out.append("")
            for param in Params.items():
                out.append("")
            return tuple(out)
        

    app.callback(
        [Output(param+"-choice", "value") for param in Params.keys()] +
        [Output(param+"-dist", "value") for param in Params.keys()] +
        [Output(param+"-dist-param-0", "value") for param in Params.keys()] +
        [Output(param+"-dist-param-1", "value") for param in Params.keys()] +
        [Output(param+"-dist-param-2", "value") for param in Params.keys()] +
        [Output(param+"-dist-param-3", "value") for param in Params.keys()],
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

        # changed_id = [p["prop_id"] for p in ctx.triggered][0]
        changed_id = ctx.triggered_id
        if changed_id == "met-submit-button":

            met_datetime = datetime.replace(datetime.fromisoformat(met_date['value']+' '+met_time['value']), tzinfo=timezone.utc)

            met_data = None
            msg = ""
            match met_source['dataset']:

                case 'Wyoming':

                    if met_datetime > datetime.now(timezone.utc):
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

    def get_distributed_params(param_set):

        ret = []
        for key, value in param_set.items():
            if not value['fixed']:
                ret.append(key)
        return ret

    @app.callback(
        Output("output-select", "options"),
        Output("out-quantity", "data"),
        Input("params", "data"),
        prevent_initial_call=True,
    )
    def set_outputs(param_set):

        Outputs = ([
                {"label": "Tephra mass eruption rate", "value":"Qs0"},
                {"label": "Mass eruption rate", "value":"Q0"},
            ]
            + [{"label": label, "value": value} for value, label in observables.items()]
        )

        if param_set is not None:

            var_params = get_distributed_params(param_set)
            if var_params is not None:
            
                for label in var_params:
                    name = Params[label]['name']
                    Outputs.append({"label": name, "value": label})

        return Outputs, Outputs

    @app.long_callback(
        output=dict(df_out=Output("ensemble", "data"),
                    fig_out=Output("ensemble-image", "children"),
                    obs_ensemble_class=Output("obs-ensemble", "className"),
                    data_download_disabled=Output("data-download-button", "disabled"),
                    # color_choice_class=Output("output-choice", "className")
        ),
        inputs=dict(clicked=Input("run-ensemble-button", "n_clicks"),
                    Hvar=Input("obs-select", "value"),
                    Cvar=Input("output-select", "value"),
        ),
        state=dict(met_data=State("met", "data"),
                   params_data=State("params", "data"),
                   ensemble_size=State("ensemble-size", "value"),
                   df_in=State("ensemble", "data"),
                   out_options=State("out-quantity", "data")
        ),
        running=[
            # (Output("run-ensemble-button", "disabled"), True, False),
            (Output("cancel-ensemble-button", "disabled"), False, True),
        ],
        cancel=[Input("cancel-ensemble-button", "n_clicks")],
        prevent_initial_call=True,
        manager=long_callback_manager,
    )
    def ensemble_run(clicked, Hvar, Cvar, met_data, params_data, ensemble_size, df_in, out_options):
        

        if ctx.triggered_id in ["run-ensemble-button", "obs-select", "output-select"]:

            if ctx.triggered_id == "run-ensemble-button":

                df = solve_ensemble(params_data, ensemble_size, met_data)

                data_download_disabled = False

            else:  #ctx.triggered_id in ["obs-select", "output-select"]:

                df = DataFrame(df_in)


            fig = make_ensemble_figure(df, Hvar, color_column=Cvar, out_options=out_options, param_set=params_data)

            images = html.Div(
                        id="result-plot",
                        className="imagehp",
                        children=[dcc.Graph(id="ensemble-plot", figure=fig)],
                    )
            obs_ensemble_class = "column column-four"

            return dict(
                df_out=df.to_dict('records'),
                fig_out=images,
                obs_ensemble_class=obs_ensemble_class,
                data_download_disabled=data_download_disabled,
            )


    def make_ensemble_figure(df_in, Hvar, color_column, out_options, param_set):

        df = df_in.copy()
        
        df['Htop_label'] = df["Htop"].apply(lambda x: sci_string(x/1e3, precision=2, start_sci=2))
        df['Hn_label'] = df["Hn"].apply(lambda x: 0 if x<0 else sci_string(x/1e3, precision=2, start_sci=2))
        df['HtopLg_label'] = df["HtopLg"].apply(lambda x: 0 if x<0 else sci_string(x/1e3, precision=2, start_sci=2))
        df["Q_label"] = df["Q0"].apply(lambda x: sci_string(x, precision=2, start_sci=2))
        df["Qs_label"] = df["Qs0"].apply(lambda x: sci_string(x, precision=2, start_sci=2))
        df["Collapse_label"] = df["Collapse"].apply(lambda x: 'Collapsing column' if x==1.0 else 'Buoyant column')
        df[Hvar] = df[Hvar]

        color_title = next(item['label'] for item in out_options if item['value'] == color_column)
        
        match color_column:
            case 'Qs0' | 'Q0':
                df['color'] = np.log10(df[color_column])
                df['color_label'] = df[color_column].apply(lambda x: sci_string(x, precision=2, start_sci=2))
                Cmin = nice_round_down(df[color_column].min())
                Cmax = nice_round_up(df[color_column].max())
                Cmin = np.minimum(Cmin, Cmax/10)
                Cmax = np.maximum(Cmax, Cmin*10)
                Clevels = np.log10(log_steps(Cmin, Cmax, 2))
                Cmin = np.log10(Cmin)
                Cmax = np.log10(Cmax)
                Clabels = [sci_string(10**x, precision=2, start_sci=2) for x in Clevels]
                color_unit = 'kg/s'
            case 'Htop' | 'HtopLg' | 'Hn':
                df['color'] = df[color_column].apply(lambda x: 0 if x<0 else x/1e3)
                df['color_label'] = df[color_column].apply(lambda x: 0 if x<0 else sci_string(x/1e3, precision=2, start_sci=2))
                Cmin = nice_round_down(df['color'].min())
                Cmax = nice_round_up(df['color'].max())
                Cmin = np.min([0, Cmin, Cmax-1])
                Cmax = np.maximum(Cmax, Cmin+1)
                Clevels = np.linspace(Cmin, Cmax, 5)
                Clabels = [sci_string(x, precision=1, start_sci=2) for x in Clevels]
                color_unit = 'km'
            case _:
                df['color'] = df[color_column]
                df['color_label'] = df[color_column].apply(lambda x: sci_string(x, precision=2, start_sci=3))
                color_unit = Params[color_column]['unit']
                Cmin = param_set[color_column]['dist-param-0']
                Cmax = param_set[color_column]['dist-param-1']
                Clevels = np.linspace(Cmin, Cmax, 5)
                Clabels = [sci_string(x, precision=0, start_sci=3) for x in Clevels]

        fig = go.Figure(
            data=go.Scattergl(
                x=df['Qs0'],
                y=df[Hvar]/1e3,
                mode='markers',
                marker=dict(
                    color=df['color'],
                    colorscale='matter_r',
                    opacity=0.5,
                    line_width=1,
                    showscale=True,
                ),
                customdata=np.stack((df['Q_label'],
                                     df['Qs_label'],
                                     df['Htop_label'],
                                     df['Hn_label'],
                                     df['Collapse_label'],
                                     df['color_label']), axis=-1
                ),
                # hovertemplate="Total mass eruption rate = %{customdata[0]} kg/s<br>"
                #     + "Tephra mass eruption rate = %{customdata[1]} kg/s<br>"
                #     + "Plume top height = %{customdata[2]} km a.s.l.<br>"
                #     + "Neutral buoyancy height = %{customdata[3]} km a.s.l.<br>"
                #     + f"{color_title} = %{customdata[5]} {color_unit} <br>"
                #     + "%{customdata[4]}<br>"
                #     + "<extra></extra>",
            )
        )

        hvtmp = "Total mass eruption rate = %{customdata[0]} kg/s<br>"
        hvtmp += "Tephra mass eruption rate = %{customdata[1]} kg/s<br>"
        hvtmp += "Plume top height = %{customdata[2]} km a.s.l.<br>"
        hvtmp += "Neutral buoyancy height = %{customdata[3]} km a.s.l.<br>"
        if color_column in Params.keys():
            hvtmp += color_title + " = %{customdata[5]} " + color_unit +"<br>"
        hvtmp += "%{customdata[4]}<br>"
        hvtmp += "<extra></extra>"
        fig.update_traces(hovertemplate=hvtmp)

        Qmin = nice_round_down(df['Qs0'].min())
        Qmax = nice_round_up(df['Qs0'].max())
        Hmin = nice_round_down(df[Hvar].min()/1e3)
        Hmax = nice_round_up(df[Hvar].max()/1e3)

        fig.update_layout(
            plot_bgcolor="#efebe7",
            paper_bgcolor="#efebe7",
            height=400,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
            hoverdistance=10,
            hoverlabel=dict(
                bgcolor="#f8f6f4",
                font_size=12,
                font_family="Source Sans",
            ),
        )
        
        fig.update_traces(selector=dict(type='scattergl'),
            marker_colorbar_title_text = f"{color_title} ({color_unit})",
            marker_colorbar_title_side = 'right',
            marker_cmin = Cmin,
            marker_cmax = Cmax,
            marker_colorbar_tickmode = 'array',
            marker_colorbar_tickvals = Clevels,
            marker_colorbar_ticktext = Clabels,
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
            title=observables[Hvar]+" (km a.s.l.)",
            minor=dict(ticklen=4, tickcolor="LightGrey"),
        )
        return fig

    @app.callback(
        output=dict(fig_out=Output("obs-image", "children")),
        inputs=dict(
            clicked=Input("obs-button", "n_clicks"),
            Hvar=Input("obs-select", "value"),
            out_var=Input("output-select", "value"),
        ),
        state=dict(
                   df_in=State("ensemble", "data"),
                   params_data=State("params", "data"),
                   obs_min=State("obs-min-store","data"),
                   obs_max=State("obs-max-store","data"),
                   out_options=State("out-quantity", "data")
        ),
        prevent_initial_call=True,
    )
    def infer_from_ensemble(clicked, Hvar, out_var, df_in, params_data, obs_min, obs_max, out_options):
        changed_id = ctx.triggered_id

        if (changed_id == "obs-button") or (changed_id in ["obs-select", "output-select"] and clicked>0):

            hmin = float(obs_min['value'])
            hmax = float(obs_max['value'])

            df = DataFrame(df_in)
            
            df = df.loc[(df[Hvar]>= hmin*1000) & (df[Hvar]<=hmax*1000)]

            xlabel = next(item['label'] for item in out_options if item['value'] == out_var)

            log_scale = False

            match out_var:
                case 'Qs0' | 'Q0':
                    df['out'] = np.log10(df[out_var])
                    xlabel += ' (kg/s)'
                    log_scale = True
                case 'Htop' | 'HtopLg' | 'Hn':
                    df['out'] = df[out_var].apply(lambda x: 0 if x<0 else x/1e3)
                    xlabel += ' (km)'
                case _:
                    df['out'] = df[out_var]
                    out_unit = Params[out_var]['unit']
                    xlabel += f' ({out_unit})'

            # Qmin = nice_round_down(df['log_Qs0'].min())
            # Qmax = nice_round_up(df['log_Qs0'].min())

            hist, bin_edges = np.histogram(df['out'], density=False)
            if df.shape[0]>0:
                hist = hist/df.shape[0]
            bin_mid = 0.5*(bin_edges[1:]+bin_edges[:-1])
            bin_width = bin_edges[1:]-bin_edges[:-1]
            if out_var in ['Qs0', 'Q0']:
                bins = np.power(10, bin_edges)
            else:
                bins = bin_edges
            bin_mid = 0.5*(bins[1:]+bins[:-1])
            bin_width = bins[1:]-bins[:-1]
            bins = [sci_string(b, precision=2, start_sci=3) for b in bins]

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x = bin_mid,
                    y = hist,
                    width = bin_width,
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
                plot_bgcolor="#efebe7",
                paper_bgcolor="#efebe7",
                height=400,
                margin={"l": 20, "r": 20, "t": 40, "b": 20},
                hoverdistance=10,
                hoverlabel=dict(
                    bgcolor="#f8f6f4",
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
                type="log" if log_scale else "linear",
                title=xlabel,
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
        else:
            return dict(fig_out=None)

    @app.callback(
        Output("export-select", "options"),
        Input("obs-button", "n_clicks"),
        prevent_inital_call=True,
    )
    def data_export_options(obs_clicked):
        if obs_clicked>0:
            return [
                        {"label": "Full ensemble", "value":"full"},
                        {"label": "Matching runs", "value":"match"},
                    ]
        else:
            return [
                        {"label": "Full ensemble", "value":"full"},
                        {"label": "Matching runs", "value":"match", "disabled": True},
                    ]

    @app.callback(
        Output("ensemble-download", "data"),
        Input("data-download-button", "n_clicks"),
        State("export-select", "value"),
        State("ensemble", "data"),
        State("obs-select", "value"),
        State("obs-min-store","data"),
        State("obs-max-store","data"),
        prevent_initial_call=True,
    )
    def download_data(clicked, dataset, df_in, Hvar, obs_min, obs_max):
        changed_id = ctx.triggered_id
        if changed_id == "data-download-button":

            df = DataFrame(df_in)

            if dataset=="match":
                hmin = float(obs_min['value'])
                hmax = float(obs_max['value'])
                df = df.loc[(df[Hvar]>= hmin*1000) & (df[Hvar]<=hmax*1000)]

            df["Collapse"] = df["Collapse"].apply(lambda x: 'Collapse' if x==1.0 else 'Buoyant')

            df.rename(
                columns={
                    "Q0": "Mass eruption rate",
                    "Qs0": "Tephra mass eruption rate",
                    "Hn": "Neutral buoyancy height",
                    "Htop": "Plume top centreline height",
                    "HtopL": "Plume top top-hat edge height",
                    "HtopLg": "Plume top Gaussian edge height",
                    "HCondense": "Plume Condensation height",
                    "Collapse": "Buoyant/Collapse",
                },
                inplace=True,
            )

            if dataset=="full":
                return dcc.send_data_frame(df.to_csv, "plumerise_full_ensemble.csv")
            elif dataset=="match":
                return dcc.send_data_frame(df.to_csv, "plumerise_matching_ensemble.csv")


    app.run_server(host="127.0.0.1", debug=True)

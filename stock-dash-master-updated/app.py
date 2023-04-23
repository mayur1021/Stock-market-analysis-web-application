import dash
from dash.html.Img import Img
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime as dt
import yfinance as yf
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
# model
from model import prediction
from sklearn.svm import SVR

def get_stockn_price_fig(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(mode="lines", x=df["Date"], y=df["Close"]))
    return fig

def get_stock_price_fig(df):

    fig = px.line(df,
                  x="Date",
                  y=["Close", "Open"],
                  title="Closing and Openning Price vs Date")

    return fig

def get_dounts(df, label):

    non_main = 1 - df.values[0]
    labels = ["main", label]
    values = [non_main, df.values[0]]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.499)])
    return fig

def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df,
                     x="Date",
                     y="EWA_20",
                     title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig


app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Archivo+Black&family=Russo+One&display=swap"
    ])
server = app.server
# html layout of site
app.layout = html.Div(
    [
        
        html.Div(
            [
                # Navigation
                html.P("BULLSTOCKS", className="start"),
                html.Div([
                    
                    html.Div([
                        dcc.Input(placeholder =" Enter stock code ", id="dropdown_tickers", type="text"),
                        html.Button("Submit", id='submit'),
                    ],
                             className="form")
                ],
                         className="input-place"),
                html.Div([
                    dcc.DatePickerRange(id='my-date-picker-range',
                                        min_date_allowed=dt(1995, 8, 5),
                                        max_date_allowed=dt.now(),
                                        initial_visible_month=dt.now(),
                    ),
                ],
                         className="date"),
                html.Div([
                    html.Button(
                        "Stocksn", className="stockn-btn", id="stockn"),
                    html.Button(
                         className="indicatorsn-btn", id="indicatorsn"),
                    html.Button(
                        "Stock Price", className="stock-btn", id="stock"),
                    
                    html.Button("Indicator",
                                className="indicators-btn",
                                id="indicators"),
                    dcc.Input(id="n_days",
                              type="text",
                              placeholder=" Number of days"),
                    html.Button(
                        "Forecast", className="forecast-btn", id="forecast")
                ],
                         className="buttons"),
                # here
            ],
            className="nav"),
    
    

        # content
        
        html.Div(
            [
                html.Div(
                    [  # header
                        html.Img(id="logo"),
                        html.P(id="ticker")
                    ],
                    className="header"),
                html.Div(id="description", className="decription_ticker"),
                html.Div([], id="graphs-contentn"),
                html.Div([], id="graphs-content"),
                html.Div([], id="main-content"),
                html.Div([], id="main-contentn"),
                html.Div([], id="forecast-content")
            ],
            className="content"),

            ####################contentn#######
        
                 
           
           #####################contentn#########
    ],
    className="container")


# callback for company info
@app.callback([
    Output("description", "children"),
    Output("logo", "src"),
    Output("ticker", "children"),
    Output("stock", "n_clicks"),
    Output("indicators", "n_clicks"),
    Output("forecast", "n_clicks")
], [Input("submit", "n_clicks")], [State("dropdown_tickers", "value")])
def update_data(n, val):  # input parameter(s)
    if n == None:
        return html.Div([
            html.H1(["Instruction :", html.Br(), html.Br()]),
            html.P(" 1. Enter the stock code."),
            html.P(" 2. Click on Stocksn button to see the price of stock till date, in form of graph."),
            html.P(" 3. Click on Stock Price button after selecting the duration, i.e. selecting the dates from which we want to see the price of stock."),
            html.P(" 4. Click on view button to see the Main-Margin and Main-Payout dounut diagram."),
            html.P(" 5. Click on indicator button to see Average vs Date graph"),
            html.P([" 6. Click on Forecast button after selecting numbers of days for which you want to predict the value of stock", html.Br(), html.Br()]),
            html.P(" (NOTE: This can predict the value of stocks for next 60 days) ")
            

        ],className="AboutUs"), "https://htmlcolorcodes.com/assets/images/colors/white-color-solid-background-1920x1080.png", None, None, None, None
        # raise PreventUpdate
    else:
        if val == None:
            raise PreventUpdate
        else:
            ticker = yf.Ticker(val)
            inf = ticker.info
            df = pd.DataFrame().from_dict(inf, orient="index").T
            df[[ 'longName', 'shortName', 'longBusinessSummary']]
            return df['longBusinessSummary'].values[0], df['longName'].values[
                0], df['shortName'].values[0], None, None, None


#callback for stockn graphs
@app.callback(
            [Output("graphs-contentn", "children")],
            [Input("stockn", "n_clicks")],
            [State("dropdown_tickers", "value")]
)

def stockn_price(v, v2):
    if v == None:
        raise PreventUpdate
    if v2 == None:
        raise PreventUpdate

    df = yf.download(v2)
    df.reset_index(inplace=True)

    fig = get_stockn_price_fig(df)

    return [dcc.Graph(figure=fig)]

# callback for stocks graphs
@app.callback([
    Output("graphs-content", "children"),
], [
    Input("stock", "n_clicks"),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date')
], [State("dropdown_tickers", "value")])
def stock_price(n, start_date, end_date, val):
    if n == None:
        return [""]
        #raise PreventUpdate
    if val == None:
        raise PreventUpdate
    else:
        if start_date != None:
            df = yf.download(val, str(start_date), str(end_date))
        else:
            df = yf.download(val)

    df.reset_index(inplace=True)
    fig = get_stock_price_fig(df)
    return [dcc.Graph(figure=fig)]

#################dounuts##########

@app.callback(
            [Output("main-contentn", "children"), Output("stockn", "n_clicks")],
            [Input("indicatorsn", "n_clicks")],
            [State("dropdown_tickers", "value")]
)

def indicatorsn(v, v2):
    if v == None:
        raise PreventUpdate
    if v2 == None:
        raise PreventUpdate
    ticker = yf.Ticker(v2)


    df_calendar = ticker.calendar.T
    df_info = pd.DataFrame.from_dict(ticker.info, orient="index").T
    df_info.to_csv("test.csv")
    df_info = df_info[["priceToBook", "profitMargins", "bookValue", "enterpriseToEbitda", "shortRatio", "beta", "payoutRatio", "trailingEps"]]
    
    

    df_calendar["Earnings Date"] = pd.to_datetime(df_calendar["Earnings Date"])
    df_calendar["Earnings Date"] = df_calendar["Earnings Date"].dt.date

    tbl = html.Div([
             html.Div([
        html.Div([
            html.H4("Price To Book"),
            html.P(df_info["priceToBook"])
        ]),
        html.Div([
            html.H4("Enterprise to Ebitda"),
            html.P(df_info["enterpriseToEbitda"])
        ]),
        html.Div([
            html.H4("Beta"),
            html.P(df_info["beta"])
        ]),
    ], className="kpi"), 
        html.Div([
            dcc.Graph(figure=get_dounts(df_info["profitMargins"], "Margin")),
            dcc.Graph(figure=get_dounts(df_info["payoutRatio"], "Payout"))
        ], className="dounuts")
        ])
       
    
    return [
        html.Div([tbl], id="graphs-contentn")], None





# callback for indicators
@app.callback([Output("main-content", "children")], [
    Input("indicators", "n_clicks"),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date')
], [State("dropdown_tickers", "value")])
def indicators(n, start_date, end_date, val):
    if n == None:
        return [""]
    if val == None:
        return [""]

    if start_date == None:
        df_more = yf.download(val)
    else:
        df_more = yf.download(val, str(start_date), str(end_date))

    df_more.reset_index(inplace=True)
    fig = get_more(df_more)
    return [dcc.Graph(figure=fig)]


# callback for forecast
@app.callback([Output("forecast-content", "children")],
              [Input("forecast", "n_clicks")],
              [State("n_days", "value"),
               State("dropdown_tickers", "value")])
def forecast(n, n_days, val):
    if n == None:
        return [""]
    if val == None:
        raise PreventUpdate
    fig = prediction(val, int(n_days) + 1)
    return [dcc.Graph(figure=fig)]


if __name__ == '__main__':
    app.run_server(debug=True)
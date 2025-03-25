# Import required libraries
import dash
from dash import Dash, html, dcc, dash_table  # Import Dash and its components
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import numpy as np


# Define external stylesheets (Bootstrap)
external_stylesheets = ["https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"]
# Dash App
app = Dash(__name__, external_stylesheets=external_stylesheets)
# app.title = "North Tower Energy Forecast Dashboard"
app.config.suppress_callback_exceptions = True  # Avoids callback warnings
server = app.server  # This exposes the Flask server instance
# Load Data
df = pd.read_csv('nt_2019_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Hour'] = df['Date'].dt.hour  # Extract hour from the Date column

# Load training data for user-selected model training
dftest = pd.read_csv('train_data_dashboard')  # Ensure the file name matches exactly
dftest['Date'] = pd.to_datetime(dftest['Date'])
dftest['Hour'] = dftest['Date'].dt.hour  # Extract hour from the Date column

# Rename the column from "Power (kW)" to "North Tower (kWh)"
dftest = dftest.rename(columns={'Power (kW)': 'North Tower (kWh)'})
dftest = dftest.rename(columns={'Temp': 'temp_C'})

# Load feature scores data
feature_scores = pd.read_csv('feature_scores.csv')
feature_scores_wrapper = pd.read_csv('feature_scores_wrapper.csv')
feature_scores_ensemble = pd.read_csv('feature_scores_ensemble.csv')

# Define IST color palette
ist_colors = ['#109EE0', '#2E3242', '#AFB5BC', "#000000", "#FF0000"]

# Available features for selection (including Hour)
available_features = ['Sin_hour', 'Gaussian', 'Power-1', 'temp_C', 'Hour']

# Available models for selection
available_models = [
    {'label': 'Decision Tree', 'value': 'DecisionTree'},
    {'label': 'Random Forest', 'value': 'RandomForest'},
    {'label': 'Neural Network', 'value': 'NeuralNetwork'}
]

# Available error metrics for selection
available_error_metrics = [
    {'label': 'MAE', 'value': 'MAE'},
    {'label': 'MBE', 'value': 'MBE'},
    {'label': 'MSE', 'value': 'MSE'},
    {'label': 'RMSE', 'value': 'RMSE'},
    {'label': 'cvRMSE', 'value': 'cvRMSE'},
    {'label': 'NMBE', 'value': 'NMBE'}
]

# Available plot types for Tab 1
available_plot_types = [
    {'label': 'Line Plot', 'value': 'line'},
    {'label': 'Filter Feature Scores Histogram', 'value': 'histogram_scores'},
    {'label': 'Wrapper Feature Scores Histogram', 'value': 'histogram_wrapper'},
    {'label': 'Ensemble Feature Scores Histogram', 'value': 'histogram_ensemble'}
]


# App Layout
app.layout = html.Div([
    html.Div(className="container mt-4", children=[
        html.H1("🔋 North Tower Energy Forecast Dashboard", className="text-center text-primary fw-bold"),
        html.P("📊 Visualizing historical data, forecasts, and error metrics (Jan - Mar 2019).",
               className="text-center text-secondary fs-5"),

        # Date Picker
        html.Div(className="d-flex justify-content-center my-3", children=[
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=df['Date'].min(),
                max_date_allowed=df['Date'].max(),
                start_date=df['Date'].min(),
                end_date=df['Date'].max(),
                display_format="YYYY-MM-DD",
            )
        ]),

        # Tabs
        dcc.Tabs(id='tabs', value='tab-1', className="nav nav-tabs", children=[
            dcc.Tab(label='📈 Raw Data', value='tab-1', className="nav-item"),
            dcc.Tab(label='🔧 Train Your Own Model', value='tab-2', className="nav-item"),
            dcc.Tab(label='📉 Error Metrics', value='tab-3', className="nav-item"),  # New Tab 3
        ]),

        html.Div(id='tabs-content', className="mt-4"),

        # Store to hold error metrics data
        dcc.Store(id='error-metrics-store'),  # Store for error metrics
    ])
])


@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def render_content(tab, start_date, end_date):
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    fig1 = px.line(filtered_df, x="Date",
                   y=["North Tower (kWh)", "Sin_hour", "Gaussian", "Power-1", "temp_C", "Hour"],
                   template="plotly_white", markers=False,
                   color_discrete_sequence=ist_colors)

    fig1.for_each_trace(lambda t: t.update(
        name="Sine of the hour" if t.name == "Sin_hour" else
        "Gaussian function hour" if t.name == "Gaussian" else
        "Previous Hour Power (kWh)" if t.name == "Power-1" else
        "Temperature (°C)" if t.name == "temp_C" else
        "Hour" if t.name == "Hour" else t.name,
        visible="legendonly"
    ))

    fig1.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.2, xanchor="center", x=0.5))

    plot_type_dropdown = dcc.Dropdown(
        id='plot-type-dropdown',
        options=available_plot_types,
        value='line',  # Default to line plot
        placeholder="Select a plot type"
    )

    if tab == 'tab-1':
        return html.Div([
            html.H4('📊 North Towers Raw Data', className="text-center text-dark fw-bold"),
            plot_type_dropdown,
            html.Div(id='plot-container'),  # Container for the selected plot
        ], className="p-3 bg-white rounded shadow")

    elif tab == 'tab-2':
        return html.Div([
            html.H4('🔧 Train Your Own Model', className="text-center text-dark fw-bold"),
            dcc.Dropdown(
                id='train-feature-dropdown',
                options=[{'label': f, 'value': f} for f in available_features],
                multi=True,
                placeholder="Select up to 3 features",
            ),
            dcc.Dropdown(
                id='model-dropdown',
                options=available_models,
                multi=True,
                placeholder="Select model(s) to train",
            ),
            html.Button("Train Models", id="train-button", className="btn btn-primary mt-3"),
            dcc.Loading(id="loading-train", children=[html.Div(id="train-output")], type="circle"),
            html.Div(id='forecast-graph-container'),  # Container for the forecast graph in Tab 2
        ], className="p-3 bg-white rounded shadow")

    elif tab == 'tab-3':
        return html.Div([
            html.H4('📉 Error Metrics', className="text-center text-dark fw-bold"),
            dcc.Dropdown(
                id='error-metrics-dropdown',
                options=available_error_metrics,
                multi=True,
                placeholder="Select error metrics to display",
            ),
            html.Div(id='error-metrics-table'),  # Container for the error metrics table in Tab 3
        ], className="p-3 bg-white rounded shadow")


@app.callback(
    Output('plot-container', 'children'),
    [Input('plot-type-dropdown', 'value')]
)
def update_plot(plot_type):
    if plot_type == 'line':
        filtered_df = df[(df['Date'] >= df['Date'].min()) & (df['Date'] <= df['Date'].max())]
        fig = px.line(filtered_df, x="Date",
                      y=["North Tower (kWh)", "Sin_hour", "Gaussian", "Power-1", "temp_C", "Hour"],
                      template="plotly_white", markers=False,
                      color_discrete_sequence=ist_colors)
        fig.for_each_trace(lambda t: t.update(
            name="Sine of the hour" if t.name == "Sin_hour" else
            "Gaussian function hour" if t.name == "Gaussian" else
            "Previous Hour Power (kWh)" if t.name == "Power-1" else
            "Temperature (°C)" if t.name == "temp_C" else
            "Hour" if t.name == "Hour" else t.name,
            visible="legendonly"
        ))
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.2, xanchor="center", x=0.5))
        return dcc.Graph(id='line-plot', figure=fig)

    elif plot_type == 'histogram_scores':
        fig = px.bar(feature_scores, x='Feature', y='Score', title='Feature Scores', template="plotly_white")
        return dcc.Graph(id='histogram-scores', figure=fig)

    elif plot_type == 'histogram_wrapper':
        fig = px.bar(feature_scores_wrapper, x='Feature', y='Score', title='Wrapper Feature Scores', template="plotly_white")
        return dcc.Graph(id='histogram-wrapper', figure=fig)

    elif plot_type == 'histogram_ensemble':
        fig = px.bar(feature_scores_ensemble, x='Feature', y='Score', title='Ensemble Feature Scores', template="plotly_white")
        return dcc.Graph(id='histogram-ensemble', figure=fig)

    return "Please select a plot type."


@app.callback(
    [Output("train-output", "children"),
     Output("forecast-graph-container", "children"),
     Output("error-metrics-store", "data")],  # Store error metrics data
    [Input("train-button", "n_clicks")],
    [State("train-feature-dropdown", "value"),
     State("model-dropdown", "value")]
)
def train_models(n_clicks, selected_features, selected_models):
    if not n_clicks or not selected_features or len(selected_features) > 3 or not selected_models:
        return "⚠️ Please select up to 3 features and at least one model.", None, None  # Return None for the forecast graph and error metrics

    # Check if the required column exists
    if 'North Tower (kWh)' not in dftest.columns:
        return "⚠️ Column 'North Tower (kWh)' not found in the training data.", None, None

    # Prepare data
    X2 = dftest[selected_features]
    Y = dftest["North Tower (kWh)"]
    X_train, X_test, y_train, y_test = train_test_split(X2, Y, test_size=0.2)

    # Initialize models
    models = {}
    if 'DecisionTree' in selected_models:
        models['DecisionTree'] = DecisionTreeRegressor(min_samples_leaf=5)
    if 'RandomForest' in selected_models:
        models['RandomForest'] = RandomForestRegressor(n_estimators=200, min_samples_split=15, max_features='sqrt', max_depth=20)
    if 'NeuralNetwork' in selected_models:
        models['NeuralNetwork'] = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=300)

    # Train selected models
    for model_name, model in models.items():
        model.fit(X_train, y_train)

    # Predict on the 2019 data (df)
    X_2019 = df[selected_features]  # Use the same features for prediction
    df_forecast = pd.DataFrame({"Date": df["Date"], "Actual": df["North Tower (kWh)"]})

    for model_name, model in models.items():
        y_pred = model.predict(X_2019)
        df_forecast[model_name] = y_pred

    # Create the forecast figure
    fig2 = px.line(df_forecast, x="Date", y=["Actual"] + list(models.keys()),
                   title="Forecast vs Actual Energy Consumption (2019 Data)",
                   template="plotly_white", markers=False, color_discrete_sequence=ist_colors)

    # Calculate error metrics
    def calculate_metrics(y_true, y_pred):
        mae = metrics.mean_absolute_error(y_true, y_pred)
        mbe = np.mean(y_true - y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        cvrmse = rmse / np.mean(y_true)
        nmbe = mbe / np.mean(y_true)
        return mae, mbe, mse, rmse, cvrmse, nmbe

    # Calculate metrics for each model
    df_metrics = pd.DataFrame(columns=["Model", "MAE", "MBE", "MSE", "RMSE", "cvRMSE", "NMBE"])
    for model_name in models.keys():
        metrics_values = calculate_metrics(df_forecast["Actual"], df_forecast[model_name])
        df_metrics = pd.concat([df_metrics, pd.DataFrame({
            "Model": [model_name],
            "MAE": [metrics_values[0]],
            "MBE": [metrics_values[1]],
            "MSE": [metrics_values[2]],
            "RMSE": [metrics_values[3]],
            "cvRMSE": [metrics_values[4]],
            "NMBE": [metrics_values[5]]
        })], ignore_index=True)

    # Return the training output message, forecast graph, and error metrics data
    return (
        f"✅ Model Training Complete! Features used: {', '.join(selected_features)}",
        dcc.Graph(id='forecast-graph', figure=fig2),  # Render forecast graph in Tab 2
        df_metrics.to_dict('records')  # Store error metrics data
    )


@app.callback(
    Output('error-metrics-table', 'children'),
    [Input('error-metrics-store', 'data'),
     Input('error-metrics-dropdown', 'value')]
)
def display_error_metrics(data, selected_metrics):
    if not data:
        return "No error metrics available. Please train a model first."

    # Create a DataFrame from the stored data
    df_metrics = pd.DataFrame(data)

    # Filter columns based on selected metrics
    if selected_metrics:
        columns_to_display = ["Model"] + selected_metrics
        df_metrics = df_metrics[columns_to_display]

    # Create a Dash DataTable for error metrics
    error_table = dash_table.DataTable(
        id='error-table',
        columns=[{"name": col, "id": col} for col in df_metrics.columns],
        data=df_metrics.to_dict('records'),
        style_table={'overflowX': 'auto', 'margin': '20px'},
        style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'center'}
    )

    return error_table


if __name__ == '__main__':
    app.run_server(debug=True)

import os
import dash
from dash import html

app = dash.Dash(__name__)
app.layout = html.Div("Hello, Render!")

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))

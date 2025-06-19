import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs import Layout
import math

def plot_bubble(df, size_col, label_col=None, x_col=None, y_col=None, size_scale=None, colors=None):
    
    default_colors = [
        'rgb(31, 119, 180)',
        'rgb(174, 199, 232)',
        'rgb(255, 127, 14)',
        'rgb(255, 187, 120)',
        'rgb(44, 160, 44)',
        'rgb(152, 223, 138)',
        'rgb(214, 39, 40)',
        'rgb(255, 152, 150)',
        'rgb(148, 103, 189)',
        'rgb(197, 176, 213)',
        'rgb(140, 86, 75)',
        'rgb(196, 156, 148)',
        'rgb(227, 119, 194)',
        'rgb(247, 182, 210)',
        'rgb(127, 127, 127)',
        'rgb(199, 199, 199)',
        'rgb(188, 189, 34)',
        'rgb(219, 219, 141)',
        'rgb(23, 190, 207)',
        'rgb(158, 218, 229)',
    ]
    colors = default_colors if colors is None else colors
    
    N = len(df)
    x = list(range(N)) if x_col is None else df[x_col].tolist()
    y = [0]*N if y_col is None else df[y_col].tolist()
    labels = [''] * N if label_col is None else df[label_col].tolist()
        
    sizes = df[size_col].tolist()
    sizes = [math.sqrt(size) for size in sizes] # bubble size by area, not by radius
    if size_scale:
        sizes = [size*size_scale for size in sizes] # bubble size by area, not by radius

    layout = Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig = go.Figure(data=[go.Scatter(x=x, y=y, text=labels, mode='markers+text', 
                                     marker=dict(size=sizes, color=colors),
                                    opacity=1)],
                   layout=layout)
    fig.update_traces(textposition='middle center')
    
    fig.show()

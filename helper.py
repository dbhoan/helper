import pandas as pd
import numpy as np
import os
from os.path import join as jn
import plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def trim_str(s, keep_chars=20):
    return s[:(keep_chars-1)] + '…' if len(s) > keep_chars else s

def keep_top_values(df, cols, weight_col=None, keep=10, other_label='%d others', show_n_others=True):
    """
    Args:
        df: input pandas dataframe
        cols: list of column names, or can be a string to specify only one column
        weight_col: name of weight column (for data that has already been aggregated)
        keep: number of most frequent values to keep, can be a list if specifying for multiple columns
        other_label: label to replace the rest of the value with
        show_n_others: if True add number of other unique values to other_label
        
    Returns:
        out: dataframe where most frequent values are kept and others are grouped
        
    """

    if type(cols) is str:
        cols = [cols]
    if type(keep) is int:
        keep = [keep]* len(cols)
        
    assert len(cols)==len(keep), 'Mismatched lengths between cols and keep.'

    out = df.copy()
    for i, c in enumerate(cols):
        if weight_col is None:
            vc = out[c].value_counts()
        else:
            vc = out.groupby(c)[weight_col].sum().sort_values(ascending=False)
        if len(vc) <= keep[i]:
            continue
        if show_n_others:
            if '%d' in other_label:
                label = other_label % (len(vc)-keep[i])
            else:
                label = other_label + str(len(vc)-keep[i])
        mapping = {s:label for s in vc.index[keep[i]:]}
        out[c] = out[c].replace(mapping)
        
    return out

def plot_sankey(input_dataframe, cols, weight_col=None, keep=20, n_chars=30, show_count=True,
    font_size=10, width=800, height=600, col_labels=None, col_label_size=12,
    show=True, save=True, save_path='./plots/', save_file=None, auto_open=False):
    
    """
    Args:
        input_dataframe: pandas dataframe input
        cols: list of column names (required at least 2) where values/counts are extracted
        weight_col: name of weight column (for data that has already been aggregated)
        keep: number of most frequent values to keep per column (the rest are grouped as others)
            can also be a list to specify for individual columns
        n_chars: values with long names will be trimmed to keep first n_chars characters
        show_count: if True show the count in value name
        col_labels: labels to show for each column, default values are column names
        col_label_size: font size for column labels
        show: if True show plot
        save: if True save plot as HTML file as save_path\save_file
        
    """
    
    col_labels = cols if col_labels is None else col_labels
    
    assert type(cols) is list, 'Input cols should be a list of column names.'
    assert len(cols) >= 2, 'Need to specify a list of at least 2 columns.'
    assert len(cols)==len(col_labels), 'Number of column labels must match number of columns.'
    
    # Prepare data for plotting
    
    df0 = keep_top_values(input_dataframe, cols, weight_col=weight_col, keep=keep)
    
    value_counts_list = []
    for c in cols:
        if weight_col is None:
            value_count = df0[c].value_counts()
        else:
            value_count = df0.groupby(c)[weight_col].sum().sort_values(ascending=False)
        value_count.index = [(c+'>>'+str(s)) for s in value_count.index]
        value_counts_list.append(value_count)
    vc = pd.concat(value_counts_list, axis=0)

    dfs = []
    for i in range(len(cols)-1):
        c1, c2 = cols[i], cols[i+1]
        if weight_col is None:
            df2 = df0[[c1,c2]]
            df2['count'] = 1
        else:
            df2 = df0[[c1,c2,weight_col]]
            df2['count'] = df2[weight_col]
        df2 = df2.groupby([c1,c2])['count'].sum().reset_index()[[c1,c2,'count']]
        df2[c1] = c1 + '>>' + df2[c1].astype(str)
        df2[c2] = c2 + '>>' + df2[c2].astype(str)
        df2 = df2.rename(columns={c1:'source',c2:'target'})
        dfs.append(df2)
        
    df = pd.concat(dfs).sort_values('count', ascending=False)
    labels_full = set(list(df['source'].unique()) + list(df['target'].unique()))
    label_mapping = {v:k for k,v in enumerate(labels_full)}
    plot_data = df.replace(label_mapping)
    
    if show_count:
        labels = ['%s (%d)'%(trim_str(s.split('>>')[1],n_chars),vc[s]) for s in labels_full]
    else:
        labels = [trim_str(s.split('>>')[1],n_chars) for s in labels_full]  
    
    # Create sankey plot
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            label = labels, pad = 15, thickness = 20,
            line = dict(color = "black", width = 0.5),
        ),
        link = dict(
            source = list(plot_data['source'].values),
            target = list(plot_data['target'].values),
            value = list(plot_data['count'].values)
    ))])
        
    # Add annotations for column labels
    
    for i, c in enumerate(cols):
        text = '%s (%d)' % (col_labels[i], len(input_dataframe[c].unique()))
        x = i/(len(cols)-1) 
        fig.add_annotation(x=x, y=1, text=text, showarrow=False, yshift=30, font_size=col_label_size)
    
    fig.update_layout(font_size=font_size, width=width, height=height)
    
    if show:
        fig.show()
    
    # Save output plot as an HTML file
    
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_file is None:
            save_file = 'sankey-'+'-'.join(cols) + '.html'
        plotly.offline.plot(fig, filename=jn(save_path,save_file), auto_open=auto_open)
        print('Output plot saved as %s' % jn(save_path,save_file))

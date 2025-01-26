import pandas as pd
import numpy as np
import os
from os.path import join as jn
from datetime import datetime as dt
import plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# =====================================================================================
#  DATA CLEAN-UP AND PROCESSING
# =====================================================================================

def remove_rows_cols(input_df):   
    df = input_df.copy()
    
    # Remove rows that contain only NaN
    df = df.dropna(axis=0, how='all')

    # Remove columns that contain only NaN
    df = df.dropna(axis=1, how='all')

    # Remove columns that contain only a single value besides NaN
    single_value_cols = [c for c in df.columns if len(df[c].dropna().unique()) == 1]
    df = df.drop(columns=single_value_cols)

    print('Data size before :', input_df.shape)
    print('Data size after  :', df.shape)
    
    return df


def convert_mixed_type_to_str(df):
    count = 0
    for c in df.columns:
        if df[c].dtype == 'O':
            df[c] = df[c].astype(str)
            count += 1
    
    print('Converted %d columns to string type.' % count)
            
    return df


def convert_to_datetime(df, cols=None):
    # Automatically detect and convert if columns are not specified
    if cols is None:
        converted_cols = []
        for c in df.columns:
            if df[c].dtype == 'O':
                try:
                    df[c] = pd.to_datetime(df[c])
                    converted_cols.append(c)
                except ValueError:
                    pass
        print('Columns converted to datetime:', converted_cols)
                
    # Convert specified columns to datetime
    else:
        for c in datetime_cols:
            df[c] = pd.to_datetime(df[c])
        print('Converted %d columns to datetime type.' % len(cols))
        
    return df


def clean(df, saveas='', datetime_cols=None):
    df = remove_rows_cols(df)
    df = convert_mixed_type_to_str(df)
    df = convert_to_datetime(df, datetime_cols)
    
    # Save as parquet file if filename is specified
    if saveas != '':
        df.to_parquet(saveas, engine='fastparquet')
        print('Data frame saved as parquet file %s')
    return df


def search(df, s):
    """ 
    Search for specific string value in dataframe column names or cell values
    Args:
        df: pandas dataframe input
        s: string to search for in column names and values
        
    """ 
    print('Columns with matched names:')
    for c in df.columns:
        if s.lower() in c.lower():
            print('  %s' % c)
    print('Columns with matched values:')
    for c in df.columns:
        value_found = False
        for v in df[c].astype(str).values:
            if s.lower() in v.lower():
                value_found = True
        if value_found:
            print('  %s' % c)

            
def summarize_statistics(df, n_top=5, n_chars=20, save=True, output='summary_statistics.xlsx'):
    """
    Calculate statistics for each column and save as Excel file
    
    Args:
        df: pandas dataframe input
        n_top: number of top values to show
        n_chars: number of characters to keep if string values are too long
        save: if True save output as Excel file
        output: file name to save
        
    Return:
        ss: pandas dataframe containing summary statistics
    """
    
    cols = ['Field','Type','Count','Empty','Min','Max','Mean','Unique']
    for i in range(n_top):
        cols.append('%%%d' % (i+1))        
        cols.append('Value%d' % (i+1))
    ss = pd.DataFrame(columns=cols)
    n = len(df)
    
    for k, c in enumerate(df.columns):
        print('Processing column %d/%d (\'%s\')%40s' % (k, len(df.columns),c,'') ,end='\r')
        
        # String
        if df[c].dtype.name in ['object']:
            vc = df[c].value_counts()
            vc = vc[~vc.index.str.strip().str.lower().isin(['nan',''])]
            n_unique = len(vc)
            count = vc.sum()
            missing = n - count
            top_vals = []
            fracs = []
            for i,v in zip(vc.head(n_top).index, vc.head(n_top).values):
                top_vals.append(trim_str(i, n_chars))
                fracs.append(v/n)
            if len(top_vals) < n_top:
                top_vals += ['' for i in range(n_top-len(top_vals))]
                fracs += ['' for i in range(n_top-len(fracs))]
            new_row = [c, 'string', count, missing/n, '', '', '', n_unique]
            for v,f in zip(top_vals, fracs):
                new_row.append(f)
                new_row.append(v)
            ss.loc[len(ss)] = new_row
            
        # Numeric
        elif df[c].dtype.name in ['int64', 'float64']:
            vc = df[c].value_counts()
            n_unique = len(vc)
            count = vc.sum()
            missing = n - count
            top_vals = []
            fracs = []
            for i,v in zip(vc.head(n_top).index, vc.head(n_top).values):
                top_vals.append(i)
                fracs.append(v/n)
            if len(top_vals) < n_top:
                top_vals += ['' for i in range(n_top-len(top_vals))]
                fracs += ['' for i in range(n_top-len(fracs))]
            new_row = [c, 'numeric', count, missing/n, df[c].min(), df[c].max(), df[c].mean(), n_unique]
            for v,f in zip(top_vals, fracs):
                new_row.append(f)
                new_row.append(v)
            ss.loc[len(ss)] = new_row
            
        # Dates
        elif df[c].dtype.name in ['datetime64[ns]']:
            vc = df[c].value_counts()
            n_unique = len(vc)
            count = vc.sum()
            missing = n - count
            for i,v in zip(vc.head(n_top).index, vc.head(n_top).values):
                top_vals.append(i)
                fracs.append(v/n)
            if len(top_vals) < n_top:
                top_vals += ['' for i in range(n_top-len(top_vals))]
                fracs += ['' for i in range(n_top-len(fracs))]
            date_min = df[c].min()
            date_min = date_min.date() if date_min is not None else ''
            date_max = df[c].max()
            date_max = date_max.date() if date_max is not None else ''
            new_row = [c, 'datetime', count, missing/n, date_min, date_max, '', n_unique]
            new_row += ['']*2*n_top
            ss.loc[len(ss)] = new_row
        
    ss = ss.drop(columns=['Empty'])
    
    if save:
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        wb = writer.book
        ss.to_excel(writer, index=False)

        # Define colors and base formats
        light_green, light_blue, light_purple = '#DDEBDD', '#D8DCE8', '#DED3E8'
        f0 = wb.add_format({'bg_color':'white'})
        f_header = wb.add_format({'bg_color':'#595959','font_color':'white','bold':True})
        f_perc = wb.add_format({'bg_color':'white','num_format':'0%'})
        f_float = wb.add_format({'bg_color':'white','num_format':'0.0'})
        f_numeric = wb.add_format({'bg_color': light_blue})
        f_string = wb.add_format({'bg_color': 'white'})
        f_datetime = wb.add_format({'bg_color': light_green})

        # Define conditional formats
        cf1 = {
            'type':'data_bar','bar_solid':True,'bar_color':'#7399BF',
            'min_type':'num','min_value':0,'max_type':'num','max_value':n}
        cf2 = {
            'type':'2_color_scale','min_color':'#FFFFFF','max_color':'#F8696B',
            'min_type':'num','min_value':0,'max_type':'num','max_value':1}
        cf3 = {
            'type':'data_bar','bar_solid':True,'bar_color':'#7399BF','bar_only':True,
            'min_type':'num','min_value':0,'max_type':'num','max_value':1}
        cfs = {
            'numeric': {'type':'formula', 'criteria':'=$Z$1=""', 'format':f_numeric},
            'string': {'type':'formula', 'criteria':'=$Z$1=""', 'format':f_string},
            'datetime': {'type':'formula', 'criteria':'=$Z$1=""', 'format':f_datetime}}

        # Specify column widths and base formats
        col_widths = [18,8,10,11,11,11,8] + [3.5,20]*n_top
        col_formats = [f0,f0,f0,f0,f0,f_float,f0] + [f0,f0]*n_top

        # Loop through all sheets
        for name, ws in writer.sheets.items():
            
            # Format header and set column width for each column
            for i, col_name in enumerate(ss.columns.values):
                ws.write(0, i , col_name, f_header)
                ws.set_column(i, i, col_widths[i], col_formats[i])

            # Set conditional format based on data type for all except Count column
            for i, row in ss.iterrows():
                ws.conditional_format('A%d:B%d'%(i+2,i+2), cfs[row['Type']])
                ws.conditional_format('D%d:Q%d'%(i+2,i+2), cfs[row['Type']])

            # Set data_bar for Count column
            ws.conditional_format('C2:C1000', cf1)

            # Set data_bar for value percentage columns
            for i in range(n_top):
                col = 7+i*2
                ws.conditional_format(1,col,1000, col, cf3)    

#             # Set 2_color_scale for Empty column
#             ws.conditional_format('D2:D1000', cf2)

        writer.save()
        print('Output saved as %s' % output + ' '*30)
    
    return ss        


def trim_str(s, keep_chars=20):
    return s[:(keep_chars-1)] + '…' if len(s) > keep_chars else s

            
def keep_top_values(df, cols, keep=10, other_label='%d others', show_n_others=True):
    """
    Args:
        df: input pandas dataframe
        cols: list of column names, or can be a string to specify only one column
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
        vc = out[c].value_counts()
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


# =====================================================================================
#  PLOT FUNCTIONS
# =====================================================================================


def plot_count_by_time(df, date_col, freq='week', groupby='', keep=6, kind='bar',
    figsize=(8,5), style='fivethirtyeight', ylabel='Count', title='',):
    dfs, labels = [df], ['Total']
    if groupby != '':
        if groupby in df.columns:
            dfs, labels = [], []
            kind = 'line'
            top_vals = list(df[groupby].value_counts().head(keep).index)
            labels += top_vals # for legend
            for val in top_vals:
                dfs.append(df[df[groupby]==val])
        else:
            print('Groupby column not found in dataframe.')
    if freq == 'day':
        xlabel = 'By date'
        kind = 'line'
        count_dfs = [x.groupby([x[date_col].dt.date])[date_col].count() for x in dfs]
    elif freq == 'week':
        xlabel = 'By week'
        count_dfs = [x.groupby([x[date_col].dt.year, x[date_col].dt.strftime('%U')])[date_col].count() for x in dfs]
    elif freq == 'month':
        xlabel = 'By month'    
        count_dfs = [x.groupby([x[date_col].dt.year, x[date_col].dt.month])[date_col].count() for x in dfs]
        for df in count_dfs:
            df.index = [dt(int(y),int(m),1).strftime('%b %Y') for y,m in df.index]

    fig, ax = plt.subplots(figsize=figsize)
    plt.style.use(style)
    fig.autofmt_xdate()
    for count_df, label in zip(count_dfs, labels):
        count_df.plot(label=label, kind=kind)
    ax.tick_params(axis='x', labelsize=10, rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if groupby != '':
        plt.legend(loc='upper right')


def plot_sankey(input_dataframe, cols, keep=20, n_chars=30, show_count=True,
    font_size=10, width=1400, height=800, col_labels=None, col_label_size=12,
    show=True, save=True, save_path='./plots/', save_file=None, auto_open=False):
    
    """
    Args:
        input_dataframe: pandas dataframe input
        cols: list of column names (required at least 2) where values/counts are extracted
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
    
    df0 = keep_top_values(input_dataframe, cols, keep)
    
    value_counts_list = []
    for c in cols:
        value_count = df0[c].value_counts()
        value_count.index = [(c+'>>'+str(s)) for s in value_count.index]
        value_counts_list.append(value_count)
    vc = pd.concat(value_counts_list, axis=0)

    dfs = []
    for i in range(len(cols)-1):
        c1, c2 = cols[i], cols[i+1]
        df2 = df0[[c1,c2]]
        df2['count'] = 1
        df2 = df2.groupby([c1,c2]).count().reset_index()[[c1,c2,'count']]
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

    
def plot_pie(df, col, keep=10, n_chars=15, show=True, 
    other_label='%d others', show_n_others=True, textinfo='label+value',
    font_size=10, width=400, height=400, title=None, title_offset=0,   
    save=False, save_path='./plots/', save_file=None, auto_open=False):
    
    """
    Args:
        df: pandas dataframe input
        col: column from which values are count and plotted
        keep: number of most frequent values to keep, the rest is grouped as "others"
        n_chars: values with long names will be trimmed to keep first n_chars characters
        show: if True show plot
        other_label: label to replace the rest of the value with
        show_n_others: if True add number of other unique values to other_label
        save: if True save plot as HTML file as save_path\save_file
    
    """    
    if title is None:
        title = col      
    
    df_count = df[col].value_counts()
    if len(df_count) > keep:
        if show_n_others and ('%d' in other_label):
            other_label = other_label % (len(df_count)-keep)
        top = df_count.head(keep)
        bottom = pd.Series([df_count.iloc[keep-1:].sum()], index=[other_label])
        df_count = pd.concat([top,bottom])

    labels = df_count.index.tolist()
    labels = [trim_str(str(s),n_chars) for s in labels]
    values = df_count.values.tolist()

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo=textinfo, hole=.55)])
    fig.update_layout(
        font_size=font_size, width=width, height=height, template='seaborn', showlegend=False,
        title={'text': title,'y':0.48,'x':0.5 + title_offset,'xanchor': 'center','yanchor': 'middle'},
    )
    if show:
        fig.show()

    # Save output plot as an HTML file
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_file is None:
            save_file = 'pie-' + col + '.html'
        plotly.offline.plot(fig, filename=jn(save_path,save_file), auto_open=auto_open)
        print('Output plot saved as %s' % jn(save_path,save_file))

        
def plot_bar(series, n_chars=16, color='#30A2DA',
    threshold=None, color_above='#E5AE38', color_below='#30A2DA',
    figsize=(11,2.5), style='fivethirtyeight', title=None, ylabel=None):

    df_plot = series.copy()
    df_plot.index = [trim_str(s,n_chars) for s in df_plot.index]

    fig, ax = plt.subplots(figsize=figsize)
    plt.style.use(style)
    fig.autofmt_xdate()

    if threshold is not None:
        ax.axhline(threshold, linewidth=2, linestyle=':')
        colors = [color_below if v < threshold else color_above for v in df_plot.values]
    else:
        colors = color
        
    df_plot.plot(kind='bar', color=colors)

    ax.tick_params(axis='x', labelsize=10, rotation=45)        
    plt.title(title, size=14)
    plt.ylabel(ylabel, size=14)
    plt.show()        

    
def plot_grouped_bar(df, cols, labels=None, colors=None, width = 0.3, 
    figsize=(14,2.5), style='fivethirtyeight', title=None, ylabel=None):
    
    n = len(cols)
    x_coord = np.arange(len(df))
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.autofmt_xdate()
    plt.style.use(style)

    for i,c in enumerate(cols):
        bar = ax.bar(x_coord - (n+1)*width/2 + (i+1)*width, df[c], width, color=colors[i], label=labels[i])
    
    ax.tick_params(axis='x', labelsize=10, rotation=45)
    ax.set_xticks(x_coord)
    ax.set_xticklabels([trim_str(s,16) for s in df.index])

    plt.title(title, size=14)
    plt.ylabel(ylabel, size=14)
    plt.legend(fontsize=10, loc='upper right')
    plt.show()        

    
def plot_time_return_map(input_df, col, window_size=1, add_random_noise=True,
    figsize=(6,6), colors='grey', sizes=3 ,alpha=0.4, title=None, 
    ticks = [1, 60, 3600, 3600*24], ticklabels = ['1 sec', '1 min', '1 hour', '1 day']):
    
    df = input_df[[col]]
    if add_random_noise:
        df[col] = df[col] + pd.to_timedelta(np.random.rand(len(df)), unit='seconds')
        
    df = df.sort_values(col).reset_index()

    window_size = 1
    for i in range(1,window+1):
        df['x%d' % i] = (df[col] - df[col].shift(i)).dt.total_seconds()
        df['y%d' % i] = (df[col].shift(-i) - df[col]).dt.total_seconds()

    x_cols = ['x%d' % i for i in range(1, window_size + 1)]
    y_cols = ['y%d' % i for i in range(1, window_size + 1)]
    df['x'] = df[x_cols].mean(axis=1)
    df['y'] = df[y_cols].mean(axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(df['x'], df['y'], c=colors, alpha=alpha, s=sizes)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    ax.set_xlim(0.5,3600*24*3)
    ax.set_ylim(0.5,3600*24*3)
    plt.title(title)
    plt.show()    

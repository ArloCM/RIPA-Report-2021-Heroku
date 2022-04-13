### Imports
# import math
from itertools import count
from nntplib import NNTPPermanentError
from tracemalloc import stop
import numpy as np
import pandas as pd
from scipy import stats
# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()
import geopandas as gpd
# from shapely import wkt
import pytz
from astral import LocationInfo
import datetime
from astral.sun import sun
from typing import List, Tuple, Optional, Union
# from sodapy import Socrata
from matplotlib import font_manager
font_manager.fontManager.addfont('Nunito-VariableFont_wght.ttf')
import matplotlib.pyplot as plt
params = {'legend.fontsize':14,
          'figure.figsize':(12, 9),
          'axes.labelsize':24,
          'axes.titlesize':28,
          'xtick.labelsize':12,
          'ytick.labelsize':12,
          'font.family':'Nunito, Tahoma, Arial'}
plt.rcParams.update(params)
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

### Data Analysis Functions
def get_stops(stops_df:pd.DataFrame,
          vehicle:bool = True,
          pedestrian:bool = True,
          freq:str = 'W',
          beats:bool = False,
          year:List[int] = list(range(2015, 2021))) -> pd.DataFrame:
    '''This function takes the stops df and returns a df
       of total stops by beat by race.
       
       Inputs:
       stops: stops Data from Berkeley Open Data, cleaned
       freq: grouping of observations by time period (W = weeks, M = months, Y= years)
       
       Returns:
       Dataframe of searches by race by time/beat
    '''
    stops_df = stops_df[stops_df['datetime'].dt.year.isin(year)] # Restricting observations to years specified in parameters
    if vehicle & pedestrian:
        pass
    elif vehicle:
        stops_df = stops_df[stops_df['vehicle'] == 1]
    else:
        stops_df = stops_df[stops_df['pedestrian'] == 1]
    
    if beats == True:
        stops_df = stops_df[stops_df['datetime'].dt.year.isin(year)] # Restricting observations to when a search has occurred    
    
    if beats == False:
        stops = (stops_df.groupby(pd.Grouper(key='datetime', freq = freq))
                     .sum()
                     [['asian', 'black', 'hispanic', 'other', 'white']]) # Selecting race variables
    else:
        stops = (stops_df.groupby(['beat'])
                     .sum()
                     [['asian', 'black', 'hispanic', 'other', 'white']]) # Selecting race variables

    
    return stops

def get_searches(stops_df:pd.DataFrame,
             vehicle:bool = True,
             pedestrian:bool = True,
             freq:str = 'W',
             beats:bool = False,
             year:List[int] = list(range(2015, 2021))) -> pd.DataFrame:
    
    '''This function takes the stops df and returns a df
       of total searches by beat by race.
       
       Inputs:
       stops: stops Data from Berkeley Open Data, cleaned
       freq: grouping of observations by time period (W = weeks, M = months, Y= years)
       
       Returns:
       Dataframe of searches by race by time/beat
    '''
    stops_df = stops_df[stops_df['datetime'].dt.year.isin(year)] # Restricting observations to years specified in parameters
    searches = stops_df[stops_df['searches'] == 1] # Filtering for searches
    
    if beats == True:
        searches = searches[searches['datetime'].dt.year.isin(year)] # Restricting observations to when a search has occurred

    ### Restricting according to types of stops parameters
    if vehicle & pedestrian:
        pass
    elif vehicle:
        searches = searches[searches['vehicle'] == 1]
    else:
        searches = searches[searches['pedestrian'] == 1]
    
    ### Grouping according to parameters
    if beats == False:
        searches = (searches.groupby(pd.Grouper(key='datetime', freq = freq))
                     .sum()
                     [['asian', 'black', 'hispanic', 'other', 'white']]) # Selecting race variables
    else:
        searches = (searches.groupby(['beat'])
                     .sum()
                     [['asian', 'black', 'hispanic', 'other', 'white']]) # Selecting race variables
    
    return searches

def get_outcomes(stops_df:pd.DataFrame,
             vehicle:bool = True,
             pedestrian:bool = True,
             arrests:bool = True,
             citations:bool = True,
             contraband:bool = False,
             freq:str = 'W',
             beats:bool = False,
             year:List[int] = list(range(2015, 2021))) -> pd.DataFrame:
    
    '''This function takes in a df of stop records from BPD 
       plus selections of what to include in the analysis. 
       df must have datetime column titled 'datetime'. 
       It returns a df of the yield rates at the selected frequency.
       
       Inputs:
       vehicle/pedestrian: what types of stops to include.
       arrests/citations: whay types of outcomes to inclue.
       freq: grouping of observations by time period (W = weeks, M = months, Y= years)
       year: list of years to restrict the Data
       
       Returns:
       Dataframe of outcomes by race by time/beat
    '''
    
    stops_df = stops_df[stops_df['searches'] == 1] # Restricting observations to when a search has occurred
    stops_df = stops_df[stops_df['datetime'].dt.year.isin(year)] # Restricting observations to when a search has occurred

    ### Restricting according to types of stops parameters
    if vehicle & pedestrian:
        pass
    elif vehicle:
        stops_df = stops_df[stops_df['vehicle'] == 1]
    else:
        stops_df = stops_df[stops_df['pedestrian'] == 1]

    
    ### Restricting according to types of outcomes parameters
    if arrests & citations & contraband:
        outcomes = stops_df[(stops_df['arrests'] == 1) | (stops_df['citations'] == 1) | (stops_df['contraband'] == 1)]
    elif arrests & citations:
        outcomes = stops_df[(stops_df['arrests'] == 1) | (stops_df['citations'] == 1)]
    elif arrests & contraband:
        outcomes = stops_df[(stops_df['arrests'] == 1) | (stops_df['contraband'] == 1)]
    elif citations & contraband:
        outcomes = stops_df[(stops_df['citations'] == 1) | (stops_df['contraband'] == 1)]
    elif arrests:
        outcomes = stops_df[stops_df['citations'] == 1]
    elif citations:
        outcomes = stops_df[stops_df['citations'] == 1]
    else:
        outcomes = stops_df[stops_df['contraband'] == 1]
        
    ### Grouping according to parameters
    if beats == False:
        outcomes = (outcomes.groupby(pd.Grouper(key='datetime', freq = freq))
                     .sum()
                     [['asian', 'black', 'hispanic', 'other', 'white']]) # Selecting race variables
    else:
        outcomes = (outcomes.groupby(['beat'])
                     .sum()
                     [['asian', 'black', 'hispanic', 'other', 'white']]) # Selecting race variables

    return outcomes


def get_yields(stops_df:pd.DataFrame,
           vehicle:bool = True,
           pedestrian:bool = True,
           arrests:bool = True,
           citations:bool = True,
           contraband:bool = False,
           freq:str = 'W',
           beats:bool = False,
           year:List[int] = list(range(2015, 2021))) -> pd.DataFrame:
    
    '''This function takes in a df of stop records from BPD 
       plus selections of what to include in the analysis. 
       df must have datetime column titled 'datetime'. 
       It returns a df of the yield rates at the selected frequency.
       
       Inputs:
       vehicle/pedestrian: what types of stops to include.
       arrests/citations: whay types of outcomes to inclue.
       freq: grouping of observations by time period (W = weeks, M = months, Y= years)
       year: list of years to restrict the Data
       
       Returns:
       Dataframe of yield rates by race
    '''

     ### Getting search denominators
    searches = get_searches(stops_df,
                            vehicle, pedestrian,
                            freq, beats, year)
    outcomes = get_outcomes(stops_df,
                            vehicle, pedestrian,
                            arrests, citations, contraband,
                            freq, beats, year)
    
    ### Reshaping the outcome df to get the numerator; outcomes by freq by race
    
    yields = pd.DataFrame()
    
    yields['white'] = outcomes['white']/searches['white']
    yields['black'] = outcomes['black']/searches['black']
    yields['hispanic'] = outcomes['hispanic']/searches['hispanic']
    yields['asian'] = outcomes['asian']/searches['asian']
    yields['other'] = outcomes['other']/searches['other']
    
    return yields

### Visualization Functions

def plot_yield_after_search(stops_df: pd.DataFrame,
                            vehicle:bool = True,
                            pedestrian:bool = True,
                            arrests:bool = True,
                            citations:bool = True,
                            contraband:bool = True,
                            rolling_days:int = 365,
                            freq:str = 'W',
                            beats:bool = False,
                            year:List[int] = list(range(2015, 2020))) -> None:

    '''
    '''
    
    ### Setting label variables according to parameters    
    if vehicle & pedestrian:
        stop_types = 'Vehicle and Pedestrian'
    elif vehicle:
        stop_types = 'Vehicle'
    else:
        stop_types = 'Pedestrian'
    
    if arrests & citations & contraband:
        outcome_types = 'Arrests, Citations or Contraband'
    elif arrests & citations:
        outcome_types = 'Arrests or Citations'
    elif arrests & contraband:
        outcome_types = 'Arrests or Contraband'
    elif citations & contraband:
        outcome_types = 'Citaions or Contraband'
    elif arrests:
        outcome_types = 'Arrests'
    elif citations:
        outcome_types = 'Citations'
    else:
        outcome_types = 'Contraband'
    
    if freq == 'W':
        time = 'Week'
    elif freq == 'M':
        time = 'Month'
    else:
        time = 'Year'
        
    ### Initializing plot
    plt.figure(figsize = (12,9))
    ax = plt.gca()
    
    ### Getting Data
    yields = get_yields(stops_df,
                        vehicle, pedestrian,
                        arrests, citations, contraband,
                        freq, beats, year)
    stops = get_stops(stops_df,
                      vehicle, pedestrian,
                      freq, beats, year)
    searches = get_searches(stops_df,
                            vehicle, pedestrian,
                            freq, beats, year)
    outcomes = get_outcomes(stops_df,
                            vehicle, pedestrian,
                            arrests, citations, contraband,
                            freq, beats, year)
    
    if beats:
        x = yields['white']
        y = yields['black']
        ax = sns.scatterplot(x = x, y = y,
                             size = (searches['white'] + searches['black']),
                             sizes = (400, 4000), alpha = .6)

        if len(year) == 1:
            plt.title(f'Yield Rates After Search {year[0]}')
        else:
            plt.title(f'Yield Rates After Search {year[0]}-{year[-1]}')
        plt.xlabel('White Yield Rate')
        plt.ylabel('Black Yield Rate')
        plt.legend(markerscale = .25, loc = 'upper right', title = '# of Searches', title_fontsize = 14)
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls = '--', c = '.3')

        props = dict(boxstyle = 'square', facecolor='white', edgecolor = 'gray', alpha=0.5)
        textstr = f'''{outcome_types} after \n{stop_types} Searches'''
        ax.text(0.015, 0.977, textstr, transform = ax.transAxes,
                verticalalignment = 'top', horizontalalignment = 'left',
                bbox = props, fontsize = 14)

        for line in range(len(x)):
            ax.text(x.iloc[line], y.iloc[line], x.index[line],
                    horizontalalignment = 'center', verticalalignment = 'center',
                    size = 'large', color='black', weight='semibold')

        if len(year) == 1:
            plt.savefig(f'Figures/{outcome_types}_by_beat_{year[0]}.png')
        else:
            plt.savefig(f'Figures/{outcome_types}_by_beat_{year[0]}-{year[-1]}.png')
        return None
    
    ### Plot yield rate ratio
    x = yields.index # Getting freq measures
    y = yields['white'] / yields['black'] # Calculating Yield Rate Ratio
    sns.lineplot(x = x, y = y,
                 color = 'grey', alpha = 0.5,
                 label = f'{time}ly Ratio',
                 ax = ax)

    ### Calculate 5 year stats
    median = yields['white'].median() / yields['black'].median()
    mean = yields['white'].mean() / yields['black'].mean()
    
    ### Plot rolling mean
    data = y.rolling(f'{str(rolling_days)}D') # Create rolling object by number of days
    data = data.median() # Calling the mean from the rolling object
    sns.lineplot(data = data,
                 color = 'black',
                 label = f'{rolling_days} Day Rolling Median Ratio',
                 ax = ax)
    ax.lines[1].set_linewidth(3) # Setting linewidth

    ### Plotting benchmark lines
    plt.axhline(y = 1,
                linestyle = '--', c = 'cornflowerblue',
                label = '1:1 Ratio')
    plt.axhline(y = mean,
                linestyle = '--', c = 'firebrick', 
                label = f'4.5 Year Average Ratio ({round(mean, 2)})')

    ### Setting consistent plot boundaries
    plt.ylim(bottom = 0, top = 10)
    
    ### Axes labels
    plt.ylabel(f'White:Black Ratio')
    plt.xlabel(f'{time}')
    plt.title(f'Yield Rate Analysis')

    ### Explanatory boxes
    plt.legend(loc = 'upper right')
    props = dict(boxstyle = 'square', facecolor='white', edgecolor = 'gray', alpha=0.5)
    textstr = f'''{outcome_types} after \n{stop_types} Searches'''
    ax.text(0.015, 0.977, textstr,
            transform = ax.transAxes,
            verticalalignment = 'top', horizontalalignment = 'left',
            bbox = props, fontsize = 14)
    
    ### Creating second Y axes 
    ax2 = ax.twinx()
    
    ### Calculating trend lines
    series_sum = stops['white'] + stops['black']
    x = np.arange(len(series_sum))
    fit = np.polyfit(x, series_sum, 3)
    fit_fn = np.poly1d(fit)
    plt.plot(series_sum.index, fit_fn(x), ':',
             color = 'black', label = f'Stops/{time}')
    
    series_sum = searches['white'] + searches['black']
    x = np.arange(len(series_sum))
    fit = np.polyfit(x, series_sum, 3)
    fit_fn = np.poly1d(fit)
    plt.plot(series_sum.index, fit_fn(x), ':',
             color = 'black', label = f'Searches/{time}')
        
    series_sum = outcomes['white'] + outcomes['black']
    x = np.arange(len(series_sum))
    fit = np.polyfit(x, series_sum, 3)
    fit_fn = np.poly1d(fit)
    plt.plot(series_sum.index, fit_fn(x), '--', color = 'black',
             label = f'{outcome_types}/{time}')

    plt.ylabel(f'{time}ly Stops and Outcomes')
    plt.legend(loc = 'lower left')
    
    plt.savefig(f'Figures/Ratio of {outcome_types} After {stop_types} Search.png')
    
    
def plotly_yield_after_search_beat(stops_df,
                                   vehicle = True,
                                   pedestrian = True,
                                   arrests = True,
                                   citations = True,
                                   contraband = True,
                                   freq = 'W',
                                   beats = True,
                                   minority = ['black'],
                                   year = list(range(2015, 2020))):

    '''
    '''
    
    if vehicle & pedestrian:
        stop_types = 'Vehicle and Pedestrian'
    elif vehicle:
        stop_types = 'Vehicle'
    else:
        stop_types = 'Pedestrian'

    if arrests & citations & contraband:
        outcome_types = 'Arrests, Citations or Contraband'
    elif arrests & citations:
        outcome_types = 'Arrests or Citations'
    elif arrests & contraband:
        outcome_types = 'Arrests or Contraband'
    elif citations & contraband:
        outcome_types = 'Citaions or Contraband'
    elif arrests:
        outcome_types = 'Arrests'
    elif citations:
        outcome_types = 'Citations'
    else:
        outcome_types = 'Contraband'

    if freq == 'W':
        time = 'Week'
    elif freq == 'M':
        time = 'Month'
    else:
        time = 'Year'

    yields = get_yields(stops_df,
                        vehicle, pedestrian,
                        arrests, citations, contraband,
                        freq, beats, year)
    stops = get_stops(stops_df,
                          vehicle, pedestrian,
                          freq, beats, year)
    searches = get_searches(stops_df,
                            vehicle, pedestrian,
                            freq, beats, year)
    outcomes = get_outcomes(stops_df,
                            vehicle, pedestrian,
                            arrests, citations, contraband,
                            freq, beats, year)

    x = round(yields['white'], 2)
    y = round(yields[minority].mean(axis = 1), 2)
    size = (searches['white'] + searches[minority].sum(axis = 1))/10

    text = [f'''
    Beat {int(beat)} <br>
    Ratio: {str(round(ratio, 2))} <br>
    White Searches/Stops: {str(round(white_searches, 2))}/{str(round(white_stops, 2))} <br>
    Black Searches/Stops: {str(round(black_searches, 2))}/{str(round(black_stops, 2))} <br>
    '''
            for ratio, white_searches, black_searches, white_stops, black_stops, beat
            in zip(list(yields['white'] / yields[minority].mean(axis = 1)),
                   list(searches['white']),
                   list(searches[minority].sum(axis = 1)),
                   list(stops['white']),
                   list(stops['black']),
                   list(yields.index))]

    trace_1 = go.Scatter(x = x, y = y, text = list(range(1,17)),
                             mode = 'markers + text',
                             marker = dict(size = size,
                                           sizemode = 'area',
                                           sizeref = 2. * max(size) / (75 ** 2),
                                           color = 'rgb(93, 164, 214)'),
                             textposition = 'middle center',
                             textfont = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                             hoverinfo = 'text',
                             hovertext = text,
                             hoverlabel = {'bgcolor': 'white',
                                           'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                             showlegend = False)

    x_lim = ( round(min(x.dropna()) - .05, 1), round(max(x.dropna()) + .05, 1) )
    x_lim_frac = (x_lim[1] - x_lim[0])/5

    annotations = go.layout.Annotation(text = f'{outcome_types} after <br>{stop_types} Searches',
                                                          font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                                                          bgcolor = 'rgba(250, 250, 250, 0.8)',
                                                          align = 'left',
                                                          showarrow = False,
                                                          xref = 'paper',
                                                          yref = 'paper',
                                                          x = .01,
                                                          y = 0.99,
                                                          bordercolor = 'black',
                                                          borderwidth = 1)
    shapes = go.layout.Shape(type = 'line',
                  x0 = -0.1, y0 = -0.1,
                  x1 = 1.1, y1 = 1.1,
                  line = dict(color = 'black',
                              width = 1,
                              dash = 'dot'))

    if len(year) == 1:
        title = f'Yield Rates After Search {year[0]}'
    else:
        title = f'Yield Rates After Search {year[0]} - {year[-1]}'

    font = 'Nunito, Tahoma, Arial'

    title = go.layout.Title(text = title,
                            font = {'family': font,
                                    'size': 28},
                            y = 0.9,
                            x = 0.5,
                            xanchor = 'center',
                            yanchor = 'top')
    
    xtitle = go.layout.xaxis.Title(text = 'White Yield Rate',
                             font = {'family': font,
                                    'size': 24})
    
    ytitle = go.layout.yaxis.Title(text = 'Black Yield Rate',
                             font = {'family': font,
                                    'size': 24})

    xaxis = go.layout.XAxis(title = xtitle,
                            range = [-0.1, 1.1],
                            zeroline = False)

    yaxis = go.layout.YAxis(title = ytitle,
                            range = [-0.1, 1.1],
                            zeroline = False)

    layout = go.Layout(annotations = [annotations],
                       shapes = [shapes],
                       title = title,
                       xaxis = xaxis,
                       yaxis = yaxis,
                       autosize = True,
                       height = 600,
                       transition = {'duration': 500})

    fig = go.Figure(data = [trace_1], layout = layout)
    return fig

def plotly_yield_after_search(stops_df,
                              vehicle = True,
                              pedestrian = True,
                              arrests = True,
                              citations = True,
                              contraband = False,
                              rolling_days = 365,
                              freq = 'W',
                              beats = False,
                              minority = ['black'],
                              year = list(range(2015, 2021))):

    '''
    '''
    ### Setting label variables according to parameters    
    if vehicle & pedestrian:
        stop_types = 'Vehicle and Pedestrian'
    elif vehicle:
        stop_types = 'Vehicle'
    else:
        stop_types = 'Pedestrian'

    if arrests & citations & contraband:
        outcome_types = 'Arrests, Citations or Contraband'
    elif arrests & citations:
        outcome_types = 'Arrests or Citations'
    elif arrests & contraband:
        outcome_types = 'Arrests or Contraband'
    elif citations & contraband:
        outcome_types = 'Citaions or Contraband'
    elif arrests:
        outcome_types = 'Arrests'
    elif citations:
        outcome_types = 'Citations'
    else:
        outcome_types = 'Contraband'

    if freq == 'W':
        time = 'Week'
    elif freq == 'M':
        time = 'Month'
    else:
        time = 'Year'


    yields = get_yields(stops_df,
                        vehicle, pedestrian,
                        arrests, citations, contraband,
                        freq, beats, year)
    stops = get_stops(stops_df,
                          vehicle, pedestrian,
                          freq, beats, year)
    searches = get_searches(stops_df,
                            vehicle, pedestrian,
                            freq, beats, year)
    outcomes = get_outcomes(stops_df,
                            vehicle, pedestrian,
                            arrests, citations, contraband,
                            freq, beats, year)

    ### Plot yield rate ratio
    x = yields.index # Getting freq measures
    y = yields['white'] / yields[minority].mean(axis = 1) # Calculating Yield Rate Ratio

    trace_1 = go.Scatter(x = x, y = y, name = f'{time}ly Ratio',
                         mode = 'lines',
                         line = dict(color = 'grey', width = 2),
                         opacity = .6,
    #                      hovertext = text,
                         hoverlabel = {'bgcolor': 'white',
                                       'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                         showlegend = True,
                         yaxis = 'y1')

    ### Calculate 5 year stats
    median = yields['white'].median() / yields[minority].mean(axis = 1).median()
    mean = yields['white'].mean() / yields[minority].mean(axis = 1).mean()

    ### Plot rolling mean
    y = y.rolling(f'{str(rolling_days)}D') # Create rolling object by number of days
    y = y.median() # Calling the mean from the rolling object

    trace_2 = go.Scatter(x = x, y = y, name = f'{rolling_days} Day Rolling Median Ratio',
                         mode = 'lines',
                         line = dict(color = 'black', width = 4),
    #                      hovertext = text,
                         hoverlabel = {'bgcolor': 'white',
                                       'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                         showlegend = True,
                         yaxis = 'y1',
                         visible = 'legendonly')

    ### Plotting benchmark lines
    trace_3 = go.Scatter(x = x, y = [1] * len(x), name = '1:1 Ratio',
                         mode = 'lines',
                         line = dict(color = 'cornflowerblue', width = 4, dash = 'dash'),
    #                      hovertext = text,
                         hoverlabel = {'bgcolor': 'white',
                                       'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                         showlegend = True,
                         yaxis = 'y1')

    trace_4 = go.Scatter(x = x, y = [mean] * len(x), name = f'Average Ratio ({round(mean, 2)})',
                         mode = 'lines',
                         line = dict(color = 'firebrick', width = 4, dash = 'dash'),
    #                      hovertext = text,
                         hoverlabel = {'bgcolor': 'white',
                                       'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                         showlegend = True,
                         yaxis = 'y1')

    ### Calculating trend lines
    series_sum = stops['white'] + stops[minority].sum(axis = 1)
    x_line = np.arange(len(series_sum))
    fit = np.polyfit(x_line, series_sum, 3)
    fit_fn = np.poly1d(fit)
    trace_5 = go.Scatter(x = x, y = fit_fn(x_line), name = f'Stops/{time}',
                         mode = 'lines',
                         line = dict(color = 'black', width = 1, dash = 'dot'),
    #                      hovertext = text,
                         hoverlabel = {'bgcolor': 'white',
                                       'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                         showlegend = True,
                         yaxis = 'y2',
                         visible = 'legendonly')

    series_sum = searches['white'] + searches[minority].sum(axis = 1)
    x_line = np.arange(len(series_sum))
    fit = np.polyfit(x_line, series_sum, 3)
    fit_fn = np.poly1d(fit)
    trace_6 = go.Scatter(x = x, y = fit_fn(x_line), name = f'Searches/{time}',
                         mode = 'lines',
                         line = dict(color = 'black', width = 1, dash = 'dot'),
    #                      hovertext = text,
                         hoverlabel = {'bgcolor': 'white',
                                       'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                         showlegend = True,
                         yaxis = 'y2',
                         visible = 'legendonly')

    series_sum = outcomes['white'] + outcomes[minority].sum(axis = 1)
    x_line = np.arange(len(series_sum))
    fit = np.polyfit(x_line, series_sum, 3)
    fit_fn = np.poly1d(fit)
    trace_7 = go.Scatter(x = x, y = fit_fn(x_line), name = f'{outcome_types}/{time}',
                         mode = 'lines',
                         line = dict(color = 'black', width = 1, dash = 'dot'),
    #                      hovertext = text,
                         hoverlabel = {'bgcolor': 'white',
                                       'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                         showlegend = True,
                         yaxis = 'y2',
                         visible = 'legendonly')

    annotations = go.layout.Annotation(text = f'{outcome_types} after <br>{stop_types} Searches',
                                                          font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                                                          bgcolor = 'rgba(250, 250, 250, 0.8)',
                                                          align = 'left',
                                                          showarrow = False,
                                                          xref = 'paper',
                                                          yref = 'paper',
                                                          x = .01,
                                                          y = 0.99,
                                                          bordercolor = 'black',
                                                          borderwidth = 1)
    title = 'Yield Rate Analysis'

    font = 'Nunito, Tahoma, Arial'

    title = go.layout.Title(text = title,
                            font = {'family': font,
                                    'size': 28},
                            y = 0.9,
                            x = 0.5,
                            xanchor = 'center',
                            yanchor = 'top')

    xtitle = go.layout.xaxis.Title(text = f'{time}',
                             font = {'family': font,
                                     'size': 24})

    ytitle = go.layout.yaxis.Title(text = f'White:Black Ratio',
                             font = {'family': font,
                                     'size': 24})

    ytitle2 = go.layout.yaxis.Title(text = f'# Stops and Outcomes',
                              font = {'family': font,
                                      'size': 24})

    xaxis = go.layout.XAxis(title = xtitle,
    #                         range = [-0.1, 1.1],
                            zeroline = False)

    yaxis = go.layout.YAxis(title = ytitle,
                            #range = [0, 10],
                            zeroline = False)

    yaxis2 = go.layout.YAxis(title = ytitle2,
                             #range = [0, 150],
                             zeroline = False,
                             overlaying = 'y',
                             side = 'right')

    legend = go.layout.Legend(font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                              bgcolor = 'rgba(250, 250, 250, 0.8)',
                              x = .99,
                              xanchor = 'right',
                              y = 0.99,
                              bordercolor = 'black',
                              borderwidth = 1)

    layout = go.Layout(annotations = [annotations],
                       title = title,
                       xaxis = xaxis,
                       yaxis = yaxis,
                       yaxis2 = yaxis2,
                       legend = legend,
                       autosize = True,
                       height = 600,
                       transition = {'duration': 500})

    fig = go.Figure(data = [trace_1, trace_2, trace_3, trace_4, trace_5, trace_6, trace_7],
                    layout = layout)
    return fig

def plotly_veil_of_darkness(stops_df,
                            year = [2019]):
    df = stops_df[stops_df['datetime'].dt.year.isin(year)] # Restricting observations to years specified in parameters

    city = LocationInfo("Berkeley", "USA", pytz.timezone('America/Los_Angeles'), 37.87, -122.27)
    light = []
    dusk = []
    dawn = []
    for date_time in df['datetime']:
        if pd.isnull(date_time):
            light.append(np.nan)
            continue
        s = sun(city.observer, date = date_time.date(), tzinfo = pytz.timezone('America/Los_Angeles'))
        dusk.append((s['dusk']).time())
        dawn.append((s['dawn']).time())
        if s['dawn'] < date_time < s['dusk']:
            light.append(1)
        else:
            light.append(0)

    dusk_min = str(min(dusk))[:5]
    dusk_max = str(max(dusk))[:5]
    dawn_min = str(min(dawn))[:5]
    dawn_max = str(max(dawn))[:5]

    df = df.copy()
    df['light'] = light
    sizes = df.groupby('beat').sum()
    size = sizes['white'] + sizes['black'] + sizes['hispanic'] + sizes['asian'] + sizes['other']

    df = df.set_index('datetime').between_time(dusk_min, dusk_max).groupby(['beat', 'light']).sum()
    df['prct_white'] = df['white']/(df['white'] + df['black'] + df['hispanic'] + df['asian'] + df['other'])

    y = df.loc[pd.IndexSlice[:, [1]], :]['prct_white']
    x = df.loc[pd.IndexSlice[:, [0]], :]['prct_white']

    text = [f'''
    Beat {str(beat)} <br>
    White Stops: {str(round(white_stops, 2))} <br>
    Black Stops: {str(round(minority_stops, 2))} <br>
    '''
            for white_stops, minority_stops, beat
            in zip(list(df['white']),
                   list(df['black']),
                   list(sizes.index))]

    trace_1 = go.Scatter(x = x, y = y, text = size.index,
                             mode = 'markers + text',
                             marker = dict(size = size,
                                           sizemode = 'area',
                                           sizeref = 2. * max(size) / (75 ** 2),
                                           color = 'rgb(93, 164, 214)'),
                             textposition = 'middle center',
                             textfont = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                             hoverinfo = 'text',
                             hovertext = text,
                             hoverlabel = {'bgcolor': 'white',
                                           'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                             showlegend = False)

    annotations = go.layout.Annotation(text = f'Percent of stops that are of White people <br>when race is visible vs. when race is not visible',
                                                          font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                                                          bgcolor = 'rgba(250, 250, 250, 0.8)',
                                                          align = 'left',
                                                          showarrow = False,
                                                          xref = 'paper',
                                                          yref = 'paper',
                                                          x = .01,
                                                          y = 0.99,
                                                          bordercolor = 'black',
                                                          borderwidth = 1)
    shapes = go.layout.Shape(type = 'line',
                  x0 = -0.1, y0 = -0.1,
                  x1 = 1.1, y1 = 1.1,
                  line = dict(color = 'black',
                              width = 1,
                              dash = 'dot'))

    if len(year) == 1:
        title = f'Percent of Stops, White {year[0]}'
    else:
        title = f'Percent of Stops, White {year[0]}-{year[-1]}'

    font = 'Nunito, Tahoma, Arial'

    title = go.layout.Title(text = title,
                            font = {'family': font,
                                    'size': 28},
                            y = 0.9,
                            x = 0.5,
                            xanchor = 'center',
                            yanchor = 'top')

    xtitle = go.layout.xaxis.Title(text = 'Stops in the Dark',
                             font = {'family': font,
                                    'size': 24})

    ytitle = go.layout.yaxis.Title(text = 'Stops in the Light',
                             font = {'family': font,
                                    'size': 24})

    xaxis = go.layout.XAxis(title = xtitle,
                            range = [-0.1, 1.1],
                            zeroline = False)

    yaxis = go.layout.YAxis(title = ytitle,
                            range = [-0.1, 1.1],
                            zeroline = False)

    layout = go.Layout(annotations = [annotations],
                       shapes = [shapes],
                       title = title,
                       xaxis = xaxis,
                       yaxis = yaxis,
                       autosize = True,
                       height = 600,
                       transition = {'duration': 500})

    fig = go.Figure(data = [trace_1], layout = layout)
    return fig

    df = stops_df[stops_df['datetime'].dt.year.isin(year)] # Restricting observations to years specified in parameters

    city = LocationInfo("Berkeley", "USA", pytz.timezone('America/Los_Angeles'), 37.87, -122.27)
    light = []
    dusk = []
    dawn = []
    for date_time in df['datetime']:
        if pd.isnull(date_time):
            light.append(np.nan)
            continue
        s = sun(city.observer, date = date_time.date(), tzinfo = pytz.timezone('America/Los_Angeles'))
        dusk.append((s['dusk']).time())
        dawn.append((s['dawn']).time())
        if s['dawn'] < date_time < s['dusk']:
            light.append(1)
        else:
            light.append(0)

    dusk_min = str(min(dusk))[:5]
    dusk_max = str(max(dusk))[:5]
    dawn_min = str(min(dawn))[:5]
    dawn_max = str(max(dawn))[:5]

    df = df.copy()
    df['light'] = light

    df = df.set_index('datetime').between_time(dusk_min, dusk_max).reset_index().groupby([pd.Grouper(key = 'datetime', freq = freq), 'light']).sum()
    df['prct_white'] = df['white']/(df['white'] + df['black'] + df['hispanic'] + df['asian'] + df['other'])

    y = df.loc[pd.IndexSlice[:, [1]], :]['prct_white']
    x = df.loc[pd.IndexSlice[:, [0]], :]['prct_white']

    text = [f'''
    Beat {str(beat)} <br>
    White Stops: {str(round(white_stops, 2))} <br>
    Black Stops: {str(round(minority_stops, 2))} <br>
    '''
            for white_stops, minority_stops, beat
            in zip(list(df['white']),
                   list(df['black']),
                   list(sizes.index))]

    trace_1 = go.Scatter(x = x, y = y, text = size.index,
                             mode = 'markers + text',
                             marker = dict(size = size,
                                           sizemode = 'area',
                                           sizeref = 2. * max(size) / (75 ** 2),
                                           color = 'rgb(93, 164, 214)'),
                             textposition = 'middle center',
                             textfont = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                             hoverinfo = 'text',
                             hovertext = text,
                             hoverlabel = {'bgcolor': 'white',
                                           'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                             showlegend = False)

    annotations = go.layout.Annotation(text = f'Percent of stops that are of White people <br>when race is visible vs. when race is not visible',
                                                          font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                                                          bgcolor = 'rgba(250, 250, 250, 0.8)',
                                                          align = 'left',
                                                          showarrow = False,
                                                          xref = 'paper',
                                                          yref = 'paper',
                                                          x = .01,
                                                          y = 0.99,
                                                          bordercolor = 'black',
                                                          borderwidth = 1)
    shapes = go.layout.Shape(type = 'line',
                  x0 = -0.1, y0 = -0.1,
                  x1 = 1.1, y1 = 1.1,
                  line = dict(color = 'black',
                              width = 1,
                              dash = 'dot'))

    if len(year) == 1:
        title = f'Percent of Stops, White {year[0]}'
    else:
        title = f'Percent of Stops, White {year[0]}-{year[-1]}'

    font = 'Nunito, Tahoma, Arial'

    title = go.layout.Title(text = title,
                            font = {'family': font,
                                    'size': 28},
                            y = 0.9,
                            x = 0.5,
                            xanchor = 'center',
                            yanchor = 'top')

    xtitle = go.layout.xaxis.Title(text = 'Stops in the Dark',
                             font = {'family': font,
                                    'size': 24})

    ytitle = go.layout.yaxis.Title(text = 'Stops in the Light',
                             font = {'family': font,
                                    'size': 24})

    xaxis = go.layout.XAxis(title = xtitle,
                            range = [-0.1, 1.1],
                            zeroline = False)

    yaxis = go.layout.YAxis(title = ytitle,
                            range = [-0.1, 1.1],
                            zeroline = False)

    layout = go.Layout(annotations = [annotations],
                       shapes = [shapes],
                       title = title,
                       xaxis = xaxis,
                       yaxis = yaxis,
                       autosize = True,
                       height = 600,
                       transition = {'duration': 500})

    fig = go.Figure(data = [trace_1], layout = layout)
    return fig


def beat_demographics(census_df, beats, tracts):
    
    inter = gpd.overlay(census_df, beats, how = 'intersection')
    geometry = inter['geometry']
    geometry_area = inter['geometry'].area / 5280 ** 2
    beat_nums = inter['beat']
    temp = inter.iloc[:,0:22]
    
    temp = temp.multiply(geometry_area, axis = 0)
    
    temp['beat'] = beat_nums
    temp['geometry'] = geometry
    temp = temp.dissolve(by = 'beat', aggfunc = 'sum')
    
    geometry = temp['geometry']
    geometry_area = temp['geometry'].area
    temp = temp.iloc[:, 1:]
    
    inter = temp.divide(geometry_area, axis = 0)
    inter = inter * 5280 ** 2
    
    const = tracts['totalpop'].sum() / inter['totalpopulation'].sum()
    
    inter = inter * const
    inter = inter.reset_index()
    inter['geometry'] = geometry.values
    
    return inter

def plot_beat_dems(census_df, beats, tracts):
    beat_dems = beat_demographics(census_df, beats, tracts)

    beat_dems_geojson = beat_dems[['beat', 'geometry']].__geo_interface__

    mapbox = go.layout.Mapbox(style = 'stamen-toner',
                              center = go.layout.mapbox.Center(lat = 37.8715, lon = -122.2730),
                              zoom = 12)

    trace_1 = go.Choroplethmapbox(geojson = beat_dems_geojson, locations = beat_dems.index, z = beat_dems['totalpopulation'],
                                  marker_opacity = 0.5, marker_line_width = 4, text = beat_dems['beat'])

    title = 'Population by Beat'

    font = 'Nunito, Tahoma, Arial'

    title = go.layout.Title(text = title,
                            font = {'family': font,
                                    'size': 28},
                            y = 0.9,
                            x = 0.5,
                            xanchor = 'center',
                            yanchor = 'top')

    layout = go.Layout(mapbox = mapbox,
                       geo = dict(fitbounds = 'locations', visible = False),
                       title = title,
                       autosize = True)

    fig = go.Figure(data = [trace_1],
                    layout = layout)
    return fig

### OVERVIEW ###

def plot_stops(df, pedestrian, vehicle, searches, no_searches, arrests, citations, mh_hold, no_enforcement, population, freq, user_title=None):

    stops_df = df.copy()
    
    if vehicle & pedestrian:
        stop_types = 'Vehicle and pedestrian stops'
    elif vehicle:
        stop_types = 'Vehicle stops'
    elif pedestrian:
        stop_types = 'Pedestrian stops'
    else:
        stop_types = 'All stops'

    if mh_hold:
        if arrests & citations & no_enforcement & mh_hold:
            outcome_types = ' that result in an<br>arrest, citation, 5150 hold, or no enforcement'    
        elif arrests & citations & mh_hold:
            outcome_types = ' that result in an<br>arrest, citation, or 5150 hold'
        elif arrests & no_enforcement & mh_hold:
            outcome_types = ' that result in an<br>arrest, 5150 hold or no enforcement'
        elif citations & no_enforcement & mh_hold:
            outcome_types = ' that result in a<br>citation, 5150 hold or no enforcement'
        elif arrests & citations & no_enforcement:
            outcome_types = ' that result in an<br>arrest, citation, or no enforcement'    
        elif arrests & citations:
            outcome_types = ' that result in an<br>arrest or citation'
        elif arrests & no_enforcement:
            outcome_types = ' that result in an<br>arrest or no enforcement'
        elif citations & no_enforcement:
            outcome_types = ' that result in a<br>citation or no enforcement'
        elif arrests & mh_hold:
            outcome_types = ' that result in an<br>arrest or 5150 hold'
        elif citations & mh_hold:
            outcome_types = ' that result in a<br>citation or 5150 hold'
        elif no_enforcement & mh_hold:
            outcome_types = ' that result in<br>a 5150 hold or no enforcement'
        elif mh_hold:
            outcome_types = ' that result in<br>a 5150 hold'
        elif arrests:
            outcome_types = ' that result in an<br>arrest'
        elif citations:
            outcome_types = ' that result in a<br>citation'
        elif no_enforcement:
            outcome_types = ' that result in<br>no enforcement'
        else:
            outcome_types = ''
    else:
        if arrests & citations & no_enforcement:
            outcome_types = ' that result in an<br>arrest, citation, or no enforcement'    
        elif arrests & citations:
            outcome_types = ' that result in an<br>arrest or citation'
        elif arrests & no_enforcement:
            outcome_types = ' that result in an<br>arrest or no enforcement'
        elif citations & no_enforcement:
            outcome_types = ' that result in a<br>citation or no enforcement'
        elif arrests:
            outcome_types = ' that result in an<br>arrest'
        elif citations:
            outcome_types = ' that result in a<br>citation'
        elif no_enforcement:
            outcome_types = ' that result in<br>no enforcement'
        else:
            outcome_types = ''

    if freq == 'W':
        time = 'Week'
    elif freq == 'M':
        time = 'Month'
    else:
        time = 'Year'
        
    if searches == False:
        searched = ' and no search'
    elif searches == True:
        searched = ' and a search'
    elif (searches == True) & (no_searches == True):
        searched = ''
    else:
        pass
    
    berkeley_pop = 120463

    if population == 'berkeley':
        black_pct = .079
        white_pct = .588
        hispanic_pct = .11
        asian_pct = .21
        pop_title = 'Berkeley Demographics'
    elif population == 'ala_ccc':
        black_pct = ( ( (1150000 * .085) + (1670000 * .101) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        white_pct = ( ( (1150000 * .43) + (1670000 * .309) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        hispanic_pct = ( ( (1150000 * .258) + (1670000 * .224) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        asian_pct = ( ( (1150000 * .168) + (1670000 * .306) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        pop_title = 'Alameda and Contra Costa County Demographics'
    elif population == 'ala_ccc_sfo':
        black_pct = ( ( (1150000 * .085) + (1670000 * .101) + (883000 * .049) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        white_pct = ( ( (1150000 * .43) + (1670000 * .309) + (883000 * .4) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        hispanic_pct = ( ( (1150000 * .258) + (1670000 * .224) + (883000 * .152) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        asian_pct = ( ( (1150000 * .168) + (1670000 * .306) + (883000 * .341) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        pop_title = 'Alameda, Contra Costa and San Francisco County Demographics'
    elif population == 'met_stat_area':
        black_pct = .074
        white_pct = .39
        hispanic_pct = .219
        asian_pct = .263
        pop_title = 'Metropolitan Statiscal Area Demographics'
    elif population == 'oak_berk_rich':
        black_pct = ( ( (111701*.0352 * .2) + (444956*.0778 * .24) + (120463*.6835 * .083) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        white_pct = ( ( (111701*.0352 * .36) + (444956*.0778 * .35) + (120463*.6835 * .591) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        hispanic_pct = ( ( (111701*.0352 * .42) + (444956*.0778 * .27) + (120463*.6835 * .11) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        asian_pct = ( ( (111701*.0352 * .23) + (444956*.0778 * .17) + (120463*.6835 * .21) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        pop_title = 'Oakland, Berkeley and Richmond Demographics'
    elif population == 'other':
        black_pct = 0.519997
        white_pct = 0.274169
        hispanic_pct = 0.118260
        asian_pct = 0.028824
        pop_title = 'Victim-Described Suspect Demographics'
    elif population == 'None':
        black_pct = 1/berkeley_pop*1000
        white_pct = 1/berkeley_pop*1000
        hispanic_pct = 1/berkeley_pop*1000
        asian_pct = 1/berkeley_pop*1000
        pop_title = 'Count'
    else:
        pass

    # df = stops_df
    if (vehicle == False) & (pedestrian == False):
        stops_df = stops_df
    elif (vehicle == True) & (pedestrian == True):
        stops_df = stops_df[(stops_df['pedestrian'] == 1) | (stops_df['vehicle'] == 1)]
    elif (vehicle == True) & (pedestrian == False):
        stops_df = stops_df[(stops_df['vehicle'] == 1) & (stops_df['pedestrian'] == 0)]
    elif (vehicle == False) & (pedestrian == True):
        stops_df = stops_df[(stops_df['pedestrian'] == 1) & (stops_df['vehicle'] == 0)]
    else:
        pass

        
    if (searches == False) & (no_searches == True):
        stops_df = stops_df[stops_df['searches'] == 0]
    elif (searches == True) & no_searches == False:
        stops_df = stops_df[stops_df['searches'] == 1]
    elif (searches == True) & (no_searches == True):
        stops_df = stops_df
    else:
        pass

    if mh_hold:    
        if (arrests == False) & (citations == False) & (no_enforcement == True) & (mh_hold == False):
            stops_df = stops_df[( (stops_df['no_action'] == 1) ) & ( (stops_df['arrests'] == 0) & (stops_df['citations'] == 0) & (stops_df['5150'] == 0) ) ]
        elif (arrests == True) & (citations == False) & (no_enforcement == False) & (mh_hold == False):
            stops_df = stops_df[( (stops_df['arrests'] == 1) ) & ( (stops_df['no_action'] == 0) & (stops_df['citations'] == 0) & (stops_df['5150'] == 0) ) ]
        elif (arrests == False) & (citations == True) & (no_enforcement == False) & (mh_hold == False):
            stops_df = stops_df[( (stops_df['citations'] == 1) ) & ( (stops_df['no_action'] == 0) & (stops_df['arrests'] == 0) & (stops_df['5150'] == 0) ) ]
        elif (arrests == False) & (citations == False) & (no_enforcement == False) & (mh_hold == True):
            stops_df = stops_df[( (stops_df['5150'] == 1) ) & ( (stops_df['no_action'] == 0) & (stops_df['arrests'] == 0) & (stops_df['citations'] == 0) ) ]
        elif (arrests == True) & (citations == True) & (no_enforcement == False) & (mh_hold == False):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['citations'] == 1) ) & ( (stops_df['no_action'] == 0) & (stops_df['5150'] == 0) ) ]
        elif (arrests == True) & (citations == False) & (no_enforcement == True) & (mh_hold == False):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['citations'] == 0) & (stops_df['5150'] == 0) ) ]
        elif (arrests == False) & (citations == True) & (no_enforcement == True) & (mh_hold == False):
            stops_df = stops_df[( (stops_df['citations'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['arrests'] == 0) & (stops_df['5150'] == 0) ) ]
        elif (arrests == False) & (citations == False) & (no_enforcement == True) & (mh_hold == True):
            stops_df = stops_df[( (stops_df['5150'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['arrests'] == 0) & (stops_df['citations'] == 0) ) ]
        elif (arrests == True) & (citations == False) & (no_enforcement == False) & (mh_hold == True):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['5150'] == 1) ) & ( (stops_df['no_action'] == 0) & (stops_df['citations'] == 0) ) ]
        elif (arrests == False) & (citations == True) & (no_enforcement == False) & (mh_hold == True):
            stops_df = stops_df[( (stops_df['citations'] == 1) | (stops_df['5150'] == 1) ) & ( (stops_df['no_action'] == 0) & (stops_df['arrests'] == 0) ) ]
        elif (arrests == True) & (citations == True) & (no_enforcement == False) & (mh_hold == True):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['citations'] == 1) | (stops_df['5150'] == 1) ) & ( (stops_df['no_action'] == 0) ) ]
        elif (arrests == True) & (citations == False) & (no_enforcement == True) & (mh_hold == True):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['5150'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['citations'] == 0) ) ]
        elif (arrests == False) & (citations == True) & (no_enforcement == True) & (mh_hold == True):
            stops_df = stops_df[( (stops_df['citations'] == 1) | (stops_df['5150'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['arrests'] == 0) ) ]
        elif (arrests == True) & (citations == True) & (no_enforcement == True) & (mh_hold == False):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['citations'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['5150'] == 0) ) ]
        elif (arrests == True) & (citations == True) & (no_enforcement == True) & (mh_hold == True):
            stops_df = stops_df
        else:
            pass
    else:
        if (arrests == False) & (citations == False) & (no_enforcement == True):
            stops_df = stops_df[( (stops_df['no_action'] == 1) ) & ( (stops_df['arrests'] == 0) & (stops_df['citations'] == 0) ) ]
        elif (arrests == True) & (citations == False) & (no_enforcement == False):
            stops_df = stops_df[( (stops_df['arrests'] == 1) ) & ( (stops_df['no_action'] == 0) & (stops_df['citations'] == 0) ) ]
        elif (arrests == False) & (citations == True) & (no_enforcement == False):
            stops_df = stops_df[( (stops_df['citations'] == 1) ) & ( (stops_df['no_action'] == 0) & (stops_df['arrests'] == 0) ) ]
        elif (arrests == True) & (citations == True) & (no_enforcement == False):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['citations'] == 1) ) & ( (stops_df['no_action'] == 0) ) ]
        elif (arrests == True) & (citations == False) & (no_enforcement == True):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['citations'] == 0) ) ]
        elif (arrests == False) & (citations == True) & (no_enforcement == True):
            stops_df = stops_df[( (stops_df['citations'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['arrests'] == 0) ) ]
        elif (arrests == True) & (citations == True) & (no_enforcement == False):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['citations'] == 1) ) & ( (stops_df['no_action'] == 0) ) ]
        elif (arrests == True) & (citations == False) & (no_enforcement == True):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['citations'] == 0) ) ]
        elif (arrests == False) & (citations == True) & (no_enforcement == True):
            stops_df = stops_df[( (stops_df['citations'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['arrests'] == 0) ) ]
        elif (arrests == True) & (citations == True) & (no_enforcement == True):
            stops_df = stops_df
        else:
            pass
    # if searches == '1':
    #     breakdown = stops_df.groupby(pd.Grouper(key='datetime', freq = freq)).sum()[['asian', 'black', 'hispanic', 'other', 'white']]
    #     baseline = df.groupby(pd.Grouper(key='datetime', freq = freq)).sum()[['asian', 'black', 'hispanic', 'other', 'white']]
    #     breakdown['asian'] = breakdown['asian'] / baseline['asian']
    #     breakdown['black'] = breakdown['black'] / baseline['black']
    #     breakdown['hispanic'] = breakdown['hispanic'] / baseline['hispanic']
    #     breakdown['white'] = breakdown['white'] / baseline['white']
    # else:    
    breakdown = stops_df.groupby(pd.Grouper(key='datetime', freq = freq)).sum()[['asian', 'black', 'hispanic', 'other', 'white']]
    breakdown['asian'] = breakdown['asian'] / ( (berkeley_pop * asian_pct) / 1000)
    breakdown['black'] = breakdown['black'] / ( (berkeley_pop * black_pct) / 1000)
    breakdown['hispanic'] = breakdown['hispanic'] / ( (berkeley_pop * hispanic_pct) / 1000)
    breakdown['white'] = breakdown['white'] / ( (berkeley_pop * white_pct) / 1000)
    
    y = breakdown['black']
    x = breakdown.index
    trace_1 = go.Scatter(x = x, y = y,
                         mode = 'lines',
                         name = 'Black',
                         hoverlabel = {'bgcolor': 'white',
                                        'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}})
    
    y = breakdown['white']
    x = breakdown.index
    trace_2 = go.Scatter(x = x, y = y,
                         mode = 'lines',
                         name = 'White',
                         hoverlabel = {'bgcolor': 'white',
                                        'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}})
    
    y = breakdown['hispanic']
    x = breakdown.index
    trace_3 = go.Scatter(x = x, y = y,
                         mode = 'lines',
                         name = 'Hispanic',
                         hoverlabel = {'bgcolor': 'white',
                                        'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                         visible = 'legendonly')
    
    y = breakdown['asian']
    x = breakdown.index
    trace_4 = go.Scatter(x = x, y = y,
                         mode = 'lines',
                         name = 'Asian',
                         hoverlabel = {'bgcolor': 'white',
                                        'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                         visible = 'legendonly')
   
    annotations = go.layout.Annotation(text = f'{stop_types}{outcome_types}{searched}',
                                        font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                                        bgcolor = 'rgba(250, 250, 250, 0.8)',
                                        align = 'left',
                                        showarrow = False,
                                        xref = 'paper',
                                        yref = 'paper',
                                        x = .01,
                                        y = 0.99,
                                        bordercolor = 'black',
                                        borderwidth = 1)
    if arrests and not citations:
        title = f'{stop_types} by Race'
    elif citations and not arrests:
        title = f'{stop_types} by Race'
    elif not arrests and not citations and not vehicle and not pedestrian and searches == 1:
        title = f'{stop_types} by Race'
    else:
        title = f'Stops by Race ({pop_title})'

    if user_title:
        title = go.layout.Title(text = user_title,
                            font = {'family': font,
                                    'size': 28},
                            y = 0.9,
                            x = 0.5,
                            xanchor = 'center',
                            yanchor = 'top')

    font = 'Nunito, Tahoma, Arial'

    title = go.layout.Title(text = title,
                            font = {'family': font,
                                    'size': 28},
                            y = 0.9,
                            x = 0.5,
                            xanchor = 'center',
                            yanchor = 'top')

    xtitle = go.layout.xaxis.Title(text = f'{time}',
                             font = {'family': font,
                                     'size': 24})

    if population == 'None':
        ytitle = go.layout.yaxis.Title(text = f'# of Stops',
                            font = {'family': font,
                                    'size': 24})
    else:
        ytitle = go.layout.yaxis.Title(text = f'Stops per 1000',
                                font = {'family': font,
                                        'size': 24})

    xaxis = go.layout.XAxis(title = xtitle,
    #                         range = [-0.1, 1.1],
                            zeroline = False)

    yaxis = go.layout.YAxis(title = ytitle,
#                             range = [0, 10],
                            zeroline = False)

    legend = go.layout.Legend(font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                              bgcolor = 'rgba(250, 250, 250, 0.8)',
                              x = .99,
                              xanchor = 'right',
                              y = 0.99,
                              bordercolor = 'black',
                              borderwidth = 1)

    layout = go.Layout(annotations = [annotations],
                       title = title,
                    #    xaxis = xaxis,
                       margin_b = 0,
                       yaxis = yaxis,
                       legend = legend,
                       autosize = True,
                    #    height = 600,
                       transition = {'duration': 500})
    
    fig = go.Figure(data = [trace_1, trace_2, trace_3, trace_4],
                    layout = layout)
    return fig

def get_initial_outcome_values(stops_df, vehicle, pedestrian, population, freq, minority):
    
    if (vehicle == False) & (pedestrian == False):
        stops_df = stops_df
    elif (vehicle == True) & (pedestrian == True):
        stops_df = stops_df[(stops_df['pedestrian'] == 1) | (stops_df['vehicle'] == 1)]
    elif (vehicle == True) & (pedestrian == False):
        stops_df = stops_df[stops_df['vehicle'] == 1]
    elif (vehicle == False) & (pedestrian == True):
        stops_df = stops_df[stops_df['pedestrian'] == 1]
    
    
    berkeley_pop = 120463

    if population == 'berkeley':
        black_pct = .079
        white_pct = .588
        hispanic_pct = .11
        asian_pct = .21
        pop_title = 'Berkeley'
    elif population == 'ala_cc':
        black_pct = ( ( (1150000 * .085) + (1670000 * .101) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        white_pct = ( ( (1150000 * .43) + (1670000 * .309) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        hispanic_pct = ( ( (1150000 * .258) + (1670000 * .224) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        asian_pct = ( ( (1150000 * .168) + (1670000 * .306) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        pop_title = 'Alameda and Contra Costa County Demographics'
    elif population == 'ala_ccc_sfo':
        black_pct = ( ( (1150000 * .085) + (1670000 * .101) + (883000 * .049) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        white_pct = ( ( (1150000 * .43) + (1670000 * .309) + (883000 * .4) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        hispanic_pct = ( ( (1150000 * .258) + (1670000 * .224) + (883000 * .152) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        asian_pct = ( ( (1150000 * .168) + (1670000 * .306) + (883000 * .341) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        pop_title = 'Alameda, Contra Costa and San Francisco County Demographics'
    elif population == 'met_stat_area':
        black_pct = .074
        white_pct = .39
        hispanic_pct = .219
        asian_pct = .263
        pop_title = 'Metropolitan Statiscal Area Demographics'
    elif population == 'oak_berk_rich':
        black_pct = ( ( (111701*.0352 * .2) + (444956*.0778 * .24) + (120463*.6835 * .083) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        white_pct = ( ( (111701*.0352 * .36) + (444956*.0778 * .35) + (120463*.6835 * .591) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        hispanic_pct = ( ( (111701*.0352 * .42) + (444956*.0778 * .27) + (120463*.6835 * .11) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        asian_pct = ( ( (111701*.0352 * .23) + (444956*.0778 * .17) + (120463*.6835 * .21) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        pop_title = 'Oakland, Berkeley and Richmond Demographics'
    elif population == 'other':
        black_pct = 0.519997
        white_pct = 0.274169
        hispanic_pct = 0.118260
        asian_pct = 0.028824
        pop_title = 'Victim-Described Suspect Demographics'

        
    breakdown = stops_df.groupby(pd.Grouper(key='datetime', freq = freq)).sum()[['asian', 'black', 'hispanic', 'white']]
    breakdown['asian'] = breakdown['asian'] / ( (berkeley_pop * asian_pct) / 1000)
    breakdown['black'] = breakdown['black'] / ( (berkeley_pop * black_pct) / 1000)
    breakdown['hispanic'] = breakdown['hispanic'] / ( (berkeley_pop * hispanic_pct) / 1000)
    breakdown['white'] = breakdown['white'] / ( (berkeley_pop * white_pct) / 1000)    
        
    searches = stops_df[(stops_df['searches'] == 1)].groupby(pd.Grouper(key='datetime', freq = freq)).sum()[['asian', 'black', 'hispanic', 'white']]
    searches['asian'] = searches['asian'] / ( (berkeley_pop * asian_pct) / 1000)
    searches['black'] = searches['black'] / ( (berkeley_pop * black_pct) / 1000)
    searches['hispanic'] = searches['hispanic'] / ( (berkeley_pop * hispanic_pct) / 1000)
    searches['white'] = searches['white'] / ( (berkeley_pop * white_pct) / 1000)  

    
    search_arrest = stops_df[(stops_df['searches'] == 1) & (stops_df['arrests'] == 1)].groupby(pd.Grouper(key='datetime', freq = freq)).sum()[['asian', 'black', 'hispanic', 'white']]
    search_arrest['asian'] = search_arrest['asian'] / ( (berkeley_pop * asian_pct) / 1000)
    search_arrest['black'] = search_arrest['black'] / ( (berkeley_pop * black_pct) / 1000)
    search_arrest['hispanic'] = search_arrest['hispanic'] / ( (berkeley_pop * hispanic_pct) / 1000)
    search_arrest['white'] = search_arrest['white'] / ( (berkeley_pop * white_pct) / 1000)

    search_citation = stops_df[(stops_df['searches'] == 1) & (stops_df['citations'] == 1)].groupby(pd.Grouper(key='datetime', freq = freq)).sum()[['asian', 'black', 'hispanic', 'white']]
    search_citation['asian'] = search_citation['asian'] / ( (berkeley_pop * asian_pct) / 1000)
    search_citation['black'] = search_citation['black'] / ( (berkeley_pop * black_pct) / 1000)
    search_citation['hispanic'] = search_citation['hispanic'] / ( (berkeley_pop * hispanic_pct) / 1000)
    search_citation['white'] = search_citation['white'] / ( (berkeley_pop * white_pct) / 1000)

    stops_min_1000 = breakdown[minority].mean(axis = 1)
    stops_wht_1000 = breakdown['white']

    search_pct_min = searches[minority].mean(axis = 1) / breakdown[minority].mean(axis = 1)
    search_pct_wht = searches['white'] / breakdown['white']
    
    search_arrest_pct_min = search_arrest[minority].mean(axis = 1) / searches[minority].mean(axis = 1)
    search_arrest_pct_wht = search_arrest['white'] / searches['white']

    search_citation_pct_min = search_citation[minority].mean(axis = 1) / searches[minority].mean(axis = 1)
    search_citation_pct_wht = search_citation['white'] / searches['white']
    
    return (stops_min_1000, stops_wht_1000,
            search_pct_min, search_pct_wht,
            search_arrest_pct_min, search_arrest_pct_wht,
            search_citation_pct_min, search_citation_pct_wht)

def plot_ratios(#vehicle, pedestrian, 
                population, freq, years,
                stops_blk_1000, stops_wht_1000,
                search_pct_blk, search_pct_wht,
                search_arrest_pct_blk, search_arrest_pct_wht,
                search_citation_pct_blk, search_citation_pct_wht): 
    
#     if vehicle & pedestrian:
#         stop_types = 'Vehicle and Pedestrian'
#     elif vehicle:
#         stop_types = 'Vehicle'
#     else:
#         stop_types = 'Pedestrian'


    if freq == 'W':
        time = 'Week'
    elif freq == 'M':
        time = 'Month'
    else:
        time = 'Year'
    
    berkeley_pop = 120463

    if population == 'berkeley':
        black_pct = .083
        white_pct = .591
        pop_title = 'Berkeley'
    elif population == 'ala_cc':
        black_pct = ( ( (1150000 * .095) + (1666000 * .112) )/ 2 ) / ( (1150000 + 1666000) / 2 )
        white_pct = ( ( (1150000 * .655) + (1666000 * .497) )/ 2 ) / ( (1150000 + 1666000) / 2 )
        pop_title = 'Alameda and Contra Costa County Demographics'
    elif population == 'ala_ccc_sfo':
        black_pct = ( ( (1150000 * .095) + (1666000 * .112) + (883000 * .056) ) / 3 ) / ( (1150000 + 1666000 + 883000) / 3 )
        white_pct = ( ( (1150000 * .655) + (1666000 * .497) + (883000 * .529) ) / 3 ) / ( (1150000 + 1666000 + 883000) / 3 )
        pop_title = 'Alameda, Contra Costa and San Francisco County Demographics'
    elif population == 'met_stat_area':
        black_pct = .079
        white_pct = .578
        pop_title = 'Metropolitan Statiscal Area Demographics'
    elif population == 'nine_counties':
        black_pct = .06
        white_pct = .41
        pop_title = 'Nine County Bay Area Demographics'
    elif population == 'oak_berk_rich':
        black_pct = ( ( (111701*.0352 * .2) + (444956*.0778 * .24) + (120463*.6835 * .083) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        white_pct = ( ( (111701*.0352 * .36) + (444956*.0778 * .35) + (120463*.6835 * .591) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        pop_title = 'Oakland, Berkeley and Richmond Demographics'
    elif population == 'other':
        met_stat_area_black_pct = ( ( (110000 * .20) + (444956*.0778 * .236) + (121643 * .083) ) / 3 ) / ( (110000 + 444956*.0778 + 121643) / 3 )
        met_stat_area_white_pct = ( ( (110000 * .372) + (444956*.0778 * .361) + (121643 * .591) ) / 3 ) / ( (110000 + 444956*.0778 + 883000) / 3 )
        black_pct = ( ( (4 * .20) + (16 * .236) + (53 * .083) + (27 * met_stat_area_black_pct) ) / 4 ) / ( (4 + 16 + 53 + 27) / 4 )
        white_pct = ( ( (4 * .372) + (16 * .361) + (53 * .591) + (27 * met_stat_area_white_pct) ) / 4 ) / ( (4 + 16 + 53 + 27) / 4 )
        pop_title = 'Victim-Described Suspect Demographics'

#     df = stops_df
#     if (vehicle == True) & (pedestrian == True):
#         stops_df = stops_df
#     elif (vehicle == True) & (pedestrian == True):
#         stops_df = stops_df[(stops_df['pedestrian'] == 1) | (stops_df['vehicle'] == 1)]
#     elif (vehicle == True) & (pedestrian == False):
#         stops_df = stops_df[stops_df['vehicle'] == 1]
#     elif (vehicle == False) & (pedestrian == True):
#         stops_df = stops_df[stops_df['pedestrian'] == 1]
    
    # stops_blk_1000 = INPUT
    # stops_wht_1000 = INPUT
    stops_ratio = stops_blk_1000 / stops_wht_1000

    # search_pct_blk = INPUT
    # search_pct_wht = INPUT

    search_blk_1000 = stops_blk_1000 * search_pct_blk
    search_wht_1000 = stops_wht_1000 * search_pct_blk
    search_ratio = search_blk_1000 / search_wht_1000

    no_search_pct_blk = 1 - search_pct_blk
    no_search_pct_wht = 1 - search_pct_wht

    no_search_blk_1000 = stops_blk_1000 * no_search_pct_blk
    no_search_wht_1000 = stops_wht_1000 * no_search_pct_blk
    no_search_ratio = no_search_blk_1000 / no_search_wht_1000

    # search_arrest_pct_blk = INPUT
    # search_arrest_pct_wht = INPUT

    search_arrest_blk_1000 = search_blk_1000 * search_arrest_pct_blk
    search_arrest_wht_1000 = search_wht_1000 * search_arrest_pct_wht
    search_arrest_ratio = search_arrest_blk_1000 / search_arrest_wht_1000

    no_search_arrest_pct_blk = 1 - search_arrest_pct_blk
    no_search_arrest_pct_wht = 1 - search_arrest_pct_wht

    no_search_arrest_blk_1000 = no_search_blk_1000 * no_search_arrest_pct_blk
    no_search_arrest_wht_1000 = no_search_wht_1000 * no_search_arrest_pct_wht
    no_search_arrest_ratio = no_search_arrest_blk_1000 / no_search_arrest_wht_1000

    # search_citation_pct_blk = INPUT
    # search_citation_pct_wht = INPUT

    search_citation_blk_1000 = search_blk_1000 * search_citation_pct_blk
    search_citation_wht_1000 = search_wht_1000 * search_citation_pct_wht
    search_citation_ratio = search_citation_blk_1000 / search_citation_wht_1000

    no_search_citation_pct_blk = 1 - search_citation_pct_blk
    no_search_citation_pct_wht = 1 - search_citation_pct_wht

    no_search_citation_blk_1000 = no_search_blk_1000 * no_search_citation_pct_blk
    no_search_citation_wht_1000 = no_search_wht_1000 * no_search_citation_pct_wht
    no_search_citation_ratio = no_search_citation_blk_1000 / no_search_citation_wht_1000

    arrests_blk = search_arrest_blk_1000 + no_search_arrest_blk_1000
    arrests_wht = search_arrest_wht_1000 + no_search_arrest_wht_1000
    arrests_ratio = arrests_blk / arrests_wht

    citations_blk = search_citation_blk_1000 + no_search_citation_blk_1000
    citations_wht = search_citation_wht_1000 + no_search_citation_wht_1000
    citations_ratio = citations_blk / citations_wht

    x = stops_ratio.index
    y = stops_ratio

    trace_1 = go.Scatter(x = x, y = y, name = f'Stops Ratio',
                         mode = 'lines',
    #                      line = dict(color = 'grey', width = 2),
                         opacity = .6,
                         hoverlabel = {'bgcolor': 'white',
                                       'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                         showlegend = True)

    x = search_ratio.index
    y = search_ratio

    trace_2 = go.Scatter(x = x, y = y, name = f'Searches Ratio',
                         mode = 'lines',
    #                      line = dict(color = 'grey', width = 2),
                         opacity = .6,
                         hoverlabel = {'bgcolor': 'white',
                                       'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                         showlegend = True,
                         visible = 'legendonly')

    x = arrests_ratio.index
    y = arrests_ratio

    trace_3 = go.Scatter(x = x, y = y, name = f'Arrests Ratio',
                         mode = 'lines',
    #                      line = dict(color = 'grey', width = 2),
                         opacity = .6,
                         hoverlabel = {'bgcolor': 'white',
                                       'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                         showlegend = True,
                         visible = 'legendonly')

    x = citations_ratio.index
    y = citations_ratio

    trace_4 = go.Scatter(x = x, y = y, name = f'Citations Ratio',
                         mode = 'lines',
    #                      line = dict(color = 'grey', width = 2),
                         opacity = .6,
                         hoverlabel = {'bgcolor': 'white',
                                       'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                         showlegend = True,
                         visible = 'legendonly')

    trace_5 = go.Scatter(x = x, y = [1] * len(x), name = f'1:1 Ratio',
                         mode = 'lines',
    #                      line = dict(color = 'grey', width = 2),
                         opacity = .6,
                         hoverlabel = {'bgcolor': 'white',
                                       'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                         showlegend = True)

    # annotations = go.layout.Annotation(text = f'{outcome_types} after <br>{stop_types} Searches',
    #                                                           font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
    #                                                           bgcolor = 'rgba(250, 250, 250, 0.8)',
    #                                                           align = 'left',
    #                                                           showarrow = False,
    #                                                           xref = 'paper',
    #                                                           yref = 'paper',
    #                                                           x = .01,
    #                                                           y = 0.99,
    #                                                           bordercolor = 'black',
    #                                                           borderwidth = 1)
    title = 'Enforcement Ratios'

    font = 'Nunito, Tahoma, Arial'

    title = go.layout.Title(text = title,
                            font = {'family': font,
                                    'size': 28},
                            y = 0.9,
                            x = 0.5,
                            xanchor = 'center',
                            yanchor = 'top')

    xtitle = go.layout.xaxis.Title(text = f'{time}',
                             font = {'family': font,
                                     'size': 24})

    ytitle = go.layout.yaxis.Title(text = f'Black:White Ratio',
                             font = {'family': font,
                                     'size': 24})

    xaxis = go.layout.XAxis(title = xtitle,
    #                         range = [-0.1, 1.1],
                            zeroline = False)

    yaxis = go.layout.YAxis(title = ytitle,
                            range = [0, 12],
                            zeroline = False)

    legend = go.layout.Legend(font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                              bgcolor = 'rgba(250, 250, 250, 0.8)',
                              x = .99,
                              xanchor = 'right',
                              y = 0.99,
                              bordercolor = 'black',
                              borderwidth = 1)

    layout = go.Layout(#annotations = [annotations],
                       title = title,
                       xaxis = xaxis,
                       yaxis = yaxis,
                       legend = legend,
                       autosize = True,
                       height = 600,
                       transition = {'duration': 500})

    fig = go.Figure(data = [trace_1, trace_2, trace_3, trace_4, trace_5],
                    layout = layout)

    return fig

def make_traces(stops_df, pedestrian, vehicle, searches, no_searches, arrests, citations, no_enforcement, population, freq):
    
    if vehicle & pedestrian:
        stop_types = 'Vehicle and pedestrian stops'
    elif vehicle:
        stop_types = 'Vehicle stops'
    elif pedestrian:
        stop_types = 'Pedestrian stops'
    else:
        stop_types = 'All stops'

    if arrests & citations & no_enforcement:
        outcome_types = ' that result in an<br>arrest, citation, or no enforcement'    
    elif arrests & citations:
        outcome_types = ' that result in an<br>arrest or citation'
    elif arrests & no_enforcement:
        outcome_types = ' that result in an<br>arrest or no enforcement'
    elif citations & no_enforcement:
        outcome_types = ' that result in a<br>citation or no enforcement'
    elif arrests:
        outcome_types = ' that result in an<br>arrest'
    elif citations:
        outcome_types = ' that result in a<br>citation'
    elif no_enforcement:
        outcome_types = ' that result in<br> no enforcement'
    else:
        outcome_types = ''

    if freq == 'W':
        time = 'Week'
    elif freq == 'M':
        time = 'Month'
    else:
        time = 'Year'
        
    if searches == False:
        searched = ' and no search'
    if searches == True:
        searched = ' and a search'
    if (searches == True) & (no_searches == True):
        searched = ''
    
    berkeley_pop = 120463

    if population == 'berkeley':
        black_pct = .079
        white_pct = .588
        hispanic_pct = .11
        asian_pct = .21
        pop_title = 'Berkeley'
    elif population == 'ala_ccc':
        black_pct = ( ( (1150000 * .085) + (1670000 * .101) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        white_pct = ( ( (1150000 * .43) + (1670000 * .309) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        hispanic_pct = ( ( (1150000 * .258) + (1670000 * .224) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        asian_pct = ( ( (1150000 * .168) + (1670000 * .306) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        pop_title = 'Alameda and Contra Costa County Demographics'
    elif population == 'ala_ccc_sfo':
        black_pct = ( ( (1150000 * .085) + (1670000 * .101) + (883000 * .049) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        white_pct = ( ( (1150000 * .43) + (1670000 * .309) + (883000 * .4) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        hispanic_pct = ( ( (1150000 * .258) + (1670000 * .224) + (883000 * .152) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        asian_pct = ( ( (1150000 * .168) + (1670000 * .306) + (883000 * .341) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        pop_title = 'Alameda, Contra Costa and San Francisco County Demographics'
    elif population == 'met_stat_area':
        black_pct = .074
        white_pct = .39
        hispanic_pct = .219
        asian_pct = .263
        pop_title = 'Metropolitan Statiscal Area Demographics'
    elif population == 'oak_berk_rich':
        black_pct = ( ( (111701*.0352 * .2) + (444956*.0778 * .24) + (120463*.6835 * .083) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        white_pct = ( ( (111701*.0352 * .36) + (444956*.0778 * .35) + (120463*.6835 * .591) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        hispanic_pct = ( ( (111701*.0352 * .42) + (444956*.0778 * .27) + (120463*.6835 * .11) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        asian_pct = ( ( (111701*.0352 * .23) + (444956*.0778 * .17) + (120463*.6835 * .21) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        pop_title = 'Oakland, Berkeley and Richmond Demographics'
    elif population == 'other':
        black_pct = 0.519997
        white_pct = 0.274169
        hispanic_pct = 0.118260
        asian_pct = 0.028824
        pop_title = 'Victim-Described Suspect Demographics'
    elif population == 'None':
        black_pct = 1/berkeley_pop*1000
        white_pct = 1/berkeley_pop*1000
        hispanic_pct = 1/berkeley_pop*1000
        asian_pct = 1/berkeley_pop*1000
        pop_title = 'Count'

    df = stops_df
    if (vehicle == False) & (pedestrian == False):
        stops_df = stops_df
    elif (vehicle == True) & (pedestrian == True):
        stops_df = stops_df[(stops_df['pedestrian'] == 1) | (stops_df['vehicle'] == 1)]
    elif (vehicle == True) & (pedestrian == False):
        stops_df = stops_df[stops_df['vehicle'] == 1]
    elif (vehicle == False) & (pedestrian == True):
        stops_df = stops_df[stops_df['pedestrian'] == 1]
        
    if searches == False:
        stops_df = stops_df[stops_df['searches'] == 0]
    if searches == True:
        stops_df = stops_df[stops_df['searches'] == 1]
    if (searches == True) & (no_searches == True):
        stops_df = stops_df
        
    if (arrests == False) & (citations == False) & (no_enforcement == True):
        stops_df = stops_df[(stops_df['arrests'] == 0) & (stops_df['citations'] == 0)]
    elif (arrests == True) & (citations == False) & (no_enforcement == False):
        stops_df = stops_df[stops_df['arrests'] == 1]
    elif (arrests == False) & (citations == True) & (no_enforcement == False):
        stops_df = stops_df[stops_df['citations'] == 1]
    elif (arrests == True) & (citations == True) & (no_enforcement == False):
        stops_df = stops_df[(stops_df['arrests'] == 1) | (stops_df['citations'] == 1)]
    elif (arrests == True) & (citations == False) & (no_enforcement == True):
        stops_df = stops_df[(stops_df['arrests'] == 1) | ((stops_df['arrests'] == 0) & (stops_df['citations'] == 0))]
    elif (arrests == False) & (citations == True) & (no_enforcement == True):
        stops_df = stops_df[(stops_df['citations'] == 1) | ((stops_df['arrests'] == 0) & (stops_df['citations'] == 0))]
    elif (arrests == True) & (citations == True) & (no_enforcement == True):
        stops_df = stops_df
    
    if searches == 1:
        breakdown = stops_df.groupby(pd.Grouper(key='datetime', freq = freq)).sum()[['asian', 'black', 'hispanic', 'other', 'white']]
        baseline = df.groupby(pd.Grouper(key='datetime', freq = freq)).sum()[['asian', 'black', 'hispanic', 'other', 'white']]
        breakdown['asian'] = breakdown['asian'] / baseline['asian']
        breakdown['black'] = breakdown['black'] / baseline['black']
        breakdown['hispanic'] = breakdown['hispanic'] / baseline['hispanic']
        breakdown['white'] = breakdown['white'] / baseline['white']
    else:    
        breakdown = stops_df.groupby(pd.Grouper(key='datetime', freq = freq)).sum()[['asian', 'black', 'hispanic', 'other', 'white']]
        breakdown['asian'] = breakdown['asian'] / ( (berkeley_pop * asian_pct) / 1000)
        breakdown['black'] = breakdown['black'] / ( (berkeley_pop * black_pct) / 1000)
        breakdown['hispanic'] = breakdown['hispanic'] / ( (berkeley_pop * hispanic_pct) / 1000)
        breakdown['white'] = breakdown['white'] / ( (berkeley_pop * white_pct) / 1000)
    
    y = breakdown['black']
    x = breakdown.index

    trace_1 = go.Scatter(x = x, y = y,
                         mode = 'lines',
                         name = 'Black',
                         visible = False)
    
    y = breakdown['white']
    x = breakdown.index
    trace_2 = go.Scatter(x = x, y = y,
                         mode = 'lines',
                         name = 'White',
                         visible = False)
    
    annotations = go.layout.Annotation(text = f'{stop_types}{outcome_types}{searched}',
                                      font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                                      bgcolor = 'rgba(250, 250, 250, 0.8)',
                                      align = 'left',
                                      showarrow = False,
                                      xref = 'paper',
                                      yref = 'paper',
                                      x = .01,
                                      y = 0.99,
                                      bordercolor = 'black',
                                      borderwidth = 1)

    font = 'Nunito, Tahoma, Arial'

    title = go.layout.Title(text = title,
                            font = {'family': font,
                                    'size': 28},
                            y = 0.9,
                            x = 0.5,
                            xanchor = 'center',
                            yanchor = 'top')

    xtitle = go.layout.xaxis.Title(text = f'{time}',
                             font = {'family': font,
                                     'size': 24})

    ytitle = go.layout.yaxis.Title(text = f'Stops per 1000 Residents',
                             font = {'family': font,
                                     'size': 24})

    xaxis = go.layout.XAxis(title = xtitle,
                            zeroline = False)

    yaxis = go.layout.YAxis(title = ytitle,
                            zeroline = False)

    legend = go.layout.Legend(font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                              bgcolor = 'rgba(250, 250, 250, 0.8)',
                              x = .99,
                              xanchor = 'right',
                              y = 0.99,
                              bordercolor = 'black',
                              borderwidth = 1)
    
    return trace_1, trace_2, annotations, title, xaxis, yaxis, legend
    
def get_outcomes_breakdown(stops_df, pedestrian, vehicle, searches, no_searches, arrests, citations, no_enforcement, mh_hold, population, freq):
    
    if vehicle & pedestrian:
        stop_types = 'Vehicle and pedestrian stops'
    elif vehicle:
        stop_types = 'Vehicle stops'
    elif pedestrian:
        stop_types = 'Pedestrian stops'
    else:
        stop_types = 'All stops'

    if mh_hold:
        if arrests & citations & no_enforcement & mh_hold:
            outcome_types = ' that result in an<br>arrest, citation, 5150 hold, or no enforcement'    
        elif arrests & citations & mh_hold:
            outcome_types = ' that result in an<br>arrest, citation, or 5150 hold'
        elif arrests & no_enforcement & mh_hold:
            outcome_types = ' that result in an<br>arrest, 5150 hold or no enforcement'
        elif citations & no_enforcement & mh_hold:
            outcome_types = ' that result in a<br>citation, 5150 hold or no enforcement'
        elif arrests & citations & no_enforcement:
            outcome_types = ' that result in an<br>arrest, citation, or no enforcement'    
        elif arrests & citations:
            outcome_types = ' that result in an<br>arrest or citation'
        elif arrests & no_enforcement:
            outcome_types = ' that result in an<br>arrest or no enforcement'
        elif citations & no_enforcement:
            outcome_types = ' that result in a<br>citation or no enforcement'
        elif arrests & mh_hold:
            outcome_types = ' that result in an<br>arrest or 5150 hold'
        elif citations & mh_hold:
            outcome_types = ' that result in a<br>citation or 5150 hold'
        elif no_enforcement & mh_hold:
            outcome_types = ' that result in<br>a 5150 hold or no enforcement'
        elif mh_hold:
            outcome_types = ' that result in<br>a 5150 hold'
        elif arrests:
            outcome_types = ' that result in an<br>arrest'
        elif citations:
            outcome_types = ' that result in a<br>citation'
        elif no_enforcement:
            outcome_types = ' that result in<br>no enforcement'
        else:
            outcome_types = ''
    else:
        if arrests & citations & no_enforcement:
            outcome_types = ' that result in an<br>arrest, citation, or no enforcement'    
        elif arrests & citations:
            outcome_types = ' that result in an<br>arrest or citation'
        elif arrests & no_enforcement:
            outcome_types = ' that result in an<br>arrest or no enforcement'
        elif citations & no_enforcement:
            outcome_types = ' that result in a<br>citation or no enforcement'
        elif arrests:
            outcome_types = ' that result in an<br>arrest'
        elif citations:
            outcome_types = ' that result in a<br>citation'
        elif no_enforcement:
            outcome_types = ' that result in<br>no enforcement'
        else:
            outcome_types = ''
    if freq == 'W':
        time = 'Week'
    elif freq == 'M':
        time = 'Month'
    else:
        time = 'Year'
        
    if (searches == False) & (no_searches == True):
        searched = ' and no search'
    elif (searches == True) & (no_searches == False):
        searched = ' and a search'
    elif (searches == True) & (no_searches == True):
        searched = ''
    
    berkeley_pop = 120463

    if population == 'berkeley':
        black_pct = .079
        white_pct = .588
        hispanic_pct = .11
        asian_pct = .21
        pop_title = 'Berkeley Demographics'
    elif population == 'ala_ccc':
        black_pct = ( ( (1150000 * .085) + (1670000 * .101) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        white_pct = ( ( (1150000 * .43) + (1670000 * .309) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        hispanic_pct = ( ( (1150000 * .258) + (1670000 * .224) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        asian_pct = ( ( (1150000 * .168) + (1670000 * .306) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        pop_title = 'Alameda and Contra Costa County Demographics'
    elif population == 'ala_ccc_sfo':
        black_pct = ( ( (1150000 * .085) + (1670000 * .101) + (883000 * .049) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        white_pct = ( ( (1150000 * .43) + (1670000 * .309) + (883000 * .4) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        hispanic_pct = ( ( (1150000 * .258) + (1670000 * .224) + (883000 * .152) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        asian_pct = ( ( (1150000 * .168) + (1670000 * .306) + (883000 * .341) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        pop_title = 'Alameda, Contra Costa and San Francisco County Demographics'
    elif population == 'met_stat_area':
        black_pct = .074
        white_pct = .39
        hispanic_pct = .219
        asian_pct = .263
        pop_title = 'Metropolitan Statiscal Area Demographics'
    elif population == 'oak_berk_rich':
        black_pct = ( ( (111701*.0352 * .2) + (444956*.0778 * .24) + (120463*.6835 * .083) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        white_pct = ( ( (111701*.0352 * .36) + (444956*.0778 * .35) + (120463*.6835 * .591) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        hispanic_pct = ( ( (111701*.0352 * .42) + (444956*.0778 * .27) + (120463*.6835 * .11) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        asian_pct = ( ( (111701*.0352 * .23) + (444956*.0778 * .17) + (120463*.6835 * .21) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        pop_title = 'Oakland, Berkeley and Richmond Demographics'
    elif population == 'other':
        black_pct = 0.519997
        white_pct = 0.274169
        hispanic_pct = 0.118260
        asian_pct = 0.028824
        pop_title = 'Victim-Described Suspect Demographics'
    elif population == 'None':
        black_pct = 1/berkeley_pop*1000
        white_pct = 1/berkeley_pop*1000
        hispanic_pct = 1/berkeley_pop*1000
        asian_pct = 1/berkeley_pop*1000
        pop_title = 'Count'

    df = stops_df
    if (vehicle == False) & (pedestrian == False):
        stops_df = stops_df
    elif (vehicle == True) & (pedestrian == True):
        stops_df = stops_df[(stops_df['pedestrian'] == 1) | (stops_df['vehicle'] == 1)]
    elif (vehicle == True) & (pedestrian == False):
        stops_df = stops_df[stops_df['vehicle'] == 1]
    elif (vehicle == False) & (pedestrian == True):
        stops_df = stops_df[stops_df['pedestrian'] == 1]
        
    if (searches == False) & (no_searches == True):
        stops_df = stops_df[stops_df['searches'] == 0]
    elif (searches == True) & (no_searches == False):
        stops_df = stops_df[stops_df['searches'] == 1]
    elif (searches == True) & (no_searches == True):
        stops_df = stops_df
        
    if mh_hold:    
        if (arrests == False) & (citations == False) & (no_enforcement == True) & (mh_hold == False):
            stops_df = stops_df[( (stops_df['no_action'] == 1) ) & ( (stops_df['arrests'] == 0) & (stops_df['citations'] == 0) & (stops_df['5150'] == 0) ) ]
        elif (arrests == True) & (citations == False) & (no_enforcement == False) & (mh_hold == False):
            stops_df = stops_df[( (stops_df['arrests'] == 1) ) & ( (stops_df['no_action'] == 0) & (stops_df['citations'] == 0) & (stops_df['5150'] == 0) ) ]
        elif (arrests == False) & (citations == True) & (no_enforcement == False) & (mh_hold == False):
            stops_df = stops_df[( (stops_df['citations'] == 1) ) & ( (stops_df['no_action'] == 0) & (stops_df['arrests'] == 0) & (stops_df['5150'] == 0) ) ]
        elif (arrests == False) & (citations == False) & (no_enforcement == False) & (mh_hold == True):
            stops_df = stops_df[( (stops_df['5150'] == 1) ) & ( (stops_df['no_action'] == 0) & (stops_df['arrests'] == 0) & (stops_df['citations'] == 0) ) ]
        elif (arrests == True) & (citations == True) & (no_enforcement == False) & (mh_hold == False):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['citations'] == 1) ) & ( (stops_df['no_action'] == 0) & (stops_df['5150'] == 0) ) ]
        elif (arrests == True) & (citations == False) & (no_enforcement == True) & (mh_hold == False):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['citations'] == 0) & (stops_df['5150'] == 0) ) ]
        elif (arrests == False) & (citations == True) & (no_enforcement == True) & (mh_hold == False):
            stops_df = stops_df[( (stops_df['citations'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['arrests'] == 0) & (stops_df['5150'] == 0) ) ]
        elif (arrests == False) & (citations == False) & (no_enforcement == True) & (mh_hold == True):
            stops_df = stops_df[( (stops_df['5150'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['arrests'] == 0) & (stops_df['citations'] == 0) ) ]
        elif (arrests == True) & (citations == False) & (no_enforcement == False) & (mh_hold == True):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['5150'] == 1) ) & ( (stops_df['no_action'] == 0) & (stops_df['citations'] == 0) ) ]
        elif (arrests == False) & (citations == True) & (no_enforcement == False) & (mh_hold == True):
            stops_df = stops_df[( (stops_df['citations'] == 1) | (stops_df['5150'] == 1) ) & ( (stops_df['no_action'] == 0) & (stops_df['arrests'] == 0) ) ]
        elif (arrests == True) & (citations == True) & (no_enforcement == False) & (mh_hold == True):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['citations'] == 1) | (stops_df['5150'] == 1) ) & ( (stops_df['no_action'] == 0) ) ]
        elif (arrests == True) & (citations == False) & (no_enforcement == True) & (mh_hold == True):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['5150'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['citations'] == 0) ) ]
        elif (arrests == False) & (citations == True) & (no_enforcement == True) & (mh_hold == True):
            stops_df = stops_df[( (stops_df['citations'] == 1) | (stops_df['5150'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['arrests'] == 0) ) ]
        elif (arrests == True) & (citations == True) & (no_enforcement == True) & (mh_hold == False):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['citations'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['5150'] == 0) ) ]
        elif (arrests == True) & (citations == True) & (no_enforcement == True) & (mh_hold == True):
            stops_df = stops_df
        else:
            pass
    else:
        if (arrests == False) & (citations == False) & (no_enforcement == True):
            stops_df = stops_df[( (stops_df['no_action'] == 1) ) & ( (stops_df['arrests'] == 0) & (stops_df['citations'] == 0) ) ]
        elif (arrests == True) & (citations == False) & (no_enforcement == False):
            stops_df = stops_df[( (stops_df['arrests'] == 1) ) & ( (stops_df['no_action'] == 0) & (stops_df['citations'] == 0) ) ]
        elif (arrests == False) & (citations == True) & (no_enforcement == False):
            stops_df = stops_df[( (stops_df['citations'] == 1) ) & ( (stops_df['no_action'] == 0) & (stops_df['arrests'] == 0) ) ]
        elif (arrests == True) & (citations == True) & (no_enforcement == False):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['citations'] == 1) ) & ( (stops_df['no_action'] == 0) ) ]
        elif (arrests == True) & (citations == False) & (no_enforcement == True):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['citations'] == 0) ) ]
        elif (arrests == False) & (citations == True) & (no_enforcement == True):
            stops_df = stops_df[( (stops_df['citations'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['arrests'] == 0) ) ]
        elif (arrests == True) & (citations == True) & (no_enforcement == False):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['citations'] == 1) ) & ( (stops_df['no_action'] == 0) ) ]
        elif (arrests == True) & (citations == False) & (no_enforcement == True):
            stops_df = stops_df[( (stops_df['arrests'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['citations'] == 0) ) ]
        elif (arrests == False) & (citations == True) & (no_enforcement == True):
            stops_df = stops_df[( (stops_df['citations'] == 1) | (stops_df['no_action'] == 1) ) & ( (stops_df['arrests'] == 0) ) ]
        elif (arrests == True) & (citations == True) & (no_enforcement == True):
            stops_df = stops_df
        else:
            pass

    breakdown = stops_df.groupby(pd.Grouper(key='datetime', freq = freq)).sum()[['asian', 'black', 'hispanic', 'other', 'white']]
    breakdown['asian'] = breakdown['asian'] / ( (berkeley_pop * asian_pct) / 1000)
    breakdown['black'] = breakdown['black'] / ( (berkeley_pop * black_pct) / 1000)
    breakdown['hispanic'] = breakdown['hispanic'] / ( (berkeley_pop * hispanic_pct) / 1000)
    breakdown['white'] = breakdown['white'] / ( (berkeley_pop * white_pct) / 1000)
        
    return breakdown
    
def make_ratio_traces(stops_df, vehicle, pedestrian, population, freq, minority, mh_holds = True):

    if freq == 'W':
        time = 'Week'
    elif freq == 'M':
        time = 'Month'
    else:
        time = 'Year'
    
    berkeley_pop = 120463

    if mh_holds:
        stops = get_outcomes_breakdown(stops_df, pedestrian = pedestrian, vehicle = vehicle,
                                    searches = True, no_searches = True,
                                    arrests = True, citations = True, no_enforcement = True, mh_hold = True,
                                    population = population, freq = freq)
        stops_ratio = stops[minority].mean(axis = 1) / stops['white']
        x = stops_ratio.index
        y = stops_ratio
        trace_1 = go.Scatter(x = x, y = y, name = f'Stops Ratio',
                            mode = 'lines',
                            opacity = .6,
                            hoverlabel = {'bgcolor': 'white',
                                        'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                            showlegend = True,
                            visible = True)

        searches = get_outcomes_breakdown(stops_df, pedestrian = pedestrian, vehicle = vehicle,
                                        searches = True, no_searches = False,
                                        arrests = True, citations = True, no_enforcement = True, mh_hold = True,
                                        population = population, freq = freq)
        search_ratio = (searches[minority].mean(axis = 1) / stops[minority].mean(axis = 1)) / (searches['white'] / stops['white'])
        x = search_ratio.index
        y = search_ratio
        trace_2 = go.Scatter(x = x, y = y, name = f'Searches Ratio',
                            mode = 'lines',
                            opacity = .6,
                            hoverlabel = {'bgcolor': 'white',
                                        'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                            showlegend = True,
                            visible = 'legendonly')

        arrests = get_outcomes_breakdown(stops_df, pedestrian = pedestrian, vehicle = vehicle,
                                        searches = True, no_searches = True,
                                        arrests = True, citations = False, no_enforcement = False, mh_hold = False,
                                        population = population, freq = freq)
        arrests_ratio = (arrests[minority].mean(axis = 1) / stops[minority].mean(axis = 1)) / (arrests['white'] / stops['white'])
        x = arrests_ratio.index
        y = arrests_ratio
        trace_3 = go.Scatter(x = x, y = y, name = f'Arrests Ratio',
                            mode = 'lines',
                            opacity = .6,
                            hoverlabel = {'bgcolor': 'white',
                                        'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                            showlegend = True,
                            visible = 'legendonly')
        
        citations = get_outcomes_breakdown(stops_df, pedestrian = pedestrian, vehicle = vehicle,
                                        searches = True, no_searches = True,
                                        arrests = False, citations = True, no_enforcement = False, mh_hold = False,
                                        population = population, freq = freq)
        citations_ratio = (citations[minority].mean(axis = 1) / stops[minority].mean(axis = 1)) / (citations['white'] / stops['white'])
        x = citations_ratio.index
        y = citations_ratio
        trace_4 = go.Scatter(x = x, y = y, name = f'Citations Ratio',
                            mode = 'lines',
                            opacity = .6,
                            hoverlabel = {'bgcolor': 'white',
                                        'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                            showlegend = True,
                            visible = 'legendonly')

        no_enforcements = get_outcomes_breakdown(stops_df, pedestrian = pedestrian, vehicle = vehicle,
                                        searches = True, no_searches = True,
                                        arrests = False, citations = False, no_enforcement = True, mh_hold = False,
                                        population = population, freq = freq)
        no_enforcements_ratio = (no_enforcements[minority].mean(axis = 1) / stops[minority].mean(axis = 1)) / (no_enforcements['white'] / stops['white'])
        x = no_enforcements_ratio.index
        y = no_enforcements_ratio
        trace_5 = go.Scatter(x = x, y = y, name = f'No Enforcement Ratio',
                            mode = 'lines',
                            opacity = .6,
                            hoverlabel = {'bgcolor': 'white',
                                        'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                            showlegend = True,
                            visible = 'legendonly')

        mh_holds = get_outcomes_breakdown(stops_df, pedestrian = pedestrian, vehicle = vehicle,
                                        searches = True, no_searches = True,
                                        arrests = False, citations = False, no_enforcement = False, mh_hold = True,
                                        population = population, freq = freq)
        mh_holds_ratio = (mh_holds[minority].mean(axis = 1) / stops[minority].mean(axis = 1)) / (mh_holds['white'] / stops['white'])
        x = mh_holds_ratio.index
        y = mh_holds_ratio
        trace_6 = go.Scatter(x = x, y = y, name = f'5150 Hold Ratio',
                            mode = 'lines',
                            opacity = .6,
                            hoverlabel = {'bgcolor': 'white',
                                        'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                            showlegend = True,
                            visible = 'legendonly')
    else:
        stops = get_outcomes_breakdown(stops_df, pedestrian = pedestrian, vehicle = vehicle,
                                    searches = True, no_searches = True,
                                    arrests = True, citations = True, no_enforcement = True, mh_hold = None,
                                    population = population, freq = freq)
        stops_ratio = stops[minority].mean(axis = 1) / stops['white']
        x = stops_ratio.index
        y = stops_ratio
        trace_1 = go.Scatter(x = x, y = y, name = f'Stops Ratio',
                            mode = 'lines',
                            opacity = .6,
                            hoverlabel = {'bgcolor': 'white',
                                        'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                            showlegend = True,
                            visible = True)

        searches = get_outcomes_breakdown(stops_df, pedestrian = pedestrian, vehicle = vehicle,
                                        searches = True, no_searches = False,
                                        arrests = True, citations = True, no_enforcement = True, mh_hold = None,
                                        population = population, freq = freq)
        search_ratio = (searches[minority].mean(axis = 1) / stops[minority].mean(axis = 1)) / (searches['white'] / stops['white'])
        x = search_ratio.index
        y = search_ratio
        trace_2 = go.Scatter(x = x, y = y, name = f'Searches Ratio',
                            mode = 'lines',
                            opacity = .6,
                            hoverlabel = {'bgcolor': 'white',
                                        'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                            showlegend = True,
                            visible = 'legendonly')

        arrests = get_outcomes_breakdown(stops_df, pedestrian = pedestrian, vehicle = vehicle,
                                        searches = True, no_searches = True,
                                        arrests = True, citations = False, no_enforcement = False, mh_hold = None,
                                        population = population, freq = freq)
        arrests_ratio = (arrests[minority].mean(axis = 1) / stops[minority].mean(axis = 1)) / (arrests['white'] / stops['white'])
        x = arrests_ratio.index
        y = arrests_ratio
        trace_3 = go.Scatter(x = x, y = y, name = f'Arrests Ratio',
                            mode = 'lines',
                            opacity = .6,
                            hoverlabel = {'bgcolor': 'white',
                                        'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                            showlegend = True,
                            visible = 'legendonly')
        
        citations = get_outcomes_breakdown(stops_df, pedestrian = pedestrian, vehicle = vehicle,
                                        searches = True, no_searches = True,
                                        arrests = False, citations = True, no_enforcement = False, mh_hold = None,
                                        population = population, freq = freq)
        citations_ratio = (citations[minority].mean(axis = 1) / stops[minority].mean(axis = 1)) / (citations['white'] / stops['white'])
        x = citations_ratio.index
        y = citations_ratio
        trace_4 = go.Scatter(x = x, y = y, name = f'Citations Ratio',
                            mode = 'lines',
                            opacity = .6,
                            hoverlabel = {'bgcolor': 'white',
                                        'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                            showlegend = True,
                            visible = 'legendonly')

        no_enforcements = get_outcomes_breakdown(stops_df, pedestrian = pedestrian, vehicle = vehicle,
                                        searches = True, no_searches = True,
                                        arrests = False, citations = False, no_enforcement = True, mh_hold = None,
                                        population = population, freq = freq)
        no_enforcements_ratio = (no_enforcements[minority].mean(axis = 1) / stops[minority].mean(axis = 1)) / (no_enforcements['white'] / stops['white'])
        x = no_enforcements_ratio.index
        y = no_enforcements_ratio
        trace_5 = go.Scatter(x = x, y = y, name = f'No Enforcement Ratio',
                            mode = 'lines',
                            opacity = .6,
                            hoverlabel = {'bgcolor': 'white',
                                        'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                            showlegend = True,
                            visible = 'legendonly')

        trace_6 = []


    annotations = go.layout.Annotation(text = f'Black:White Ratios<br>Enforcement Actions',
                                                              font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                                                              bgcolor = 'rgba(250, 250, 250, 0.8)',
                                                              align = 'left',
                                                              showarrow = False,
                                                              xref = 'paper',
                                                              yref = 'paper',
                                                              x = .01,
                                                              y = 0.99,
                                                              bordercolor = 'black',
                                                              borderwidth = 1)
    
    title = f'Enforcement Ratios'

    font = 'Nunito, Tahoma, Arial'

    title = go.layout.Title(text = title,
                            font = {'family': font,
                                    'size': 28},
                            y = 0.9,
                            x = 0.5,
                            xanchor = 'center',
                            yanchor = 'top')

    xtitle = go.layout.xaxis.Title(text = f'{time}',
                             font = {'family': font,
                                     'size': 24})

    ytitle = go.layout.yaxis.Title(text = f'Black:White',
                             font = {'family': font,
                                     'size': 24})

    xaxis = go.layout.XAxis(title = xtitle,
                            zeroline = False)

    yaxis = go.layout.YAxis(title = ytitle,
                            zeroline = False)

    legend = go.layout.Legend(font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                              bgcolor = 'rgba(250, 250, 250, 0.8)',
                              x = .99,
                              xanchor = 'right',
                              y = 0.99,
                              bordercolor = 'black',
                              borderwidth = 1)

    return trace_1, trace_2, trace_3, trace_4, trace_5, trace_6, annotations, title, xaxis, yaxis, legend

def plot_problem(stops_df, pedestrian, vehicle, searches, no_searches, arrests, citations, no_enforcement, population, freq, user_title=None):
    
    stops_trace_1, stops_trace_2, stops_annotations, stops_title, stops_xaxis, stops_yaxis, stops_legend = make_traces(stops_df,
                                                                             pedestrian = pedestrian,
                                                                             vehicle = vehicle,
                                                                             searches = searches,
                                                                             no_searches = no_searches,
                                                                             arrests = arrests,
                                                                             citations = citations,
                                                                             no_enforcement = no_enforcement,
                                                                             population = population,
                                                                             freq = freq)
    searches_trace_1, searches_trace_2, searches_annotations, searches_title, searches_xaxis, searches_yaxis, searches_legend = make_traces(stops_df,
                                                                             pedestrian = pedestrian,
                                                                             vehicle = vehicle,
                                                                             searches = searches,
                                                                             no_searches = False,
                                                                             arrests = arrests,
                                                                             citations = citations,
                                                                             no_enforcement = no_enforcement,
                                                                             population = population,
                                                                             freq = freq)
    citations_trace_1, citations_trace_2, citations_annotations, citations_title, citations_xaxis, citations_yaxis, citations_legend = make_traces(stops_df,
                                                                             pedestrian = pedestrian,
                                                                             vehicle = vehicle,
                                                                             searches = searches,
                                                                             no_searches = no_searches,
                                                                             arrests = False,
                                                                             citations = True,
                                                                             no_enforcement = False,
                                                                             population = population,
                                                                             freq = freq)
    arrests_trace_1, arrests_trace_2, arrests_annotations, arrests_title, arrests_xaxis, arrests_yaxis, arrests_legend = make_traces(stops_df,
                                                                             pedestrian = pedestrian,
                                                                             vehicle = vehicle,
                                                                             searches = searches,
                                                                             no_searches = no_searches,
                                                                             arrests = True,
                                                                             citations = False,
                                                                             no_enforcement = False,
                                                                             population = population,
                                                                             freq = freq)
    ratios_trace_1, ratios_trace_2, ratios_trace_3, ratios_trace_4, ratios_annotations, ratios_title, ratios_xaxis, ratios_yaxis, ratios_legend = make_ratio_traces(stops_df = stops_df, vehicle = False, pedestrian = False, population = population, freq = freq, minority = ['black'])
    
    fig = go.Figure(data = [stops_trace_1, stops_trace_2,
                            searches_trace_1, searches_trace_2,
                            citations_trace_1, citations_trace_2,
                            arrests_trace_1, arrests_trace_2,
                            ratios_trace_1, ratios_trace_2, ratios_trace_3, ratios_trace_4])
    
    layout = go.Layout(autosize = True,
#                        height = 600,
                       transition = {'duration': 500})
    fig.update_layout(
        updatemenus=[
            dict(
                active = 0,
                buttons = list([
                    dict(label = 'Stops',
                         method = 'update',
                         args=[{'visible': [True, True,
                                            False, False,
                                            False, False,
                                            False, False,
                                            False, False, False, False, False
                                           ]},
                               {'title': stops_title,
                                'annotations': [stops_annotations],
                                'xaxis': stops_xaxis,
                                'yaxis': stops_yaxis,
                                'legend': stops_legend}]),
                    dict(label = 'Searches',
                         method = 'update',
                         args=[{'visible': [False, False,
                                            True, True,
                                            False, False,
                                            False, False,
                                            False, False, False, False, False
                                           ]},
                               {'title': searches_title,
                                'annotations': [searches_annotations],
                                'xaxis': searches_xaxis,
                                'yaxis': searches_yaxis,
                                'legend': searches_legend}]),
                    dict(label = 'Citations',
                         method = 'update',
                         args=[{'visible': [False, False,
                                            False, False,
                                            True, True,
                                            False, False,
                                            False, False, False, False, False
                                           ]},
                               {'title': citations_title,
                                'annotations': [citations_annotations],
                                'xaxis': citations_xaxis,
                                'yaxis': citations_yaxis,
                                'legend': citations_legend}]),
                    dict(label = 'Arrests',
                         method = 'update',
                         args=[{'visible': [False, False,
                                            False, False,
                                            False, False,
                                            True, True,
                                            False, False, False, False, False
                                           ]},
                               {'title': arrests_title,
                                'annotations': [arrests_annotations],
                                'xaxis': arrests_xaxis,
                                'yaxis': arrests_yaxis,
                                'legend': arrests_legend}]),
                    dict(label = 'Ratios',
                         method = 'update',
                         args=[{'visible': [False, False,
                                            False, False,
                                            False, False,
                                            False, False,
                                            True, True, True, True, True
                                           ]},
                               {'title': ratios_title,
                                'annotations': [ratios_annotations],
                                'xaxis': ratios_xaxis,
                                'yaxis': ratios_yaxis,
                                'legend': ratios_legend}])                ]),
                type = 'buttons',
                direction = 'right',
                showactive = True,
                x = 0.5,
                y = -0.5,
                xanchor = 'center',
                yanchor = 'bottom',
                font = {'family':'Nunito, Tahoma, Arial'}
                )
        ])
    if user_title:
        font = 'Nunito, Tahoma, Arial'
        stops_title = go.layout.Title(text = user_title,
                            font = {'family': font,
                                    'size': 28},
                            y = 0.9,
                            x = 0.5,
                            xanchor = 'center',
                            yanchor = 'top')
    fig.update_layout({'title': stops_title,
                        'annotations': [stops_annotations],
                        'xaxis': stops_xaxis,
                        'yaxis': stops_yaxis,
                        'legend': stops_legend})

    return fig

def plot_ratio_time(stops_df, vehicle, pedestrian, population, freq, ripa, minority, user_title = None):
    
    if ripa == True:
        (ratios_trace_1, ratios_trace_2,
        ratios_trace_3, ratios_trace_4,
        ratios_trace_5, ratios_trace_6,
        ratios_annotations, ratios_title,
        ratios_xaxis, ratios_yaxis,
        ratios_legend) = make_ratio_traces(stops_df, vehicle, pedestrian, population, freq, minority)
    if ripa == False:
        (ratios_trace_1, ratios_trace_2,
        ratios_trace_3, ratios_trace_4,
        ratios_trace_5, ratios_trace_6,
        ratios_annotations, ratios_title,
        ratios_xaxis, ratios_yaxis,
        ratios_legend) = make_ratio_traces(stops_df, vehicle, pedestrian, population, freq, minority, mh_holds = None)


    layout = go.Layout(#title = ratios_title,
                       annotations = [ratios_annotations],
                       xaxis = ratios_xaxis,
                       margin_t = 0,
                       yaxis = ratios_yaxis,
                       legend = ratios_legend,
                       autosize = True,
                       transition = {'duration': 500})
    
    if ripa == True:
        fig = go.Figure(data = [ratios_trace_1, ratios_trace_2, ratios_trace_3,
                                ratios_trace_4, ratios_trace_5, ratios_trace_6],
                        layout = layout)
    if ripa == False:
        fig = go.Figure(data = [ratios_trace_1, ratios_trace_2, ratios_trace_3,
                                ratios_trace_4],
                        layout = layout)

    return fig

def scatter_beat_dems(beat_dems,  calls_grouped, df_size, weighted = True, part_one = True):
    
    if calls_grouped is not None:
        if weighted:
            y = calls_grouped['weighted_count']
            y.index = y.index.astype(int)
            y = y.sort_index()
        else:
            y = calls_grouped['count']
            y.index = y.index.astype(int)
            y = y.sort_index()
    else:
        y = beat_dems['totalpopulation']

    x = beat_dems['black'] / (beat_dems['totalpopulation'])


    rg = sns.regplot(x = x, y = y)
    plt.close()
    
    X = rg.get_lines()[0].get_xdata()
    Y = rg.get_lines()[0].get_ydata()
    P = rg.get_children()[1].get_paths()
    
    p_codes={1:'M', 2: 'L', 79: 'Z'}#dict to get the Plotly codes for commands to define the svg path
    path = ''
    for s in P[0].iter_segments():
        c = p_codes[s[1]]
        xx, yy = s[0]
        path += c + str('{:.5f}'.format(xx))+' '+str('{:.5f}'.format(yy))

    shapes = [dict(type = 'path',
                   path = path,
                   line = dict(width=0.1,color='rgba(68, 122, 219, 0.25)' ),
                   fillcolor = 'rgba(68, 122, 219, 0.25)')]   
    
    if 'weighted_count' in df_size.columns:
        s = df_size.groupby('beat').sum()['part_one']
        if not part_one:
            s = df_size.groupby('beat').sum()['count']
    else:
        s = df_size.groupby('beat').size()
    s.index = s.index.astype(int)
    s = s.sort_index()
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    if part_one == True:
        hovertext = 'Part 1 cases'
    elif 'weighted_count' in df_size.columns:
        hovertext = 'cases'
    else:
        hovertext = 'stops'
        
    trace_1 = go.Scatter(x = x, y = y, text = s.index, name = 'Beat',
                             mode = 'markers + text',
                             marker = dict(size = s,
                                           sizemode = 'area',
                                           sizeref = 2. * max(s) / (75 ** 2),
                                           color = 'rgb(93, 164, 214)'),
                             textposition = 'middle center',
                             textfont = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                             hoverinfo = 'text',
                             hovertext = [f'{s.iloc[i]} {hovertext}' for i in range(len(s))],
                             hoverlabel = {'bgcolor': 'white',
                                           'font': {'family': 'Nunito, Tahoma, Arial', 'size': 12}},
                             showlegend = False)
    
    trace_2 = go.Scatter(x = X, y = Y,
                         mode = 'lines',
                         showlegend = False)

    annotations = go.layout.Annotation(text = f'''R-Squared:{round(r_value**2, 3)}<br>P-Value: {round(p_value, 3)}''',
                                                          font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                                                          bgcolor = 'rgba(250, 250, 250, 0.8)',
                                                          align = 'left',
                                                          showarrow = False,
                                                          xref = 'paper',
                                                          yref = 'paper',
                                                          x = .01,
                                                          y = 0.99,
                                                          bordercolor = 'black',
                                                          borderwidth = 1)
    

    
    title = f'Beat Proportion Black by Total Population'

    font = 'Nunito, Tahoma, Arial'

    title = go.layout.Title(text = title,
                            font = {'family': font,
                                    'size': 28},
                            y = 0.9,
                            x = 0.5,
                            xanchor = 'center',
                            yanchor = 'top')
    
    xtitle = go.layout.xaxis.Title(text = '% Black Residents',
                             font = {'family': font,
                                    'size': 24})
    
    if calls_grouped is None:
        ytitle = go.layout.yaxis.Title(text = 'Total Population',
                                 font = {'family': font,
                                        'size': 24})
    else:
        ytitle = go.layout.yaxis.Title(text = 'Calls for Service',
                                 font = {'family': font,
                                        'size': 24})
    xaxis = go.layout.XAxis(title = xtitle,
                            zeroline = False)

    yaxis = go.layout.YAxis(title = ytitle,
                            zeroline = False)

    layout = go.Layout(annotations = [annotations],
                       shapes = shapes,
                       title = title,
                       xaxis = xaxis,
                       yaxis = yaxis,
                       autosize = True,
                       height = 600,
                       transition = {'duration': 500})

    fig = go.Figure(data = [trace_1, trace_2], layout = layout)
    return fig


    
def plot_stops_ratios(stops_df, population, freq,
                      pedestrian, vehicle, searches, no_searches, arrests, citations, no_enforcement):

    breakdown = get_outcomes_breakdown(stops_df, pedestrian, vehicle, searches, no_searches, arrests, citations, no_enforcement, population, freq)

    y = breakdown['black']
    x = breakdown.index
    trace_1 = go.Scatter(x = x, y = y,
                         mode = 'lines',
                         name = 'Black')
    
    y = breakdown['white']
    x = breakdown.index
    trace_2 = go.Scatter(x = x, y = y,
                         mode = 'lines',
                         name = 'White')
    
    y = breakdown['hispanic']
    x = breakdown.index
    trace_3 = go.Scatter(x = x, y = y,
                         mode = 'lines',
                         name = 'Hispanic')
    
    y = breakdown['asian']
    x = breakdown.index
    trace_4 = go.Scatter(x = x, y = y,
                         mode = 'lines',
                         name = 'Asian')

    if vehicle & pedestrian:
        stop_types = 'Vehicle and pedestrian stops'
    elif vehicle:
        stop_types = 'Vehicle stops'
    elif pedestrian:
        stop_types = 'Pedestrian stops'
    else:
        stop_types = 'All stops'

    if arrests & citations:
        outcome_types = ' that result in <br>arrests and citations'
    elif arrests:
        outcome_types = ' that result in <br>arrests'
    elif citations:
        outcome_types = ' that result in <br>citations'
    else:
        outcome_types = ''

    if freq == 'W':
        time = 'Week'
    elif freq == 'M':
        time = 'Month'
    else:
        time = 'Year'
        
    if searches == 0:
        searched = '<br>with no search'
    if searches == 1:
        searched = '<br>with a search'
    if searches == 2:
        searched = ''
   
    annotations = go.layout.Annotation(text = f'{stop_types}{outcome_types}{searched}',
                                      font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                                      bgcolor = 'rgba(250, 250, 250, 0.8)',
                                      align = 'left',
                                      showarrow = False,
                                      xref = 'paper',
                                      yref = 'paper',
                                      x = .01,
                                      y = 0.99,
                                      bordercolor = 'black',
                                      borderwidth = 1)
    title = 'Outcomes by Race'

    font = 'Nunito, Tahoma, Arial'

    title = go.layout.Title(text = title,
                            font = {'family': font,
                                    'size': 28},
                            y = 0.9,
                            x = 0.5,
                            xanchor = 'center',
                            yanchor = 'top')

    xtitle = go.layout.xaxis.Title(text = f'{time}',
                             font = {'family': font,
                                     'size': 24})

    ytitle = go.layout.yaxis.Title(text = f'Outcomes per 1000',
                             font = {'family': font,
                                     'size': 24})

    xaxis = go.layout.XAxis(title = xtitle,
    #                         range = [-0.1, 1.1],
                            zeroline = False)

    yaxis = go.layout.YAxis(title = ytitle,
#                             range = [0, 10],
                            zeroline = False)

    legend = go.layout.Legend(font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                              bgcolor = 'rgba(250, 250, 250, 0.8)',
                              x = .99,
                              xanchor = 'right',
                              y = 0.99,
                              bordercolor = 'black',
                              borderwidth = 1)

    layout = go.Layout(annotations = [annotations],
                       title = title,
                       xaxis = xaxis,
                       yaxis = yaxis,
                       legend = legend,
                       autosize = True,
                       height = 600,
                       transition = {'duration': 500})
    
    fig = go.Figure(data = [trace_1, trace_2, trace_3, trace_4],
                    layout = layout)
    return fig

def plot_traffic_violation_offenses(df, population, top_type = 'Disparity'):

    berkeley_pop = 120463

    if population == 'berkeley':
        black_pct = .079
        white_pct = .588
        hispanic_pct = .11
        asian_pct = .21
        pop_title = 'Berkeley Demographics'
    elif population == 'ala_ccc':
        black_pct = ( ( (1150000 * .085) + (1670000 * .101) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        white_pct = ( ( (1150000 * .43) + (1670000 * .309) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        hispanic_pct = ( ( (1150000 * .258) + (1670000 * .224) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        asian_pct = ( ( (1150000 * .168) + (1670000 * .306) )/ 2 ) / ( (1150000 + 1670000) / 2 )
        pop_title = 'Alameda and Contra Costa County Demographics'
    elif population == 'ala_ccc_sfo':
        black_pct = ( ( (1150000 * .085) + (1670000 * .101) + (883000 * .049) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        white_pct = ( ( (1150000 * .43) + (1670000 * .309) + (883000 * .4) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        hispanic_pct = ( ( (1150000 * .258) + (1670000 * .224) + (883000 * .152) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        asian_pct = ( ( (1150000 * .168) + (1670000 * .306) + (883000 * .341) ) / 3 ) / ( (1150000 + 1670000 + 883000) / 3 )
        pop_title = 'Alameda, Contra Costa and San Francisco County Demographics'
    elif population == 'met_stat_area':
        black_pct = .074
        white_pct = .39
        hispanic_pct = .219
        asian_pct = .263
        pop_title = 'Metropolitan Statiscal Area Demographics'
    elif population == 'oak_berk_rich':
        black_pct = ( ( (111701*.0352 * .2) + (444956*.0778 * .24) + (120463*.6835 * .083) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        white_pct = ( ( (111701*.0352 * .36) + (444956*.0778 * .35) + (120463*.6835 * .591) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        hispanic_pct = ( ( (111701*.0352 * .42) + (444956*.0778 * .27) + (120463*.6835 * .11) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        asian_pct = ( ( (111701*.0352 * .23) + (444956*.0778 * .17) + (120463*.6835 * .21) ) / 3 ) / ( (111701*.0352 + 444956*.0778 + 120463*.6835) / 3 )
        pop_title = 'Oakland, Berkeley and Richmond Demographics'
    elif population == 'other':
        black_pct = 0.519997
        white_pct = 0.274169
        hispanic_pct = 0.118260
        asian_pct = 0.028824
        pop_title = 'Victim-Described Suspect Demographics'
    elif population == 'None':
        black_pct = 1/berkeley_pop*1000
        white_pct = 1/berkeley_pop*1000
        hispanic_pct = 1/berkeley_pop*1000
        asian_pct = 1/berkeley_pop*1000
        pop_title = 'Count'


    if top_type == 'Disparity':
        breakdown = df.groupby('traffic_violation_offense').sum()[['asian', 'black', 'hispanic', 'white']]
        breakdown['asian'] = breakdown['asian'] / ( (berkeley_pop * asian_pct) / 1000)
        breakdown['black'] = breakdown['black'] / ( (berkeley_pop * black_pct) / 1000)
        breakdown['hispanic'] = breakdown['hispanic'] / ( (berkeley_pop * hispanic_pct) / 1000)
        breakdown['white'] = breakdown['white'] / ( (berkeley_pop * white_pct) / 1000)
        disparity = ((breakdown['black'] - breakdown['white']).sort_values(ascending = False)[:10])
        count = df[df['traffic_violation_offense'].isin(list(disparity.index))]['traffic_violation_offense'].value_counts()
    elif top_type == 'Count':
        count = df['traffic_violation_offense'].value_counts()[:10]
        breakdown = df[df['traffic_violation_offense'].isin(count.index)].groupby('traffic_violation_offense').sum()[['asian', 'black', 'hispanic', 'white']]
        breakdown['asian'] = breakdown['asian'] / ( (berkeley_pop * asian_pct) / 1000)
        breakdown['black'] = breakdown['black'] / ( (berkeley_pop * black_pct) / 1000)
        breakdown['hispanic'] = breakdown['hispanic'] / ( (berkeley_pop * hispanic_pct) / 1000)
        breakdown['white'] = breakdown['white'] / ( (berkeley_pop * white_pct) / 1000)
        disparity = (breakdown['black'] - breakdown['white'])
        
    data = pd.concat([disparity, count], axis = 1).set_axis(['Disparity', 'Count'], axis=1)

    trace_disparity = go.Bar(name = 'Disparity', x = disparity.index, y = disparity, yaxis = 'y', offsetgroup = 1)
    trace_count = go.Bar(name = 'Count', x = count.index, y = count, yaxis = 'y2', offsetgroup = 2)

    title = f'Top Traffic Violation Offenses, 2021 ({pop_title})'

    font = 'Nunito, Tahoma, Arial'

    title = go.layout.Title(text = title,
                            font = {'family': font,
                                    'size': 28},
                            y = 0.9,
                            x = 0.5,
                            xanchor = 'center',
                            yanchor = 'top')
    
    xtitle = go.layout.xaxis.Title(text = 'Traffic Violation Offense',
                             font = {'family': font,
                                    'size': 24})
    
    ytitle = go.layout.yaxis.Title(text = '# Stops, Black - White',
                                 font = {'family': font,
                                        'size': 24})
    ytitle2 = go.layout.yaxis.Title(text = 'Total # Stops',
                                 font = {'family': font,
                                        'size': 24})

    xaxis = go.layout.XAxis(title = xtitle,
                            zeroline = False)

    yaxis = go.layout.YAxis(title = ytitle,
                            zeroline = False)
    
    yaxis2 = go.layout.YAxis(title = ytitle2,
                            zeroline = False,
                            overlaying = 'y',
                            side = 'right')

    legend = go.layout.Legend(font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                          bgcolor = 'rgba(250, 250, 250, 0.8)',
                          x = .99,
                          xanchor = 'right',
                          y = 0.99,
                          bordercolor = 'black',
                          borderwidth = 1)
    
    layout = go.Layout(title = title,
                    #    xaxis = xaxis,
                       yaxis = yaxis,
                       yaxis2 = yaxis2,
                       autosize = True,
                       height = 600,
                       legend = legend,
                       barmode = 'group',
                       transition = {'duration': 500})
   
    if top_type == 'Count':
        fig = go.Figure(data = [trace_count, trace_disparity], layout = layout)
    elif top_type == 'Disparity':
        fig = go.Figure(data = [trace_disparity, trace_count], layout = layout)
    return fig


def plot_force(df):
    
    traces = [go.Bar(name = df.columns[idx], x = df.index, y = df[column])
              for idx, column in enumerate(df.columns)]

    title = f'Use of Force by Type'

    font = 'Nunito, Tahoma, Arial'

    title = go.layout.Title(text = title,
                            font = {'family': font,
                                    'size': 28},
                            y = 0.9,
                            x = 0.5,
                            xanchor = 'center',
                            yanchor = 'top')
    
    xtitle = go.layout.xaxis.Title(text = 'Year',
                             font = {'family': font,
                                    'size': 24})
    
    ytitle = go.layout.yaxis.Title(text = 'Count',
                                 font = {'family': font,
                                        'size': 24})
    xaxis = go.layout.XAxis(title = xtitle,
                            zeroline = False)

    yaxis = go.layout.YAxis(title = ytitle,
                            zeroline = False)

    legend = go.layout.Legend(font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                          bgcolor = 'rgba(250, 250, 250, 0.8)',
                          x = .99,
                          xanchor = 'right',
                          y = 0.99,
                          bordercolor = 'black',
                          borderwidth = 1)
    
    layout = go.Layout(title = title,
                       xaxis = xaxis,
                       yaxis = yaxis,
                       autosize = True,
                       height = 600,
                       legend = legend,
                       barmode = 'group',
                       transition = {'duration': 500})

    fig = go.Figure(data = traces, layout = layout)
    return fig

def plot_complaints(df):
    
    traces = [go.Bar(name = df.columns[idx], x = df.index, y = df[column])
              for idx, column in enumerate(df.columns)]

    title = f'Use of Force, Complaints'

    font = 'Nunito, Tahoma, Arial'

    title = go.layout.Title(text = title,
                            font = {'family': font,
                                    'size': 28},
                            y = 0.9,
                            x = 0.5,
                            xanchor = 'center',
                            yanchor = 'top')
    
    xtitle = go.layout.xaxis.Title(text = 'Year',
                             font = {'family': font,
                                    'size': 24})
    
    ytitle = go.layout.yaxis.Title(text = 'Count',
                                 font = {'family': font,
                                        'size': 24})
    xaxis = go.layout.XAxis(title = xtitle,
                            zeroline = False)

    yaxis = go.layout.YAxis(title = ytitle,
                            zeroline = False)

    legend = go.layout.Legend(font = {'family': 'Nunito, Tahoma, Arial', 'size': 12},
                          bgcolor = 'rgba(250, 250, 250, 0.8)',
                          x = .99,
                          xanchor = 'right',
                          y = 0.99,
                          bordercolor = 'black',
                          borderwidth = 1)
    
    layout = go.Layout(title = title,
                       xaxis = xaxis,
                       yaxis = yaxis,
                       autosize = True,
                       height = 600,
                       legend = legend,
                       barmode = 'group',
                       transition = {'duration': 500})

    fig = go.Figure(data = traces, layout = layout)
    return fig

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()
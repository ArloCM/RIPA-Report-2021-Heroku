import streamlit as st
import pandas as pd
import numpy as np
from Functions import *

@st.cache
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True)
    df['datetime'] = df['datetime'].dt.tz_localize(tz = pytz.timezone('America/Los_Angeles'), ambiguous = 'NaT')
    return df

def app():
        
    filepath = 'stops.csv'
    stops = load_data(filepath)

    st.sidebar.markdown('---')

    st.sidebar.subheader('Filter Data')
    option1 = st.sidebar.multiselect(
                                    'Select a Stop Type',
                                    ['Vehicle', 'Pedestrian'],
                                    ['Vehicle', 'Pedestrian'])
    if option1 == ['Vehicle']:
        pedestrian = False
        vehicle = True
    elif option1 == ['Pedestrian']:
        pedestrian = True
        vehicle = False
    elif option1 == ['Vehicle', 'Pedestrian']:
        pedestrian = True
        vehicle = True
    else:
        st.error('Error: Please select a Stop Type')

    option2 = st.sidebar.multiselect(
        'Select a Stop Outcome',
        ['Arrest', 'Citation', 'No Enforcement'],
        ['Arrest', 'Citation', 'No Enforcement'])
    if option2 == ['Arrest']:
        arrests = True
        citations = False
        no_enforcement = False
    elif option2 == ['Citation']:
        arrests = False
        citations = True
        no_enforcement = False
    elif option2 == ['No Enforcement']:
        arrests = False
        citations = False
        no_enforcement = True
    elif option2 == ['Arrest', 'Citation']:
        arrests = True
        citations = True
        no_enforcement = False
    elif option2 == ['Arrest', 'No Enforcement']:
        arrests = True
        citations = False
        no_enforcement = True
    elif option2 == ['Citation', 'No Enforcement']:
        arrests = False
        citations = True
        no_enforcement = True
    elif option2 == ['Arrest', 'Citation', 'No Enforcement']:
        arrests = True
        citations = True
        no_enforcement = True
    else:
        st.error('Error: Please select a Stop Outcome')

    option3 = st.sidebar.multiselect(
        'Includes a Search',
        ['Search', 'No Search'],
        ['Search', 'No Search'])
    if option3 == ['Search']:
        searches = True
        no_searches = False
    elif option3 == ['No Search']:
        searches = False
        no_searches = True
    elif option3 == ['Search', 'No Search']:
        searches = True
        no_searches = True
    else:
        st.error('Error: Please select whether or not stops include a search')

    option4 = st.sidebar.selectbox('Select a demographic baseline',
                            ['Berkeley',
                            'Stop Data Residence Demographics',
                            'Victim Described Suspect Demographics',
                            'Count'],
                            3
                            )
    if option4 == 'Berkeley':
        population = 'berkeley'
    elif option4 == 'Stop Data Residence Demographics':
        population = 'oak_berk_rich'
    elif option4 == 'Victim Described Suspect Demographics':
        population = 'other'
    elif option4 == 'Count':
        population = 'None'
    else:
        st.error('Error: Please select a demographic baseline')

    option5 = st.sidebar.radio('Select measurement frequency',
                                ('Year', 'Month', 'Week', 'Day'),
                                1)
    if option5 == 'Year':
        freq = 'Y'
    elif option5 == 'Month':
        freq = 'M'
    elif option5 == 'Week':
        freq = 'W'
    elif option5 == 'Day':
        freq = 'D'
    else:
        st.error('Please select a measurement frequency')


    with st.container():
        overview_intro = read_markdown_file('Overview_Intro.md')
        st.markdown(overview_intro, unsafe_allow_html=True)
        st.image('Berkeley-Sunset-2.jpg')
        st.header('Stops Disparities')

        overview_intro = read_markdown_file('Disparities.md')
        st.markdown(overview_intro, unsafe_allow_html=True)

    st.markdown('---')

    try:
        fig1 = plot_stops(stops,
                        pedestrian = pedestrian,
                        vehicle = vehicle,
                        searches = searches,
                        no_searches = no_searches,
                        arrests = arrests,
                        citations = citations,
                        no_enforcement= no_enforcement,
                        mh_hold = None,
                        population = population,
                        freq = freq)
        st.plotly_chart(fig1)    
    except NameError:
        st.error('Error: Please make a selection for each data filter')

            
    st.markdown('---')

    try:
        fig2 = plot_ratio_time(stops,
                            vehicle = vehicle,
                            pedestrian = pedestrian,
                            population = population,
                            freq = freq,
                            ripa = False,
                            minority = ['black'])
        st.plotly_chart(fig2)
    except NameError:
        st.error('Error: Please make a selection for each data filter')

    st.markdown('---')


    overview_analysis = read_markdown_file('Overview_Analysis.md')
    st.markdown(overview_analysis, unsafe_allow_html=True)

    with st.container():
        st.header('Officer Discretion')

        discretion_intro = read_markdown_file('Discretion_Intro.md')
        st.markdown(discretion_intro, unsafe_allow_html=True)

        discretion_yield_rate = read_markdown_file('Discretion_Yield_Rate.md')
        st.markdown(discretion_yield_rate, unsafe_allow_html=True)
    st.markdown('---')

    option8 = st.selectbox('Select a view for Yield Rate Analysis',
                            ['Through time',
                            'By beat'],
                            0
                            )
    if option8 == 'Through time':
        chart_placeholder1 = st.empty()

        rolling_days = st.slider('Select # of days for rolling median',
                14, 730, 365, 1)

        try:
            fig3 = plotly_yield_after_search(stops,
                                            vehicle = vehicle,
                                            pedestrian = pedestrian,
                                            arrests = arrests,
                                            citations = citations,
                                            contraband = False,
                                            rolling_days = rolling_days,
                                            freq = freq,
                                            beats = False,
                                            minority = ['black'],
                                            year = list(range(2015, 2023)))
            chart_placeholder1.plotly_chart(fig3)
        except NameError:
            st.error('Error: Please make a selection for each data filter')
    elif option8 == 'By beat':
        chart_placeholder2 = st.empty()

        year = st.slider('Select Years',
                2015, 2021, (2015,2021))

        try:
            fig4 = plotly_yield_after_search_beat(stops,
                                                vehicle = vehicle,
                                                pedestrian = pedestrian,
                                                arrests = arrests,
                                                citations = citations,
                                                contraband = False,
                                                freq = freq,
                                                beats = True,
                                                minority = ['black'],
                                                year = year)
            chart_placeholder2.plotly_chart(fig4)
        except NameError:
            st.error('Error: Please make a selection for each data filter')

    discretion_yield_rate_analysis = read_markdown_file('Discretion_Yield_Rate_Analysis.md')
    st.markdown(discretion_yield_rate_analysis, unsafe_allow_html=True)

    st.markdown('---')

    discretion_veil = read_markdown_file('Discretion_Veil.md')
    st.markdown(discretion_veil, unsafe_allow_html=True)

    try:
        fig5 = plotly_veil_of_darkness(stops, year = list(range(2015, 2023)))
        st.plotly_chart(fig5)
    except NameError:
        st.error('Error: Please make a selection for each data filter')

    st.markdown('---')

    with st.container():
        st.header('Traffic Enforcement')

        traffic_intro = read_markdown_file('Traffic_Enforcement.md')
        st.markdown(traffic_intro, unsafe_allow_html=True)

    embed_code = '''
    <iframe src="https://storymaps.arcgis.com/stories/54e56f63ae25455180045f852b44b9dd?cover=false" width="100%" height="500px" frameborder="0" allowfullscreen allow="geolocation"></iframe>
    '''
    src = 'https://storymaps.arcgis.com/stories/54e56f63ae25455180045f852b44b9dd?cover=false'
    st.components.v1.iframe(src, width=None, height=500, scrolling=False)

    st.markdown('---')
    # st.image('Berkeley_Banner.jpg')
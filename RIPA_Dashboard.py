import streamlit as st
import pandas as pd
import numpy as np
from Functions import *

@st.cache
def load_data(filepath, year = None):
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True)
    df['datetime'] = df['datetime'].dt.tz_localize(tz = pytz.timezone('America/Los_Angeles'), ambiguous = 'NaT')
    if year:
        df = df[df['year'] == year]
    return df

def app():
        
    filepath = 'ripa.csv'
    stops = load_data(filepath, 2021)
    
    st.sidebar.markdown('---')
    st.sidebar.subheader('Filter Data')


    option6 = st.sidebar.multiselect('Stop is made in response to call for service',
                                    ['Yes', 'No'],
                                    ['Yes', 'No'])
    if option6 == ['Yes']:
        stops = stops[stops['is_stop_made_in_response'] == True]
    elif option6 == ['No']:
        stops = stops[stops['is_stop_made_in_response'] == False]
    elif set(option6) == set(['Yes', 'No']):
        pass
    else:
        st.error('Please select whether to include stops that are in reponse to a call for service')

    option7 = st.sidebar.multiselect('Information based stops | Note: Only available after May 2021',
                                    ['Yes', 'No'],
                                    [])
    if option7 == ['Yes']:
        stops = stops[stops['InfoBasedStop'] == 'yes']
    elif option7 == ['No']:
        stops = stops[stops['InfoBasedStop'] == 'no']
    elif set(option7) == set(['Yes', 'No']):
        stops = stops[(stops['InfoBasedStop'] == 'no') | (stops['InfoBasedStop'] == 'yes')]
    elif option7 == []:
        pass
    else:
        st.error('Please select whether to include information based stops')

    option1 = st.sidebar.multiselect(
                                    'Select a Stop Type',
                                    ['Vehicle', 'Pedestrian'],
                                    ['Vehicle', 'Pedestrian'])
    if set(option1) == set(['Vehicle', 'Pedestrian']):
        pedestrian = True
        vehicle = True
    elif option1 == ['Vehicle']:
        pedestrian = False
        vehicle = True
    elif option1 == ['Pedestrian']:
        pedestrian = True
        vehicle = False
    else:
        st.error('Error: Please select a Stop Type')

    option7 = st.sidebar.multiselect('Select a traffic violation type',
                                    ['Moving Violation', 'Equipment Violation', 'Non-moving violation / including registration', 'Other'],
                                    ['Moving Violation', 'Equipment Violation', 'Non-moving violation / including registration', 'Other'])
    if 'Other' in option7:
        stops = stops[(stops['traffic_violation_type'].isin(option7)) | stops['traffic_violation_type'].isna()]
    else:        
        stops = stops[stops['traffic_violation_type'].isin(option7)]

    option2 = st.sidebar.multiselect(
        'Select a Stop Outcome',
        ['Arrest', 'Citation', 'No Enforcement', '5150'],
        ['Arrest', 'Citation', 'No Enforcement', '5150'])
    if option2 == ['Arrest']:
        arrests = True
        citations = False
        no_enforcement = False
        mh_hold = False
    elif option2 == ['Citation']:
        arrests = False
        citations = True
        no_enforcement = False
        mh_hold = False
    elif option2 == ['No Enforcement']:
        arrests = False
        citations = False
        no_enforcement = True
        mh_hold = False
    elif option2 == ['5150']:
        arrests = False
        citations = False
        no_enforcement = False
        mh_hold = True
    elif set(option2) == set(['Arrest', 'Citation']):
        arrests = True
        citations = True
        no_enforcement = False
        mh_hold = False
    elif set(option2) == set(['Arrest', 'No Enforcement']):
        arrests = True
        citations = False
        no_enforcement = True
        mh_hold = False
    elif set(option2) == set(['Citation', 'No Enforcement']):
        arrests = False
        citations = True
        no_enforcement = True
        mh_hold = False
    elif set(option2) == set(['Arrest', 'Citation', 'No Enforcement']):
        arrests = True
        citations = True
        no_enforcement = True
        mh_hold = False
    elif set(option2) == set(['Arrest', '5150']):
        arrests = True
        citations = False
        no_enforcement = False
        mh_hold = True
    elif set(option2) == set(['Citation', '5150']):
        arrests = False
        citations = True
        no_enforcement = False
        mh_hold = True
    elif set(option2) == set(['No Enforcement', '5150']):
        arrests = False
        citations = False
        no_enforcement = True
        mh_hold = True
    elif set(option2) == set(['Arrest', 'Citation', '5150']):
        arrests = True
        citations = True
        no_enforcement = False
        mh_hold = True
    elif set(option2) == set(['Arrest', 'No Enforcement', '5150']):
        arrests = True
        citations = False
        no_enforcement = True
        mh_hold = True
    elif set(option2) == set(['Citation', 'No Enforcement', '5150']):
        arrests = False
        citations = True
        no_enforcement = True
        mh_hold = True
    elif set(option2) == set(['Arrest', 'Citation', 'No Enforcement', '5150']):
        arrests = True
        citations = True
        no_enforcement = True
        mh_hold = True
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
    elif set(option3) == set(['Search', 'No Search']):
        searches = True
        no_searches = True
    else:
        st.error('Error: Please select whether or not stops include a search')

    option4 = st.sidebar.selectbox('Select a demographic baseline',
                            ['Berkeley',
                            'Alameda and Contra Costa Counties',
                            'Alameda, Contra Costa and San Francisco Counties',
                            'Metropolitan Statiscal Area',
                            'Oakland, Berkeley and Richmond',
                            'Victim Described Suspect Demographics',
                            'Count'],
                            6
                            )
    if option4 == 'Berkeley':
        population = 'berkeley'
    elif option4 == 'Alameda and Contra Costa Counties':
        population = 'ala_ccc'
    elif option4 == 'Alameda, Contra Costa and San Francisco Counties':
        population = 'ala_ccc_sfo'
    elif option4 == 'Metropolitan Statiscal Area':
        population = 'met_stat_area'
    elif option4 == 'Oakland, Berkeley and Richmond':
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
        fig1 = plot_stops(df = stops,
                        pedestrian = pedestrian,
                        vehicle = vehicle,
                        searches = searches,
                        no_searches = no_searches,
                        arrests = arrests,
                        citations = citations,
                        mh_hold = mh_hold,
                        no_enforcement= no_enforcement,
                        population = population,
                        freq = freq)
        st.plotly_chart(fig1)    
    except NameError:
        st.error('Error: Please make a selection for each data filter')

    try:
        fig2 = plot_ratio_time(stops,
                            vehicle = vehicle,
                            pedestrian = pedestrian,
                            population = population,
                            freq = freq,
                            ripa = True,
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

    yield_type = st.multiselect('Select search yield type',
                                ['Citation', 'Arrest', 'Contraband'],
                                ['Contraband'])
    if set(yield_type) == set(['Citation', 'Arrest', 'Contraband']):
        arrests = True
        citations = True
        contraband = True
    elif set(yield_type) == set(['Citation', 'Arrest']):
        arrests = True
        citations = True
        contraband = False
    elif set(yield_type) == set(['Contraband', 'Arrest']):
        arrests = True
        citations = False
        contraband = True
    elif set(yield_type) == set(['Citation', 'Contraband']):
        arrests = False
        citations = True
        contraband = True
    elif yield_type == ['Arrest']:
        arrests = True
        citations = False
        contraband = False
    elif yield_type == ['Citation']:
        arrests = False
        citations = True
        contraband = False
    elif yield_type == ['Contraband']:
        arrests = False
        citations = False
        contraband = True
    else:
        st.error('Please select a search yield type')

    option8 = st.selectbox('Select a view for Yield Rate Analysis',
                            ['Through time',
                            'By beat'],
                            0
                            )
    if option8 == 'Through time':
        chart_placeholder1 = st.empty()

        try:
            fig3 = plotly_yield_after_search(stops,
                                            vehicle = vehicle,
                                            pedestrian = pedestrian,
                                            arrests = arrests,
                                            citations = citations,
                                            contraband = contraband,
                                            rolling_days = 365,
                                            freq = freq,
                                            beats = False,
                                            minority = ['black'],
                                            year = [2021])
            chart_placeholder1.plotly_chart(fig3)
        except NameError:
            st.error('Error: Please make a selection for each data filter')
    elif option8 == 'By beat':
        chart_placeholder2 = st.empty()

        try:
            fig4 = plotly_yield_after_search_beat(stops,
                                                vehicle = vehicle,
                                                pedestrian = pedestrian,
                                                arrests = arrests,
                                                citations = citations,
                                                freq = freq,
                                                beats = True,
                                                minority = ['black'],
                                                year = [2021])
            chart_placeholder2.plotly_chart(fig4)
        except NameError:
            st.error('Error: Please make a selection for each data filter')

    st.markdown('---')

    try:
        fig5 = plotly_veil_of_darkness(stops, year = [2021])
        st.plotly_chart(fig5)
    except NameError:
        st.error('Error: Please make a selection for each data filter')

    discretion_veil = read_markdown_file('Discretion_Veil.md')
    st.markdown(discretion_veil, unsafe_allow_html=True)
    st.markdown('---')
    
    with st.container():
        st.header('Traffic Enforcement')

        traffic_intro = read_markdown_file('Traffic_Enforcement.md')
        st.markdown(traffic_intro, unsafe_allow_html=True)

    embed_code = '''
    <iframe src="https://storymaps.arcgis.com/stories/d63658136bb746a89616b52510a63aae?cover=false" width="100%" height="500px" frameborder="0" allowfullscreen allow="geolocation"></iframe>
    '''
    src = 'https://storymaps.arcgis.com/stories/d63658136bb746a89616b52510a63aae?cover=false'
    st.components.v1.iframe(src, width=None, height=500, scrolling=False)

    st.markdown('---')

    # option9 = st.selectbox('Select how to sort traffic violation offenses',
    #                         ['Top ten by disparity',
    #                         'Top ten by count'],
    #                         0
    #                         )
    # if option9 == 'Top ten by disparity':
    #     fig6 = plot_traffic_violation_offenses(stops, population, top_type = 'Disparity')
    #     st.plotly_chart(fig6)
    # elif option9 == 'Top ten by count':
    #     fig6 = plot_traffic_violation_offenses(stops, population, top_type = 'Count')
    #     st.plotly_chart(fig6)

    # traffic_violations = read_markdown_file('Traffic_Violations.md')
    # st.markdown(traffic_violations, unsafe_allow_html=True)

    st.image('Berkeley_Banner.jpg')
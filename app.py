import streamlit as st

# Custom imports 
from multipage import MultiPage
import RIPA_Dashboard, Historical_RIPA_Dashboard

# Create an instance of the app 
app = MultiPage()

hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .reportview-container .main footer {visibility: hidden;}    
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

custom_font_style = ''' <style>
@font-face {
  font-family: 'Nunito ExtraLight', 'Tahoma', 'Arial';
  src: url('https://fonts.googleapis.com/css2?family=Nunito:ital,wght@0,200;0,300;0,400;0,600;0,700;1,200;1,300;1,400;1,500;1,600&display=swap');
}

    html, body, [class*="css"]  {
    font-family: 'Nunito ExtraLight', 'Tahoma', 'Arial';
    }
    </style> '''

st.markdown(custom_font_style, unsafe_allow_html=True)
st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)

# Title of the main page
st.title("BPD Stop Data Analysis")
# col1, col2, col3 = st.sidebar.columns([1,1,6])
# col3.image('Berkeley PD.png', width=100)
st.sidebar.markdown('*BPD Racial and Identity Profiling Act (RIPA) Interactive Report, 2021*')
# Add all your applications (pages) here
app.add_page("RIPA 2021", RIPA_Dashboard.app)
app.add_page("2015-2021", Historical_RIPA_Dashboard.app)

# The main app
app.run()
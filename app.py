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
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
st.markdown(""" <style>
@font-face {
  font-family: 'Oswald';
  src: url('https://fonts.googleapis.com/css2?family=Oswald:wght@200;300;400;500;600;700&display=swap');
}

    html, body, [class*="css"]  {
    font-family: 'Oswald';
    }
    </style> """, unsafe_allow_html=True)
# Title of the main page
st.title("BPD Stop Data Analysis")
# col1, col2, col3 = st.sidebar.columns([1,1,6])
# col3.image('Berkeley PD.png', width=100)
st.sidebar.markdown('*BPD 2021 Racial and Identity Profiling Act (RIPA) Analysis Dashboard*')
# Add all your applications (pages) here
app.add_page("RIPA 2021", RIPA_Dashboard.app)
app.add_page("2015-Current", Historical_RIPA_Dashboard.app)

# The main app
app.run()
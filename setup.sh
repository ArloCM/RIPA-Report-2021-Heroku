mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
[theme]\n\
base = 'light'\n\
primaryColor = '#3b7b9b'\n\
secondaryBackgroundColor = '#f0f0f0'\n\
textColor = '#0d1d29'\n\
" > ~/.streamlit/config.toml
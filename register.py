import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

with open(r'config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

try:
    email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(
        pre_authorization=False)
    if email_of_registered_user:
        st.success('User registered successfully')

        with open(r'config.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

except Exception as e:
    st.error(e)


if st.button("Login"):
    st.switch_page("Login.py")

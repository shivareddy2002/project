import streamlit as st
import sqlite3
import bcrypt
import re

# Database setup
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Hash password
def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

# Verify password
def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password)

# Add user to database
def add_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    conn.close()

# Check if user exists
def user_exists(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return result is not None

# Validate username
def validate_username(username):
    if len(username) < 4:
        return "Username must be at least 4 characters long."
    if not re.match("^[a-zA-Z0-9_]+$", username):
        return "Username can only contain letters, numbers, and underscores."
    return None

# Validate password
def validate_password(password):
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if not re.search("[A-Z]", password):
        return "Password must contain at least one uppercase letter."
    if not re.search("[a-z]", password):
        return "Password must contain at least one lowercase letter."
    if not re.search("[0-9]", password):
        return "Password must contain at least one digit."
    if not re.search("[!@#$%^&*()]", password):
        return "Password must contain at least one special character."
    return None

# Initialize database
init_db()

def main():
    # Configure page settings
    st.set_page_config(
        page_title="Tomato Stress Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS styling
    st.markdown("""
    <style>
        .header {
            background: white;
            padding: 1rem;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            position: fixed;
            width: 100%;
            top: 0;
            z-index: 1000;
        }
        .nav-link {
            margin: 0 1rem;
            color: #333;
            text-decoration: none;
        }
        .main-banner {
            padding: 6rem 0 2rem;
            background: #f8f9fa;
        }
        .service-card {
            padding: 2rem;
            margin: 1rem;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        }
        .footer {
            background: #333;
            color: white;
            padding: 2rem;
            text-align: center;
            margin-top: 4rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state for authentication and page routing
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "page" not in st.session_state:
        st.session_state.page = "signup"  # Start with the Sign Up page

    # Sign Up Page
    if st.session_state.page == "signup":
        st.title("Sign Up")
        new_username = st.text_input("Choose a Username", key="signup_username")
        new_password = st.text_input("Choose a Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Sign Up", key="signup_button"):
            username_error = validate_username(new_username)
            password_error = validate_password(new_password)
            
            if username_error:
                st.error(username_error)
            elif password_error:
                st.error(password_error)
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif user_exists(new_username):
                st.error("Username already exists. Please choose another.")
            else:
                add_user(new_username, new_password)
                st.success("Account created successfully! Please sign in.")
                st.session_state.page = "login"  # Redirect to the Login page
                st.rerun()

    # Login Page
    elif st.session_state.page == "login":
        st.title("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_button"):
            if user_exists(username):
                conn = sqlite3.connect("users.db")
                c = conn.cursor()
                c.execute("SELECT password FROM users WHERE username = ?", (username,))
                hashed_password = c.fetchone()[0]
                conn.close()
                if verify_password(password, hashed_password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.page = "home"  # Redirect to the Home page
                    st.rerun()
                else:
                    st.error("Invalid password")
            else:
                st.error("Username does not exist")

    # Home Page
    elif st.session_state.page == "home":
        # Create a layout with columns for the Sign Out button
        col1, col2 = st.columns([4, 1])  # Adjust the ratio as needed
        with col1:
            st.markdown("<div class='main-banner'>", unsafe_allow_html=True)
            
            # Banner section
            st.title("Welcome to Dashboard")
            st.markdown("""
            Classification and Forecasting of Water Stress in Tomato Plants 
            Using Bioristor Data
            """)
            st.image("static/123.webp", width=500)
            
            st.markdown("</div>", unsafe_allow_html=True)

            # File Uploader Section
            st.header("Upload Files")
            st.write("Upload your files for analysis.")

            # File uploader for CSV and PDF
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["csv", "pdf"],
                accept_multiple_files=False
            )

            if uploaded_file is not None:
                # Display file details
                st.write(f"File uploaded successfully!")

        # Sign Out Button at the top-right corner
        with col2:
            if st.button("Sign Out", key="sign_out_button"):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.session_state.page = "signup"  # Redirect to the Sign Up page
                st.rerun()

    # Footer
    st.markdown("""
    <div class='footer'>
        <p>Copyright © 2023 Plant Analysis System</p>
        <p>Design: TemplateMo</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
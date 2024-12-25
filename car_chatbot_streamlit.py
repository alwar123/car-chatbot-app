import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Setup SSL for downloading NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Expanded intents
intents = [
     {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there!", "Hello! How can I assist you with your car today?", "Hey! Need help with your car?"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye! Drive safe!", "See you later! Stay safe on the road!", "Take care!"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome!", "No problem!", "Glad I could help!"]
    },
    {
        "tag": "engine_issues",
        "patterns": ["My car engine is making noise", "Why is my engine overheating?", "The engine won't start", "My engine light is on"],
        "responses": ["Check coolant levels for overheating, or try starting the car with headlights off.", "If the engine light is on, a diagnostic scan might help.", "Engine noise may be due to low oil or damage. Check the oil level or visit a mechanic."]
    },
    {
        "tag": "tire_issues",
        "patterns": ["My tire is flat", "The tire pressure is low", "How do I check tire pressure?", "What do I do about a puncture?"],
        "responses": ["Use a spare tire for a flat or contact roadside assistance.", "Check tire pressure at a gas station.", "Use a tire pressure gauge to check PSI levels.", "For punctures, use a sealant or get the tire repaired."]
    },
    {
        "tag": "battery_issues",
        "patterns": ["My car battery is dead", "The battery won't charge", "How do I jump-start my car?", "Why is my battery draining?"],
        "responses": ["Jump-start the car with another vehicle’s battery.", "If the battery won’t charge, check the alternator.", "Turn off lights and accessories when the engine is off to avoid draining the battery.", "Replace old batteries if they fail to hold a charge."]
    },
    {
        "tag": "oil_change",
        "patterns": ["When should I change my oil?", "How do I check oil levels?", "The oil light is on", "What type of oil does my car need?"],
        "responses": ["Change oil every 5,000 to 7,500 miles, or as recommended.", "Use the dipstick to check oil levels.", "Stop driving if the oil light is on and check oil levels immediately.", "Refer to your manual for the recommended oil type."]
    },
    {
        "tag": "brake_issues",
        "patterns": ["My brakes are squeaking", "The brake pedal feels soft", "The car doesn't stop properly", "Brake light is on"],
        "responses": ["Squeaking brakes may need cleaning or new pads.", "A soft pedal could mean low brake fluid or air in the lines.", "If the car doesn’t stop well, get the brakes inspected.", "The brake light might indicate low fluid or other issues."]
    },
    {
        "tag": "general_maintenance",
        "patterns": ["What maintenance does my car need?", "How often should I service my car?", "Tips for car care?", "What’s included in a service?"],
        "responses": ["Maintenance includes oil changes, tire rotations, and brake checks.", "Service every 6 months or 10,000 miles.", "Keep your car clean, check fluids regularly, and inspect tires.", "A service includes an oil change, filter replacement, and system checks."]
    },
    {
        "tag": "fuel_economy",
        "patterns": ["How can I improve my car's fuel economy?", "Why is my car using so much gas?", "Best ways to save fuel?"],
        "responses": ["Ensure tires are inflated properly, and reduce extra weight.", "Avoid aggressive driving and use cruise control on highways.", "Regular servicing and proper motor oil can save fuel."]
    },
    {
        "tag": "ac_issues",
        "patterns": ["Why is my car AC not cooling?", "The AC smells bad", "How do I fix my car's air conditioning?"],
        "responses": ["Refrigerant recharge or filter replacement might help.", "A bad smell could be mold or a clogged filter.", "Consult a professional if the AC doesn’t work after basic checks."]
    }
]

# Train the machine learning model
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
    return "I'm sorry, I don't understand that."

# Streamlit UI
st.title("Car Chatbot")
st.write("Ask your car-related questions, and I'll try to help!")

# Sidebar with car logo and more details
# Replace this placeholder URL with your car logo URL or local file path
st.sidebar.image("car_logo.png", width=150)  # Relative path
  # Car logo image

# Sidebar content
st.sidebar.title("About")
st.sidebar.write("This is an AI-powered car troubleshooting chatbot.")
st.sidebar.write("It can provide suggestions for various car issues such as engine problems, tire issues, and more!")

st.sidebar.title("Features")
st.sidebar.write("- Get quick solutions for common car problems.")
st.sidebar.write("- Ask about your car’s maintenance needs.")
st.sidebar.write("- Get assistance with engine, tire, and battery issues.")

st.sidebar.title("Mechanic Shops Near You")
st.sidebar.write("Find a nearby mechanic shop to help with repairs.")

# Session state for keeping track of login status
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_credentials = {}

# Login/Signup logic
if not st.session_state.logged_in:
    st.subheader("Login/Signup")
    login_signup = st.radio("Choose an option", ["Login", "Sign Up"])

    if login_signup == "Login":
        username = st.text_input("Username:").strip()
        password = st.text_input("Password:", type="password").strip()

        if st.button("Login"):
            if username in st.session_state.user_credentials and st.session_state.user_credentials[username] == password:
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials, please try again.")

    elif login_signup == "Sign Up":
        new_username = st.text_input("New Username:").strip()
        new_password = st.text_input("New Password:", type="password").strip()
        if st.button("Sign Up"):
            if new_username and new_password:
                st.session_state.user_credentials[new_username] = new_password
                st.success(f"Account created for {new_username}!")
else:
    # Main display with suggestions for tags above the input section
    st.write("### Choose a car issue or ask a question:")

    # Allow users to select a car issue but don’t show the response immediately
    selected_tag = st.selectbox("Select a car issue", [""] + [intent["tag"] for intent in intents])

    # Only show suggestions if the tag is selected
    if selected_tag and selected_tag != "":
        st.write(f"You selected: {selected_tag}")
        # Here we are NOT generating any response yet, just showing the selected tag

    # User input for chatbot interaction (only after user queries)
    user_input = st.text_input("You:", "")

    # Only show chatbot response if user_input exists
    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot Response:", response, height=100)

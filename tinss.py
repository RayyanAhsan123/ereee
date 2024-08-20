import streamlit as st

# Display the GIF directly using the path
st.image("C:/Users/rayya/OneDrive/Desktop/a4d19890287115.5e139f5047d55.gif", caption="Happy Birthday!", use_column_width=True)

# Display animated text using HTML/CSS with pink shades
st.markdown(
    """
    <style>
    .blinking {
        animation: blinker 1.5s linear infinite;
        background: -webkit-linear-gradient(#ff80ab, #ff4081, #f50057);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 30px;
    }
    @keyframes blinker {
        50% { opacity: 0; }
    }
    </style>
    <p class="blinking">🎉 Wishing you a very Happy Birthday! 🎂 Enjoy your special day! 🥳<br>
    Happy birthday to the one who always lifts my spirits. Cheers to many more years of laughter and good times together!<br>
    Wishing you a day filled with love, cake, joy, and all the other wonderful things you deserve. Have a great birthday, dear friend.</p>
    """,
    unsafe_allow_html=True
)

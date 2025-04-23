import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime

st.set_page_config(layout="wide")

# Dashboard title
st.title("Social Media Sentiment Dashboard")

# Load processed data
df = pd.read_csv("./processed_posts.csv")
df["Time"] = pd.to_datetime(df["Time"])

# Sentiment filter
sentiment_filter = st.selectbox(
    "Select Sentiment", ["All", "POSITIVE", "NEGATIVE", "NEUTRAL"]
)
topic_filter = st.multiselect(
    "Select Topics", df["Topic"].unique(), default=df["Topic"].unique()
)

filtered_df = df.copy()
if sentiment_filter != "All":
    filtered_df = filtered_df[filtered_df["Sentiment"] == sentiment_filter]
if topic_filter:
    filtered_df = filtered_df[filtered_df["Topic"].isin(topic_filter)]

# Display data table
st.write(filtered_df[["Time", "Title", "Sentiment", "Topic"]])

fig, ax = plt.subplots(figsize=(10, 6))
filtered_df.groupby(
    [pd.Grouper(key="Time", freq="H"), "Sentiment"]
).size().unstack().plot(ax=ax)
ax.set_title("Sentiment Distribution by Hour")
st.pyplot(fig)

# Sentiment pie charts side by side
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    filtered_df["Sentiment"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")  # Hide y-label for aesthetics
    st.pyplot(fig)

with col2:
    st.markdown(
        "<br><br><br><span style='font-size:19px; font-style:italic;'>This pie chart shows the distribution of sentiment categories in the dataset (Positive, Neutral, and Negative). The percentages indicate the relative frequency of each sentiment.</span>",
        unsafe_allow_html=True,
    )

# Topic bar chart (Debugging added)
fig, ax = plt.subplots()
st.markdown("### 📊 Topic Frequency Table")
topic_counts = filtered_df["Topic"].value_counts()

styled_table = topic_counts.reset_index()
styled_table.columns = ["Topic", "Count"]
st.dataframe(
    styled_table.style.set_properties(
        **{
            "background-color": "#f0f2f6",
            "color": "#333333",
            "border-color": "black",
            "text-align": "center",
        }
    ).set_table_styles(
        [
            {
                "selector": "th",
                "props": [("font-size", "14px"), ("text-align", "center")],
            },
            {"selector": "td", "props": [("text-align", "center")]},
        ]
    )
)

if not topic_counts.empty:
    topic_counts.plot.bar(ax=ax)
    st.pyplot(fig)
else:
    st.write("No data available for the selected sentiment.")

# Clear Matplotlib cache
plt.close(fig)

# Word cloud
wordcloud = WordCloud(width=1600, height=800, background_color="white").generate(
    " ".join(filtered_df["Title"])
)
fig, ax = plt.subplots()
ax.imshow(wordcloud)
ax.axis("off")
st.pyplot(fig)

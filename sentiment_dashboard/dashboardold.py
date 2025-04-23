import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
# # import re
# # import nltk
# # from nltk.corpus import stopwords

# st.write("Hello, Streamlit is working!")
# st.balloons()
st.snow()
# Dashboard title
st.title("Social Media Sentiment Dashboard")

# Load processed data
df = pd.read_csv("./processed_posts.csv")

# df["Sentiment"]

# Sentiment filter
sentiment_filter = st.selectbox(
    "Select Sentiment", ["All", "POSITIVE", "NEGATIVE", "NEUTRAL"]
)
if sentiment_filter != "All":
    df = df[df["Sentiment"] == sentiment_filter]

# Display data table
st.write(df[["Time", "Title", "Sentiment", "Topic"]])

# Sentiment pie charts side by side
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    df["Sentiment"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")  # Hide y-label for aesthetics
    st.pyplot(fig)

with col2:
    st.markdown(
        "<br><br><br><span style='font-size:19px; font-style:italic;'>This pie chart shows the distribution of sentiment categories in the dataset (Positive, Neutral, and Negative). The percentages indicate the relative frequency of each sentiment.</span>",
        unsafe_allow_html=True,
    )

# with col2:
#     fig2, ax2 = plt.subplots()
#     wedges, texts, autotexts = ax2.pie(
#         df["Sentiment"].value_counts(),
#         autopct="%1.1f%%",
#         startangle=90,
#         wedgeprops=dict(width=0.4),
#     )
#     ax2.legend(
#         wedges,
#         df["Sentiment"].value_counts().index,
#         title="Sentiment",
#         loc="center left",
#         bbox_to_anchor=(1, 0, 0.5, 1),
#     )
#     ax2.axis("equal")
#     st.pyplot(fig2)
#     st.markdown(
#         "This doughnut-style chart represents the same sentiment distribution with an added legend for clarity. It's useful for a quick visual comparison between sentiment proportions."
#     )

# Topic bar chart (Debugging added)
fig, ax = plt.subplots()
topic_counts = df["Topic"].value_counts()

# Debugging: Print topic data before plotting
# st.write("Filtered DataFrame (first 5 rows):")
# st.write(df.head())

st.write("Topic Counts:")
st.write(topic_counts)

if not topic_counts.empty:
    topic_counts.plot.bar(ax=ax)
    st.pyplot(fig)
else:
    st.write("No data available for the selected sentiment.")

# Clear Matplotlib cache
plt.close(fig)

# Word cloud
wordcloud = WordCloud().generate(" ".join(df["Title"]))
fig, ax = plt.subplots()
ax.imshow(wordcloud)
ax.axis("off")
st.pyplot(fig)

# New Code

# st.title("Social Media Sentiment Dashboard")
# st.set_page_config(layout="wide")

# df = pd.read_csv("../processed_posts.csv")
# df["Time"] = pd.to_datetime(df["Time"])

# sentiment_filter = st.selectbox(
#     "Select Sentiment", ["All", "POSITIVE", "NEGATIVE", "NEUTRAL"]
# )
# topic_filter = st.multiselect(
#     "Select Topics", df["Topic"].unique(), default=df["Topic"].unique()
# )

# filtered_df = df.copy()
# if sentiment_filter != "All":
#     filtered_df = filtered_df[filtered_df["Sentiment"] == sentiment_filter]
# if topic_filter:
#     filtered_df = filtered_df[filtered_df["Topic"].isin(topic_filter)]

# st.write(filtered_df[["Time", "Title", "Sentiment", "Topic"]])

# fig, ax = plt.subplots(figsize=(10, 6))
# filtered_df.groupby(
#     [pd.Grouper(key="Time", freq="H"), "Sentiment"]
# ).size().unstack().plot(ax=ax)
# ax.set_title("Sentiment Distribution by Hour")
# st.pyplot(fig)

# fig, ax = plt.subplots()
# filtered_df["Sentiment"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
# st.pyplot(fig)

# fig, ax = plt.subplots()
# filtered_df["Topic"].value_counts().plot.bar(ax=ax)
# st.pyplot(fig)

# wordcloud = WordCloud().generate(" ".join(filtered_df["Title"]))
# fig, ax = plt.subplots()
# ax.imshow(wordcloud)
# ax.axis("off")
# st.pyplot(fig)

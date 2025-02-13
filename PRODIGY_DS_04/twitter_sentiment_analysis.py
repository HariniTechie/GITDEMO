# Import necessary libraries
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Step 1: Load the dataset
# Replace 'twitter_training.csv' with the actual path to your file
df = pd.read_csv('twitter_training.csv', header=None)

# Add column names (based on the dataset description)
df.columns = ['Tweet ID', 'Entity', 'Sentiment', 'Tweet Text']

# Display the first few rows
print("First few rows of the dataset:")
print(df.head())

# Step 2: Explore the dataset
# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Drop rows with missing values (if any)
df = df.dropna()

# Check the distribution of sentiments
print("\nSentiment distribution:")
print(df['Sentiment'].value_counts())

# Step 3: Preprocess the data
# Function to clean tweet text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Apply the cleaning function to the 'Tweet Text' column
df['Cleaned Text'] = df['Tweet Text'].apply(clean_text)

# Display the cleaned text
print("\nSample cleaned text:")
print(df[['Tweet Text', 'Cleaned Text']].head())

# Step 4: Analyze sentiment distribution
# Group by Entity and Sentiment
sentiment_distribution = df.groupby(['Entity', 'Sentiment']).size().unstack(fill_value=0)

# Display the sentiment distribution
print("\nSentiment distribution by entity:")
print(sentiment_distribution)

# Step 5: Visualize sentiment patterns
# Set the style for visualizations
sns.set(style="whitegrid")

# 1. Bar Chart: Sentiment Distribution by Entity
plt.figure(figsize=(12, 8))
sns.countplot(data=df, x='Entity', hue='Sentiment', palette='viridis')
plt.title('Sentiment Distribution by Entity')
plt.xticks(rotation=45)
plt.savefig('sentiment_distribution.png')  # Save the bar chart
plt.show()

# 2. Pie Chart: Overall Sentiment Distribution
plt.figure(figsize=(6, 6))
df['Sentiment'].value_counts().plot.pie(autopct='%1.1f%%', colors=['green', 'red', 'blue', 'orange'])
plt.title('Overall Sentiment Distribution')
plt.savefig('overall_sentiment.png')  # Save the pie chart
plt.show()

# 3. Word Cloud: Most Frequent Words in Positive and Negative Tweets
# Positive Tweets
positive_tweets = df[df['Sentiment'] == 'Positive']['Cleaned Text'].str.cat(sep=' ')
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_tweets)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Positive Tweets')
plt.savefig('positive_wordcloud.png')  # Save the word cloud for positive tweets
plt.show()

# Negative Tweets
negative_tweets = df[df['Sentiment'] == 'Negative']['Cleaned Text'].str.cat(sep=' ')
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_tweets)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Negative Tweets')
plt.savefig('negative_wordcloud.png')  # Save the word cloud for negative tweets
plt.show()

print("Visualizations saved as PNG files.")
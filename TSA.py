# import the necessary libraries
import os
import pandas as pd
import string
import unicodedata
import re
import nltk
from indicnlp.tokenize import indic_tokenize
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize, TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
import easyocr
import os
import re
import pickle
from langdetect import detect





nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')



# TO LOAD THE EXCEL FILES AND DISPLAY ITS HEAD.

# Path to the folder containing the excel files
folder_path = r'C:\Users\Admin\Desktop\end sem\INPUT'  # Replace with your actual folder path

# Get a list of all CSV files in the folder
excel_files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx'))]

# Check if there are at least 3 excel files in the folder
if len(excel_files) < 3:
    print("There are less than 3 excel files in the folder. Please add more files.")
else:
    # Read the first 3 excel files and assign them to separate variables
    try:
        # Read and assign the first excel file to df1
        df1 = pd.read_excel(os.path.join(folder_path, excel_files[0]))
        print(f"Data from {excel_files[0]} (First 5 rows):")
        print(df1.head(5))
        print("\n" + "=" * 50 + "\n")

        # Read and assign the second CSV file to df2
        df2 = pd.read_excel(os.path.join(folder_path, excel_files[1]))
        print(f"Data from {excel_files[1]} (First 5 rows):")
        print(df2.head(5))
        print("\n" + "=" * 50 + "\n")

        # Read and assign the third CSV file to df3
        df3 = pd.read_excel(os.path.join(folder_path, excel_files[2]))
        print(f"Data from {excel_files[2]} (First 5 rows):")
        print(df3.head(5))
        print("\n" + "=" * 50 + "\n")

    except Exception as e:
        print(f"Error reading files: {e}")

# Now, you can use df1, df2, or df3 for further operations



df1.info()
df2.info()
df3.info()


df1.describe()
df2.describe()
df3.describe()



print(df1.describe(include='all'))
print(df2.describe(include='all'))
print(df3.describe(include='all'))



print(df1['LABEL'].value_counts())
print(df2['LABEL'].value_counts())
print(df3['LABEL'].value_counts())




# Assuming df1, df2, df3 are already loaded and each has a 'LABEL' column
dataframes = [df1, df2, df3]  # List of DataFrames (you can add more here if needed)
file_names = ['df1', 'df2', 'df3']  # List of the corresponding file names

# Create a figure with subplots (1 row, len(dataframes) columns)
fig, axes = plt.subplots(1, len(dataframes), figsize=(15, 6))

# Set a default value for axes if there's only one subplot (to avoid errors)
if len(dataframes) == 1:
    axes = [axes]

# Iterate through the DataFrames and create plots
for i, (df, ax) in enumerate(zip(dataframes, axes)):
    # Sentiment Distribution Calculation
    sentiment_counts = df['LABEL'].value_counts()

    # Mapping sentiment values to labels
    sentiment_labels = {1: 'Positive', 0: 'Negative', 2: 'Neutral'}
    sentiment_counts.index = sentiment_counts.index.map(sentiment_labels)

    # Plotting Pie Chart
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#F66900', '#19D9FF', '#6653B9'])
    ax.set_title(f'Sentiment Distribution ({file_names[i]})', fontsize=14)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Adjust layout for better spacing between subplots
plt.tight_layout()
plt.show()

# Optionally, for Bar Graphs as well:
# Create a figure with subplots for bar graphs (1 row, len(dataframes) columns)
fig, axes = plt.subplots(1, len(dataframes), figsize=(15, 6))

# Set a default value for axes if there's only one subplot (to avoid errors)
if len(dataframes) == 1:
    axes = [axes]

# Iterate through the DataFrames and create bar plots
for i, (df, ax) in enumerate(zip(dataframes, axes)):
    # Sentiment Distribution Calculation
    sentiment_counts = df['LABEL'].value_counts()

    # Mapping sentiment values to labels
    sentiment_labels = {1: 'Positive', 0: 'Negative', 2: 'Neutral'}
    sentiment_counts.index = sentiment_counts.index.map(sentiment_labels)

    # Plotting Bar Graph
    sentiment_counts.plot(kind='bar', ax=ax, color=['#F66900', '#19D9FF', '#6653B9'])
    ax.set_title(f'Sentiment Distribution ({file_names[i]})', fontsize=14)
    ax.set_xlabel('Sentiment', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)

    # Set the xticks with both positions and labels
    ticks = ax.get_xticks()  # Get the current tick positions
    labels = sentiment_counts.index  # Your existing labels
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=12)

# Adjust layout for better spacing between subplots
plt.tight_layout()
plt.show()















# Define tokenizers
hindi_marathi_tokenizer = TweetTokenizer()

# Path to the folder containing the excel files
folder_path = r'C:\Users\Admin\Desktop\end sem\INPUT'  # Replace with actual folder path

# Marathi Stopwords
marathi_stopwords = set([
    'तुमचे', 'आहे', 'तुमची', 'तुम्ही', 'आहात', 'माझे', 'तुमच्या', 'तुमचा', 'त्या' , 'त्याला'
     'आहेत', 'तिच्या', 'त्याचे', 'तो', 'मी', 'हे', 'तिने', 'माझ्या', 'तुम्हाला' , 'तुमचं' , 'मला' , 'आहेत' , 'माझ्याकडे',
    'तिला', 'मला', 'हे', 'तुझ्या', 'माझे', 'त्याचा', 'माझा', 'माझी' , 'आम्ही' , 'त्याचं' , 'त्याने' , 'त्याच्या'
])

# Hindi Stopwords
hindi_stopwords = set([
    'आपका', 'है', 'आपकी',  'यह', 'आपके', 'आपने', 'आपमें', 'मैं', 'हूँ', 'वह', 'उनका' , 'उनकी' , 'आपको' , 'उसको' , 'उनको'
    'वे', 'मैंने', 'उसकी', 'मुझे', 'इससे', 'आपसे' , 'आप' , 'आपमे' , 'हैं' , 'उन्होंने ' , 'उसने' , 'हम' , 'यहाँ' , 'वहाँ',
    'आप', 'मुझे'
])


english_stopwords = set([
    'your' , 'a' , 'is' , 'the' , 'my' , 'you' , 'me' , 'my' , 'this' , 'that' , 'i' , 'are' , 'an' , 'by' , 'for' ,
    'it', 'of' , 'to' , 'her' , 'him' , 'his' , 'here' , 'there' , 'their' , 'they' , 'she' , 'he' , 'at' , 'in' ,
    'has' , 'have' , 'had' , 'on' , 'was' , 'were', 'be'
 ])


# Get a list of all excel files in the folder
excel_files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx'))]

# Function to remove punctuation from tokenized words (including '|')
def remove_punctuation(tokens):
    return [word for word in tokens if word not in string.punctuation and word != '।']

# Function to remove stopwords based on language
def remove_stopwords(tokens, language='english'):
    if language == 'english':
        #stop_words = set(stopwords.words('english'))
        stop_words = english_stopwords
        return [word for word in tokens if word.lower() not in stop_words]
    elif language == 'marathi':
        return [word for word in tokens if word.lower() not in marathi_stopwords]
    elif language == 'hindi':
        return [word for word in tokens if word.lower() not in hindi_stopwords]
    else:
        return tokens

# Loop through each file and process
for file in excel_files:
    file_path = os.path.join(folder_path, file)

    try:
        # Read the CSV file
        df = pd.read_excel(file_path)

        # Assuming the first column contains text, rename for consistency
        df.columns = ['Text', 'Label']  # Adjust column names if necessary

        print(f"Processing file: {file}\n")

        # Tokenization logic and language-based stopword removal
        if 'english' in file.lower():
            df['Tokens'] = df['Text'].astype(str).apply(word_tokenize)  # Tokenize English
            print(f"Tokenized English Text (First 5 rows):\n{df[['Text', 'Tokens']].head()}\n")
            df['Tokens'] = df['Tokens'].apply(lambda tokens: remove_stopwords(tokens, language='english'))
        elif 'marathi' in file.lower():
            df['Tokens'] = df['Text'].astype(str).apply(hindi_marathi_tokenizer.tokenize)  # Tokenize Marathi
            print(f"Tokenized Marathi Text (First 5 rows):\n{df[['Text', 'Tokens']].head()}\n")
            df['Tokens'] = df['Tokens'].apply(lambda tokens: remove_stopwords(tokens, language='marathi'))
        elif 'hindi' in file.lower():
            df['Tokens'] = df['Text'].astype(str).apply(hindi_marathi_tokenizer.tokenize)  # Tokenize Hindi
            print(f"Tokenized Hindi Text (First 5 rows):\n{df[['Text', 'Tokens']].head()}\n")
            df['Tokens'] = df['Tokens'].apply(lambda tokens: remove_stopwords(tokens, language='hindi'))

        # Remove punctuation from the tokens
        df['Tokens'] = df['Tokens'].apply(remove_punctuation)
        print(f"Tokens after Removing Punctuation (First 5 rows):\n{df[['Text', 'Tokens']].head()}\n")

        # Print tokens after removing stopwords
        print(f"Tokens after Removing Stopwords (First 5 rows):\n{df[['Text', 'Tokens']].head()}\n")

        # Convert tokens back to a string
        df['Cleaned_Text'] = df['Tokens'].apply(lambda x: ' '.join(x))

        # Bag of Words (BoW)
        vectorizer = CountVectorizer()
        X_bow = vectorizer.fit_transform(df['Cleaned_Text'])
        bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())
        final_bow_df = pd.concat([df[['Text', 'Label']], bow_df], axis=1)

        # TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        X_tfidf = tfidf_vectorizer.fit_transform(df['Cleaned_Text'])
        tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        final_tfidf_df = pd.concat([df[['Text', 'Label']], tfidf_df], axis=1)

        # Print sample outputs
        print(f"Bag of Words Representation (First 5 rows):\n{final_bow_df.head()}\n")
        print(f"TF-IDF Representation (First 5 rows):\n{final_tfidf_df.head()}\n")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"Failed to read or process {file}: {e}")





# WORDCLOUD FOR ENGLISH

# Path to the English file (replace with actual path)
english_file_path = r'C:\Users\Admin\Desktop\end sem\INPUT\ENGLISH.xlsx'  # Replace with actual file path

# Read the English CSV file
english_df = pd.read_excel(english_file_path)

# Assuming the first column contains text (replace with actual column name if needed)
text_column = english_df.columns[0]  # Adjust if necessary
english_text = ' '.join(english_df[text_column].dropna())  # Join all rows of text into one string

# Tokenization and stopword removal
english_stopwords = set(stopwords.words('english'))
tokens = word_tokenize(english_text)
cleaned_tokens = [word for word in tokens if word.lower() not in english_stopwords and word not in string.punctuation]

# Join cleaned tokens back into a single string
cleaned_english_text = ' '.join(cleaned_tokens)

# Generate Word Cloud for English text
wordcloud_english = WordCloud(width=800, height=400, background_color='white').generate(cleaned_english_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_english, interpolation='bilinear')
plt.axis('off')
plt.title('English Word Cloud', fontsize=16)
plt.show()










# WORDCLOUD FOR HINDI

# Path to the Hindi file (replace with actual path)
hindi_file_path = r'C:\Users\Admin\Desktop\end sem\INPUT\HINDI.xlsx'  # Replace with actual file path

# Read the Hindi CSV file
hindi_df = pd.read_excel(hindi_file_path)

# Assuming the first column contains text (replace with actual column name if needed)
text_column = hindi_df.columns[0]  # Adjust if necessary
hindi_text = ' '.join(hindi_df[text_column].dropna())  # Join all rows of text into one string

# Hindi Stopwords
hindi_stopwords = set([
    'आपका', 'है', 'आपकी',  'यह', 'आपके', 'आपने', 'आपमें', 'मैं', 'हूँ', 'वह', 'उनका' , 'उनकी' , 'आपको' , 'उसको' , 'उनको'
    'वे', 'मैंने', 'उसकी', 'मुझे', 'इससे', 'आपसे' , 'आप' , 'आपमे' , 'हैं' , 'उन्होंने ' , 'उसने' , 'हम' , 'यहाँ' , 'वहाँ',
    'आप', 'मुझे'
])

# Tokenization and stopword removal
tokens = word_tokenize(hindi_text)
cleaned_tokens = [word for word in tokens if word.lower() not in hindi_stopwords and word not in string.punctuation]

# Join cleaned tokens back into a single string
cleaned_hindi_text = ' '.join(cleaned_tokens)

# Generate Word Cloud for Hindi text
wordcloud_hindi = WordCloud(width=800, height=400,font_path=r'C:\Users\Admin\Desktop\end sem\INPUT\Mangal Regular.ttf', background_color='white').generate(cleaned_hindi_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_hindi, interpolation='bilinear')
plt.axis('off')
plt.title('Hindi Word Cloud', fontsize=16)
plt.show()









# WORDCLOUD FOR MARATHI

# Path to the Marathi file (replace with actual path)
marathi_file_path = r'C:\Users\Admin\Desktop\end sem\INPUT\MARATHI.xlsx'  # Replace with actual file path

# Read the Marathi excel file
marathi_df = pd.read_excel(marathi_file_path)

# Assuming the first column contains text (replace with actual column name if needed)
text_column = marathi_df.columns[0]  # Adjust if necessary
marathi_text = ' '.join(marathi_df[text_column].dropna())  # Join all rows of text into one string

# Marathi stopwords (can be customized if needed)
marathi_stopwords = set([
    'तुमचे', 'आहे', 'तुमची', 'तुम्ही', 'आहात', 'माझे', 'तुमच्या', 'तुमचा', 'त्या' , 'त्याला'
     'आहेत', 'तिच्या', 'त्याचे', 'तो', 'मी', 'हे', 'तिने', 'माझ्या', 'तुम्हाला' , 'तुमचं' , 'मला' , 'आहेत' , 'माझ्याकडे',
    'तिला', 'मला', 'हे', 'तुझ्या', 'माझे', 'त्याचा', 'माझा', 'माझी' , 'आम्ही' , 'त्याचं' , 'त्याने' , 'त्याच्या'
])

# Tokenization and stopword removal
tokens = word_tokenize(marathi_text)
cleaned_tokens = [word for word in tokens if word.lower() not in marathi_stopwords and word not in string.punctuation]

# Join cleaned tokens back into a single string
cleaned_marathi_text = ' '.join(cleaned_tokens)

# Generate Word Cloud for Marathi text
wordcloud_marathi = WordCloud(width=800, height=400, font_path=r'C:\Users\Admin\Desktop\end sem\INPUT\Mangal Regular.ttf', background_color='white').generate(cleaned_marathi_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_marathi, interpolation='bilinear')
plt.axis('off')
plt.title('Marathi Word Cloud', fontsize=16)
plt.show()






# RESULT MATRICES WITH CONFUSION MATRIX FOR ALL


# Function to load and preprocess data from Excel files
def load_data_from_excel(file_path):
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Assuming the Excel sheet has 'text' and 'label' columns
        if 'SENTENCE' in df.columns and 'LABEL' in df.columns:
            return df
        else:
            print(f"Columns 'SENTENCE' and 'LABEL' not found in {file_path}")
            return None
    except Exception as e:
        print(f"Error loading data from Excel: {e}")
        return None

# Function for training and evaluating models
def train_and_evaluate_models(X_train, y_train, X_test, y_test, file_name):
    # here we are using 5 algorithms models for taring and testing the data for sentiment analysis
    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    # giving colors to algorithms to differentiate the algorithms results such that it is to visulaize too.
    color_map = {
        "Logistic Regression": 'Reds',
        "SVM": 'Grays',
        "Naive Bayes": 'Purples',
        "Decision Tree": 'Blues',
        "Random Forest": 'Oranges'
    }

    print(f"\nResults for {file_name}:")

    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"\n{model_name} Accuracy: {accuracy}")
            print(f"Classification Report for {model_name}:")
            print(classification_report(y_test, y_pred))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"Confusion Matrix for {model_name}:")
            print(cm)

            # Optional: Visualize confusion matrix with a custom color map
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(cmap=color_map[model_name])  # Set the color based on the model
            plt.title(f"Confusion Matrix: {model_name}")
            plt.show()

        except Exception as e:
            print(f"Error with {model_name}: {e}")

# Main function to run the process
def main():
    try:
        # Folder containing Excel files
        folder_path = r'C:\Users\Admin\Desktop\end sem\INPUT'  # Replace with your folder path containing Excel files
        for filename in os.listdir(folder_path):
            if filename.endswith(".xlsx"):
                file_path = os.path.join(folder_path, filename)

                # Load data from each Excel file
                df = load_data_from_excel(file_path)

                if df is not None:
                    # Preprocessing
                    print(f"Data loaded successfully from {filename}. Starting preprocessing...")

                    # TF-IDF Vectorization
                    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
                    X = tfidf.fit_transform(df['SENTENCE'])

                    # Encoding labels (Languages or Sentiment labels)
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(df['LABEL'])

                    # Splitting into train and test sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Train and evaluate models for each file
                    train_and_evaluate_models(X_train, y_train, X_test, y_test, filename)

    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()








# getting the model ready to use as pkl models are getting ready
# Function for basic text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Function to load and preprocess data from Excel files
def load_data_from_excel(file_path):
    try:
        df = pd.read_excel(file_path)

        if 'SENTENCE' in df.columns and 'LABEL' in df.columns:
            return df
        else:
            print(f"Columns 'SENTENCE' and 'LABEL' not found in {file_path}")
            return None
    except Exception as e:
        print(f"Error loading data from Excel: {e}")
        return None

# Function for training and evaluating models
def train_and_evaluate_models(X_train, y_train, X_test, y_test, vectorizer, label_encoder, file_name):
    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    best_model = None
    best_accuracy = 0

    print(f"\nResults for {file_name}:")

    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"\n{model_name} Accuracy: {accuracy}")
            print(f"Classification Report for {model_name}:")
            print(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            print(f"Confusion Matrix for {model_name}:\n", cm)

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(cmap='Blues')
            plt.title(f"Confusion Matrix: {model_name}")
            plt.show()

            # Save the best performing model
            if accuracy > best_accuracy:
                best_model = model
                best_accuracy = accuracy

        except Exception as e:
            print(f"Error with {model_name}: {e}")

    # Save the best model and vectorizer
    if best_model:
        with open("sentiment_model.pkl", "wb") as model_file:
            pickle.dump(best_model, model_file)

        with open("vectorizer.pkl", "wb") as vec_file:
            pickle.dump(vectorizer, vec_file)

        with open("label_encoder.pkl", "wb") as enc_file:
            pickle.dump(label_encoder, enc_file)

        print(f"Best model ({best_model.__class__.__name__}) saved successfully.")

# Main function to train and evaluate
def main():
    try:
        folder_path = r'C:\Users\Admin\Desktop\end sem\INPUT'  # Update with your folder path
        for filename in os.listdir(folder_path):
            if filename.endswith(".xlsx"):
                file_path = os.path.join(folder_path, filename)
                df = load_data_from_excel(file_path)

                if df is not None:
                    print(f"Data loaded successfully from {filename}. Starting preprocessing...")

                    df['SENTENCE'] = df['SENTENCE'].apply(preprocess_text)

                    tfidf = TfidfVectorizer(stop_words=None, max_features=1000)
                    X = tfidf.fit_transform(df['SENTENCE'])

                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(df['LABEL'])

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    train_and_evaluate_models(X_train, y_train, X_test, y_test, tfidf, label_encoder, filename)

    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()





# for text testing 
# now this code is perfect to test the model

# Function for basic text preprocessing (same as training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Function to predict sentiment of unseen text
def predict_sentiment(text):
    try:
        # Load saved model and vectorizer
        with open("sentiment_model.pkl", "rb") as model_file:
            loaded_model = pickle.load(model_file)

        with open("vectorizer.pkl", "rb") as vec_file:
            loaded_vectorizer = pickle.load(vec_file)

        # Define a mapping for numeric predictions to sentiment labels
        sentiment_mapping = {0: "negative", 1: "positive", 2: "neutral"}

        processed_text = preprocess_text(text)
        vectorized_text = loaded_vectorizer.transform([processed_text])
        prediction = loaded_model.predict(vectorized_text)

        return sentiment_mapping.get(prediction[0], "Unknown")

    except Exception as e:
        return f"त्रुटि: {e}"  # Error message in Hindi/Marathi

# Function to print sentiment in selected language
def print_sentiment(language, text, messages):
    sentiment = predict_sentiment(text)

    if language in messages:
        print(f"\n{messages[language]['output'].format(messages[language].get(sentiment, 'Unknown'))}")
        print(messages[language]['sentence'].format(messages[language].get(sentiment, 'Unknown')))

if __name__ == "__main__":
    # Language-specific messages
    messages = {
        "hindi": {
            "positive": "सकारात्मक",
            "negative": "नकारात्मक",
            "neutral": "तटस्थ",
            "output": "पूर्वानुमानित भावना: {}",
            "sentence": "दिया गया वाक्य {} है।",
            "exit_prompt": "\nवाक्य प्रविष्ट करें (बंद करने के लिए 'exit' टाइप करें): "
        },
        "marathi": {
            "positive": "सकारात्मक",
            "negative": "नकारात्मक",
            "neutral": "तटस्थ",
            "output": "भाकित भावना: {}",
            "sentence": "दिलेले वाक्य {} आहे।",
            "exit_prompt": "\nवाक्य प्रविष्ट करा (बंद करण्यासाठी 'exit' टाइप करा): "
        },
        "english": {
            "positive": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
            "output": "Predicted Sentiment: {}",
            "sentence": "The given sentence is {}.",
            "exit_prompt": "\nEnter a sentence for sentiment analysis (or type 'exit' to quit): "
        }
    }

    # Keep asking until a valid language is selected
    while True:
        print("Select Language: English / हिंदी / मराठी")
        language = input("Enter language (english/hindi/marathi): ").strip().lower()

        if language in messages:
            break
        else:
            print("Not a valid option. Try again!\n")

    # Start sentiment analysis loop
    while True:
        sample_text = input(messages[language]["exit_prompt"])

        if sample_text.lower() == 'exit':
            break
        print_sentiment(language, sample_text, messages)




# for images tesing after detecting the languages

# Initialize the OCR reader with English ('en'), Hindi ('hi'), and Marathi ('mr')
reader = easyocr.Reader(['en', 'hi', 'mr'])  # Specify languages: 'en' for English, 'hi' for Hindi, 'mr' for Marathi

# Folder containing the images
folder_path = r"C:\Users\Admin\Desktop\end sem\INPUT\IN"  # Update with the path to your folder containing images

# Check if folder exists and contains files
if not os.path.exists(folder_path):
    print(f"Error: The folder path '{folder_path}' does not exist.")
else:
    print(f"Processing images in folder: {folder_path}")

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        # Convert filename to lowercase and check for image extensions (case-insensitive)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")  # Print the image being processed

            try:
                # Read text from the image
                result = reader.readtext(image_path)

                # Check if text was detected
                if not result:
                    print(f"No text detected in {filename}")
                else:
                    # Join all the detected text from the image
                    text = " ".join([detection[1] for detection in result])
                    print(f"Detected text from {filename}: {text}")

                    # Detect the language of the extracted text
                    try:
                        detected_language = detect(text)
                        print(f"Detected language: {detected_language}")
                    except Exception as e:
                        detected_language = "en"  # Default to English if detection fails
                        print(f"Error detecting language: {e}")

                    # Call the sentiment prediction function
                    # Pass the detected language and text to the print_sentiment function
                    if detected_language == 'en':
                        print_sentiment('english', text, messages)
                    elif detected_language == 'hi':
                        print_sentiment('hindi', text, messages)
                    elif detected_language == 'mr':
                        print_sentiment('marathi', text, messages)
                    else:
                        print("Unable to detect the sentiment for this language.")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
            print("-" * 50)  # Print a separator for readability
        else:
            print(f"Skipping non-image file: {filename}")


# Function for basic text preprocessing (same as training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Function to predict sentiment of unseen text
def predict_sentiment(text):
    try:
        # Load saved model and vectorizer
        with open("sentiment_model.pkl", "rb") as model_file:
            loaded_model = pickle.load(model_file)

        with open("vectorizer.pkl", "rb") as vec_file:
            loaded_vectorizer = pickle.load(vec_file)

        # Define a mapping for numeric predictions to sentiment labels
        sentiment_mapping = {0: "negative", 1: "positive", 2: "neutral"}

        processed_text = preprocess_text(text)
        vectorized_text = loaded_vectorizer.transform([processed_text])
        prediction = loaded_model.predict(vectorized_text)

        return sentiment_mapping.get(prediction[0], "Unknown")

    except Exception as e:
        return f"त्रुटि: {e}"  # Error message in Hindi/Marathi

# Function to print sentiment in selected language
def print_sentiment(language, text, messages):
    sentiment = predict_sentiment(text)

    if language in messages:
        print(f"\n{messages[language]['output'].format(messages[language].get(sentiment, 'Unknown'))}")
        print(messages[language]['sentence'].format(messages[language].get(sentiment, 'Unknown')))

# Language-specific messages
messages = {
    "hindi": {
        "positive": "सकारात्मक",
        "negative": "नकारात्मक",
        "neutral": "तटस्थ",
        "output": "पूर्वानुमानित भावना: {}",
        "sentence": "दिया गया वाक्य {} है।",
        "exit_prompt": "\nवाक्य प्रविष्ट करें (बंद करने के लिए 'exit' टाइप करें): "
    },
    "marathi": {
        "positive": "सकारात्मक",
        "negative": "नकारात्मक",
        "neutral": "तटस्थ",
        "output": "भाकित भावना: {}",
        "sentence": "दिलेले वाक्य {} आहे।",
        "exit_prompt": "\nवाक्य प्रविष्ट करा (बंद करण्यासाठी 'exit' टाइप करा): "
    },
    "english": {
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
        "output": "Predicted Sentiment: {}",
        "sentence": "The given sentence is {}.",
        "exit_prompt": "\nEnter a sentence for sentiment analysis (or type 'exit' to quit): "
    }
}









# now its time to go to the user interface for any kind of sentence testing or for image testing 

import easyocr
import io
import os
import re
import string
import pickle
from langdetect import detect

# ----------------------------
# Load the OCR reader
# ----------------------------
reader = easyocr.Reader(['en', 'hi', 'mr'])

# ----------------------------
# Language-specific messages
# ----------------------------
messages = {
    "hindi": {
        "positive": "सकारात्मक",
        "negative": "नकारात्मक",
        "neutral": "तटस्थ",
        "output": "पूर्वानुमानित भावना: {}",
        "sentence": "दिया गया वाक्य {} है।",
        "exit_prompt": "\nवाक्य प्रविष्ट करें (बंद करने के लिए 'exit' टाइप करें): "
    },
    "marathi": {
        "positive": "सकारात्मक",
        "negative": "नकारात्मक",
        "neutral": "तटस्थ",
        "output": "भाकित भावना: {}",
        "sentence": "दिलेले वाक्य {} आहे।",
        "exit_prompt": "\nवाक्य प्रविष्ट करा (बंद करण्यासाठी 'exit' टाइप करा): "
    },
    "english": {
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
        "output": "Predicted Sentiment: {}",
        "sentence": "The given sentence is {}.",
        "exit_prompt": "\nEnter a sentence for sentiment analysis (or type 'exit' to quit): "
    }
}

# ----------------------------
# Text Preprocessing Function
# ----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# ----------------------------
# Sentiment Prediction Function
# ----------------------------
def predict_sentiment(text):
    try:
        with open("sentiment_model.pkl", "rb") as model_file:
            loaded_model = pickle.load(model_file)
        with open("vectorizer.pkl", "rb") as vec_file:
            loaded_vectorizer = pickle.load(vec_file)

        sentiment_mapping = {0: "negative", 1: "positive", 2: "neutral"}
        processed_text = preprocess_text(text)
        vectorized_text = loaded_vectorizer.transform([processed_text])
        prediction = loaded_model.predict(vectorized_text)

        return sentiment_mapping.get(prediction[0], "Unknown")

    except Exception as e:
        return f"त्रुटि: {e}"

# ----------------------------
# Analyze Sentiment from Text
# ----------------------------
def analyze_sentiment(text, lang=None):
    try:
        # detected_language = detect(text)
        detected_language = 'en' #lang if lang else detect(text)
        print(f"Detected language: {detected_language}")
    except Exception as e:
        detected_language = "en"
        print(f"Error detecting language: {e}")

    if detected_language == 'en':
        lang_key = 'english'
    elif detected_language == 'hi':
        lang_key = 'hindi'
    elif detected_language == 'mr':
        lang_key = 'marathi'
    else:
        print("Unsupported language for sentiment analysis.")
        return "Unsupported language for sentiment analysis."

    sentiment = predict_sentiment(text)

    if lang_key in messages:
        print(f"\n{messages[lang_key]['output'].format(messages[lang_key].get(sentiment, 'Unknown'))}")
        return(messages[lang_key]['sentence'].format(messages[lang_key].get(sentiment, 'Unknown')))
    else:
        return("No message template found for this language.")

# ----------------------------
# Analyze Sentiment from Image
# ----------------------------
def analyze_sentiment_from_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: The image path '{image_path}' does not exist.")
        return

    try:
        result = reader.readtext(image_path)

        if not result:
            print("No text detected in the image.")
            return

        # Combine all detected text
        detected_text = " ".join([detection[1] for detection in result])
        print(f"Detected text: {detected_text}")

        # Call the sentiment analyzer
        analyze_sentiment(detected_text)

    except Exception as e:
        print(f"Error processing image: {e}")

# Add this function to your maincode.py for more accurate language detection
def validate_language(text, expected_lang_code):
    """
    Validates if the provided text is in the expected language.
    
    Args:
        text (str): The text to validate
        expected_lang_code (str): The expected language code ('en', 'hi', 'mr')
        
    Returns:
        bool: True if text is in the expected language, False otherwise
    """
    try:
        # For English text, be more lenient as langdetect often misclassifies short English text
        if expected_lang_code == 'en':
            # Check if text contains mostly Latin characters (English)
            english_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-_'\"\n\t ")
            devanagari_chars = set(range(0x0900, 0x097F))  # Devanagari Unicode range
            
            # Count characters in different scripts
            eng_count = sum(1 for c in text if c in english_chars)
            dev_count = sum(1 for c in text if ord(c) in devanagari_chars)
            
            # If mostly English characters, consider it English
            if eng_count > (len(text) * 0.6):  # If more than 60% are English characters
                return True
            elif dev_count > (len(text) * 0.3):  # If more than 30% are Devanagari, it's not English
                return False
            else:
                return True  # Default to accepting if uncertain
                
        # For Hindi/Marathi, we can be more strict since they use Devanagari script
        elif expected_lang_code in ['hi', 'mr']:
            # For short texts, check character sets
            if len(text.strip()) < 20:
                english_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
                eng_count = sum(1 for c in text if c in english_chars)
                dev_count = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
                
                # If more Devanagari than English, it's likely Hindi/Marathi
                return dev_count > eng_count
            else:
                # For longer texts, use langdetect but be forgiving
                try:
                    detected_lang = detect(text)
                    # Hindi and Marathi can sometimes be confused with each other
                    if expected_lang_code == 'hi' and detected_lang in ['hi', 'mr', 'ne']:
                        return True
                    elif expected_lang_code == 'mr' and detected_lang in ['mr', 'hi', 'ne']:
                        return True
                    else:
                        return detected_lang == expected_lang_code
                except:
                    # If detection fails, check character set as fallback
                    dev_count = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
                    return dev_count > (len(text) * 0.5)
        else:
            # For other languages, default to langdetect
            detected_lang = detect(text)
            return detected_lang == expected_lang_code
    except Exception as e:
        print(f"Language validation error: {str(e)}")
        # If detection fails, default to accepting the input
        return True

# ----------------------------
# Example Usage
# ----------------------------
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def index():
    return "Hello!"

@app.route('/analyze-text', methods=['POST'])
def analyze():
    try:
        message = request.get_json()
        lang = message["language"]
        text = message["text"]
        
        # Convert frontend language names to language codes
        language_code_map = {
            'english': 'en',
            'hindi': 'hi',
            'marathi': 'mr'
        }
        
        lang_code = language_code_map.get(lang, lang)
        print(f"Text: {text}")
        print(f"Selected language: {lang_code}")
        
        # Skip language validation for very short texts
        if len(text.strip()) < 5:
            return analyze_sentiment(text, lang_code)
            
        # Use our improved validation function
        is_valid_language = validate_language(text, lang_code)
        print(f"Language validation result: {is_valid_language}")
        
        if not is_valid_language:
            # Language mismatch
            if lang_code == 'en':
                return "Error: Please enter English text only."
            elif lang_code == 'hi':
                return "त्रुटि: कृपया केवल हिंदी पाठ दर्ज करें।"
            elif lang_code == 'mr':
                return "त्रुटी: कृपया फक्त मराठी मजकूर प्रविष्ट करा."
            else:
                return f"Error: Please enter text in {lang} only."
        
        # If language matches, proceed with sentiment analysis
        return analyze_sentiment(text, lang_code)
        
    except Exception as e:
        print(f"Error analyzing text: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return {"error": "No image file provided"}, 400
            
        image = request.files['image']
        language = request.form.get('language', 'en').lower()

        # Map to language codes
        language_code_map = {
            'english': 'en',
            'hindi': 'hi',
            'marathi': 'mr'
        }
        lang_code = language_code_map.get(language, language)
        print(f"Language received: {language}, converted to code: {lang_code}")

        # Save image temporarily
        temp_path = "temp_image_" + str(os.getpid()) + ".png"
        image.save(temp_path)

        # OCR with all languages to get best match
        reader = easyocr.Reader(['en', 'hi', 'mr'])  # <-- Important: Use all here
        result = reader.readtext(temp_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Extract detected text
        extracted_text = ' '.join([text for _, text, _ in result])

        if not extracted_text.strip():
            return {
                "extracted_text": "",
                "sentiment": "No text detected in the image."
            }

        # ✅ Validate language using custom function
        is_valid_language = validate_language(extracted_text, lang_code)
        print(f"OCR Text: {extracted_text}")
        print(f"Language validation result: {is_valid_language}")

        if not is_valid_language:
            error_messages = {
                'en': "Error: Please upload image with English text only.",
                'hi': "त्रुटि: कृपया केवल हिंदी पाठ वाली छवि अपलोड करें।",
                'mr': "त्रुटी: कृपया फक्त मराठी मजकूर असलेली छवि अपलोड करा."
            }
            return {
                "extracted_text": extracted_text,
                "sentiment": error_messages.get(lang_code, "Error: Language mismatch.")
            }

        # ✅ Call sentiment analysis using user-selected language
        sentiment_result = analyze_sentiment(extracted_text, lang_code)
        return {
            "extracted_text": extracted_text,
            "sentiment": sentiment_result
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Error analyzing image: {str(e)}"}, 500




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# if __name__ == "__main__":
# From Text:
    # analyze_sentiment("मुझे यह फिल्म बहुत पसंद आई।")

# From Image:
    # analyze_sentiment_from_image("/content/INPUT/IN/h.PNG")

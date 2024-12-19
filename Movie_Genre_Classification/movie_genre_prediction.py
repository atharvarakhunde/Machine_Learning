import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

def load_data(file_path, is_train=True):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(":::")
            if is_train:
                if len(parts) == 4:
                    data.append({
                        'ID': parts[0].strip(),
                        'TITLE': parts[1].strip(),
                        'GENRE': parts[2].strip(),
                        'DESCRIPTION': parts[3].strip()
                    })
            else:
                if len(parts) == 3:
                    data.append({
                        'ID': parts[0].strip(),
                        'TITLE': parts[1].strip(),
                        'DESCRIPTION': parts[2].strip()
                    })
    return pd.DataFrame(data)

def load_test_solutions(file_path):
    solutions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(":::")
            if len(parts) >= 3:
                solutions.append({
                    'ID': parts[0].strip(),
                    'GENRE': parts[2].strip()
                })
    return pd.DataFrame(solutions)

train_data_path = "Movie_Genre_Prediction/Genre Classification Dataset/train_data.txt"
test_data_path = "Movie_Genre_Prediction/Genre Classification Dataset/test_data.txt"
test_solution_path = "Movie_Genre_Prediction/Genre Classification Dataset/test_data_solution.txt"

train_df = load_data(train_data_path, is_train=True)
test_df = load_data(test_data_path, is_train=False)
test_solution_df = load_test_solutions(test_solution_path)

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    text = text.lower()
    return text

train_df['DESCRIPTION'] = train_df['DESCRIPTION'].apply(clean_text)
train_df['GENRE'] = train_df['GENRE'].apply(clean_text)
test_df['DESCRIPTION'] = test_df['DESCRIPTION'].apply(clean_text)
test_solution_df['GENRE'] = test_solution_df['GENRE'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(train_df['DESCRIPTION'])
y = train_df['GENRE']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train_resampled, y_train_resampled)

y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))

test_X = vectorizer.transform(test_df['DESCRIPTION'])
test_predictions = model.predict(test_X)

test_df['GENRE_PRED'] = test_predictions
test_merged = test_df.merge(test_solution_df, on='ID', how='inner')

if test_merged['GENRE'].isnull().any():
    print("Warning: Missing genres in merged test_df. Check the test solution file formatting.")

test_accuracy = accuracy_score(test_merged['GENRE'], test_merged['GENRE_PRED'])
print("Test Accuracy:", test_accuracy)
print("Classification Report (Test):\n", classification_report(test_merged['GENRE'], test_merged['GENRE_PRED']))

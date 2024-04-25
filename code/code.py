import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


def load_dataset(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)


def explore_dataset(dataset):
    """Perform exploratory data analysis."""
    print("First few rows:")
    print(dataset.head())
    print("\nLast few rows:")
    print(dataset.tail())
    print("\nDataset shape:")
    print(dataset.shape)
    print("\nDataset information:")
    print(dataset.info())
    print("\nMissing values:")
    print(dataset.isnull().sum())


def visualize_class_distribution(dataset):
    """Visualize the class distribution."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Class', data=dataset, palette='Set1')
    plt.title('Class Distribution')
    plt.xlabel('Class (0: Non-Fraudulent, 1: Fraudulent)')
    plt.ylabel('Count')
    plt.show()


def visualize_amount_distribution(dataset):
    """Visualize the distribution of transaction amount."""
    plt.figure(figsize=(10, 6))
    sns.histplot(dataset['Amount'], bins=30, kde=True)
    plt.title('Distribution of Transaction Amount')
    plt.xlabel('Amount')
    plt.ylabel('Count')
    plt.show()


def visualize_time_vs_amount(dataset):
    """Visualize Time vs Amount by Class."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Time', y='Amount', data=dataset, hue='Class', palette='coolwarm')
    plt.title('Time vs Amount by Class')
    plt.xlabel('Time (in seconds)')
    plt.ylabel('Amount')
    plt.legend(title='Class')
    plt.show()


def visualize_correlation_heatmap(dataset):
    """Visualize the correlation heatmap."""
    plt.figure(figsize=(10, 8))
    corr = dataset.corr()
    corr.to_csv("correlation.csv", index=False)
    sns.heatmap(corr, cmap='viridis', annot=True)
    plt.title('Correlation Heatmap')
    plt.show()


def preprocess_data(dataset):
    """Perform data preprocessing."""
    scaler = StandardScaler()
    dataset['Amount'] = scaler.fit_transform(pd.DataFrame(dataset['Amount']))
    dataset = dataset.drop(['Time'], axis=1)
    dataset = dataset.drop_duplicates()
    return dataset


def prepare_data_for_model(dataset):
    """Prepare data for model training."""
    regular_trans = dataset[dataset['Class'] == 0]
    fraud_trans = dataset[dataset['Class'] == 1]
    sub_regular_trans = regular_trans.sample(n=473)
    updated_df = pd.concat([sub_regular_trans, fraud_trans], ignore_index=True)
    X = updated_df.drop('Class', axis=1)
    y = updated_df['Class']
    updated_df.to_csv("updatedDf.csv",index=False)
    return X, y


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Train and evaluate machine learning models."""
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        results[name] = {'Accuracy': accuracy, 'Recall': recall, 'F1 Score': f1, 'Precision': precision}

    return results


def visualize_model_comparison(results):
    """Visualize model comparison."""
    result_df = pd.DataFrame(results).T
    result_df.to_csv("modelcomparison.csv",index=False)
    result_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.legend(title='Metric')
    plt.xticks(rotation=0)
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset("creditcard.csv")

    # Exploratory Data Analysis
    explore_dataset(dataset)
    visualize_class_distribution(dataset)
    visualize_amount_distribution(dataset)
    visualize_time_vs_amount(dataset)
    visualize_correlation_heatmap(dataset)

    # Data Preprocessing
    processed_data = preprocess_data(dataset)

    # Prepare data for model training
    X, y = prepare_data_for_model(processed_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Train and Evaluate Models
    results = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    print(results)

    # Visualize Model Comparison
    visualize_model_comparison(results)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind

# Load the dataset
df = pd.read_csv('bank-full.csv', sep=';')

# Drop the specified columns
df.drop(columns=['contact', 'day', 'month', 'poutcome'], inplace=True)

# Check for missing values
print(df.isnull().sum())

# Basic statistics for numeric columns
print(df.describe())

# Check the distribution of the target variable
print(df['y'].value_counts())

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['job', 'marital', 'education', 'default', 'housing', 'loan'], drop_first=True)

# Encode the target variable
df_encoded['y'] = df_encoded['y'].map({'no': 0, 'yes': 1})

# Scale Numerical Features
scaler = StandardScaler()
numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# Separate features (X) and target (y)
X = df_encoded.drop(columns=['y'])
y = df_encoded['y']

# Split the dataset into training and test sets before applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize and apply SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Display new class counts in the training set
print("Class distribution after SMOTE:")
print(y_train_balanced.value_counts())

# Univariate Feature Analysis with Statistical Tests
def plot_kde_with_stats(X_train, y_train, features):
    data = pd.concat([X_train[features], y_train], axis=1)
    data['y'] = data['y'].map({0: 'No', 1: 'Yes'})

    plt.figure(figsize=(14, 10))
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 3, i)
        sns.kdeplot(data=data, x=feature, hue='y', fill=True)
        plt.title(f'Distribution of {feature} by Subscription')
    plt.tight_layout()
    plt.show()

    print("Statistical Tests (T-Test Results):\n")
    for feature in features:
        class_no = data[data['y'] == 'No'][feature]
        class_yes = data[data['y'] == 'Yes'][feature]
        t_stat, p_value = ttest_ind(class_no, class_yes, equal_var=False)
        significance = "Significant" if p_value < 0.05 else "Not Significant"
        print(f"Feature: {feature}, T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4e}, Result: {significance}\n")

plot_kde_with_stats(pd.DataFrame(X_train_balanced, columns=X_train.columns), y_train_balanced, numerical_cols)

# Box Plot for Feature Distribution
def plot_box_features(df, features):
    plt.figure(figsize=(14, 10))
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x='y', y=feature, data=df)
        plt.title(f'Box Plot of {feature} by Subscription')
    plt.tight_layout()
    plt.show()

# Pair Plot for Key Features
def plot_pairplot(df):
    sns.pairplot(df, vars=['age', 'balance', 'duration'], hue='y', diag_kind='kde')
    plt.suptitle('Pair Plot of Key Features', y=1.02)
    plt.show()

# Combine the balanced features and target into one DataFrame for visualization
df_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns)
df_balanced['y'] = y_train_balanced

# Execute the plots
plot_box_features(df_balanced, ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous'])
plot_pairplot(df_balanced)

# Correlation Matrix Heatmap
corr_matrix = df_balanced.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
plt.title('Correlation Matrix Heatmap Including One-Hot Encoded Variables')
plt.show()

# Clustering Analysis
X = df_balanced.drop('y', axis=1)

def plot_elbow_method(X):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

plot_elbow_method(X)

# Perform K-means clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)
df_balanced['Cluster'] = clusters

# Evaluate clustering with Silhouette Score
silhouette_avg = silhouette_score(X, clusters)
print(f'Silhouette Score: {silhouette_avg:.2f}')

# Visualize clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=50)
plt.title('Clusters Visualization with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Model Training and Evaluation
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"
    }

for model_name, metrics in results.items():
    print(f"\n{model_name} Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

# Hyperparameter Tuning with GridSearchCV for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
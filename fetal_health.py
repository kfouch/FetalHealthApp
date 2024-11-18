import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data (Cache data as it's static)
@st.cache_data
def load_data():
    df = pd.read_csv('fetal_health.csv')
    return df

# Train and Evaluate Models (Cache resource as it involves models)
def train_models(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    ab = AdaBoostClassifier(random_state=42)

    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    ab.fit(X_train, y_train)

    # Soft Voting Classifier with weighted averages
    models = [("Random Forest", rf), ("Decision Tree", dt), ("AdaBoost", ab)]
    f1_scores = [model.score(X_train, y_train) for _, model in models]
    weights = [score / sum(f1_scores) for score in f1_scores]

    svc = VotingClassifier(estimators=models, voting='soft', weights=weights)
    svc.fit(X_train, y_train)

    return rf, dt, ab, svc, weights

def compute_feature_importances(models, weights):
    # Check if models have feature_importances_ attribute
    feature_importances = np.zeros(models[0].feature_importances_.shape)
    for model, weight in zip(models, weights):
        feature_importances += weight * model.feature_importances_
    return feature_importances

def display_classification_report(y_test, y_pred):
    target_names = ["Normal", "Suspect", "Pathological"]
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Color-coding the classification report
    def color_coding(val):
        if isinstance(val, str):  # Ignore non-numeric entries
            return ""
        gradient = (val - report_df.drop("support", axis=1).values.min()) / (
                report_df.drop("support", axis=1).values.max() - report_df.drop("support", axis=1).values.min())
        color = plt.cm.viridis(gradient)[:3]  # Viridis colormap
        return f"background-color: rgba({color[0] * 255}, {color[1] * 255}, {color[2] * 255}, 0.6);"

    def color_support(val):
        gradient = (val - report_df["support"].min()) / (report_df["support"].max() - report_df["support"].min())
        color = plt.cm.viridis(gradient)[:3]  # Viridis colormap
        return f"background-color: rgba({color[0] * 255}, {color[1] * 255}, {color[2] * 255}, 0.6);"

    styled_df = report_df.style.applymap(color_coding, subset=["precision", "recall", "f1-score"]).applymap(
        color_support, subset=["support"])
    
    return styled_df

def display_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
                xticklabels=["Normal", "Suspect", "Pathological"], yticklabels=["Normal", "Suspect", "Pathological"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    return plt

def display_feature_importance(X_test, rf, dt, ab, weights):
    # Compute weighted feature importances
    importances = compute_feature_importances([rf, dt, ab], weights)
    importance_df = pd.DataFrame({"Feature": X_test.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)

    # Generate the feature importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
    plt.title("Feature Importance (Weighted)")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    return plt

# Streamlit App
def main():
    st.title("Fetal Health Classification App")
    
    # Display GIF Image at the top
    st.image("fetal_health_image.gif", width=700)

    # Sidebar for Input Features
    st.sidebar.header("Upload Input Data and Select Model")
    
    # File upload in the sidebar
    uploaded_file = st.sidebar.file_uploader("Upload CSV for Predictions", type=["csv"])
    model_choice = st.sidebar.selectbox(
        "Choose a model for classification",
        ("Random Forest", "Decision Tree", "AdaBoost", "Soft Voting Classifier")
    )

    # Load and Prepare Data
    df = load_data()
    st.sidebar.write("### Sample Data Preview")
    st.sidebar.dataframe(df.head())

    # Prepare Data
    X = df.drop(columns=["fetal_health"])
    y = df["fetal_health"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Models
    rf, dt, ab, svc, weights = train_models(X_train, y_train)

    # Handle user-uploaded data
    if uploaded_file:
        user_data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(user_data)

        # Use selected model for predictions
        if model_choice == "Random Forest":
            selected_model = rf
        elif model_choice == "Decision Tree":
            selected_model = dt
        elif model_choice == "AdaBoost":
            selected_model = ab
        elif model_choice == "Soft Voting Classifier":
            selected_model = svc

        # Make predictions using the selected model
        predictions = selected_model.predict(user_data)

        # Get prediction probabilities
        prediction_probs = selected_model.predict_proba(user_data)

        # Convert numeric predictions to text labels
        prediction_labels = ["Normal" if pred == 1 else "Suspect" if pred == 2 else "Pathological" for pred in predictions]
        user_data["Predicted Class"] = prediction_labels
        
        # Add a column with the probability of the predicted class
        user_data["Prediction Probability"] = [prob[selected_model.classes_.tolist().index(pred)] for prob, pred in zip(prediction_probs, predictions)]

        # Color the "Predicted Class" column
        def color_predicted_class(val):
            color_map = {
                "Normal": 'background-color: lime;',  # Normal
                "Suspect": 'background-color: yellow;',  # Suspect
                "Pathological": 'background-color: orange;'  # Pathological
            }
            return color_map.get(val, '')

        st.write("### Predictions on Uploaded Data")
        styled_df = user_data.style.applymap(color_predicted_class, subset=["Predicted Class"])
        st.dataframe(styled_df)

        # Generate predictions for the test set (y_pred)
        y_pred = selected_model.predict(X_test)

        # Display model results and visualizations
        st.write(f"## Results for {model_choice}")

        # Tabs for Classification Report, Confusion Matrix, and Feature Importance
        tab1, tab2, tab3 = st.tabs(["Classification Report", "Confusion Matrix", "Feature Importance"])

        with tab1:
            st.write("### Classification Report")
            st.dataframe(display_classification_report(y_test, y_pred))

        with tab2:
            st.write("### Confusion Matrix")
            st.pyplot(display_confusion_matrix(y_test, y_pred))

        with tab3:
            st.write("### Feature Importance")
            st.pyplot(display_feature_importance(X_test, rf, dt, ab, weights))

if __name__ == "__main__":
    main()
from helperdata import *
def execution():
    # Load dataset
    df = pd.read_csv("Customer_Sentiment_csv.csv")

    print("Dataset Loaded Successfully!")
    print("\nMissing Values:\n", df.isnull().sum())

    # 1 Sentiment Graph
    sentiment_distribution(df)

    # 2 Customer Rating Graph
    customer_rating_distribution(df)

    # 3 Complaint + Review Length Graph
    complaint_registered(df)

    # 4 Assign Issue Type
    df['issue_type'] = df['review_text'].apply(assign_issue_type)

    # 5 Issue Type Distribution Graph
    plot_issue_distribution(df)

    # Clean Text
    df['clean_text'] = df['review_text'].apply(clean_text)

    # 6 Train Model + Confusion Matrix Graph
    train_model(df)


# Run the system
if __name__ == "__main__":
    execution()
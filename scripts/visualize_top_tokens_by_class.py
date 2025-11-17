import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_token_freqs(texts):
    tokens = [token for msg in texts for token in msg.split()]
    return pd.Series(tokens).value_counts()

def main():
    df = pd.read_csv("datasets/processed/sms_spam_clean.csv")
    freqs_ham = get_token_freqs(df[df["0"] == "ham"]["text_clean"])
    freqs_spam = get_token_freqs(df[df["0"] == "spam"]["text_clean"])
    top_ham = freqs_ham.head(20)
    top_spam = freqs_spam.head(20)
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=False)
    colors = sns.color_palette("viridis", 20)
    top_ham[::-1].plot.barh(ax=axes[0], color=colors)
    axes[0].set_title("Top 20 Tokens: Ham")
    axes[0].set_xlabel("Frequency")
    axes[0].set_ylabel("Token")
    axes[0].tick_params(labelsize=10)
    top_spam[::-1].plot.barh(ax=axes[1], color=colors)
    axes[1].set_title("Top 20 Tokens: Spam")
    axes[1].set_xlabel("Frequency")
    axes[1].set_ylabel("")
    axes[1].tick_params(labelsize=10)
    plt.tight_layout()
    os.makedirs("reports/visualizations", exist_ok=True)
    plt.savefig("reports/visualizations/top_tokens_by_class.png", dpi=120)
    plt.close()
    print("Saved: reports/visualizations/top_tokens_by_class.png")

if __name__ == "__main__":
    main()

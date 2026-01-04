from churnguard.data import load_raw_csv

def main():
    df = load_raw_csv("telco_churn.csv")
    print("Loaded dataset:", df.shape)
    print(df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()

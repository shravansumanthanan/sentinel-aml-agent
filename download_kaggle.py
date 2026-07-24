import os
import shutil
import glob
import kagglehub

def download_and_ingest_kaggle_dataset():
    print("Downloading IBM Transactions for AML dataset from Kaggle via kagglehub...")
    path = kagglehub.dataset_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
    print("Path to dataset files:", path)

    target_dir = "/Users/sterlingsuman/Desktop/projectx/data"
    os.makedirs(target_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True) + glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        # Check all files in path
        csv_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]

    print(f"Found {len(csv_files)} CSV files in Kaggle download cache.")
    for csv_file in csv_files:
        dest = os.path.join(target_dir, "ibm_aml_transactions.csv")
        shutil.copy(csv_file, dest)
        print(f"✅ Copied {csv_file} -> {dest}")
        break

if __name__ == "__main__":
    download_and_ingest_kaggle_dataset()

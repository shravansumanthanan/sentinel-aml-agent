import os
import shutil
import glob
import kagglehub

def download_and_ingest_kaggle_dataset(target_dir: str = None):
    print("Downloading IBM Transactions for AML dataset from Kaggle via kagglehub...")
    path = kagglehub.dataset_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
    print("Path to dataset files:", path)

    # Default to <repo_root>/data next to this script
    if target_dir is None:
        target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(target_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True) + glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        csv_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]

    if not csv_files:
        print("ERROR: No CSV files found in the downloaded dataset.")
        return

    print(f"Found {len(csv_files)} CSV files in Kaggle download cache.")
    for csv_file in csv_files:
        dest = os.path.join(target_dir, os.path.basename(csv_file))
        shutil.copy(csv_file, dest)
        print(f"✅ Copied {csv_file} -> {dest}")

if __name__ == "__main__":
    download_and_ingest_kaggle_dataset()

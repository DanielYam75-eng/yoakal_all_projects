from dagshub import get_repo_bucket_client
from list_files import load_files
from upload_file import info_file_string
import argparse


def download(username, bucketname, filepath, key_name):
    existing_files = load_files(username, bucketname)
    if key_name not in existing_files.index:
        print(f"Key '{key_name}' not found in the bucket '{bucketname}'.")
        return

    key_row = existing_files.loc[key_name]
    key = info_file_string(key_name, key_row['Source'], key_row['Creation Date'], key_row['Template'])


    boto_client = get_repo_bucket_client(username + "/" + bucketname)
    boto_client.download_file(bucketname, key, filepath)

def main():
    parser = argparse.ArgumentParser(description="Download a file from a DagsHub bucket.")
    # Hardcoded username and bucketname
    username = 'yoacal.data.science'
    bucketname = 'exp-repo'

    parser.add_argument("key", type=str, help="Key of the file in the bucket")
    parser.add_argument("-o", "--output", type=str, help="Local path to save the downloaded file")


    args = parser.parse_args()

    download(username, bucketname, args.output, args.key)

if __name__ == "__main__":
    main()


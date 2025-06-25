from dagshub import get_repo_bucket_client
from list_files import load_files
from upload_file import info_file_string
import argparse


def download(username, bucketname, filepath, key):
    boto_client = get_repo_bucket_client(username + "/" + bucketname)
    boto_client.download_file(bucketname, key, filepath)

def main():
    parser = argparse.ArgumentParser(description="Download a file from a DagsHub bucket.")
    # Hardcoded username and bucketname
    username = 'yoacal.data.science'
    bucketname = 'new-repo'

    parser.add_argument("key", type=str, help="Key of the file in the bucket")
    parser.add_argument("-o", "--output", type=str, help="Local path to save the downloaded file")

    args = parser.parse_args()

    existing_files = load_files(username, bucketname)
    if args.key not in existing_files.index:
        print(f"Key '{args.key}' not found in the bucket '{bucketname}'.")
        return

    key_row = existing_files.loc[args.key]
    key = info_file_string(args.key, key_row['Source'], key_row['Creation Date'], key_row['Template'])

    download(username, bucketname, args.output, key)

if __name__ == "__main__":
    main()


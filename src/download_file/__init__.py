from dagshub import get_repo_bucket_client
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

    download(username, bucketname, args.filepath, args.key)

if __name__ == "__main__":
    main()


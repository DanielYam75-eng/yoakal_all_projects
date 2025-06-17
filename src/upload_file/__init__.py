from dagshub import get_repo_bucket_client
import argparse


def upload(username, bucketname, filepath, key):
    boto_client = get_repo_bucket_client(username + "/" + bucketname)
    boto_client.upload_file(bucketname, key, filepath)

def main():
    parser = argparse.ArgumentParser(description="upload a file to a DagsHub bucket.")
    # Hardcoded username and bucketname
    username = 'yoacal.data.science'
    bucketname = 'new-repo'

    parser.add_argument("key", type=str, help="Key of the file in the bucket")
    parser.add_argument("-i", "--input", type=str, help="local path of file to input")

    args = parser.parse_args()

    upload(username, bucketname, args.input, args.key)

if __name__ == "__main__":
    main()


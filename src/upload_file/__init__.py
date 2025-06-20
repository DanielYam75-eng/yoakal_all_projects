from dagshub import get_repo_bucket_client
import argparse


def upload(username, bucketname, filepath, key):
    boto_client = get_repo_bucket_client(username + "/" + bucketname)
    boto_client.upload_file(filepath, bucketname, key)

def info_file_string(source,creation_date,template):
    info_source_format = f"source={source}"
    info_source_creation_date = f"creation_date={creation_date}"
    info_source_template = f"template={template}"
    combined_file_info = f"my-dataset^{info_source_format}^{info_source_creation_date}^{info_source_template}"
    return combined_file_info

def main():
    parser = argparse.ArgumentParser(description="upload a file to a DagsHub bucket.")
    # Hardcoded username and bucketname
    username = 'yoacal.data.science'
    bucketname = 'new-repo'

    parser.add_argument("-i", "--input", type=str, help="local path of file to input")

    args = parser.parse_args()

    source = input("Enter source: ")
    creation_date = input("Enter creation date(format: YYYY-MM-DD): ")
    template = input("Enter template date: ")


    info_of_file_as_string = info_file_string(source,creation_date,template)
    upload(username, bucketname, args.input, info_of_file_as_string)

if __name__ == "__main__":
    main()
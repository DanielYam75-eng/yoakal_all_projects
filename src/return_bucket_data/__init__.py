import io
from dagshub import get_repo_bucket_client
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="read a file from a DagsHub bucket.")

    username = 'yoacal.data.science'
    bucketname = 'exp-repo'
        
    parser.add_argument("keyname", type=str, help="key name for the uploaded file") 

    args = parser.parse_args()

    boto_client = get_repo_bucket_client(username + "/" + bucketname)

    obj = boto_client.get_object(Bucket=bucketname, Key=args.keyname)
    data = pd.read_csv(io.BytesIO(obj['Body'].read()))
    print(data)
    
if __name__ == "__main__":
    main()
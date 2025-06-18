from dagshub import get_repo_bucket_client
import argparse
from datetime import datetime


def list_objects(username, bucketname):
    boto_client = get_repo_bucket_client(username + "/" + bucketname)
    response = boto_client.list_objects_v2(Bucket=bucketname)
    return response

def getting_contents_response(response):
    contents_response = response.get('Contents', [])
    contents_info = contents_response[0]
    return contents_info

def getting_tags_info(contents_info):
    name = contents_info.get('Key', '')
    date = contents_info.get('LastModified', None)
    size = contents_info.get('Size', 0)
    return name, date, size

def main():
    username = 'yoacal.data.science'
    bucketname = 'new-repo'

    response = list_objects(username, bucketname)
    contents_info = getting_contents_response(response)
    name, date, size = getting_tags_info(contents_info)

    formatted_dict = {
        'name': name,
        'date': date.isoformat(),
        'size': size
    }

    response['Contents'][0]['Key'] = formatted_dict #update tags
    print(response)

if __name__ == "__main__":
    main()
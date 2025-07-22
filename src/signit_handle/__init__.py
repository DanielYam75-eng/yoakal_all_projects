from dagshub import get_repo_bucket_client

username = 'yoacal.data.science'
bucketname = 'exp-repo'
boto_client = get_repo_bucket_client(username + "/" + bucketname)

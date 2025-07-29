from dagshub import get_repo_bucket_client
from httpx import ConnectError
import sys

username = 'yoacal.data.science'
bucketname = 'exp-repo'
try:
    boto_client = get_repo_bucket_client(username + "/" + bucketname)
except ConnectError:
    print("Could not connect to DagsHub. Please check your internet connection or DagsHub status.", file=sys.stderr)
    sys.exit(0)


import boto3
from datetime import date

BUCKET = "hab-bloom-db-2026"
PREFIX = "daily/2026/"

s3 = boto3.client("s3")

def get_last_available_date():
    """
    Returns latest date available in 2026 only.
    """

    continuation_token = None
    dates = set()

    while True:
        if continuation_token:
            resp = s3.list_objects_v2(
                Bucket=BUCKET,
                Prefix=PREFIX,
                ContinuationToken=continuation_token
            )
        else:
            resp = s3.list_objects_v2(
                Bucket=BUCKET,
                Prefix=PREFIX
            )

        if "Contents" in resp:
            for obj in resp["Contents"]:
                parts = obj["Key"].split("/")
                # Expected: daily/2026/MM/DD/file.nc
                if len(parts) >= 5:
                    try:
                        y = 2026
                        m = int(parts[2])
                        d = int(parts[3])
                        dates.add(date(y, m, d))
                    except:
                        continue

        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break

    return max(dates) if dates else None
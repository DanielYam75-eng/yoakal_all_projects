import glob
import os

def main():
    # רשימת תבניות קבצים למחיקה
    patterns = [
        "*forcast*.csv",
        "*actual*.csv",
        "*full*.csv",
        "*grades*.csv",
        "*result*.csv"
    ]

    # מחיקה לפי כל תבנית
    for pattern in patterns:
        for file in glob.glob(pattern):
            os.remove(file)
            print(f"Deleted: {file}")

if __name__ == "__main__":
    main()
    print("Done")

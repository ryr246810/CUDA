#Python file for Cleaning Results
#Author:   yogurt
#Created:  2022-11-28

import os
import shutil

def Clean_Log():
    NeedCleanDir = "./Log_Cmd"

    if os.path.exists(NeedCleanDir):
        shutil.rmtree(NeedCleanDir)

if __name__ == "__main__":
    print("\nStart to clean Log_Cmd.")
    print("......")
    Clean_Log()
    print("......")
    print("Clean Log_Cmd successfully.\n")
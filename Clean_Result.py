#Python file for Cleaning Results
#Author:   yogurt
#Created:  2021-11-28

import os
import shutil

def Clean_Results():
    NeedCleanDir = [
        "./bin/BWO_FZQ2/result",
        "./bin/BWO_Test/result",
        "./bin/BWO_Test_anios",
        "./bin/BWO_Test_anios2",
        "./bin/BWO_Test_anios3",
        "./bin/new_Bwo/result",
        "./bin/new_Bwo_1/result",
        "./bin/PtclMoveTest/result",
        "./bin/testCoaxialLine/result",
        "./bin/testDiode/result",
        "./bin/testPolyModel_MUR/result"
    ]

    for tmpdir in NeedCleanDir:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    print("\nStart to clean results.")
    print("......")
    Clean_Results()
    print("......")
    print("Clean results successfully.\n")
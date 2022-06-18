import os
import time

for i in range(2):

    os.system("git add .")
    os.system("git commit -m \"Upload data\"")
    os.system("git push --all")
    time.sleep(10)


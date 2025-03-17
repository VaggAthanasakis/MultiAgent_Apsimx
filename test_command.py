import subprocess
import numpy as np
import requests_cache
import pandas as pd
from retry_requests import retry
import re
import sqlite3


# apsim_exe = r"c:\Users\vagga\Desktop\test_apsim_GUI\program\APSIM2025.1.7644.0\bin\Models.exe"
# commands_file = r"C:\Users\vagga\Desktop\test_apsim_GUI\Python_Integration\APSIM_FILES\pear_commands"
apsim_exe = r"/usr/local/lib/apsim/2025.2.7659.0/bin/Models"
commands_file = r"/home/eathanasakis/Intership/MultiAgent_Apsimx/APSIM_FILES/commands"


subprocess.run([apsim_exe, ' ','--apply', commands_file], check=True)


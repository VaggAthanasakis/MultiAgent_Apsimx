import subprocess

apsim_exe = r"c:\Users\vagga\Desktop\test_apsim_GUI\program\APSIM2025.1.7644.0\bin\Models.exe"
commands_file = r"C:\Users\vagga\Desktop\test_apsim_GUI\Python_Integration\APSIM_FILES\pear_commands"

subprocess.run([apsim_exe, ' ','--apply', commands_file], check=True)

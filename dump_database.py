import os
import subprocess
import json

mainDir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where the script is located

# Load database connection details from the configuration file
with open(os.path.join(mainDir, 'config.json'), 'r') as config_file:
    config = json.load(config_file)

# Extract database connection parameters
db_host = config['database']['host']
db_user = config['database']['user']
db_password = config['database']['password']
db_name = config['database']['database_name']

# Specify the path to the SQL dump file
dump_file_path = os.path.join(mainDir, 'results', 'database_dump.sql')

# Specify the full path to the mysqldump executable
mysqldump_path = r'C:\Program Files\MySQL\MySQL Server 8.0\bin\mysqldump.exe'  # Replace with the actual path

# Construct the mysqldump command
mysqldump_cmd = [
    mysqldump_path,
    '-h', db_host,
    '-u', db_user,
    f'--password={db_password}',
    db_name,
    '--result-file=' + dump_file_path,
]

# Execute the mysqldump command to export the database contents to the SQL file
try:
    subprocess.run(mysqldump_cmd, check=True)
    print(f"Database contents dumped to {dump_file_path}")
except subprocess.CalledProcessError as err:
    print(f"Error: {err}")

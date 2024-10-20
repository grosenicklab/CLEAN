# Copyright (C) Responsive Capital Management - All Rights Reserved.
# Unauthorized copying of any code in this directory, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Logan Grosenick <logan@responsive.ai>, 2019.

# Run an individual command.  Command output is returned as a       
# string, along with a bool indicated whether the command ran              
# successfully.                                                                    
def run_command_wrapper(cmd):
    try:
        import subprocess
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)  # Grab stderr as well...  
        return (True, '\n'+result, ''.join(["%s " % el for el in cmd]))
    except subprocess.CalledProcessError as e:                              # Handle error conditions 
        return (False, str(e) + '\nProgram output was:\n' + e.output, ''.join(["%s " % el for el in cmd]))

def ensure_path(path):
    """ Check if a path exists. If not, create the necessary directories,   
    but if the path includes a file, don't create the file"""
    import os, errno
    dir_path = os.path.dirname(path)
    if len(dir_path) > 0:  # Only run makedirs if there is a directory to create!                         
        try:
            os.makedirs(dir_path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
    return path

if __name__ == "__main__":
    pass

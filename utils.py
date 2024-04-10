import numpy as np
import pandas as pd
import os
import re

def find_specific_txt_files(folder_path: str) -> list:
    """
    Scans the given folder for .txt files that contain 'MemoryCapacity' in their names
    but not 'log'.
    
    Parameters:
    - folder_path: str - The path to the folder to be scanned.
    
    Returns:
    - List of paths to the files that match the criteria.
    """
    # List all files in the directory
    all_files = os.listdir(folder_path)
    
    # Compile a regular expression to match the criteria
    pattern = re.compile(r'MemoryCapacity(?!.*log).*\.txt$', re.IGNORECASE)
    
    # Filter files using the regular expression
    matching_files = [file for file in all_files if pattern.search(file)]
    
    # Prepend the folder path to each filename
    full_paths = [os.path.join(folder_path, file) for file in matching_files]
    
    return full_paths
#!/usr/bin/env python3
"""
Script to replace user IDs and emails in data files
"""
import os
import re
import csv
from pathlib import Path

# User mapping
user_mapping = {
    'user1': 'user1',  # No replacement
    'user2': 'abimanyu',
    'user3': 'abishek',
    'user4': 'arun',
    'user5': 'ajeeth',
    'user6': 'bharani',
    'user7': 'bharathkumar',
    'user8': 'rahul',
    'user9': 'maha',
    'user10': 'sanjay',
    'user11': 'nandhakumar',
    'user12': 'dhamu',
    'user13': 'balaji',
    'user14': 'priya',
    'user15': 'karthi',
    'user16': 'udhaya',
    'user17': 'jeeva',
    'user18': 'anuja',
    'user19': 'sandhana',
    'user20': 'thangakumar',
}

# Email mapping
email_mapping = {f'user{i}@company.com': f'{user_mapping[f"user{i}"]}@company.com' 
                 for i in range(1, 21)}

data_dir = Path('data')

# Files to process
csv_files = [
    'emails.csv',
    'logins.csv',
    'file_access.csv',
    'usb_usage.csv',
    'anomaly_scores.csv',
    'features.csv',
    'merged_features.csv',
    'red_team_users.csv'
]

def replace_in_file(file_path):
    """Replace user IDs and emails in a CSV file"""
    print(f"Processing {file_path.name}...")
    
    try:
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Replace emails first (to avoid replacing user IDs within emails)
        for old_email, new_email in email_mapping.items():
            content = content.replace(old_email, new_email)
        
        # Replace user IDs
        for old_user, new_user in user_mapping.items():
            # Use word boundaries to avoid partial replacements
            pattern = rf'\b{old_user}\b'
            content = re.sub(pattern, new_user, content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✓ Updated {file_path.name}")
        else:
            print(f"✗ No changes in {file_path.name}")
            
    except Exception as e:
        print(f"ERROR in {file_path.name}: {e}")

# Process all CSV files
for csv_file in csv_files:
    file_path = data_dir / csv_file
    if file_path.exists():
        replace_in_file(file_path)
    else:
        print(f"⚠ File not found: {file_path}")

print("\nReplacement complete!")

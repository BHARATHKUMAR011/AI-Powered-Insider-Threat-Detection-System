import pandas as pd
import numpy as np
import random
import os
from datetime import timedelta

DATA_DIR = 'data'
random.seed(99)
np.random.seed(99)

# Load logs
logins = pd.read_csv(os.path.join(DATA_DIR, 'logins.csv'), parse_dates=['login', 'logout'])
emails = pd.read_csv(os.path.join(DATA_DIR, 'emails.csv'), parse_dates=['time'])
usb_usage = pd.read_csv(os.path.join(DATA_DIR, 'usb_usage.csv'), parse_dates=['plug_time', 'unplug_time'])
file_access = pd.read_csv(os.path.join(DATA_DIR, 'file_access.csv'), parse_dates=['access_time'])

users = logins['user'].unique()

# Force user20 and user7 to be red team depending on the format. usually "user20", "user7".
# We'll dynamically determine the exact representation if it has prefix
red_users = ["user20", "user7", "user12"]
if "user20" not in users:
    red_users = random.sample(list(users), 3)

# For user7, inject after-hours suspicious emails and mass emailing
email_users = ["user7"]
for user in email_users:
    if user not in users: continue
    for i in range(10):
        day = random.choice(pd.date_range(logins['login'].min().normalize(), logins['login'].max().normalize()).to_list())
        time = day + timedelta(hours=random.randint(0, 4))  # 12am-4am
        recipient = random.choice(["unknown@external.com", "hacker@evil.com", "personal@gmail.com", "competitor@rival.com"])
        subject = random.choice(["Confidential", "Emergency", "Secret", "Password Reset", "Urgent Transfer"])
        emails = pd.concat([emails, pd.DataFrame([{'sender': user, 'recipient': recipient, 'time': time, 'subject': subject}])], ignore_index=True)

    day = random.choice(pd.date_range(logins['login'].min().normalize(), logins['login'].max().normalize()).to_list())
    for i in range(20):
        recipient = random.choice(["unknown@external.com", "hacker@evil.com", "personal@gmail.com", "competitor@rival.com"])
        time = day + timedelta(hours=10, minutes=random.randint(0, 59))
        emails = pd.concat([emails, pd.DataFrame([{'sender': user, 'recipient': recipient, 'time': time, 'subject': 'Follow up'}])], ignore_index=True)

# For user20, inject suspicious USB activity
usb_users = ["user20"]
for user in usb_users:
    if user not in users: continue
    print(f"Injecting USB for {user}")
    day = random.choice(pd.date_range(logins['login'].min().normalize(), logins['login'].max().normalize()).to_list())
    plug_time = day + timedelta(hours=2)
    unplug_time = plug_time + timedelta(minutes=70) # Over an hour after hours
    device = random.choice(usb_usage['device'].unique()) if len(usb_usage['device'].unique()) > 0 else 'usb_123'
    usb_usage = pd.concat([usb_usage, pd.DataFrame([{'user': user, 'device': device, 'plug_time': plug_time, 'unplug_time': unplug_time}])], ignore_index=True)

# For user12, inject mass file downloads
file_users = ["user12"]
for user in file_users:
    if user not in users: continue
    day = random.choice(pd.date_range(logins['login'].min().normalize(), logins['login'].max().normalize()).to_list())
    for i in range(25):
        file_list = file_access['file'].unique()
        file = random.choice(file_list) if len(file_list) > 0 else 'confidential_data.zip'
        access_time = day + timedelta(hours=10, minutes=random.randint(0, 59))
        file_access = pd.concat([file_access, pd.DataFrame([{'user': user, 'file': file, 'access_time': access_time}])], ignore_index=True)

# Save modified logs
emails.to_csv(os.path.join(DATA_DIR, 'emails.csv'), index=False)
usb_usage.to_csv(os.path.join(DATA_DIR, 'usb_usage.csv'), index=False)
file_access.to_csv(os.path.join(DATA_DIR, 'file_access.csv'), index=False)
pd.DataFrame({'user': red_users}).to_csv(os.path.join(DATA_DIR, 'red_team_users.csv'), index=False)
print('Red team behaviors injected. Red team users saved to data/red_team_users.csv') 
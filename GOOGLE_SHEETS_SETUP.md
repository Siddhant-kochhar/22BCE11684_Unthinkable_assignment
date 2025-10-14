# Google Sheets Integration Setup Guide

## Step 1: Setup Google Sheets API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google Sheets API
4. Create credentials (Service Account)
5. Download the JSON key file
6. Share your Google Sheet with the service account email

## Step 2: Get Your Google Sheet ID

Your Google Form URL: https://forms.gle/pSXvKcmBsa11Q3QY6

1. Go to Google Forms and check where responses are being stored
2. Open the Google Sheet with responses
3. Copy the Sheet ID from URL: 
   `https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit`

## Step 3: Update the Code

Replace the sheet ID in the code below:

```python
GOOGLE_SHEET_ID = "YOUR_SHEET_ID_HERE"
CREDENTIALS_FILE = "path/to/your/credentials.json"
```

## Current Setup

For now, the app uses an in-memory authentication system with these test users:
- Username: `demo`, Password: `password123`
- Username: `test`, Password: `test123`

You can test the signup/signin functionality with these credentials or create new ones!

## Next Steps

1. Get your Google Sheet ID from the form responses
2. Create Google Service Account credentials
3. Update the code with real Google Sheets integration

The current setup will work perfectly for testing!
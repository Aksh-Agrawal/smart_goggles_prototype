# Emergency Alerts Troubleshooting

If you're experiencing issues with the emergency alert system in Smart Goggles, follow these troubleshooting steps:

## Email Alerts Not Working

### Gmail-Specific Issues:

1. **App Password Required**:

   - Gmail no longer supports "less secure apps" authentication
   - You must use an "App Password" instead of your regular Gmail password
   - Generate an App Password at: https://myaccount.google.com/apppasswords
   - Update your `.env` file with the new App Password

2. **Check Email Configuration**:

   - Ensure your `.env` file has these settings correctly filled (see the `.env.example` file for reference):
     ```
     EMAIL_SENDER=your_gmail@gmail.com
     EMAIL_PASSWORD=your_app_password_here
     EMAIL_RECIPIENTS=recipient1@example.com,recipient2@example.com
     EMAIL_SERVER=smtp.gmail.com
     EMAIL_PORT=587
     ```
   - Make sure there are no typos or extra spaces

3. **2-Factor Authentication**:
   - If you use 2FA with Gmail, an App Password is required
   - Sign in to your Google Account
   - Go to Security → App passwords
   - Select "Mail" as the app and "Other" as the device
   - Generate and copy the 16-character password

### Other Email Providers:

- Check your email provider's SMTP settings
- Update `EMAIL_SERVER` and `EMAIL_PORT` accordingly
- Some providers may require different authentication methods

## SMS Alerts Not Working

### Twilio Setup Issues:

1. **Twilio Package Installation**:

   - Make sure the Twilio package is installed:

   ```
   pip install twilio
   ```

2. **Check Twilio Configuration**:

   - Verify your `.env` file has these settings:
     ```
     TWILIO_ACCOUNT_SID=your_account_sid
     TWILIO_AUTH_TOKEN=your_auth_token
     TWILIO_FROM_NUMBER=+1234567890  # Your Twilio phone number with + prefix
     TWILIO_TO_NUMBERS=+1987654321  # Recipient numbers with + prefix
     ```
   - SID and Auth Token can be found in your Twilio dashboard

3. **Phone Number Formatting**:

   - All phone numbers must include country code with + prefix
   - Example: `+1XXXXXXXXXX` for US numbers
   - Example: `+91XXXXXXXXXX` for India numbers

4. **Trial Account Limitations**:
   - Twilio trial accounts can only send SMS to verified numbers
   - Verify your recipient numbers in the Twilio console
   - Or upgrade to a paid Twilio account

## Checking Logs

For detailed error information, check the console output when running the application. Look for lines with:

- "ERROR" - Indicates a serious problem
- "WARNING" - Indicates a potential issue

The log will show specific error messages that can help diagnose the problem:

- "Email authentication failed" - Likely an issue with your email/password
- "SMTP error" - Problems connecting to the email server
- "Failed to send SMS" - Issues with Twilio configuration or network

## Testing the Alert System

To test if your emergency alert system is working:

1. Press 'E' key in the Smart Goggles application
2. Check the console for detailed logs about sending attempts
3. Look for "Email sent successfully" or "SMS sent successfully" messages

If you continue to experience issues after following these steps, please check your internet connection and firewall settings, which might be blocking outgoing connections to email servers or the Twilio API.

## Google Cloud & Gmail API setup (quick)

1. Go to https://console.cloud.google.com/ and create a project (or select existing).
2. Enable the Gmail API for the project (APIs & Services > Library > Gmail API).
3. Create OAuth Credentials (APIs & Services > Credentials > Create Credentials > OAuth client ID).
   - Application type: Desktop app
   - Name: e.g. "Email Agent Local"
4. Download the credentials JSON and save it as `client_secrets.json` in the `samples/email_agent/` folder, or set `GOOGLE_CLIENT_SECRETS_PATH` to its path.
5. Run the OAuth flow:

```powershell
python .\samples\email_agent\gmail_oauth.py
```

This will open a browser. After consenting, `token.json` will be created.

Notes:
- For production, use a web server flow and store refresh tokens in a secure secrets store.
- If refreshing fails or consent changes, delete `token.json` and re-run the flow.

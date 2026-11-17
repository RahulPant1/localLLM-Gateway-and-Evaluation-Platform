"""Connect Gmail, read-only, for the gateway's agents. Run once:

    cd backend && uv run python scripts/gmail_auth.py

It needs an OAuth client of type "Desktop app" from Google Cloud, saved as
data/gmail_client_secret.json (see docs/USAGE.md, "Connecting real apps").
The script prints a Google sign-in link. After you allow read-only access,
Google sends the browser back to a page this script serves on 127.0.0.1;
if that page does not load (WSL can block it), copy the address from the
browser's address bar and paste it here instead.

Only the gmail.readonly permission is asked for, so the gateway can search
and read mail but never send, delete or change anything. The token is saved
to data/gmail_token.json, readable only by you. To disconnect, delete that
file and remove the app at https://myaccount.google.com/permissions.
"""

import base64
import hashlib
import json
import os
import secrets
import select
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

from llm_gateway.config import get_settings
from llm_gateway.connectors.gmail import SCOPE, TOKEN_URL, load_client

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
PROFILE_URL = "https://gmail.googleapis.com/gmail/v1/users/me/profile"


def main() -> None:
    settings = get_settings()
    if not settings.gmail_client_path.is_file():
        sys.exit(
            f"Missing {settings.gmail_client_path}. See docs/USAGE.md, 'Connecting real apps'."
        )
    client = load_client(settings.gmail_client_path)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=")
    state = secrets.token_urlsafe(16)
    received: dict = {}
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - name required by http.server
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            received.update({k: v[0] for k, v in query.items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Done. You can close this tab and return to the terminal.")
            done.set()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    redirect = f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.handle_request, daemon=True).start()
    link = (
        AUTH_URL
        + "?"
        + urllib.parse.urlencode(
            {
                "client_id": client["client_id"],
                "redirect_uri": redirect,
                "response_type": "code",
                "scope": SCOPE,
                "access_type": "offline",
                "prompt": "consent",
                "code_challenge": challenge.decode(),
                "code_challenge_method": "S256",
                "state": state,
            }
        )
    )
    print("\nOpen this link, sign in and allow read-only access to Gmail:\n")
    print(link)
    webbrowser.open(link)
    print("\nWaiting… If the browser then shows an error page, paste its full address here:")
    while not done.is_set():
        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
        if ready:
            pasted = sys.stdin.readline().strip()
            if pasted:
                query = urllib.parse.parse_qs(urllib.parse.urlparse(pasted).query)
                received.update({k: v[0] for k, v in query.items()})
                break
    if received.get("error"):
        sys.exit(f"Google returned: {received['error']}")
    if received.get("state") != state or "code" not in received:
        sys.exit("The answer from Google did not match this sign-in; run the script again.")

    form = urllib.parse.urlencode(
        {
            **client,
            "code": received["code"],
            "code_verifier": verifier,
            "grant_type": "authorization_code",
            "redirect_uri": redirect,
        }
    ).encode()
    with urllib.request.urlopen(urllib.request.Request(TOKEN_URL, data=form), timeout=30) as r:
        token = json.load(r)
    if "refresh_token" not in token:
        sys.exit("Google sent no refresh token; remove the app's access and run this again.")
    profile_request = urllib.request.Request(
        PROFILE_URL, headers={"Authorization": f"Bearer {token['access_token']}"}
    )
    with urllib.request.urlopen(profile_request, timeout=30) as r:
        email = json.load(r).get("emailAddress", "")

    saved = {
        "refresh_token": token["refresh_token"],
        "access_token": token["access_token"],
        "expiry": time.time() + int(token.get("expires_in", 3600)),
        "scope": token.get("scope", SCOPE),
        "email": email,
    }
    path = settings.gmail_token_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved))
    os.chmod(path, 0o600)
    print(f"\nConnected {email}, read-only. Token saved to {path}. Restart the backend to use it.")


if __name__ == "__main__":
    main()

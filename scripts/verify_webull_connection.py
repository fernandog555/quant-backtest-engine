"""
Run this against your own Webull UAT (sandbox) credentials before trusting
the execution layer with anything real. It exercises every read-only call
the bot depends on and prints the raw response shape, so you can confirm
the field names assumed in src/execution/webull_client.py actually match
what your account/SDK version returns.

This does NOT place any orders — read-only by design.

Usage:
    cp .env.example .env   # fill in WEBULL_APP_KEY / WEBULL_APP_SECRET
    python scripts/verify_webull_connection.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from src.execution.webull_client import WebullConfig, WebullExecutionClient, WebullApiError


def main():
    print("Loading Webull config from environment...")
    config = WebullConfig.from_env()

    if config.is_live:
        print("REFUSING to run this verification script against a 'prod' environment.")
        print("Set WEBULL_ENVIRONMENT=uat in your .env before running this.")
        sys.exit(1)

    print(f"Environment: {config.environment}, region: {config.region_id}")
    print("Connecting...\n")

    client = WebullExecutionClient(config)

    print("--- get_account_list() ---")
    try:
        accounts = client.get_account_list()
        print(f"Found {len(accounts)} account(s).")
        if accounts:
            print("First account keys:", list(accounts[0].keys()))
            print("Sample:", accounts[0])
    except WebullApiError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
    print()

    account_id = client.get_account_id()

    print("--- get_account_balance() ---")
    try:
        balance = client.get_account_balance(account_id)
        print("Balance keys:", list(balance.keys()))
        print("Sample:", balance)
        if "buying_power" not in balance:
            print(
                "\n*** WARNING: 'buying_power' key not found. "
                "get_buying_power() in webull_client.py assumes this field exists — "
                "update it to match the actual key name above. ***"
            )
    except WebullApiError as e:
        print(f"FAILED: {e}")
    print()

    print("--- get_account_positions() ---")
    try:
        positions = client.get_account_positions(account_id)
        print(f"Found {len(positions)} position(s).")
        if positions:
            print("First position keys:", list(positions[0].keys()))
            print("Sample:", positions[0])
        else:
            print("(No open positions — that's fine for a fresh sandbox account.)")
    except WebullApiError as e:
        print(f"FAILED: {e}")
    print()

    print("Done. Review any WARNING lines above and update src/execution/webull_client.py")
    print("field-name assumptions if your account's response shape differs.")


if __name__ == "__main__":
    main()

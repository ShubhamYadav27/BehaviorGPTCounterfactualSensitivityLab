#!/usr/bin/env python3
"""Verify that the official BehaviorGPT SDK is installed and can authenticate.

Exit codes:
    0 - SDK installed, authenticated, inference successful
    1 - SDK not installed
    2 - API key not configured
    3 - Authentication failed (invalid key)
    4 - Inference request failed
    5 - Unexpected error
"""

import os
import sys
import time

OK = "[OK]"
FAIL = "[FAIL]"


def main() -> int:
    print("=" * 60)
    print("BehaviorGPT SDK Verification")
    print("=" * 60)

    # Step 1: Check SDK installation
    print("\n[1/4] Checking behaviorgpt package installation...")
    try:
        import behaviorgpt
        print(f"  {OK} behaviorgpt package found")
    except ImportError:
        print(f"  {FAIL} behaviorgpt package is NOT installed.")
        print("    Install with: pip install behaviorgpt")
        return 1

    # Step 2: Check API key
    print("\n[2/4] Checking API key configuration...")
    api_key = os.environ.get("UNBOXAI_API_KEY")
    if not api_key:
        print(f"  {FAIL} UNBOXAI_API_KEY environment variable is not set.")
        print("    Set it with: export UNBOXAI_API_KEY='your-key-here'")
        print("    Or on Windows: set UNBOXAI_API_KEY=your-key-here")
        return 2
    print(f"  {OK} UNBOXAI_API_KEY is set (length={len(api_key)}, prefix={api_key[:4]}...)")

    # Step 3: Initialize client
    print("\n[3/4] Initializing UnboxAIClient...")
    try:
        from behaviorgpt import UnboxAIClient, View
        client = UnboxAIClient(market="us", api_key=api_key, default_catalog_id="retail_catalog")
        print(f"  {OK} Client initialized (base_url={client.base_url})")
    except behaviorgpt.AuthenticationError as e:
        print(f"  {FAIL} Authentication error during init: {e}")
        return 3
    except Exception as e:
        print(f"  {FAIL} Unexpected error during init: {type(e).__name__}: {e}")
        return 5

    # Step 4: Execute test inference
    print("\n[4/4] Executing test inference...")
    try:
        start = time.perf_counter()
        response = client.complete(
            history=[View("test-item-001")],
            limit=5,
        )
        elapsed = (time.perf_counter() - start) * 1000

        print(f"  {OK} Inference completed in {elapsed:.1f}ms")
        print(f"    Items returned: {len(response.items)}")
        if response.items:
            first = response.items[0]
            print(f"    First item: id={first.id}, score={first.score:.6f}")
            print(f"    Item name: {first.data.get('name', 'N/A')}")
        print(f"    Response offset: {response.offset}, limit: {response.limit}")
        return 0

    except behaviorgpt.AuthenticationError as e:
        print(f"  {FAIL} Authentication failed: {e}")
        print("    Your API key may be invalid or expired.")
        return 3
    except behaviorgpt.BadRequestError as e:
        print(f"  {FAIL} Bad request: {e}")
        return 4
    except behaviorgpt.RateLimitError as e:
        print(f"  {FAIL} Rate limited: {e}")
        return 4
    except behaviorgpt.UnboxAIError as e:
        print(f"  {FAIL} SDK error: {e}")
        return 4
    except Exception as e:
        print(f"  {FAIL} Unexpected error: {type(e).__name__}: {e}")
        return 5
    finally:
        try:
            client._http_client.close()
        except Exception:
            pass


if __name__ == "__main__":
    print()
    exit_code = main()
    print()
    print("=" * 60)
    status = "PASS" if exit_code == 0 else "FAIL"
    print(f"Result: {status} (exit code {exit_code})")
    print("=" * 60)
    sys.exit(exit_code)

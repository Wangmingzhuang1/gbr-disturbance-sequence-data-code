# -*- coding: utf-8 -*-
import json
import subprocess
import os
import sys

# Ensure UTF-8 output for terminal
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def call_giiisp_api(endpoint, payload):
    """
    Robust wrapper for Giiisp Paper Search APIs (Skill.md Section)
    """
    base_url = "https://giiisp.com"
    full_url = f"{base_url}{endpoint}"
    
    cmd = [
        "curl", "-sS", "-X", "POST", full_url,
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload)
    ]
    
    try:
        # Use explicit encoding for Windows subprocess handling
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if not result.stdout.strip():
            return {"error": "Empty response from server", "status": result.returncode}
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response", "raw": result.stdout[:200]}
    except Exception as e:
        return {"error": str(e), "msg": "Execution failed"}

def search_reef_literature(query_list):
    """ OA Paper Search (Section 1) """
    endpoint = "/first/oaPaper/searchArticlesByQuery1"
    payload = {"titleAndAbs": query_list}
    return call_giiisp_api(endpoint, payload)

if __name__ == "__main__":
    print("Testing Giiisp API Skill Integration...")
    # Using broader keywords for testing
    results = search_reef_literature(["coral reef", "thermal stress"])
    if "error" in results:
        print(f"Status: {results['error']}")
    else:
        print("Success! Retrieved literature results.")
        # Print keys or summary to verify
        print(json.dumps(results, indent=2, ensure_ascii=False)[:500])

#!/usr/bin/env python3
import sys
import os

def main():
    """Simple health check - container is healthy if it's running"""
    run_mode = os.environ.get('RUN_MODE', 'TASK')
    
    if run_mode == 'SERVICE':
        print("✅ SVG animation service is healthy")
        sys.exit(0)
    else:
        print("✅ Task mode - health check N/A")
        sys.exit(0)

if __name__ == "__main__":
    main()

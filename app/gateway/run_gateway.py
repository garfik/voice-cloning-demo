#!/usr/bin/env python3
"""
Gateway runner script
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append('/app')

# Import and run the gateway
from gateway.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

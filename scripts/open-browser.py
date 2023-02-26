#!/usr/bin/env python
"""open-browser"""
# flake8: noqa
import os
import sys
from urllib.request import pathname2url
import webbrowser

URL = sys.argv[1]

FINAL_ADDRESS = f"{URL}"

print(f"FINAL_ADDRESS: {FINAL_ADDRESS}")

# MacOS
CHROME_PATH = "open -a /Applications/Google\ Chrome.app %s"  # pylint: disable=anomalous-backslash-in-string

webbrowser.get(CHROME_PATH).open(FINAL_ADDRESS)

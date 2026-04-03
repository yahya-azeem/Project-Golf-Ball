import requests
import re
import sys

def get_current_sota():
    """
    Scrapes the official OpenAI Parameter Golf leaderboard for the current SOTA BPB.
    """
    url = "https://openai-gh-parameter-golf.p-s.io/"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        
        # Regex to find BPB in the table. 
        # Usually format is 0.XXXX
        # Looking for the first occurrence which is typically #1
        match = re.search(r'> (0\.\d{4,}) <', r.text)
        if match:
            return float(match.group(1))
        
        # Fallback regex for different HTML structure
        match = re.search(r'<td>(0\.\d{4,})</td>', r.text)
        if match:
            return float(match.group(1))

        # Another fallback: search for anything that looks like a score
        scores = re.findall(r'0\.\d{4,}', r.text)
        if scores:
            # Sort scores if multiple found (lowest is best)
            f_scores = sorted([float(s) for s in scores])
            return f_scores[0]
            
        return 0.85 # Very safe fallback
    except Exception as e:
        print(f"Error fetching SOTA: {e}", file=sys.stderr)
        return 0.85 # Assume a high default if scraping fails

if __name__ == "__main__":
    sota = get_current_sota()
    # Print only the value for GHA shell capture
    print(sota)

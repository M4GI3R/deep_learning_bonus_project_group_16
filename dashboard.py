import sys
import subprocess
from pathlib import Path

# Redirect script to launch the actual dashboard inside the src/ folder
if __name__ == "__main__":
    src_dashboard_path = Path(__file__).resolve().parent / "src" / "dashboard.py"
    print(f"Launching dashboard from: {src_dashboard_path}")
    
    # Execute streamlit run src/dashboard.py passing along any arguments
    cmd = [sys.executable, "-m", "streamlit", "run", str(src_dashboard_path)] + sys.argv[1:]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install requirements
requirements = ["pandas", "numpy", "scikit-learn", "plotly"]
for req in requirements:
    install(req)

# Now run train_model
subprocess.run([sys.executable, "train_model.py"])

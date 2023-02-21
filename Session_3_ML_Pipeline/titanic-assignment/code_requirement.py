import json
import subprocess

# Load the notebook file and extract the list of requirements
with open("02-titanic-survival-pipeline-assignment.ipynb", "r") as f:
    notebook = json.load(f)

requirements = []
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        if "import" in source or "from" in source:
            requirements += [line.strip() for line in source.split("\n") if line.startswith("import") or line.startswith("from")]

# Remove duplicates and write the requirements to a file
requirements = sorted(set(requirements))
with open("requirements.txt", "w") as f:
    f.write("\n".join(requirements))

# Install the requirements with pip and log the output
subprocess.check_call(["pip", "install", "-r", "requirements.txt", "--log", "pip-log.txt"])

import re

file_path = "acmart-primary/experiment_report.tex"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Convert figure* environment to figure environment with [H] placement
# This handles \begin{figure*}[t], \begin{figure*}[ht], etc.
content = re.sub(r'\\begin\{figure\*\}\[[^\]]*\]', r'\\begin{figure}[H]', content)
# Also handle cases where there might be no options like \begin{figure*}
content = re.sub(r'\\begin\{figure\*\}', r'\\begin{figure}[H]', content)

# 2. Convert end of environment
content = re.sub(r'\\end\{figure\*\}', r'\\end{figure}', content)

# 3. Update includegraphics size to \columnwidth
# Matches \includegraphics[...]{path} and replaces the [...] part
# We use a function to replacement to avoid messing up the path
def resize_image(match):
    path = match.group(2)
    return f"\\includegraphics[width=\\columnwidth]{{{path}}}"

content = re.sub(r'\\includegraphics\[.*?\]\{(.*?)\}', resize_image, content)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully converted to single column figures with [H] placement.")

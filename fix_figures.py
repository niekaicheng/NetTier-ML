import re

file_path = "acmart-primary/experiment_report.tex"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace \begin{figure*}[H] with \begin{figure*}[t]
# Use strict regex to avoid partial matches
# Escape special chars: \begin -> \\begin, { -> \{, } -> \}, * -> \*, [ -> \[, ] -> \]
new_content = re.sub(r'\\begin\{figure\*\}\[H\]', r'\\begin{figure*}[t]', content)

# Check if any replacement occurred
if content == new_content:
    print("No changes in content. Check regex.")
else:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Successfully updated figure environments.")

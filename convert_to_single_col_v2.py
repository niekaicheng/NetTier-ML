import re

file_path = "acmart-primary/experiment_report.tex"

def convert_figures():
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Convert figure* environment to figure environment with [H] placement
    # This matches \begin{figure*} followed by optional arguments like [t]
    content = re.sub(r'\\begin\{figure\*\}\[[^\]]*\]', r'\\begin{figure}[H]', content)
    # Also handle \begin{figure*} without args if any (though usually acmart uses args)
    content = re.sub(r'\\begin\{figure\*\}', r'\\begin{figure}[H]', content)

    # 2. Convert end of environment
    content = re.sub(r'\\end\{figure\*\}', r'\\end{figure}', content)

    # 3. Update includegraphics size to \columnwidth
    # Regex explanation:
    # \\includegraphics : literal command
    # \[.*?\] : specific optional args (non-greedy)
    # \{(.*?)\} : capture the filename inside braces (group 1)
    def resize_image(match):
        path = match.group(1)
        # Force width=\columnwidth. This overrides any existing width/height/keepaspectratio
        return f"\\includegraphics[width=\\columnwidth]{{{path}}}"

    content = re.sub(r'\\includegraphics\[.*?\]\{(.*?)\}', resize_image, content)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("Successfully converted to single column figures with [H] placement.")

if __name__ == "__main__":
    convert_figures()

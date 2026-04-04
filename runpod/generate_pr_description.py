import os
import sys

def generate_description(results_summary, project_md_path):
    """
    Synthesizes a PR description from local project context and training results.
    """
    # 1. Read project context
    context = ""
    if os.path.exists(project_md_path):
        with open(project_md_path, "r") as f:
            context = f.read()
    
    # 2. Format results
    # results_summary is expected to be "Seed 0: 0.XXXX Seed 1: 0.XXXX Seed 2: 0.XXXX"
    if not seeds:
        return "Parameter Golf: Aborted/Failed Run", "No training seeds were detected. This usually indicates a crash before the first evaluation or an early worthiness abort."

    results_table = "| Seed | BPB |\n|------|-----|\n"
    total = 0
    for i, val in enumerate(seeds):
        results_table += f"| {i} | {val} |\n"
        total += float(val)

    mean_bpb = total / len(seeds)
    is_record = len(seeds) == 3
    
    # 3. Build description
    title = f"Parameter Golf Submission: {mean_bpb:.4f} BPB"
    if is_record:
        title = "🏆 RECORD BREAKING " + title
    
    description = f"""# {title}

## Performance Metrics
{results_table}
**Mean BPB ({len(seeds)} seeds): {mean_bpb:.4f}**

## Project Context & Architecture
{context}

---
*Generated automatically by Project Golf CI/CD Pipeline*
"""
    return title, description

if __name__ == "__main__":
    res_sum = sys.argv[1] # e.g. "0.8012 0.8023 0.7998"
    proj_md = "PROJECT_GOLF_BALL_EXTRACTED.md"
    
    title, body = generate_description(res_sum, proj_md)
    
    # Write to files for GHA to pick up
    with open("pr_title.txt", "w") as f:
        f.write(title)
    with open("pr_body.md", "w") as f:
        f.write(body)

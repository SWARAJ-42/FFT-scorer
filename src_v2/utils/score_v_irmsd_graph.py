import matplotlib.pyplot as plt
import numpy as np
import os

def generate_score_vs_irmsd_plot(benchmarks, output_dir="."):
    """
    Generate scatter plot: Score vs IRMSD
    Saves:
        result.png
        result.txt
    """

    scores = []
    irmsds = []

    for b in benchmarks:
        if b.irmsd is not None:
            scores.append(b.score)
            irmsds.append(b.irmsd)

    if len(scores) == 0:
        print("No valid IRMSD data found. Skipping plot.")
        return

    scores = np.array(scores)
    irmsds = np.array(irmsds)

    # Optional: log scale if scores are large
    use_log = False
    if scores.max() / max(scores.min(), 1e-6) > 100:
        use_log = True
        scores = np.log10(scores + 1e-6)

    # ── Plot ───────────────────────────────
    plt.figure(figsize=(8, 6))
    plt.scatter(irmsds, scores, alpha=0.7)

    plt.xlabel("IRMSD (Å)")
    plt.ylabel("Score (log scale)" if use_log else "Score")
    plt.title("Score vs IRMSD (Docking Decoys)")

    # Ideal trend line (just visual reference)
    sorted_idx = np.argsort(irmsds)
    plt.plot(irmsds[sorted_idx], scores[sorted_idx], linestyle="--", alpha=0.5)

    plt.grid(True)

    output_png = os.path.join(output_dir, "result.png")
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Saved plot → {output_png}")

    # ── Metadata file ──────────────────────
    output_txt = os.path.join(output_dir, "result.txt")
    with open(output_txt, "w") as f:
        f.write("Score vs IRMSD Plot\n")
        f.write("====================\n\n")
        f.write("Description:\n")
        f.write("This plot shows the relationship between docking score and IRMSD.\n")
        f.write("Each point represents a docked pose (decoy).\n\n")

        f.write("Interpretation:\n")
        f.write("- Lower IRMSD = closer to native structure (better).\n")
        f.write("- Higher score = better docking prediction.\n")
        f.write("- Ideal trend: high score at low IRMSD.\n\n")

        if use_log:
            f.write("Note: Score axis is plotted in log scale.\n\n")

        f.write(f"Total points plotted: {len(scores)}\n")

    print(f"Saved metadata → {output_txt}")
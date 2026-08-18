"""
Monitor Sentinel-2 download progress and auto-trigger the pipeline when done.
Run as: python monitor_and_run.py
"""
import pathlib, time, subprocess, sys, logging

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("monitor")

EXPECTED = 16
S2_DIR = pathlib.Path("data/raw/sentinel2")
POLL_SECS = 30

def count_complete():
    """Count fully-downloaded ZIPs (not currently being written)."""
    zips = list(S2_DIR.glob("*.zip"))
    complete = []
    for z in zips:
        # A zip is 'complete' if its size hasn't changed for 5 s
        s1 = z.stat().st_size
        time.sleep(0.5)
        s2 = z.stat().st_size
        if s1 == s2 and s1 > 50_000_000:   # > 50 MB (skip tiny partial files)
            complete.append(z)
    return complete

def run_stage(cmd, label):
    log.info(">>> STAGE: %s", label)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error("!!! Stage '%s' exited with code %d", label, result.returncode)
    else:
        log.info("    Stage '%s' completed OK.", label)
    return result.returncode == 0

def main():
    py = sys.executable
    cfg = "config.yaml"

    log.info("Monitoring %s for Sentinel-2 downloads... (expecting %d tiles)", S2_DIR, EXPECTED)

    while True:
        done = count_complete()
        log.info("%d / %d tiles complete", len(done), EXPECTED)
        if len(done) >= EXPECTED:
            break
        time.sleep(POLL_SECS)

    log.info("All %d tiles downloaded! Starting pipeline stages...", EXPECTED)

    stages = [
        ([py, "-m", "src.preprocess",        "--config", cfg], "Preprocess"),
        ([py, "-m", "src.features",           "--config", cfg], "Features"),
        ([py, "-m", "src.baseline_models",    "--config", cfg], "Tabular baselines"),
        ([py, "-m", "src.cnn_baseline",       "--config", cfg], "CNN baseline"),
        ([py, "-m", "src.compare_baselines",  "--config", cfg], "Compare baselines"),
    ]

    for cmd, label in stages:
        ok = run_stage(cmd, label)
        if not ok:
            log.error("Pipeline stopped at stage: %s. Fix errors then resume.", label)
            break
    else:
        log.info("=" * 60)
        log.info("PIPELINE COMPLETE. See results/ for metrics and plots.")
        log.info("=" * 60)

if __name__ == "__main__":
    main()

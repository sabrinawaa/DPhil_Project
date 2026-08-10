from pathlib import Path
import re
import pandas as pd

folder = Path("CLEAR_sim/1505_small/")  # change if your CSVs are in another folder
pattern = re.compile(r"^(.*)_(\d+)\.csv$")

col_names = ["x_bin", "y_bin", "z_bin", "dose"]

def read_topas_csv(path):
    # Header is the leading block of TOPAS "#" comment lines; keep it verbatim
    # (minus the newline) so a combined file can reuse it as-is.
    header_lines = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line.rstrip("\n"))
            else:
                break

    df = pd.read_csv(
        path,
        skiprows=len(header_lines),
        header=None,
        names=col_names,
        dtype={"x_bin": int, "y_bin": int, "z_bin": int, "dose": float},
    )
    return header_lines, df

def write_topas_csv(path, header_lines, data):
    with open(path, "w") as f:
        for line in header_lines:
            f.write(line + "\n")
        for row in data.itertuples(index=False):
            dose_str = "0" if row.dose == 0 else str(row.dose)
            f.write(f"{row.x_bin}, {row.y_bin}, {row.z_bin}, {dose_str}\n")

# Group files by base name
groups = {}
for path in folder.glob("*.csv"):
    m = pattern.match(path.name)
    if m:
        base = m.group(1)
        groups.setdefault(base, []).append(path)

for base, files in groups.items():
    files = sorted(files, key=lambda p: int(pattern.match(p.name).group(2)))

    empty_files = [f for f in files if f.stat().st_size == 0]
    good_files = [f for f in files if f.stat().st_size > 0]
    if empty_files:
        print(f"WARNING: {base}: skipping {len(empty_files)}/{len(files)} empty file(s): "
              f"{', '.join(f.name for f in empty_files)}")

    # Header (incl. "Parameter File") is just a placeholder, so take the first file's as-is
    header_lines, first_df = read_topas_csv(good_files[0])
    dfs = [first_df] + [read_topas_csv(f)[1] for f in good_files[1:]]
    all_data = pd.concat(dfs, ignore_index=True)
    print(f"{base}: combined {len(good_files)}/{len(files)} files")

    # Sum dose bin-by-bin across runs, keeping the original x/y/z bin ordering
    combined = (
        all_data
        .groupby(["x_bin", "y_bin", "z_bin"], as_index=False)["dose"]
        .sum()
        .sort_values(["x_bin", "y_bin", "z_bin"])
    )

    outname = folder / f"{base}_full.csv"
    write_topas_csv(outname, header_lines, combined)
    print(f"Saved {outname} from {len(good_files)} files")
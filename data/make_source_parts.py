"""Split the original WSN-DS CSV (374,661 records) into the three consecutive blocks of 124,887 records
used by this repository, as data/WSN-DS_part{1,2,3}.csv (header repeated, lines copied verbatim).
usage: python data/make_source_parts.py PATH/TO/WSN-DS.csv
Then compare:  shasum -a 256 data/WSN-DS_part*.csv   with   data/SOURCE_PART_CHECKSUMS.txt
(the loader strips column names, so a differing hash caused only by line endings does not change any result)."""
import os, sys
src = sys.argv[1]
with open(src, "rb") as fh:
    header, *rows = fh.read().splitlines(keepends=True)
assert len(rows) == 374661, f"expected 374,661 records, found {len(rows)}"
n = len(rows) // 3
here = os.path.dirname(os.path.abspath(__file__))
for i in range(3):
    with open(os.path.join(here, f"WSN-DS_part{i + 1}.csv"), "wb") as fh:
        fh.write(header); fh.writelines(rows[i * n:(i + 1) * n])
print("wrote 3 parts of", n, "records")

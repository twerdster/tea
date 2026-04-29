# HandNet Support

The old MATLAB/MEX HandNet preprocessing files have been replaced by the maintained Python converter:

```bash
python3 scripts/handnet_to_tea.py \
  --archive /path/to/ValidationData.zip \
  --output /tmp/tea-handnet-validation \
  --features 200 \
  --feature-type F_CHAR \
  --samples-per-frame 2000 \
  --force
```

The converter reads HandNet `Data_*.mat` files from `.zip`, `.rar`, or extracted directories and writes Tea's native columnar dataset format:

- `Labels.lbl`
- `Threshholds.thr`
- `Coords.co`
- `Poses.pose`
- `F_0000.feat`, `F_0001.feat`, ...
- `handnet_metadata.json`

For a no-data smoke test of the converter and Tea training path, run:

```bash
make handnet-smoke
```

For large `TrainData.rar` slices, prefer extracting the selected `Data_*.mat` files once with `bsdtar -T` and converting the extracted `TrainData/` directory. See `docs/handnet.md` for the full TrainData depth-8 example.

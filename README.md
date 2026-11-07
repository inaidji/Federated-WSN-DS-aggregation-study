# Federated anomaly detection on WSN-DS: aggregation rules, heterogeneity and poisoning

Code, run scripts, and result summaries for the article *Federated Anomaly Detection in Wireless Sensor
Networks: Aggregation Strategies under Data Heterogeneity and Poisoning Attacks*. Every number, table and
figure of the article is produced by the scripts below from the files in `analysis/out/`.

* Federated simulation: [Flower](https://flower.ai) 1.17, TensorFlow/Keras 2.16.1, Python 3.11 (`requirements.txt`).
* Data: WSN-DS (Almomani et al., *Journal of Sensors* 2016, doi:10.1155/2016/4731953). **The dataset is not
  redistributed here**; obtain it from its authors and see `data/`.
* Aggregation rules: FedAvg, FedProx, FedAdam, FedYogi, FedAvgM, FedTrimmedAvg, FedMedian, Krum.

## Layout

| Path | Content |
|---|---|
| `common/` | model (`fl_model.py`), Flower client with FedProx/poisoning/prediction dumps (`fl_client.py`), server/strategy factory (`fl_server.py`), metric suite (`metrics_suite.py`) |
| `noniid/make_partition*.py` | the K=3 highly non-IID, K=5 and K=10 Dirichlet partitions |
| `noniid/v4/` | main grid (`run_main.sh`: 11 configs x 5 seeds) and poisoning study (`run_poison.sh`: 5 rules x 5 conditions x 3 seeds); `lib.sh` runs one server + K clients, detects dead clients and retries |
| `noniid/sweep/` | server-optimizer grids, BN-averaged variants and the K=10 Dirichlet sweep (`gen_jobs.py`, `worker.sh`, `select_adaptive.py`) |
| `centralized/` | centralized CNN / MLP / gradient-boosting baselines and their results (`baselines_v4.json`) |
| `analysis/` | offline evaluation from stored probabilities (`runio.py`), aggregation scripts, figures; `analysis/out/*.json` hold the numbers behind the article's tables |
| `data/` | checksums of the source parts and of every generated partition, and `make_source_parts.py` |

## Reproduce

```bash
pip install -r requirements.txt

# 1. data: put WSN-DS.csv anywhere, then create the three source blocks (124,887 records each)
python data/make_source_parts.py /path/to/WSN-DS.csv
shasum -a 256 data/WSN-DS_part*.csv        # compare with data/SOURCE_PART_CHECKSUMS.txt

# 2. partitions (outputs match data/PARTITION_CHECKSUMS.txt and data/DIRICHLET_CHECKSUMS.txt)
python noniid/make_partition.py            # K=3  -> noniid/data
python noniid/make_partition_k5.py         # K=5  -> noniid/data_k5
for a in 0.1 0.5 5.0; do for s in 0 1 2; do
  python noniid/make_partition_dirichlet.py $a 10 $s noniid/sweep/data/a${a}_s${s}; done; done

# 3. experiments (about 4-5 min per K=3 run on an 8-core laptop, one run at a time)
bash noniid/v4/run_main.sh                 # 55 runs  -> noniid/v4/main
bash noniid/v4/run_poison.sh               # 75 runs  -> noniid/v4/poison
python centralized/train_baselines.py      # -> centralized/baselines_v4.json

cd noniid/sweep                            # heterogeneity sweep and server-optimizer tuning
python gen_jobs.py A && python gen_jobs.py C
for j in A C; do ./worker.sh jobs_$j.txt 0 2 8100 & ./worker.sh jobs_$j.txt 1 2 8101 & wait; done
python select_adaptive.py                  # validation-based choice of (eta, tau); writes selected_adaptive.json
python gen_jobs.py B
./worker.sh jobs_B.txt 0 2 8100 & ./worker.sh jobs_B.txt 1 2 8101 & wait
cd ../..

# 4. tables and figures
python analysis/analyze_main.py && python analysis/analyze_poison.py
python analysis/analyze_sweep.py && python analysis/make_figures.py
```

Re-running only step 4 on the stored predictions (Zenodo dataset, doi:10.5281/zenodo.24042305; unpack inside `noniid/`) reproduces all
reported metrics without any training. The archive keeps, for every run, the global model's predicted
probabilities on every client's validation and test sets for the last five rounds, the per-round metric
histories, and the Krum selection logs.

## Notes that matter

* **Randomness.** `WSN_SEED` fixes initialization, minibatch order, splits and partitions. TensorFlow on
  different hardware is not bit-for-bit deterministic, so individual runs can differ slightly; the
  article's conclusions rest on seed means, which are what the JSON files hold.
* **Batch-norm statistics.** Flower's FedAdam, FedYogi and FedAvgM apply the server optimizer to every model
  array, including batch-norm running means and variances, and this breaks learning on this model. The
  BN-averaged variant (`WSN_ADAPTIVE_BN_AVG=1`, `common/fl_server.py`) averages those statistics instead. The
  article reports both; Flower's default is the default here.
* **Ports.** Concurrent workers must use different `WSN_PORT`s: `lib.sh` kills whatever listens on its
  port when a run starts and ends.
* **Attackers.** Poisoned clients ignore the FedProx proximal term (`WSN_ATTACKER_IGNORES_MU=1`); the
  boosted sign-flip scales by 3 and touches trainable weights only.
* Older exploratory runs (8-round and single-seed experiments, superseded) are not included.

## Environment variables

`WSN_STRATEGY` (fedavg, fedprox, fedadam, fedyogi, fedavgm, fedmedian, fedtrimmedavg, krum), `WSN_MU`,
`WSN_ETA`, `WSN_TAU`, `WSN_BETA`, `WSN_KRUM_F`, `WSN_SEED`, `WSN_DATA_DIR`, `WSN_MIN_CLIENTS`, `WSN_PORT`,
`WSN_POISON` (none, labelflip, signflip), `WSN_POISON_PARTS`, `WSN_POISON_LAMBDA`, `WSN_VAL_FRAC`,
`WSN_DUMP_LAST`, `FL_NUM_ROUNDS`, `WSN_LOCAL_EPOCHS`, `WSN_LIGHT`.

## Citation and license

See `CITATION.cff`. Code: MIT license (`LICENSE`). The WSN-DS dataset keeps its own terms; it is not part of this repository.

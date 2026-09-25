## JevBench public benchmark

This run evaluates LLM2Jev on the 231 public items from the `original`, `easy`, and
`hard` datasets. The TypeSafe adapter sends each compiled decision to the local
LLM2Jev endpoint. The run command stores one JSONL result per decision, raw responses,
and the benchmark ledger. The summary command aggregates those results into accuracy,
calibration, and latency statistics.

Run the benchmark:

```bash
python -m jevbench.cli run \
  --tasks datasets/public/original.jsonl,datasets/public/easy.jsonl,datasets/public/hard.jsonl \
  --adapter typesafe \
  --endpoint http://127.0.0.1:30000 \
  --key-env '' \
  --model qwen3.5-4b \
  --cost-basis no_billable_account_public_endpoint \
  --reserve-usd 0 \
  --delay-s 0.2 \
  --results /data/LLM2Jev/RUN2/qwen3.5-4b-v1.jsonl \
  --raw-dir /data/LLM2Jev/RUN2/raw \
  --ledger RUN/ledger.json
```

Summarize the run:

```bash
python -m jevbench.cli summarize \
  --tasks datasets/public/original.jsonl,datasets/public/easy.jsonl,datasets/public/hard.jsonl \
  --public-export /data/LLM2Jev/RUN2/qwen3.5-4b-v1-summary.json \
  --results /data/LLM2Jev/RUN2/qwen3.5-4b-v1.jsonl
```

Accuracy is measured over the same public items used for the comparison below. Latency
is reported in seconds; the Jev result uses JevBench's published serial benchmark
measurement, while the LLM2Jev result is measured on the local endpoint.

| System | Accuracy | Latency P50 (s) | Latency P95 (s) |
|---|---:|---:|---:|
| Jev 1.13.0 (TypeSafe AI) | 85.1% | 0.652 | 0.722 |
| LLM2Jev(Qwen3.5-4B) | 76.2%  | 0.048 | 0.346|
| jev-local (Qwen3.5-9B) | 74.9% |  - | - |
| kev 8B (research preview) | 71.4%  |  - | - |
| kev 4B (research preview) | 66.2% | - | - |
| Open-Jev 2B (Zefan Cai) | 64.5%  |  - | - |
| jeff (Logan Markewich, GLiFormer 400M) | 62.8%  | - | - |
| smalljev semantic-v9 | 60.6%   |  - | - |
| Laya (Convai Innovations, ModernBERT-large 421M) | 58.4%  |  - | - |

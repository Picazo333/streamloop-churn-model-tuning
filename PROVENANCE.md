# PROVENANCE

This repository is the StreamLoop churn-tuning project. It starts from the
4Geeks Python boilerplate and the official project statement. No license is
added here, and no authors are added.

## Boilerplate

- Repository: https://github.com/4GeeksAcademy/python-hello
- GitHub template flag: `is_template=false`. The files below were copied from the git tree; the repo was not created with "Use this template".
- Commit: `142f2a913162efafce897da8bf241fe690f60395`
- Commit date: 2025-01-08 10:37:24 -0500
- Commit subject: Delete README.cn.md
- License: that commit contains no `LICENSE` file. This project does not invent one.

### Files taken from that commit

| File | Treatment |
| --- | --- |
| `server.py` | Copied unchanged. SHA-256 `28e4573330cfb9b65bf984d2c4fe9d63e9032bf2f6a9e642268edf1d9a7f04af`. It is the boilerplate static server and is not the churn entry point. |
| `main.py` | The boilerplate file was `print("Hello World")` and its README said to replace it. `main.py` is now the churn-tuning command. |
| `learn.json` | The boilerplate metadata was not kept. |
| `README.md`, `README.es.md` | Replaced with the instructions for this project. |

Upstream credits stay in the python-hello repository. They are not listed as authors of this project.

## Official statement

- Syllabus repository: https://github.com/4GeeksAcademy/ai-engineering-syllabus
- Commit: `fd2da034894ba93ba738559280ccde074ab1d953`
- `content/projects/streamloop-churn-model-tuning/README.md` SHA-256 `fb9d5b4dacad6b4781d6b870ebfd74f18a29460434e3a48a7cce8fa05b966114`
- `content/projects/streamloop-churn-model-tuning/README.es.md` SHA-256 `e4fb9ceaa54b50bc44315d0a78852b8101fe5eead281e9345bc592a82f5154b8`
- `content/projects/streamloop-churn-model-tuning/learn.json` SHA-256 `08716163a33559c30670e5b130a91b7159e38280fb8c66c1195dd4f22ea11f52`

`learn.json` in this repository is that syllabus file, byte for byte. It contains a telemetry URL from the syllabus. This project does not call it.

## Dataset

- URL required by the statement: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- No Kaggle login and no mirror. `python main.py` downloads the file, checks SHA-256 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`, and checks 7043 rows.
- The CSV is not committed. There is no serialized model in git. `results/metrics.json` stores only the headline metrics, the final hyperparameters, and the CSV SHA-256. The full search dump is not part of the submission.

## What a run writes

`python main.py` regenerates `tuning_report.md`, the short `results/metrics.json`, `docs/figures/*.png`, and the metric tables in both READMEs. A full metrics dump, when written, goes outside the repo.

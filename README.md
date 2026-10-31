# Hey Tara – local microphone test

Test one or more wake-word models with your microphone, fully offline.

## Folder structure

```
mic_test/
├── mic_test_models_local.ipynb   # the notebook
├── requirements.txt              # dependencies
├── README.md
├── pkg/
│   ├── heytara_prep.py           # preprocessing
│   ├── heytara_models.py         # model definitions
│   └── stats.json                # feature standardisation stats
└── model/
    └── your .pt files            # e.g. matchboxnet_finetuned.pt
```

Models can also sit in subfolders: `model/<model_name>/model.pt`.

## Setup

Run from the `mic_test/` folder.

macOS / Linux:
```bash
python3 -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt
```

Linux only: install PortAudio for the microphone: `sudo apt install libportaudio2`

Optional, smaller CPU-only PyTorch:
```bash
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Run

1. Copy your model files into `model/`.
2. Start Jupyter from this folder: `jupyter notebook`
3. Open `mic_test_models_local.ipynb` and run the cells top to bottom.
4. Say "hey tara" when the record cell prints `RECORDING`.

## Settings (in the notebook)

| Setting | Where | Meaning |
|---|---|---|
| `MODEL_DIR` | models cell | folder with models (default `./model`) |
| `USE` | models cell | name fragments to test; empty = all models |
| `THR_KEY` | models cell | `far1pct` (default) or `far0.1pct` (stricter) |
| `SECONDS` | record cell | length of the recording |
| `INPUT_DEVICE` | record cell | microphone; `None` = system default |
| `TEST_WAV` | wav cell | path to a local audio file to test instead |

## Notes

- Always launch Jupyter from `mic_test/`; `./model` and `./pkg` are relative paths.
- Restart the kernel after editing anything in `pkg/`.
- Recordings stay on your machine.

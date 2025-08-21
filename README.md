# to create the venv :
**Windows (PowerShell)**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/macOS (Bash)**:
```bash
python -m venv venv
source venv/bin/activate
```

# to install requirements :
## ! For GPU inference, ensure compatible versions of CUDA, torch, and torchaudio (mine CUDA 11.8). 
## Update requirements.txt with appropriate versions. !
```bash
pip install -r requirements.txt
```

# project structure :
```
poetry_app/
│── poetry_main.py               # Entry point 
│── requirements.txt             # Dependencies
│── README.md                    # Project description
│
├── poetry/                      # Main package
│   ├── config.py                # Global constants & default settings
│   ├── audio.py                 # AudioProducer, NoiseEstimator, VAD logic
│   ├── similarity.py            # Levenshtein, matching logic
│   ├── utils.py                 # Helpers functions
│   ├── models.py                # PoemData, WordMatch, MultiWordResult classes
│   ├── transcriber.py           # PoetryTranscriber class (main app logic)
│   └── state.py                 # AppState enum and related state management
│
├── data/
│   ├── poems_1.json             # Poems dataset
│   └── ...                      # Any additional poem files
│
└── tests/                       # Unit tests
    ├── test_similarity.py
    ├── test_audio.py
    └── test_transcriber.py
```

# for testing :
```bash
pytest tests/ -v
```

# example to run on CPU :
```bash
python poetry_main.py --model small --device cpu --compute int8_float32 --chunk 1 --overlap 0.5 --lang fr --poems data/poems_1.json
```

# example to run on GPU :
```bash
python poetry_main.py --model medium --device cuda --chunk 1 --overlap 0.5 --lang fr --poems data/poems_1.json
```
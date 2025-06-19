# FaceEmbed API Project Structure

## 📁 Directory Structure

```
face_embed_api/
├── 📂 src/                              # Source Code Directory
│   ├── __init__.py                      # Package Initializer
│   ├── app.py                           # FastAPI Application Main File
│   ├── database.py                      # SQLite Database Management
│   └── utils.py                         # Hailo Inference Utility Classes
│
├── 📂 data/                             # Data Directory
│   └── vectors.db                       # SQLite database file
│
├── 📂 tests/                            # Tests Directory
│   ├── __init__.py                      # Tests Package Initializer
│   ├── unit/                            # Unit Tests
│   │   ├── __init__.py
│   │   └── test_face_embed_api.py       # API and Service Unit Tests
│   └── integration/                     # Integration Tests
│       ├── __init__.py
│       └── test_api_integration.py      # Full API and DB Integration Tests
│
├── 📂 scripts/                          # Scripts Directory
│   ├── start_server.py                  # Server Start Script
│   ├── run_tests.py                     # Test Runner Script
│   ├── test_hailo_request.py            # Standalone API Request Test Script
│   └── verify_multi_model.py            # Script to verify Hailo multi-model concurrent execution
│
├── 📂 models/                           # AI Model Files
│   ├── arcface_mobilefacenet.hef        # Face Embedding Model
│   └── scrfd_10g.hef                    # Face Detection Model
│
├── 📂 docs/                             # Documentation Directory
│   ├── HAILO_SETUP.md                   # Hailo Hardware Setup Guide
│   ├── TEST_REPORT.md                   # Test Report
│   └── reference/                       # Reference Documents
│       ├── hailo_python_guide.md        # Hailo Python API Guide
│       ├── detection_with_tracker.py    # Reference Code Example
│       └── utils.py                     # Reference Utility Classes
│
├── 📂 logs/                             # Logs Directory
│   └── hailort.log                      # Hailo Runtime Log
│
├── 📂 examples/                         # Examples Directory
│   └── (Example code to be added)
│
├── 📄 README.md                         # Project README
├── 📄 PROJECT_STRUCTURE.md              # Project Structure (This file)
├── 📄 pyproject.toml                     # Project Configuration
├── 📄 requirements.txt                   # Python Dependencies
├── 📄 uv.lock                           # UV Lock File
└── 📄 .python-version                   # Python Version
```

## 🚀 Quick Start

### 1. Start the Service
```bash
# Set PYTHONPATH and start with uvicorn
source .venv/bin/activate
PYTHONPATH=src uv run uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### 2. Run Tests
```bash
# Run all tests
source .venv/bin/activate
uv run -- pytest -v
```

## 📋 File Descriptions

### Core Source Code
- **`src/app.py`**: The main FastAPI application, containing all API endpoints for embedding and database operations.
- **`src/database.py`**: Handles all interactions with the SQLite database for vector storage and searching.
- **`src/utils.py`**: Contains helper classes and functions, primarily the Hailo asynchronous inference engine.

### Test Files
- **`tests/unit/test_face_embed_api.py`**: Unit tests for individual API endpoints and service logic, using mocks to isolate functionality.
- **`tests/integration/test_api_integration.py`**: Integration tests for the complete API flow, including database interactions.

### Script Utilities
- **`scripts/start_server.py`**: Convenience script to start the server with the correct settings.
- **`scripts/run_tests.py`**: Script to execute the entire test suite.
- **`scripts/test_hailo_request.py`**: Standalone script for testing live API endpoints, including the new database functions.
- **`scripts/verify_multi_model.py`**: Script to verify concurrent execution of multiple models on Hailo hardware.

### Documentation
- **`docs/HAILO_SETUP.md`**: Detailed guide for setting up the Hailo hardware and software environment.
- **`docs/TEST_REPORT.md`**: Detailed test report.
- **`docs/reference/`**: Reference documents and code examples from third parties.

## 🔧 Development Workflow

### Adding New Features
1. Implement the feature in the `src/` directory.
2. Add corresponding unit tests in `tests/unit/`.
3. Add integration tests in `tests/integration/` to cover the new end-to-end flow.
4. Run all tests (`uv run -- pytest -v`) to ensure nothing is broken.
5. Update `README.md` and other relevant documentation.

### Deployment Preparation
1. Run the full test suite: `uv run -- pytest -v`
2. Start the service using the start script or the `uvicorn` command.

## 📊 Test Coverage

- The project contains a comprehensive suite of unit and integration tests covering all major functionalities, including face detection, embedding, and all database operations.

## 🛠️ Tech Stack

- **API Framework**: FastAPI + Uvicorn
- **AI Inference**: Hailo-8 + HailoRT
- **Database**: SQLite
- **Testing Framework**: pytest + unittest
- **Dependency Management**: UV

---

*Project structure last updated on July 28, 2024*

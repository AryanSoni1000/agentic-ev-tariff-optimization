from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
METRIC_DIR = OUTPUT_DIR / "metrics"
SIMULATION_DIR = OUTPUT_DIR / "simulation_results"
REPORT_DIR = PROJECT_ROOT / "reports"

BASE_TARIFF_PER_KWH = 0.35
MIN_TARIFF_PER_KWH = 0.20
MAX_TARIFF_PER_KWH = 0.60
DEFAULT_GRID_COST_PER_KWH = 0.14
DEFAULT_CHARGERS_PER_STATION = 4
PRICE_ELASTICITY = -0.15
RANDOM_STATE = 42
OFFICIAL_ST_EVCDP_DIR = RAW_DIR / "urbanev" / "st_evcdp"


REQUIRED_DIRS = [
    RAW_DIR,
    RAW_DIR / "acn_data",
    RAW_DIR / "urbanev",
    INTERIM_DIR,
    PROCESSED_DIR,
    MODEL_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    METRIC_DIR,
    SIMULATION_DIR,
    REPORT_DIR,
]


def ensure_directories() -> None:
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)

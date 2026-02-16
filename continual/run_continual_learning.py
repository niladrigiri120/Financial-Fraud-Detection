import pandas as pd
import time
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from continual.continual_trainer import retrain_model

STREAM_PATH = "data/continual_data.csv"
BATCH_SIZE = 200

buffer = []
stream_df = pd.read_csv(STREAM_PATH)
size = 0

for i, row in enumerate(stream_df.iterrows()):

    buffer.append(row[1])

    # 🔍 heartbeat
    if len(buffer) % 50 == 0:
        print(f"Buffered {len(buffer)} rows...")

    if len(buffer) == BATCH_SIZE:
        new_data = pd.DataFrame(buffer)
        size += len(new_data)
        retrain_model(new_data)
        buffer.clear()
        print(f"🔁 Model retrained with {size} rows")

    time.sleep(0.2)  # Simulate delay between transactions

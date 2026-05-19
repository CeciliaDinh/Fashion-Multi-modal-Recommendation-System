#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd

# ================= CONFIGURATION =================
INPUT_FILE = "clean_data.csv"
OUTPUT_DIR = "dataset/amazon_dataset"

# Output Filenames
INTER_OUT = "amazon_dataset.inter"

# Column Mapping
COL_USER = "user_id"
COL_ITEM = "asin"
COL_TIME = "timestamp"
COL_RATING = "rating"

# =================================================

def main():
    print(f"--- Starting Preprocessing for {INPUT_FILE} ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------- 1. LOAD DATA ----------------
    print("Loading raw data...")
    use_cols = [COL_USER, COL_ITEM, COL_TIME, COL_RATING]
    df = pd.read_csv(INPUT_FILE, usecols=use_cols, dtype=str)
    print(f"Original Rows: {len(df)}")

    # ---------------- 2. TYPE CONVERSION ----------------
    print("Converting types and handling missing values...")
    df[COL_TIME] = pd.to_numeric(df[COL_TIME], errors='coerce')
    df[COL_RATING] = pd.to_numeric(df[COL_RATING], errors='coerce')
    df.dropna(subset=[COL_USER, COL_ITEM, COL_TIME], inplace=True)
    print(f"Rows after dropping missing: {len(df)}")

    # ---------------- 3. GENERATE INTERACTION FILE ----------------
    print("Generating Interaction file for LightGCN...")

    inter_out = df[[COL_USER, COL_ITEM, COL_TIME, COL_RATING]].rename(columns={
        COL_USER: 'user_id:token',
        COL_ITEM: 'item_id:token',
        COL_TIME: 'timestamp:float',
        COL_RATING: 'rating:float'
    })

    inter_path = os.path.join(OUTPUT_DIR, INTER_OUT)
    inter_out.to_csv(inter_path, sep='\t', index=False)
    print(f"Saved: {inter_path}")
    print(f"Total interactions: {len(inter_out)}")

    print("--- Preprocessing Complete ---")

if __name__ == "__main__":
    main()

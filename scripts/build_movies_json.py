#!/usr/bin/env python3
"""
Build movies.json metadata file from movie_plot.csv

Your CSV format:
- movie_poster_path: e.g., "action/100108.jpg"
- movie_plot: Contains title in quotes, e.g., 'The movie "Parker" is a thrilling...'
- movie_category: e.g., "action"

This script extracts the movie title from the plot text.

Usage:
    python build_movies_json.py --csv movie_plot.csv --out movies.json
"""

import argparse
import json
import re
import os
import pandas as pd


def extract_title_from_plot(plot: str, fallback: str = "Unknown Movie") -> str:
    """
    Extract movie title from plot text.
    Expected format: 'The movie "Title Here" is a...'
    """
    if not plot:
        return fallback
    
    # Pattern to match: The movie "Title" or The film "Title"
    # Also handles: "Title" is a movie about...
    patterns = [
        r'[Tt]he (?:movie|film) ["\']([^"\']+)["\']',  # The movie "Title"
        r'^["\']([^"\']+)["\'] is',                      # "Title" is...
        r'["\']([^"\']+)["\']',                          # First quoted text as fallback
    ]
    
    for pattern in patterns:
        match = re.search(pattern, plot)
        if match:
            return match.group(1).strip()
    
    return fallback


def main():
    parser = argparse.ArgumentParser(description="Build movies.json from movie_plot.csv")
    parser.add_argument("--csv", required=True, help="Path to movie_plot.csv")
    parser.add_argument("--out", default="movies.json", help="Output JSON path")
    parser.add_argument("--max-rows", type=int, default=0, help="Debug: cap number of rows (0 = all)")
    args = parser.parse_args()

    print(f"Reading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows)
        print(f"Limited to {args.max_rows} rows for debugging")

    # Check for required columns
    required = ["movie_poster_path", "movie_plot", "movie_category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    print(f"Processing {len(df)} movies...")
    
    movies = {}
    titles_extracted = 0
    titles_fallback = 0
    
    for idx, row in df.iterrows():
        plot = str(row.get("movie_plot", "")).strip()
        poster_path = str(row.get("movie_poster_path", "")).strip()
        category = str(row.get("movie_category", "")).strip()
        
        # Generate fallback title from poster filename
        # e.g., "action/100108.jpg" -> "Movie 100108"
        filename = os.path.splitext(os.path.basename(poster_path))[0]
        fallback_title = f"Movie {filename}" if filename else f"Movie {idx}"
        
        # Try to extract title from plot
        title = extract_title_from_plot(plot, fallback=fallback_title)
        
        if title != fallback_title:
            titles_extracted += 1
        else:
            titles_fallback += 1
        
        movies[idx] = {
            "title": title,
            "plot": plot,
            "poster_path": poster_path,
            "category": category,
        }

    # Save to JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(movies)} movies to {args.out}")
    print(f"   - Titles extracted from plot: {titles_extracted}")
    print(f"   - Titles using fallback: {titles_fallback}")
    
    # Show a few samples
    print("\n📋 Sample entries:")
    for i in range(min(3, len(movies))):
        m = movies[i]
        print(f"   [{i}] {m['title']} ({m['category']})")
        print(f"       Plot preview: {m['plot'][:80]}...")


if __name__ == "__main__":
    main()
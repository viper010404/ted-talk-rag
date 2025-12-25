from collections import Counter
from pathlib import Path
from typing import Counter as CounterType
from typing import Iterable

import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def basic_overview(df: pd.DataFrame) -> None:
    print("Rows:", len(df))
    print("Columns:", ", ".join(df.columns))
    print("\nNon-null counts:")
    print(df.notnull().sum().sort_values(ascending=False))
    print("\nDtypes:")
    print(df.dtypes)


def duration_stats(df: pd.DataFrame) -> None:
    if "duration" not in df.columns:
        print("\nNo duration column found.")
        return
    durations = df["duration"].dropna()
    print("\nDuration (seconds) stats:")
    print(durations.describe())


def topic_counts(topics_series: Iterable[str], top_n: int = 15) -> CounterType[str]:
    counter: CounterType[str] = Counter()
    for raw in topics_series:
        if not isinstance(raw, str):
            continue
        for topic in raw.split(","):
            cleaned = topic.strip().lower()
            if cleaned:
                counter[cleaned] += 1
    return Counter(dict(counter.most_common(top_n)))


def transcript_length_stats(df: pd.DataFrame) -> None:
    if "transcript" not in df.columns:
        print("\nNo transcript column found.")
        return
    lengths = df["transcript"].dropna().apply(len)
    print("\nTranscript length stats (characters):")
    print(lengths.describe())


def main() -> None:
    base = Path(__file__).resolve().parent
    candidates = [base / "data/ted_talks_env.csv", base / "data/ted_talks_en.csv"]
    dataset_path = next((p for p in candidates if p.exists()), None)
    if dataset_path is None:
        raise FileNotFoundError(f"Could not find dataset at any of: {candidates}")

    df = load_dataset(dataset_path)

    print(f"Loaded dataset from {dataset_path}")
    basic_overview(df)
    duration_stats(df)
    transcript_length_stats(df)

    if "topics" in df.columns:
        top_topics = topic_counts(df["topics"])
        print("\nTop topics:")
        for topic, count in top_topics.items():
            print(f"{topic}: {count}")


if __name__ == "__main__":
    main()

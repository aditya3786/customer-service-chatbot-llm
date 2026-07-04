"""
Fetch ~3000 recent CS papers from arXiv API and save to arxiv_cs_sample.csv.
Run once from the project root: python dataset/fetch_arxiv.py
"""

import csv
import os
import time
import arxiv

CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE"]
PER_CATEGORY = 600
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arxiv_cs_sample.csv")

FIELDNAMES = ["arxiv_id", "title", "authors", "abstract", "categories", "year", "url"]


def fetch_papers():
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    papers = []
    seen_ids = set()

    for cat in CATEGORIES:
        print(f"Fetching {PER_CATEGORY} papers from {cat}...")
        search = arxiv.Search(
            query=f"cat:{cat}",
            max_results=PER_CATEGORY,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        count = 0
        for result in client.results(search):
            pid = result.entry_id.split("/")[-1]
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            papers.append({
                "arxiv_id": pid,
                "title": result.title.replace("\n", " "),
                "authors": ", ".join(a.name for a in result.authors[:5]),
                "abstract": result.summary.replace("\n", " "),
                "categories": ", ".join(result.categories),
                "year": result.published.year,
                "url": result.entry_id,
            })
            count += 1
        print(f"  → {count} papers collected (total so far: {len(papers)})")
        time.sleep(2)

    return papers


def main():
    papers = fetch_papers()
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(papers)
    print(f"\nSaved {len(papers)} papers to {OUT_PATH}")


if __name__ == "__main__":
    main()

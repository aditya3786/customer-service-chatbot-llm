"""
MedQuAD XML Parser
==================
Parses XML files from selected MedQuAD folders into a single CSV.
Run once from the dataset/ directory: python parse_medquad.py

Folders used (all have complete answers):
  1_CancerGov_QA     — Cancer (NIH/NCI)
  3_GHR_QA           — Genetics Home Reference (first 300 files)
  5_NIDDK_QA         — Diabetes, Kidney, Digestive disorders
  6_NINDS_QA         — Neurological disorders
  8_NHLBI_QA_XML     — Heart, Lung, Blood diseases
"""

import os
import csv
import xml.etree.ElementTree as ET

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDQUAD_DIR = os.path.join(BASE_DIR, "MedQuAD")
OUTPUT_CSV = os.path.join(BASE_DIR, "medquad.csv")

# (folder_name, max_files_to_use)
FOLDERS = [
    ("1_CancerGov_QA", None),
    ("3_GHR_QA", 300),
    ("5_NIDDK_QA", None),
    ("6_NINDS_QA", None),
    ("8_NHLBI_QA_XML", None),
]


def parse_xml_file(filepath):
    """Extract Q&A pairs from a single MedQuAD XML file."""
    pairs = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        focus = root.findtext("Focus", default="").strip()
        source = root.attrib.get("source", "")

        for qa in root.iter("QAPair"):
            q_elem = qa.find("Question")
            a_elem = qa.find("Answer")
            if q_elem is None or a_elem is None:
                continue
            question = (q_elem.text or "").strip()
            answer = (a_elem.text or "").strip()
            qtype = q_elem.attrib.get("qtype", "")
            if question and answer:
                pairs.append({
                    "question": question,
                    "answer": answer,
                    "focus": focus,
                    "qtype": qtype,
                    "source": source,
                })
    except ET.ParseError:
        pass
    return pairs


def main():
    all_pairs = []
    for folder_name, max_files in FOLDERS:
        folder_path = os.path.join(MEDQUAD_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Skipping missing folder: {folder_name}")
            continue

        xml_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".xml"))
        if max_files:
            xml_files = xml_files[:max_files]

        folder_pairs = []
        for fname in xml_files:
            pairs = parse_xml_file(os.path.join(folder_path, fname))
            folder_pairs.extend(pairs)

        print(f"{folder_name}: {len(xml_files)} files → {len(folder_pairs)} Q&A pairs")
        all_pairs.extend(folder_pairs)

    print(f"\nTotal Q&A pairs: {len(all_pairs)}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "focus", "qtype", "source"])
        writer.writeheader()
        writer.writerows(all_pairs)

    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

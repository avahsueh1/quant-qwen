import json
from datetime import date

document = {
    "id": "test_paper_001",

    "title": "Example Quantitative Finance Paper",
    "authors": ["Example Author"],

    "published_date": "2024-03-17",
    "retrieved_date": str(date.today()),

    "source": "test",
    "source_url": "",

    "category": "asset_pricing",
    "subcategories": [
        "factor_models"
    ],

    "document_type": "paper",

    "license": "unknown",
    "rights_status": "needs_review",

    "text": "This is example text from a quantitative finance paper.",

    "word_count": 9,
    "token_count": None,

    "quality_score": None,

    "has_equations": False,
    "has_code": False,
    "has_tables": False,

    "publication_status": "unknown",

    "split": "train"
}

output_file = "quant-corpus/metadata/documents.jsonl"

with open(output_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(document) + "\n")

print("Metadata saved!")
print(json.dumps(document, indent=2))

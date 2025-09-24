#!/usr/bin/env python3
"""
Build Golden Sets with Leakage Guard for Compliance AI Models

Creates tiny, frozen "golden" sets per taxonomy branch with leakage detection.
"""

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class GoldenSetConfig:
    """Configuration for golden set creation."""

    examples_per_branch: int = 40  # 30-50 examples per taxonomy branch
    min_examples_per_branch: int = 30
    max_examples_per_branch: int = 50
    similarity_threshold: float = 0.85  # Near-duplicate threshold
    random_seed: int = 42
    output_dir: str = "golden_sets"


class LeakageDetector:
    """Detects exact and near-duplicates across train/val/gold sets."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )

    def detect_exact_duplicates(self, examples: List[Dict]) -> Set[str]:
        """Detect exact duplicates using content hashing."""
        seen_hashes = set()
        duplicates = set()

        for example in examples:
            content = f"{example.get('instruction', '')} {example.get('response', '')}"
            content_hash = hashlib.md5(content.encode()).hexdigest()

            if content_hash in seen_hashes:
                duplicates.add(content_hash)
            else:
                seen_hashes.add(content_hash)

        return duplicates

    def detect_near_duplicates(
        self, examples: List[Dict]
    ) -> List[Tuple[int, int, float]]:
        """Detect near-duplicates using TF-IDF cosine similarity."""
        if len(examples) < 2:
            return []

        # Extract text content
        texts = []
        for example in examples:
            content = f"{example.get('instruction', '')} {example.get('response', '')}"
            texts.append(content)

        # Compute TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        # Compute cosine similarities
        similarities = cosine_similarity(tfidf_matrix)

        # Find near-duplicates
        near_duplicates = []
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                if similarities[i][j] >= self.similarity_threshold:
                    near_duplicates.append((i, j, similarities[i][j]))

        return near_duplicates


class TaxonomyBranchExtractor:
    """Extracts taxonomy branches from training examples."""

    def __init__(self):
        self.branch_patterns = {
            "PII.Contact.Email": ["email", "e-mail", "@", "mail"],
            "PII.Contact.Phone": ["phone", "telephone", "mobile", "cell"],
            "PII.Identifier.SSN": ["ssn", "social security", "tax id"],
            "PII.Identifier.Passport": ["passport", "travel document"],
            "PII.Financial.CreditCard": ["credit card", "card number", "cc"],
            "PII.Financial.BankAccount": ["bank account", "account number", "routing"],
            "PII.Demographic.Age": ["age", "birth date", "date of birth"],
            "PII.Demographic.Gender": ["gender", "sex", "male", "female"],
            "PII.Demographic.Race": ["race", "ethnicity", "ethnic"],
            "PII.Location.Address": ["address", "street", "city", "zip"],
            "PII.Location.GPS": ["gps", "coordinates", "latitude", "longitude"],
            "SECURITY.Attack.SQLInjection": [
                "sql injection",
                "sqli",
                "database attack",
            ],
            "SECURITY.Attack.XSS": ["xss", "cross-site scripting", "script injection"],
            "SECURITY.Attack.CSRF": ["csrf", "cross-site request forgery"],
            "SECURITY.Vulnerability.CVE": ["cve-", "vulnerability", "exploit"],
            "SECURITY.Vulnerability.Misconfiguration": [
                "misconfiguration",
                "config error",
            ],
            "COMPLIANCE.GDPR.Article6": ["article 6", "lawful basis", "consent"],
            "COMPLIANCE.GDPR.Article32": [
                "article 32",
                "security measures",
                "technical measures",
            ],
            "COMPLIANCE.GDPR.Article25": ["article 25", "data protection by design"],
            "COMPLIANCE.HIPAA.PHI": ["phi", "protected health", "health information"],
            "COMPLIANCE.CCPA.PersonalInfo": [
                "personal information",
                "ccpa",
                "california privacy",
            ],
            "COMPLIANCE.SOC2.Availability": ["availability", "uptime", "service level"],
            "COMPLIANCE.SOC2.Confidentiality": [
                "confidentiality",
                "data protection",
                "access control",
            ],
            "COMPLIANCE.SOC2.Integrity": [
                "integrity",
                "data accuracy",
                "processing integrity",
            ],
            "COMPLIANCE.SOC2.Security": [
                "security",
                "access controls",
                "security measures",
            ],
            "COMPLIANCE.SOC2.Privacy": ["privacy", "privacy controls", "data privacy"],
        }

    def extract_branches(self, example: Dict) -> List[str]:
        """Extract taxonomy branches from an example."""
        content = (
            f"{example.get('instruction', '')} {example.get('response', '')}".lower()
        )
        branches = []

        for branch, patterns in self.branch_patterns.items():
            if any(pattern in content for pattern in patterns):
                branches.append(branch)

        # Also extract from response JSON if available
        try:
            response_data = json.loads(example.get("response", "{}"))
            if "taxonomy" in response_data:
                branches.extend(response_data["taxonomy"])
        except (json.JSONDecodeError, TypeError):
            pass

        return list(set(branches))  # Remove duplicates


class GoldenSetBuilder:
    """Builds golden sets per taxonomy branch with leakage detection."""

    def __init__(self, config: GoldenSetConfig):
        self.config = config
        self.leakage_detector = LeakageDetector(config.similarity_threshold)
        self.branch_extractor = TaxonomyBranchExtractor()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set random seed for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

    def build_golden_sets(
        self, train_examples: List[Dict], val_examples: List[Dict]
    ) -> Dict[str, Any]:
        """Build golden sets with leakage detection."""
        print("🔥 Building golden sets with leakage detection...")

        # Combine all examples for leakage detection
        all_examples = train_examples + val_examples

        # Detect exact duplicates
        exact_duplicates = self.leakage_detector.detect_exact_duplicates(all_examples)
        print(f"⚠️ Found {len(exact_duplicates)} exact duplicates")

        # Detect near-duplicates
        near_duplicates = self.leakage_detector.detect_near_duplicates(all_examples)
        print(f"⚠️ Found {len(near_duplicates)} near-duplicate pairs")

        # Extract taxonomy branches and stratify examples
        branch_examples = defaultdict(list)
        for i, example in enumerate(all_examples):
            branches = self.branch_extractor.extract_branches(example)
            for branch in branches:
                branch_examples[branch].append((i, example))

        print(f"📊 Found {len(branch_examples)} taxonomy branches")

        # Build golden sets per branch
        golden_sets = {}
        leakage_report = {
            "exact_duplicates": len(exact_duplicates),
            "near_duplicates": len(near_duplicates),
            "branches_processed": 0,
            "examples_per_branch": {},
            "leakage_by_branch": {},
        }

        for branch, examples in branch_examples.items():
            if len(examples) < self.config.min_examples_per_branch:
                print(
                    f"⚠️ Branch {branch}: Only {len(examples)} examples (min: {self.config.min_examples_per_branch})"
                )
                continue

            # Sample examples for golden set
            num_samples = min(self.config.examples_per_branch, len(examples))
            sampled_examples = random.sample(examples, num_samples)

            # Check for leakage in this branch
            branch_leakage = self._check_branch_leakage(sampled_examples, all_examples)

            # Create golden set
            golden_set = {
                "branch": branch,
                "examples": [ex[1] for ex in sampled_examples],
                "metadata": {
                    "total_examples": len(examples),
                    "sampled_examples": num_samples,
                    "leakage_detected": branch_leakage,
                    "created_at": str(Path().cwd()),
                    "config": {
                        "examples_per_branch": self.config.examples_per_branch,
                        "similarity_threshold": self.config.similarity_threshold,
                    },
                },
            }

            golden_sets[branch] = golden_set
            leakage_report["examples_per_branch"][branch] = num_samples
            leakage_report["leakage_by_branch"][branch] = branch_leakage
            leakage_report["branches_processed"] += 1

            print(
                f"✅ Branch {branch}: {num_samples} examples, leakage: {branch_leakage}"
            )

        # Save golden sets
        self._save_golden_sets(golden_sets, leakage_report)

        return {
            "golden_sets": golden_sets,
            "leakage_report": leakage_report,
            "summary": {
                "total_branches": len(golden_sets),
                "total_examples": sum(
                    len(gs["examples"]) for gs in golden_sets.values()
                ),
                "exact_duplicates": len(exact_duplicates),
                "near_duplicates": len(near_duplicates),
            },
        }

    def _check_branch_leakage(
        self, branch_examples: List[Tuple[int, Dict]], all_examples: List[Dict]
    ) -> Dict[str, Any]:
        """Check for leakage within a branch."""
        branch_texts = [ex[1] for ex in branch_examples]

        # Check for exact duplicates within branch
        exact_dups = self.leakage_detector.detect_exact_duplicates(branch_texts)

        # Check for near-duplicates within branch
        near_dups = self.leakage_detector.detect_near_duplicates(branch_texts)

        return {
            "exact_duplicates": len(exact_dups),
            "near_duplicates": len(near_dups),
            "leakage_ratio": (
                (len(exact_dups) + len(near_dups)) / len(branch_examples)
                if branch_examples
                else 0
            ),
        }

    def _save_golden_sets(self, golden_sets: Dict, leakage_report: Dict):
        """Save golden sets and leakage report."""
        # Save individual golden sets
        for branch, golden_set in golden_sets.items():
            safe_branch = branch.replace(".", "_").replace("/", "_")
            output_file = self.output_dir / f"golden_set_{safe_branch}.json"

            with open(output_file, "w") as f:
                json.dump(golden_set, f, indent=2)

        # Save combined golden sets
        combined_file = self.output_dir / "golden_sets_combined.json"
        with open(combined_file, "w") as f:
            json.dump(golden_sets, f, indent=2)

        # Save leakage report
        leakage_file = self.output_dir / "leakage_report.json"
        with open(leakage_file, "w") as f:
            json.dump(leakage_report, f, indent=2)

        print(f"✅ Saved golden sets to {self.output_dir}")
        print(f"  - Individual sets: {len(golden_sets)} files")
        print(f"  - Combined file: {combined_file}")
        print(f"  - Leakage report: {leakage_file}")


def main():
    """Main function to build golden sets."""
    # Load training data (this would come from your actual training pipeline)
    # For now, we'll create a sample structure

    config = GoldenSetConfig(
        examples_per_branch=40, similarity_threshold=0.85, output_dir="golden_sets"
    )

    builder = GoldenSetBuilder(config)

    # Example usage (replace with actual data loading)
    print("📝 Note: Replace with actual training data loading")
    print("Example usage:")
    print("  train_examples = load_training_data('train.jsonl')")
    print("  val_examples = load_training_data('val.jsonl')")
    print("  result = builder.build_golden_sets(train_examples, val_examples)")

    # For demonstration, create sample data
    sample_examples = [
        {
            "instruction": "Map this PII detection: email address john@company.com",
            "response": json.dumps(
                {
                    "taxonomy": ["PII.Contact.Email"],
                    "scores": {"PII.Contact.Email": 0.95},
                    "confidence": 0.95,
                }
            ),
        },
        {
            "instruction": "Map this PII detection: phone number +1-555-0123",
            "response": json.dumps(
                {
                    "taxonomy": ["PII.Contact.Phone"],
                    "scores": {"PII.Contact.Phone": 0.92},
                    "confidence": 0.92,
                }
            ),
        },
    ]

    result = builder.build_golden_sets(sample_examples, [])

    print(f"\n🎉 Golden sets built successfully!")
    print(f"  Total branches: {result['summary']['total_branches']}")
    print(f"  Total examples: {result['summary']['total_examples']}")
    print(f"  Exact duplicates: {result['summary']['exact_duplicates']}")
    print(f"  Near duplicates: {result['summary']['near_duplicates']}")


if __name__ == "__main__":
    main()

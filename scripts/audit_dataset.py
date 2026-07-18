#!/usr/bin/env python3
"""
WP-05: Dataset audit utility
Generates reproducible dataset quality report without modifying data.
"""
import json
import hashlib
import sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def audit_dataset():
    """Audit dataset structure, labels, and duplicates."""
    data_root = Path(__file__).parent.parent / "data"
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "dataset_root": str(data_root),
        "splits": {},
        "class_distribution": {},
        "duplicates": [],
        "missing_files": [],
        "stats": {}
    }
    
    # Load class mapping
    class_map_path = data_root / "class_mapping.json"
    if class_map_path.exists():
        with open(class_map_path) as f:
            class_mapping = json.load(f)
        report["num_classes"] = len(class_mapping.get("class_to_idx", {}))
    else:
        report["num_classes"] = None
    
    # Audit splits
    splits_dir = data_root / "splits"
    if splits_dir.exists():
        split_files = list(splits_dir.glob("*.json"))
        report["split_files"] = [f.name for f in split_files]
    
    # Count images in train/val/test folders
    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if split_dir.exists():
            images = list(split_dir.rglob("*.jpg")) + list(split_dir.rglob("*.png"))
            report["splits"][split] = len(images)
            
            # Count by class
            class_counts = defaultdict(int)
            hashes = []
            for img_path in images:
                class_name = img_path.parent.name
                class_counts[class_name] += 1
                
                # Track file hash for duplicate detection
                try:
                    with open(img_path, 'rb') as f:
                        img_hash = hashlib.md5(f.read()).hexdigest()
                    hashes.append(img_hash)
                except Exception as e:
                    report["missing_files"].append(str(img_path))
            
            report["class_distribution"][split] = dict(class_counts)
            
            # Check for duplicate files
            hash_counts = Counter(hashes)
            for h, count in hash_counts.items():
                if count > 1:
                    report["duplicates"].append(f"{split}: {count} copies of same file (hash: {h})")
    
    # Compute overall stats
    total_images = sum(report["splits"].values())
    report["stats"] = {
        "total_images": total_images,
        "train_ratio": report["splits"].get("train", 0) / total_images if total_images else 0,
        "val_ratio": report["splits"].get("val", 0) / total_images if total_images else 0,
        "test_ratio": report["splits"].get("test", 0) / total_images if total_images else 0,
    }
    
    return report

if __name__ == "__main__":
    report = audit_dataset()
    
    # Save report
    artifact_dir = Path(__file__).parent.parent / "artifacts" / "audit"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = artifact_dir / f"dataset_audit_{timestamp}.json"
    md_path = artifact_dir / f"dataset_audit_{timestamp}.md"
    
    # JSON report
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Markdown report
    with open(md_path, 'w') as f:
        f.write("# Dataset Audit Report\n\n")
        f.write(f"**Generated:** {report['timestamp']}\n\n")
        f.write(f"**Dataset Root:** `{report['dataset_root']}`\n\n")
        
        f.write("## Split Sizes\n\n")
        for split, count in report["splits"].items():
            ratio = report["stats"].get(f"{split}_ratio", 0)
            f.write(f"| {split} | {count} images ({ratio*100:.1f}%) |\n")
        f.write(f"\n**Total:** {report['stats']['total_images']} images\n\n")
        
        f.write(f"## Classes\n\nTotal: {report['num_classes']}\n\n")
        
        for split in ["train", "val", "test"]:
            if split in report["class_distribution"]:
                f.write(f"### {split.upper()}\n\n")
                classes = report["class_distribution"][split]
                for cls, count in sorted(classes.items()):
                    f.write(f"- {cls}: {count}\n")
                f.write("\n")
        
        if report["duplicates"]:
            f.write(f"## WARNING: Duplicates Found\n\n")
            for dup in report["duplicates"]:
                f.write(f"- {dup}\n")
            f.write("\n")
        
        if report["missing_files"]:
            f.write(f"## ERROR: Missing/Inaccessible Files\n\n")
            for missing in report["missing_files"]:
                f.write(f"- {missing}\n")
            f.write("\n")
        
        f.write(f"## Files\n\n- JSON: `{json_path.name}`\n- Markdown: `{md_path.name}`\n")
    
    print(f"PASS: Dataset audit complete")
    print(f"  Total images: {report['stats']['total_images']}")
    print(f"  Classes: {report['num_classes']}")
    print(f"  Splits: train={report['splits'].get('train', 0)}, val={report['splits'].get('val', 0)}, test={report['splits'].get('test', 0)}")
    print(f"  Duplicates: {len(report['duplicates'])}")
    print(f"  Report: {json_path}")

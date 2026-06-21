import argparse
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image


# Paper Section-4 test set and expected HR sizes.
PAPER_SPECS: Dict[str, Tuple[int, int]] = {
    "Einstein": (256, 256),
    "Butterfly": (256, 256),
    "Leaves": (256, 256),
    "Bike": (256, 256),
    "Lena": (512, 512),
    "Lighthouse": (512, 512),
    "F16": (512, 512),
    "Goldhill": (512, 512),
}

# Local source mapping discovered in the project folder.
SOURCE_FILES: Dict[str, str] = {
    "Einstein": "Einstein.png",
    "Butterfly": "butterfly.png",
    "Lena": "lena_gray.bmp",
    "Lighthouse": "2.gif",
    "F16": "10.gif",  # Prefer the 512-looking version.
}


def prepare_dataset(project_dir: Path, output_dir: Path, allow_resize: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []
    missing_count = 0

    for name, expected_size in PAPER_SPECS.items():
        src_name = SOURCE_FILES.get(name, "")
        if not src_name:
            missing_count += 1
            report_lines.append(f"{name:10s} | MISSING SOURCE MAPPING")
            continue

        src_path = project_dir / src_name
        if not src_path.exists():
            missing_count += 1
            report_lines.append(f"{name:10s} | SOURCE FILE NOT FOUND: {src_name}")
            continue

        img = Image.open(src_path)
        orig_mode = img.mode
        orig_size = img.size
        gray = img.convert("L")

        final = gray
        resize_note = "no-resize"
        if gray.size != expected_size:
            if allow_resize:
                final = gray.resize(expected_size, resample=Image.Resampling.BICUBIC)
                resize_note = f"resized {gray.size}->{expected_size}"
            else:
                resize_note = f"SIZE_MISMATCH {gray.size}!={expected_size}"

        out_path = output_dir / f"{name.lower()}.png"
        final.save(out_path)

        report_lines.append(
            f"{name:10s} | src={src_name:14s} | mode={orig_mode:>4s}->{final.mode:>1s} "
            f"| size={orig_size}->{final.size} | {resize_note}"
        )

    report_path = output_dir / "dataset_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved normalized dataset to: {output_dir}")
    print(f"Saved report to: {report_path}")
    print(f"Missing or unmapped paper images: {missing_count}")
    return missing_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="../test_images/hr")
    parser.add_argument(
        "--allow_resize",
        action="store_true",
        help="If set, non-matching source sizes are resized to paper target sizes.",
    )
    args = parser.parse_args()

    prepare_dataset(
        project_dir=Path(args.project_dir),
        output_dir=Path(args.output_dir),
        allow_resize=args.allow_resize,
    )


if __name__ == "__main__":
    main()

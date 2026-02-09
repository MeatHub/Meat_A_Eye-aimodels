"""
2차 수집 등 '같은 부위' 폴더를 합칠 때 파일명이 겹치지 않게 번호를 밀어주는 스크립트.

사용법:
  python rename_to_avoid_overwrite.py <2차수집폴더> <master부위폴더경로> [접두어]
  python rename_to_avoid_overwrite.py <2차수집폴더> <시작번호> [접두어]

- 두 번째 인자가 **폴더 경로**면: master 해당 부위 폴더에서 맨 끝 번호를 찾아, 그 다음 번호부터 붙임.
- 두 번째 인자가 **숫자**면: 그 번호부터 0001, 0002... 식으로 붙임.

예:
  # master/Pork_Tenderloin 맨 끝 번호 다음부터 붙이기 (권장)
  python rename_to_avoid_overwrite.py ../data/second_batch_tenderloin ../data/master_dataset/Pork_Tenderloin Pork_Tenderloin

  # 1001번부터 수동 지정
  python rename_to_avoid_overwrite.py ../data/second_batch 1001 Pork_PicnicShoulder
"""
import re
import sys
from pathlib import Path


def find_number_in_filename(name: str) -> int:
    """파일명에서 숫자 부분 추출 (정렬용)."""
    base = Path(name).stem
    m = re.search(r"_?(\d+)$", base)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", base)
    return int(nums[-1]) if nums else 0


def get_max_number_in_folder(folder: Path) -> int:
    """폴더 안 이미지 파일명에서 가장 큰 번호 반환 (없으면 0)."""
    folder = folder.resolve()
    if not folder.is_dir():
        return 0
    max_n = 0
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            max_n = max(max_n, find_number_in_filename(f.name))
    return max_n


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    folder = Path(sys.argv[1]).resolve()
    if not folder.is_dir():
        print(f"오류: 폴더가 없습니다. {folder}")
        sys.exit(1)

    # 두 번째 인자: 숫자면 그대로 시작번호, 경로면 master 폴더에서 맨 끝 번호+1
    start_num = 1
    prefix = folder.name
    if len(sys.argv) > 2:
        arg2 = sys.argv[2]
        if arg2.isdigit():
            start_num = int(arg2)
            prefix = sys.argv[3] if len(sys.argv) > 3 else folder.name
        else:
            master_folder = Path(arg2).resolve()
            max_n = get_max_number_in_folder(master_folder)
            start_num = max_n + 1
            prefix = sys.argv[3] if len(sys.argv) > 3 else master_folder.name
            print(f"📂 master 폴더 맨 끝 번호: {max_n} → 2차는 {start_num}번부터 붙입니다.\n")

    files = sorted(
        [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda f: (find_number_in_filename(f.name), f.name),
    )
    if not files:
        print(f"해당 폴더에 이미지 파일이 없습니다: {folder}")
        sys.exit(0)

    renamed = 0
    for i, f in enumerate(files):
        new_name = f"{prefix}_{start_num + i:04d}{f.suffix}"
        new_path = f.parent / new_name
        if new_path == f:
            continue
        if new_path.exists():
            print(f"건너뜀 (이미 존재): {new_name}")
            continue
        f.rename(new_path)
        renamed += 1
        print(f"  {f.name}  →  {new_name}")

    print(f"\n완료: {renamed}개 파일 이름 변경 (접두어={prefix}, 시작번호={start_num})")


if __name__ == "__main__":
    main()

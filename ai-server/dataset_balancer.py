"""
데이터셋 균형 분할 스크립트
- raw_dataset에서 각 클래스당 100장씩 균등하게 분할
- train_dataset_1, train_dataset_2, ... 형식으로 순차 생성
- 사용된 이미지는 원본에서 삭제 (이동)
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import Optional, List, Tuple


def get_all_images_from_raw(raw_dir: str, class_name: str) -> list:
    """raw_dataset의 train/test/val 폴더에서 해당 클래스의 모든 이미지 수집"""
    images = []
    raw_path = Path(raw_dir)
    
    # train/test/val 구조인 경우
    for split in ['train', 'test', 'val']:
        class_path = raw_path / split / class_name
        if class_path.exists():
            for img in class_path.iterdir():
                if img.is_file() and img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    images.append(img)
    
    # 직접 클래스 폴더가 있는 경우
    direct_class_path = raw_path / class_name
    if direct_class_path.exists() and direct_class_path.is_dir():
        for img in direct_class_path.iterdir():
            if img.is_file() and img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                images.append(img)
    
    return images


def get_classes_from_raw(raw_dir: str) -> list:
    """raw_dataset에서 클래스 목록 가져오기"""
    raw_path = Path(raw_dir)
    classes = set()
    
    # train/test/val 구조 확인
    for split in ['train', 'test', 'val']:
        split_path = raw_path / split
        if split_path.exists():
            for d in split_path.iterdir():
                if d.is_dir() and d.name.startswith('Beef_'):
                    classes.add(d.name)
    
    # 직접 클래스 폴더 확인
    for d in raw_path.iterdir():
        if d.is_dir() and d.name.startswith('Beef_'):
            classes.add(d.name)
    
    return sorted(list(classes))


def get_next_dataset_number(data_dir: str) -> int:
    """다음 train_dataset_N 번호 찾기"""
    data_path = Path(data_dir)
    existing = []
    
    for d in data_path.iterdir():
        if d.is_dir() and d.name.startswith('train_dataset_'):
            try:
                num = int(d.name.split('_')[-1])
                existing.append(num)
            except ValueError:
                pass
    
    return max(existing) + 1 if existing else 1


def split_from_raw(
    raw_dir: str,
    data_dir: str,
    samples_per_class: int = 100,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    dataset_number: Optional[int] = None
) -> dict:
    """
    raw_dataset에서 균형 데이터셋 생성 (이동 방식)
    
    Args:
        raw_dir: raw_dataset 경로
        data_dir: data 폴더 경로 (train_dataset_N이 생성될 위치)
        samples_per_class: 클래스당 샘플 수 (기본: 100)
        train_ratio: 학습 데이터 비율 (기본 0.7)
        val_ratio: 검증 데이터 비율 (기본 0.15)
        test_ratio: 테스트 데이터 비율 (기본 0.15)
        seed: 랜덤 시드
        dataset_number: 데이터셋 번호 (None이면 자동 증가)
    """
    random.seed(seed)
    
    raw_path = Path(raw_dir)
    data_path = Path(data_dir)
    
    if not raw_path.exists():
        print(f"❌ 오류: {raw_path}가 존재하지 않습니다.")
        return {}
    
    # 클래스 목록 가져오기
    classes = get_classes_from_raw(raw_dir)
    if not classes:
        print(f"❌ 오류: raw_dataset에서 클래스를 찾을 수 없습니다.")
        return {}
    
    print(f"\n📂 raw_dataset 클래스: {len(classes)}개")
    
    # 현재 이미지 수량 확인
    class_images = {}
    print(f"\n{'클래스':<25} {'현재 수량':>10}")
    print(f"{'-'*40}")
    
    for cls in classes:
        images = get_all_images_from_raw(raw_dir, cls)
        class_images[cls] = images
        print(f"{cls:<25} {len(images):>10}장")
    
    # 최소 수량 확인
    min_count = min(len(imgs) for imgs in class_images.values())
    
    if min_count < samples_per_class:
        print(f"\n⚠️ 일부 클래스가 {samples_per_class}장 미만입니다.")
        print(f"   최소 수량: {min_count}장")
        
        insufficient = [(cls, len(imgs)) for cls, imgs in class_images.items() if len(imgs) < samples_per_class]
        print(f"\n   부족한 클래스:")
        for cls, cnt in insufficient:
            print(f"   - {cls}: {cnt}장 (추가 필요: {samples_per_class - cnt}장)")
        
        user_input = input(f"\n{min_count}장 기준으로 분할하시겠습니까? (y/n): ")
        if user_input.lower() != 'y':
            print("취소되었습니다.")
            return {}
        samples_per_class = min_count
    
    # 데이터셋 번호 결정
    if dataset_number is None:
        dataset_number = get_next_dataset_number(data_dir)
    
    output_name = f"train_dataset_{dataset_number}"
    output_path = data_path / output_name
    
    # 분할 비율 계산
    train_count = int(samples_per_class * train_ratio)
    val_count = int(samples_per_class * val_ratio)
    test_count = samples_per_class - train_count - val_count
    
    print(f"\n⚙️ 분할 설정:")
    print(f"   - 출력 폴더: {output_name}")
    print(f"   - 클래스당 총 샘플: {samples_per_class}장")
    print(f"   - Train: {train_count}장 ({train_ratio*100:.0f}%)")
    print(f"   - Val: {val_count}장 ({val_ratio*100:.0f}%)")
    print(f"   - Test: {test_count}장 ({test_ratio*100:.0f}%)")
    
    # 출력 디렉토리 생성
    for split in ['train', 'val', 'test']:
        for cls in classes:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    # 통계
    stats = {
        'dataset_name': output_name,
        'total_moved': 0,
        'remaining': {}
    }
    
    print(f"\n{'='*65}")
    print(f"{'클래스':<25} {'이동':>8} {'남은 수량':>12} {'상태':>10}")
    print(f"{'='*65}")
    
    for cls in classes:
        images = class_images[cls]
        random.shuffle(images)
        
        # 분할할 이미지 선택
        selected = images[:samples_per_class]
        remaining = len(images) - samples_per_class
        
        train_imgs = selected[:train_count]
        val_imgs = selected[train_count:train_count + val_count]
        test_imgs = selected[train_count + val_count:]
        
        # 이동 (move) - Train
        for i, img in enumerate(train_imgs):
            dst = output_path / 'train' / cls / f"{cls}_{i+1:04d}{img.suffix}"
            shutil.move(str(img), str(dst))
        
        # 이동 (move) - Val
        for i, img in enumerate(val_imgs):
            dst = output_path / 'val' / cls / f"{cls}_{i+1:04d}{img.suffix}"
            shutil.move(str(img), str(dst))
        
        # 이동 (move) - Test
        for i, img in enumerate(test_imgs):
            dst = output_path / 'test' / cls / f"{cls}_{i+1:04d}{img.suffix}"
            shutil.move(str(img), str(dst))
        
        stats['total_moved'] += samples_per_class
        stats['remaining'][cls] = remaining
        
        status = "✅" if remaining >= samples_per_class else f"📉 {remaining}장"
        print(f"{cls:<25} {samples_per_class:>8} {remaining:>12} {status:>10}")
    
    print(f"{'='*65}")
    total_remaining = sum(stats['remaining'].values())
    print(f"{'합계':<25} {stats['total_moved']:>8} {total_remaining:>12}")
    
    # 결과 요약
    print(f"\n{'='*65}")
    print(f"📊 분할 완료!")
    print(f"{'='*65}")
    print(f"   📁 생성된 데이터셋: {output_path}")
    print(f"   📦 이동된 이미지: {stats['total_moved']}장")
    print(f"   📂 raw_dataset 남은 이미지: {total_remaining}장")
    
    # 추가 분할 가능 여부
    min_remaining = min(stats['remaining'].values())
    possible_splits = min_remaining // samples_per_class
    if possible_splits > 0:
        print(f"\n   💡 추가 분할 가능: {possible_splits}회 (각 {samples_per_class}장 기준)")
    else:
        print(f"\n   ⚠️ 추가 {samples_per_class}장 분할 불가능")
        if min_remaining > 0:
            print(f"      최대 {min_remaining}장 기준으로 분할 가능")
    
    return stats


def check_raw_dataset(raw_dir: str):
    """raw_dataset 현황 확인"""
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        print(f"❌ 경로가 존재하지 않습니다: {raw_dir}")
        return
    
    classes = get_classes_from_raw(raw_dir)
    
    print(f"\n📂 raw_dataset: {raw_dir}")
    print(f"{'='*50}")
    print(f"{'클래스':<30} {'이미지 수':>15}")
    print(f"{'-'*50}")
    
    total = 0
    min_count = float('inf')
    
    for cls in classes:
        images = get_all_images_from_raw(raw_dir, cls)
        count = len(images)
        total += count
        min_count = min(min_count, count)
        print(f"{cls:<30} {count:>15}장")
    
    print(f"{'-'*50}")
    print(f"{'합계':<30} {total:>15}장")
    print(f"{'최소 클래스 수량':<30} {min_count:>15}장")
    
    # 분할 가능 횟수
    possible_100 = min_count // 100
    print(f"\n💡 100장 기준 분할 가능 횟수: {possible_100}회")


def main():
    parser = argparse.ArgumentParser(description='raw_dataset 균형 분할 도구')
    subparsers = parser.add_subparsers(dest='command', help='실행 명령')
    
    # split 명령 - raw_dataset에서 분할
    split_parser = subparsers.add_parser('split', help='raw_dataset에서 균형 데이터셋 생성')
    split_parser.add_argument('--raw', '-r', default='../data/raw_dataset', help='raw_dataset 경로 (기본: ../data/raw_dataset)')
    split_parser.add_argument('--data', '-d', default='../data', help='data 폴더 경로 (기본: ../data)')
    split_parser.add_argument('--samples', '-n', type=int, default=100, help='클래스당 샘플 수 (기본: 100)')
    split_parser.add_argument('--train-ratio', type=float, default=0.7, help='Train 비율 (기본: 0.7)')
    split_parser.add_argument('--val-ratio', type=float, default=0.15, help='Val 비율 (기본: 0.15)')
    split_parser.add_argument('--seed', type=int, default=42, help='랜덤 시드 (기본: 42)')
    split_parser.add_argument('--number', type=int, default=None, help='데이터셋 번호 (기본: 자동)')
    
    # check 명령 - raw_dataset 현황 확인
    check_parser = subparsers.add_parser('check', help='raw_dataset 현황 확인')
    check_parser.add_argument('--raw', '-r', default='../data/raw_dataset', help='raw_dataset 경로')
    
    # status 명령 - 전체 데이터셋 현황
    status_parser = subparsers.add_parser('status', help='전체 데이터셋 현황 확인')
    status_parser.add_argument('--data', '-d', default='../data', help='data 폴더 경로')
    
    args = parser.parse_args()
    
    if args.command == 'split':
        test_ratio = 1.0 - args.train_ratio - args.val_ratio
        split_from_raw(
            raw_dir=args.raw,
            data_dir=args.data,
            samples_per_class=args.samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=test_ratio,
            seed=args.seed,
            dataset_number=args.number
        )
    
    elif args.command == 'check':
        check_raw_dataset(args.raw)
    
    elif args.command == 'status':
        show_all_datasets(args.data)
    
    else:
        parser.print_help()


def show_all_datasets(data_dir: str):
    """모든 데이터셋 현황 표시"""
    data_path = Path(data_dir)
    
    print(f"\n📊 데이터셋 현황: {data_path}")
    print(f"{'='*60}")
    
    # raw_dataset 확인
    raw_path = data_path / 'raw_dataset'
    if raw_path.exists():
        classes = get_classes_from_raw(str(raw_path))
        total = sum(len(get_all_images_from_raw(str(raw_path), cls)) for cls in classes)
        min_count = min(len(get_all_images_from_raw(str(raw_path), cls)) for cls in classes) if classes else 0
        print(f"\n📁 raw_dataset")
        print(f"   총 이미지: {total}장 | 최소 클래스: {min_count}장")
        print(f"   100장 분할 가능: {min_count // 100}회")
    
    # train_dataset_N 확인
    datasets = sorted([d for d in data_path.iterdir() 
                      if d.is_dir() and d.name.startswith('train_dataset_')])
    
    if datasets:
        print(f"\n📁 분할된 데이터셋:")
        for ds in datasets:
            train_path = ds / 'train'
            if train_path.exists():
                classes = [d for d in train_path.iterdir() if d.is_dir()]
                total = sum(len(list((ds / split / cls.name).iterdir())) 
                           for split in ['train', 'val', 'test'] 
                           for cls in classes 
                           if (ds / split / cls.name).exists())
                print(f"   - {ds.name}: {total}장 ({len(classes)} 클래스)")


def check_dataset(path: str):
    """데이터셋 현황 확인"""
    dataset_path = Path(path)
    
    if not dataset_path.exists():
        print(f"❌ 경로가 존재하지 않습니다: {path}")
        return
    
    # train/val/test 구조인지 확인
    splits = ['train', 'val', 'test']
    has_splits = all((dataset_path / s).exists() for s in splits)
    
    if has_splits:
        print(f"\n📂 데이터셋: {path}")
        print(f"{'='*70}")
        
        # 클래스 목록
        classes = sorted([d.name for d in (dataset_path / 'train').iterdir() if d.is_dir()])
        
        print(f"{'클래스':<25} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
        print(f"{'-'*70}")
        
        totals = {'train': 0, 'val': 0, 'test': 0}
        
        for cls in classes:
            counts = {}
            for split in splits:
                cls_path = dataset_path / split / cls
                if cls_path.exists():
                    count = len([f for f in cls_path.iterdir() if f.is_file()])
                else:
                    count = 0
                counts[split] = count
                totals[split] += count
            
            total = sum(counts.values())
            print(f"{cls:<25} {counts['train']:>10} {counts['val']:>10} {counts['test']:>10} {total:>10}")
        
        print(f"{'-'*70}")
        print(f"{'합계':<25} {totals['train']:>10} {totals['val']:>10} {totals['test']:>10} {sum(totals.values()):>10}")
    
    else:
        # 단순 클래스 폴더 구조
        print(f"\n📂 폴더: {path}")
        print(f"{'='*50}")
        
        classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
        
        print(f"{'클래스':<35} {'이미지 수':>10}")
        print(f"{'-'*50}")
        
        total = 0
        for cls in classes:
            cls_path = dataset_path / cls
            count = len([f for f in cls_path.iterdir() if f.is_file()])
            total += count
            print(f"{cls:<35} {count:>10}")
        
        print(f"{'-'*50}")
        print(f"{'합계':<35} {total:>10}")


if __name__ == '__main__':
    main()

import os, shutil, random

def split_dataset(src_dir, out_dir, val_ratio=0.15, test_ratio=0.15):
    random.seed(42)
    classes = os.listdir(src_dir)
    counts = {}

    for cls in classes:
        cls_path = os.path.join(src_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        files = [f for f in os.listdir(cls_path)
                 if f.lower().endswith(('.jpg','.jpeg','.png'))]
        random.shuffle(files)

        n       = len(files)
        n_val   = int(n * val_ratio)
        n_test  = int(n * test_ratio)

        splits = {
            'train': files[n_val + n_test:],
            'val':   files[:n_val],
            'test':  files[n_val:n_val + n_test]
        }

        for split, imgs in splits.items():
            dest = os.path.join(out_dir, split, cls)
            os.makedirs(dest, exist_ok=True)
            for img in imgs:
                shutil.copy(os.path.join(cls_path, img),
                            os.path.join(dest, img))

        counts[cls] = {'total': n, 'train': len(splits['train']),
                       'val': len(splits['val']), 'test': len(splits['test'])}

    print("\n Dataset split complete!")
    print(f"{'Class':<20} {'Total':>6} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 48)
    for cls, c in counts.items():
        print(f"{cls:<20} {c['total']:>6} {c['train']:>6} {c['val']:>6} {c['test']:>6}")

if __name__ == "__main__":
    src = r"F:\NeuroScan\dataset\Training"
    out = r"F:\NeuroScan\dataset"
    split_dataset(src, out)

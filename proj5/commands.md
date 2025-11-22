
---

## Q1: Classification Baseline

```bash
Test Accuracy: 0.9296
```

---

## Q2: Segmentation Baseline

```bash
python eval_seg.py --exp_name q2_sample_0 --i 0 --batch_size 64     # 87.53
python eval_seg.py --exp_name q2_sample_1 --i 100 --batch_size 64   # 86.35
python eval_seg.py --exp_name q2_sample_2 --i 200 --batch_size 64   # 96.7
python eval_seg.py --exp_name q2_sample_3 --i 300 --batch_size 64   # 94.69
python eval_seg.py --exp_name q2_sample_4 --i 400 --batch_size 64   # INCORRECT: 62.6
python eval_seg.py --exp_name q2_sample_5 --i 500 --batch_size 64   # INCORRECT: 79.5

Test Accuracy: 0.8277
```

---

## Q3: Robustness Analysis

### Q3.1: Rotation Experiments (Classification)

```bash
python eval_cls.py --exp_name q3_cls_rot10 --theta 10 --i 0 --batch_size 64      # 0.9129
python eval_cls.py --exp_name q3_cls_rot30 --theta 30 --i 0 --batch_size 64      # 0.5466
python eval_cls.py --exp_name q3_cls_rot60 --theta 60 --i 0 --batch_size 64      # 0.2528
python eval_cls.py --exp_name q3_cls_rot90 --theta 90 --i 0 --batch_size 64      # 0.2130

```

### Q3.2: Rotation Experiments (Segmentation)

```bash
# Test rotation robustness on different object samples
python eval_seg.py --exp_name q3_seg_rot10 --theta 10 --i 0 --batch_size 64   # Test Accuracy: 82.10   Sample accuracy: 94.49
python eval_seg.py --exp_name q3_seg_rot30 --theta 30 --i 0 --batch_size 64   # Test Accuracy: 67.78   Sample accuracy: 64.11
python eval_seg.py --exp_name q3_seg_rot60 --theta 60 --i 0 --batch_size 64   # Test Accuracy: 29.51   Sample accuracy: 51.40
python eval_seg.py --exp_name q3_seg_rot90 --theta 90 --i 0 --batch_size 64   # Test Accuracy: 25.94   Sample accuracy: 39.28
```

### Q3.3: Point Count Experiments (Classification)

```bash
# Test point count robustness on samples from different classes
python eval_cls.py --exp_name q3_cls_points_100 --num_points 100 --i 0 --batch_size 64      # 91.50
python eval_cls.py --exp_name q3_cls_points_1000 --num_points 1000 --i 0 --batch_size 64    # 92.65
python eval_cls.py --exp_name q3_cls_points_2000 --num_points 2000 --i 0 --batch_size 64    # 92.65
python eval_cls.py --exp_name q3_cls_points_5000 --num_points 5000 --i 0 --batch_size 64    # 92.86
python eval_cls.py --exp_name q3_cls_points_10000 --num_points 10000 --i 0 --batch_size 64  # 92.96
```

### Q3.4: Point Count Experiments (Segmentation)

```bash
# Test point count robustness on different object samples
python eval_seg.py --exp_name q3_seg_points_100 --num_points 100 --i 0 --batch_size 64      # Test Accuracy: 79.87   Sample accuracy: 77
python eval_seg.py --exp_name q3_seg_points_1000 --num_points 1000 --i 0 --batch_size 64    # Test Accuracy: 82.18   Sample accuracy: 88.5
python eval_seg.py --exp_name q3_seg_points_2000 --num_points 2000 --i 0 --batch_size 64    # Test Accuracy: 82.58   Sample accuracy: 89.35
python eval_seg.py --exp_name q3_seg_points_5000 --num_points 5000 --i 0 --batch_size 64    # Test Accuracy: 82.71   Sample accuracy: 86.88
python eval_seg.py --exp_name q3_seg_points_10000 --num_points 10000 --i 0 --batch_size 64  # Test Accuracy: 82.77   Sample accuracy: 87.53
```

---

# Person Detection (People Counting) - ML Midterm

Nhan dien nguoi trong anh/webcam bang **3 thuat toan Machine Learning viet tu dau (from scratch)**:
Linear Regression, Logistic Regression, Linear SVM. Du lieu duoc **giam chieu bang PCA viet tu dau** truoc khi train.

## 1. Cau truc project

```
mid/
  app.py                  # Ung dung nhan dien (Tkinter + OpenCV)
  requirements.txt
  README.md
  src/
    pca.py                # StandardScaler + PCA (eigen decomposition) - from scratch
    models.py             # Linear Regression + Logistic Regression - from scratch
    svm.py                # Linear SVM (hinge loss + L2) - from scratch
    data_utils.py         # Liet ke/doc anh
    metrics.py            # Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix
    train.py              # Pipeline train + danh gia + luu model
  models/                 # Model da train (.npz) + preprocess.npz
  results/                # metrics.json + training_curves.png
  PeopleCountingDataSet/  # Dataset (zero_people / one_person / two_plus_people)
```

## 2. Cai dat

Yeu cau Python 3.10+:

```
python -m pip install -r requirements.txt
```

## 3. Pipeline (from scratch)

1. Doc anh grayscale, resize **48x48**, chuan hoa ve [0,1].
2. Tao dac trung (6912 chieu):
   - raw pixels (2304)
   - gradient magnitude |gx|+|gy| (2304)
   - background-difference |x - background| (2304), background = trung binh cac anh `zero_people` tren tap train.
3. Chuan hoa z-score (StandardScaler from scratch).
4. **PCA from scratch** (eigen decomposition cua ma tran hiep phuong sai) -> 100 thanh phan, giu ~55.4% phuong sai.
5. Chuan hoa dau ra PCA (giup gradient descent hoi tu on dinh).
6. Train 3 model from scratch tren dac trung PCA.

## 4. Train lai model (tuy chon)

Model da train san trong `models/`. De train lai:

```
python src/train.py
```

## 5. Chay ung dung nhan dien

```
python app.py
```

- **Bat webcam**: nhan dien truc tiep tu camera.
- **Mo anh**: nhan dien tren mot file anh.
- Man hinh hien thi ket qua cua ca 3 model (nhan + xac suat) va FPS.

## 6. Phan loai nhi phan

- `0` = KHONG CO NGUOI (zero_people)
- `1` = CO NGUOI (one_person + two_plus_people)

Tap train = 24,960 anh, tap test = 6,240 anh (ti le 80/20, chia stratified).

## 7. Ket qua tren tap test

### 7.1 Phan loai nhi phan (co nguoi / khong co nguoi)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Linear Regression | **0.7510** | 0.7707 | **0.8918** | **0.8268** | **0.7698** |
| Logistic Regression | 0.7508 | **0.7739** | 0.8846 | 0.8256 | 0.7698 |
| Linear SVM | 0.7471 | 0.7722 | 0.8803 | 0.8227 | 0.7640 |

- Baseline (luon du doan "co nguoi") = 66.67%. Ca 3 model deu vuot baseline ro ret.
- Ba model cho ket qua rat sat nhau; Linear Regression cao nhat ve Accuracy/F1/ROC-AUC.

### 7.2 Phan loai 3 lop (One-vs-Rest, Logistic Regression)

| Lop | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| zero_people | 0.584 | 0.637 | 0.610 | 2080 |
| one_person | 0.544 | 0.493 | 0.517 | 2047 |
| two_plus_people | 0.576 | 0.576 | 0.576 | 2113 |

- Accuracy 3 lop = **0.569**, macro-F1 = **0.568** (baseline ngau nhien = 33.3%).
- Phan biet "1 nguoi" vs "2+ nguoi" kho hon nhieu so voi "co/khong co nguoi".
- Chi tiet confusion matrix trong `results/metrics.json`.
- Bieu do loss va phuong sai PCA: `results/training_curves.png`.

## 8. Nhan xet

- Day la bai toan **kho** voi mo hinh tuyen tinh: nguoi trong anh nho, nam xa camera, nen tin hieu yeu.
- PCA giup giam tu 6912 -> 100 chieu, giam chi phi tinh toan va tranh overfitting.
- Background-difference huu ich vi camera co dinh (nen giong nhau giua cac frame).
- Model from scratch hoi tu on dinh sau khi chuan hoa dau ra PCA.

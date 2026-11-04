# Cotton Disease CNN

CNN yordamida paxta bargi va o‘simlik tasvirlarini to‘rtta sinfga ajratish.
Loyiha TensorFlow/Keras, NumPy, Pandas, Matplotlib va scikit-learn kutubxonalaridan
foydalanadi.

## Dataset tuzilmasi

Skript datasetdagi mavjud `train`, `val` va `test` bo‘linmalarini ishlatadi.
Har bir bo‘linmada sinf nomi bilan bir xil nomlangan papkalar bo‘lishi kerak:

```text
Cotton Disease/
  train/
    diseased cotton leaf/
    diseased cotton plant/
    fresh cotton leaf/
    fresh cotton plant/
  val/
    (shu sinf papkalari)
  test/
    (shu sinf papkalari)
```

## O‘rnatish va ishga tushirish

Windows PowerShell ichida loyiha papkasiga o‘ting, so‘ng:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
```

Standart sozlamalar: 30 epoch, 32 ta batch va 224 x 224 piksel. Parametrlarni
o‘zgartirish mumkin:

```powershell
python train.py --epochs 40 --batch-size 32 --image-size 224
python train.py --data-dir "C:\path\to\dataset" --output-dir "C:\path\to\results"
```

Tasvirlar yuklash vaqtida bir xil o‘lchamga keltiriladi va 0–1 oralig‘iga
normallashtiriladi. O‘qitishda tasodifiy aylantirish, masshtablash va gorizontal
aks ettirish qo‘llanadi; sinflar soni nomutanosibligi class weight orqali
hisobga olinadi. Eng yaxshi model validatsiya yo‘qotishi bo‘yicha saqlanadi,
yakuniy metrikalar esa faqat test to‘plamida hisoblanadi.

## Natijalar

Natijalar `outputs/` papkasida hosil bo‘ladi:

- `best_model.keras` — validatsiya bo‘yicha eng yaxshi CNN modeli.
- `class_names.json` — model chiqishlarining sinf tartibi.
- `metrics.json` — test Accuracy, macro va weighted Precision, Recall, F1-score.
- `classification_report.csv` — har bir sinf uchun Precision, Recall va F1-score.
- `confusion_matrix.png` — test to‘plami chalkashlik matritsasi.
- `training_history.png` — o‘qitish va validatsiya Accuracy/Loss grafigi.

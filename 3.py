import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, 
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
print("Loading features...")

with open("features.pkl", "rb") as f:
    data = pickle.load(f)

X = data['X']
y = data['y']
splits = np.array(data['splits'])

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_train = scaler.transform(X[splits == 'training'])
y_train = y[splits == 'training']

X_val   = scaler.transform(X[splits == 'validation'])
y_val   = y[splits == 'validation']

X_test  = scaler.transform(X[splits == 'testing'])
y_test  = y[splits == 'testing']

print(f" Loaded!")
print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")

from sklearn.metrics import roc_curve

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer * 100  # return as percentage

def evaluate_model(name, model, X_val, y_val, X_test, y_test):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    val_pred   = model.predict(X_val)
    val_proba  = model.predict_proba(X_val)[:, 1]
    val_acc    = accuracy_score(y_val, val_pred) * 100
    val_f1     = f1_score(y_val, val_pred) * 100
    val_eer    = compute_eer(y_val, val_proba)

    print(f"\n VALIDATION:")
    print(f"  Accuracy: {val_acc:.2f}%  (target ≥ 80%)")
    print(f"  F1 Score: {val_f1:.2f}%  (target ≥ 80%)")
    print(f"  EER:      {val_eer:.2f}%  (target ≤ 12%)")

    test_pred  = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_acc   = accuracy_score(y_test, test_pred) * 100
    test_f1    = f1_score(y_test, test_pred) * 100
    test_eer   = compute_eer(y_test, test_proba)

    print(f"\n TEST:")
    print(f"  Accuracy: {test_acc:.2f}%  (target ≥ 80%)")
    print(f"  F1 Score: {test_f1:.2f}%  (target ≥ 80%)")
    print(f"  EER:      {test_eer:.2f}%  (target ≤ 12%)")

    cm = confusion_matrix(y_test, test_pred)
    real_acc = cm[0,0] / cm[0].sum() * 100
    fake_acc = cm[1,1] / cm[1].sum() * 100
    print(f"\n PER CLASS (Test):")
    print(f"  Real Accuracy: {real_acc:.2f}%  (target ≥ 75%)")
    print(f"  Fake Accuracy: {fake_acc:.2f}%  (target ≥ 75%)")

    print(f"\n TARGETS MET:")
    print(f"  Accuracy ≥ 80%: {'✅' if test_acc >= 80 else '❌'}")
    print(f"  EER ≤ 12%:      {'✅' if test_eer <= 12 else '❌'}")
    print(f"  F1 ≥ 80%:       {'✅' if test_f1  >= 80 else '❌'}")
    print(f"  Per-class ≥75%: {'✅' if real_acc >= 75 and fake_acc >= 75 else '❌'}")

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Real', 'Fake'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_')}.png")
    plt.show()
    print(f"  Saved confusion matrix!")

    return {
        'name': name,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_eer': test_eer,
        'model': model
    }

print("\n Training Random Forest...")

rf_model = RandomForestClassifier(
   
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    n_jobs=-1,          
    random_state=42,
    verbose=1
)
rf_model.fit(X_train, y_train)
rf_results = evaluate_model("Random Forest", rf_model,
                             X_val, y_val, X_test, y_test)


print("\n Training XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1,
    random_state=42,
    verbosity=1
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)
xgb_results = evaluate_model("XGBoost", xgb_model,
                              X_val, y_val, X_test, y_test)


print("\n Training Neural Network (MLP)...")

mlp_model = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=50,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5,
    random_state=42,
    verbose=True
)
mlp_model.fit(X_train, y_train)
mlp_results = evaluate_model("Neural Network", mlp_model,
                              X_val, y_val, X_test, y_test)


print("\n" + "="*50)
print("  MODEL COMPARISON")
print("="*50)

all_results = [rf_results, xgb_results, mlp_results]

print(f"\n{'Model':<20} {'Accuracy':>10} {'F1':>10} {'EER':>10}")
print("-" * 52)
for r in all_results:
    print(f"{r['name']:<20} "
          f"{r['test_acc']:>9.2f}% "
          f"{r['test_f1']:>9.2f}% "
          f"{r['test_eer']:>9.2f}%")

best = max(all_results, key=lambda x: x['test_acc'])
print(f"\n Best Model: {best['name']} "
      f"({best['test_acc']:.2f}% accuracy)")

with open("best_model.pkl", "wb") as f:
    pickle.dump(best['model'], f)

print(f"Saved best_model.pkl")

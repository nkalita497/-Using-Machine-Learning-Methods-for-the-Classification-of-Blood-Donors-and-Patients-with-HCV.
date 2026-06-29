import pandas as pd
import numpy as np
import time
import joblib
import sklearn
import xgboost
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold, cross_validate,
                                     RepeatedStratifiedKFold,
                                     learning_curve)
from sklearn.preprocessing import RobustScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score,
                             f1_score, balanced_accuracy_score,
                             roc_curve, auc, precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import wilcoxon, shapiro, ttest_rel
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


print("=" * 60)
print("WERSJE BIBLIOTEK")
print("=" * 60)
print(f"sklearn:  {sklearn.__version__}")
print(f"xgboost:  {xgboost.__version__}")
print(f"pandas:   {pd.__version__}")
print(f"numpy:    {np.__version__}")


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
start_time = time.time()
EXPERIMENT_CONFIG = {
    'dataset': 'HCV Dataset (UCI)',
    'n_samples': None,
    'task': 'Multiclass classification (5 klas)',
    'cv_strategy': 'RepeatedStratifiedKFold(n_splits=5, n_repeats=3)',
    'test_size': 0.20,
    'random_state': RANDOM_STATE,
    'imbalance_handling': 'SMOTE-ENN',
    'scaling': 'RobustScaler',
    'imputation': 'SimpleImputer(median)',
    'hypothesis': 'XGBoost/RF osiagnie F1 > 0.85 na danych HCV',
    'limitation': ('Brak pelnego nested CV; wielokrotna ewaluacja na tym samym '
                   'hold-out moze dawac lekki optymizm selekcyjny')
}
STRICT_SINGLE_TEST_EVAL = True

df = pd.read_excel('hcv_data.xlsx')
df = df.drop(columns=['Unnamed: 0'])
EXPERIMENT_CONFIG['n_samples'] = int(df.shape[0])

print("\n" + "=" * 60)
print("=" * 60)
for key, value in EXPERIMENT_CONFIG.items():
    print(f"{key}: {value}")


required_cols = ['Category', 'Sex', 'Age', 'ALB', 'ALP', 'ALT', 'AST',
                 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Brak wymaganych kolumn: {missing_cols}")

expected_classes = set([0, 1, 2, 3, 4])
actual_classes = set(df['Category'].dropna().unique().tolist())
if not actual_classes.issubset(expected_classes):
    raise ValueError(f"Niepoprawne etykiety Category: {sorted(actual_classes)}")

sex_vals = set(df['Sex'].dropna().unique().tolist())
if not sex_vals.issubset({0, 1}):
    raise ValueError(
        f"Niepoprawne kodowanie Sex. Oczekiwano {{0,1}}, otrzymano {sorted(sex_vals)}"
    )

class_counts = df['Category'].value_counts()
if class_counts.min() < 3:
    raise ValueError(
        f"Za malo obserwacji w co najmniej jednej klasie (min={class_counts.min()}). "
        "Rozwaz laczenie klas lub wiecej danych."
    )

print("\n" + "=" * 60)
print("PODGLAD DANYCH")
print("=" * 60)
print(f"Wymiary zbioru: {df.shape[0]} obserwacji, {df.shape[1]} zmiennych")
print(f"\nPierwsze obserwacje:")
print(df.head())
print(f"\nTypy zmiennych:\n{df.dtypes}")
print(f"\nLiczba brakujacych wartosci:\n{df.isnull().sum()}")


print("\n" + "=" * 60)
print("ROZKLAD ZMIENNEJ DOCELOWEJ (Category)")
print("=" * 60)

category_labels = {
    0: 'Dawca krwi (zdrowy)',
    1: 'Podejrzany dawca',
    2: 'Zapalenie watroby',
    3: 'Zwloknienie watroby',
    4: 'Marskosc watroby'
}

for kod, opis in category_labels.items():
    n   = (df['Category'] == kod).sum()
    pct = n / len(df) * 100
    print(f"  Klasa {kod} - {opis}: {n} obserwacji ({pct:.1f}%)")


print("\n" + "=" * 60)
print("WERYFIKACJA KODOWANIA ZMIENNEJ Sex (m=0, f=1)")
print("=" * 60)
print(f"  Mezczyzni (0): {(df['Sex'] == 0).sum()} obserwacji")
print(f"  Kobiety   (1): {(df['Sex'] == 1).sum()} obserwacji")


feature_cols = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST',
                'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

print("\n" + "=" * 60)
print("LICZBA BRAKUJACYCH WARTOSCI PRZED IMPUTACJA")
print("=" * 60)
print(df[feature_cols].isnull().sum())

print("\n" + "=" * 60)
print("ANALIZA EKSPLORACYJNA DANYCH (EDA)")
print("=" * 60)

print("\nStatystyki opisowe cech numerycznych:")
print(df[feature_cols].describe().round(3))

fig, axes = plt.subplots(3, 4, figsize=(18, 13))
axes = axes.flatten()
colors_eda = plt.cm.tab10(np.linspace(0, 1, len(category_labels)))

for idx, col in enumerate(feature_cols):
    ax = axes[idx]
    for cat_idx, (cat, label) in enumerate(category_labels.items()):
        data = df[df['Category'] == cat][col].dropna()
        ax.hist(data, alpha=0.55, label=f'K{cat}',
                bins=20, color=colors_eda[cat_idx], edgecolor='white')
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.set_xlabel('Wartosc', fontsize=9)
    ax.set_ylabel('Liczba obserwacji', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle('Rozklady cech wedlug kategorii diagnostycznej\n'
             '(K0=Zdrowy, K1=Podejrzany, K2=Zapalenie, '
             'K3=Zwloknienie, K4=Marskosc)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
print("Zapisano: eda_distributions.png")

plt.figure(figsize=(13, 10))
corr_matrix = df[feature_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f',
            cmap='coolwarm', center=0,
            mask=mask,
            linewidths=0.5,
            annot_kws={'size': 9})
plt.title('Macierz korelacji cech biochemicznych\n'
          '(dolny trojkat - korelacje Pearsona)',
          fontsize=13)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("Zapisano: correlation_matrix.png")

fig, axes = plt.subplots(3, 4, figsize=(18, 13))
axes = axes.flatten()

for idx, col in enumerate(feature_cols):
    ax = axes[idx]
    data_per_class = [df[df['Category'] == cat][col].dropna().values
                      for cat in sorted(category_labels.keys())]
    bp = ax.boxplot(data_per_class,
                    patch_artist=True,
                    labels=[f'K{c}' for c in sorted(category_labels.keys())])
    for patch, color in zip(bp['boxes'], colors_eda):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.set_xlabel('Klasa', fontsize=9)
    ax.set_ylabel('Wartosc', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

plt.suptitle('Boxploty cech wedlug kategorii diagnostycznej',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('eda_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()
print("Zapisano: eda_boxplots.png")

# 5e. Rozklad klas - wykres kolowy i slupkowy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

counts = [( df['Category'] == k).sum() for k in sorted(category_labels.keys())]
labels_short = [f'K{k}\n{v}' for k, v in category_labels.items()]

wedges, texts, autotexts = ax1.pie(
    counts,
    labels=labels_short,
    autopct='%1.1f%%',
    colors=colors_eda,
    startangle=90,
    pctdistance=0.85
)
for text in autotexts:
    text.set_fontsize(9)
ax1.set_title('Rozklad klas diagnostycznych\n(wykres kolowy)',
              fontsize=12)

bars = ax2.bar(
    [f'K{k}' for k in sorted(category_labels.keys())],
    counts,
    color=colors_eda,
    edgecolor='white',
    linewidth=1.5
)
for bar, count in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1,
             str(count),
             ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_xlabel('Klasa diagnostyczna', fontsize=11)
ax2.set_ylabel('Liczba obserwacji', fontsize=11)
ax2.set_title('Rozklad klas diagnostycznych\n(wykres slupkowy)',
              fontsize=12)
ax2.grid(True, axis='y', alpha=0.3)

plt.suptitle('Analiza niezbalansowania klas', fontsize=13)
plt.tight_layout()
plt.savefig('eda_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("Zapisano: eda_class_distribution.png")

print("\n" + "=" * 60)
print("INZYNIERIA CECH - WSKAZNIKI BIOCHEMICZNE")
print("=" * 60)

def add_engineered_features(X_in):
    X_out = X_in.copy()
    eps = 1e-8
    den_alt = X_out['ALT'].where(X_out['ALT'].abs() > eps, np.nan)
    den_alb = X_out['ALB'].where(X_out['ALB'].abs() > eps, np.nan)
    den_ggt = X_out['GGT'].where(X_out['GGT'].abs() > eps, np.nan)

    X_out['AST_ALT_ratio'] = X_out['AST'] / den_alt
    X_out['CHE_ALB_ratio'] = X_out['CHE'] / den_alb
    X_out['ALP_GGT_ratio'] = X_out['ALP'] / den_ggt
    X_out['BIL_ALB_ratio'] = X_out['BIL'] / den_alb
    X_out = X_out.replace([np.inf, -np.inf], np.nan)
    return X_out

base_feature_cols = feature_cols
new_features = [
    'AST_ALT_ratio',
    'CHE_ALB_ratio',
    'ALP_GGT_ratio',
    'BIL_ALB_ratio'
]
feature_cols_extended = base_feature_cols + new_features

print(f"Liczba cech przed inzynieria: {len(feature_cols)}")
print(f"Liczba cech po inzynierii:    {len(feature_cols_extended)}")
print(f"\nNowe cechy:")
print(f"  AST_ALT_ratio: wskaznik De Ritisa (AST/ALT)")
print(f"  CHE_ALB_ratio: wskaznik funkcji syntetycznej (CHE/ALB)")
print(f"  ALP_GGT_ratio: wskaznik cholestazy (ALP/GGT)")
print(f"  BIL_ALB_ratio: wskaznik bilirubiny (BIL/ALB)")
print("Cechy ilorazowe beda tworzone dopiero po podziale train/test.")

df_eda_engineered = add_engineered_features(df[base_feature_cols])
df_eda_plot = df.copy()
for feat in new_features:
    df_eda_plot[feat] = df_eda_engineered[feat]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
new_feat_titles = ['Wskaznik De Ritisa\n(AST/ALT)',
                   'Funkcja syntetyczna\n(CHE/ALB)',
                   'Wskaznik cholestazy\n(ALP/GGT)',
                   'Wskaznik bilirubiny\n(BIL/ALB)']

for idx, (feat, title) in enumerate(zip(new_features, new_feat_titles)):
    ax = axes[idx]
    for cat_idx, (cat, label) in enumerate(category_labels.items()):
        data = df_eda_plot[df_eda_plot['Category'] == cat][feat].dropna()
        ax.hist(data, alpha=0.6, label=f'K{cat}',
                bins=25, color=colors_eda[cat_idx], edgecolor='white')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Wartosc wskaznika', fontsize=9)
    ax.set_ylabel('Liczba obserwacji', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Rozklady nowych wskaznikow biochemicznych wedlug klas',
             fontsize=13)
plt.tight_layout()
plt.savefig('eda_engineered_features.png', dpi=300, bbox_inches='tight')
plt.show()
print("Zapisano: eda_engineered_features.png")

X = df[base_feature_cols]
y = df['Category']

target_names = [category_labels[i] for i in sorted(category_labels)]
class_names  = category_labels

print(f"\nMacierz cech bazowych X: {X.shape[0]} obserwacji "
      f"x {X.shape[1]} zmiennych")
print("\n" + "=" * 60)
print("PODZIAL NA ZBIOR TRENINGOWY I TESTOWY")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train_feat = add_engineered_features(X_train)
X_test_feat  = add_engineered_features(X_test)
feature_cols_extended = list(X_train_feat.columns)

print(f"Zbior treningowy:  {X_train.shape[0]} obserwacji "
      f"({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Zbior testowy:     {X_test.shape[0]} obserwacji "
      f"({X_test.shape[0]/len(X)*100:.1f}%)")

print("\nRozklad klas w zbiorze treningowym:")
for k, v in sorted(y_train.value_counts().items()):
    pct = v / len(y_train) * 100
    print(f"  Klasa {k} - {category_labels[k]}: "
          f"{v} obserwacji ({pct:.1f}%)")

print("\nRozklad klas w zbiorze testowym:")
for k, v in sorted(y_test.value_counts().items()):
    pct = v / len(y_test) * 100
    print(f"  Klasa {k} - {category_labels[k]}: "
          f"{v} obserwacji ({pct:.1f}%)")

print("\n" + "=" * 60)
print("KONTROLA CECH PO INZYNIERII")
print("=" * 60)
print("Braki w nowych cechach (TRAIN):")
print(X_train_feat[new_features].isnull().sum())
print("\nBraki w nowych cechach (TEST):")
print(X_test_feat[new_features].isnull().sum())
print("\n" + "=" * 60)
print("SMOTE-ENN - KONFIGURACJA")
print("=" * 60)

min_class_size = y_train.value_counts().min()
k_neighbors_smote = min(3, min_class_size - 1)
print(f"Minimalna liczba obserwacji w klasie: {min_class_size}")
print(f"Uzywane k_neighbors dla SMOTE: {k_neighbors_smote}")

print("\nRozklad klas PRZED SMOTE-ENN:")
for k, v in sorted(pd.Series(y_train).value_counts().items()):
    pct = v / len(y_train) * 100
    print(f"  Klasa {k} - {category_labels[k]}: {v} obs ({pct:.1f}%)")
print("\nResampling SMOTE-ENN jest wykonywany wyłącznie wewnątrz pipeline CV/treningu.")

print("\n" + "=" * 60)
print("DEFINICJA MODELI")
print("=" * 60)

def make_pipeline(clf):
    """Tworzy pelny pipeline: imputer + scaler + SMOTE-ENN + clf"""
    return ImbPipeline([
        ('imputer',   SimpleImputer(strategy='median')),
        ('scaler',    RobustScaler()),
        ('smote_enn', SMOTEENN(
            random_state=RANDOM_STATE,
            smote=SMOTE(k_neighbors=k_neighbors_smote,
                        random_state=RANDOM_STATE)
        )),
        ('clf', clf)
    ])

model_constructors = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE
    ),
    'SVM': SVC(
        probability=True,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    ),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss',
        verbosity=0,
        n_jobs=1
    )
}

model_pipelines = {
    name: make_pipeline(clf)
    for name, clf in model_constructors.items()
}

for name in model_pipelines:
    print(f"  Zdefiniowano pipeline: {name}")


def bootstrap_metrics(y_true, y_pred, y_proba,
                      n_bootstrap=3000, random_state=RANDOM_STATE,
                      min_count_per_class=2, min_valid_boot=500):
    rng = np.random.RandomState(random_state)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_proba_arr = np.array(y_proba)

    classes = np.unique(y_true_arr)
    n = len(y_true_arr)
    f1_scores = []
    auc_scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true_arr[idx]
        yp = y_pred_arr[idx]
        ypr = y_proba_arr[idx]

        counts = pd.Series(yt).value_counts()
        if any(counts.get(c, 0) < min_count_per_class for c in classes):
            continue

        try:
            f1_scores.append(
                f1_score(yt, yp, average='macro', zero_division=0)
            )
            auc_scores.append(
                roc_auc_score(yt, ypr,
                              multi_class='ovr', average='macro')
            )
        except ValueError:
            continue

    if len(f1_scores) < min_valid_boot or len(auc_scores) < min_valid_boot:
        raise RuntimeError(
            f"Za malo poprawnych replik bootstrap "
            f"(F1={len(f1_scores)}, AUC={len(auc_scores)}). "
            "Zwiksze n_bootstrap lub uprosc problem klas."
        )

    f1_ci = (np.percentile(f1_scores, 2.5),
             np.percentile(f1_scores, 97.5))
    auc_ci = (np.percentile(auc_scores, 2.5),
              np.percentile(auc_scores, 97.5))
    return {'F1_ci': f1_ci, 'AUC_ci': auc_ci}

print("\n" + "=" * 60)
print("WALIDACJA KRZYZOWA (Repeated 5x3) - PIPELINE Z SMOTE-ENN")
print("=" * 60)

cv_main = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=3, random_state=RANDOM_STATE
)
cv_results_summary = {}

for name, pipe in model_pipelines.items():
    print(f"CV dla: {name}...")
    scores = cross_validate(
        pipe,
        X_train_feat.values, y_train.values,
        cv=cv_main,
        scoring={
            'f1_macro':          'f1_macro',
            'balanced_accuracy': 'balanced_accuracy',
            'roc_auc_ovr':       'roc_auc_ovr'
        },
        n_jobs=-1
    )

    cv_results_summary[name] = {
        'CV_F1_mean':      scores['test_f1_macro'].mean(),
        'F1 Macro (mean)': scores['test_f1_macro'].mean(),
        'F1 Macro (std)':  scores['test_f1_macro'].std(),
        'Bal.Acc (mean)':  scores['test_balanced_accuracy'].mean(),
        'AUC-ROC (mean)':  scores['test_roc_auc_ovr'].mean(),
        'F1_raw':          scores['test_f1_macro']
    }

    print(f"  F1 Macro:  "
          f"{cv_results_summary[name]['F1 Macro (mean)']:.4f}"
          f" +/- {cv_results_summary[name]['F1 Macro (std)']:.4f}")
    print(f"  Bal.Acc:   "
          f"{cv_results_summary[name]['Bal.Acc (mean)']:.4f}")
    print(f"  AUC-ROC:   "
          f"{cv_results_summary[name]['AUC-ROC (mean)']:.4f}")

cv_top2 = sorted(
    cv_results_summary.items(),
    key=lambda x: x[1]['CV_F1_mean'],
    reverse=True
)[:2]
if len(cv_top2) == 2:
    name_a, vals_a = cv_top2[0]
    name_b, vals_b = cv_top2[1]
    diff = vals_a['F1_raw'] - vals_b['F1_raw']
    stat_sh, p_sh = shapiro(diff)
    print("\nTest normalnosci roznic (Shapiro-Wilk) dla wynikow CV:")
    print(f"  {name_a} - {name_b}: W={stat_sh:.4f}, p={p_sh:.4f}")

    if p_sh < 0.05:
        stat_w, p_w = wilcoxon(vals_a['F1_raw'], vals_b['F1_raw'])
        print("Test Wilcoxona dla wynikow CV (F1 macro):")
        print(f"  {name_a} vs {name_b}: stat={stat_w:.4f}, p={p_w:.4f}")
        if p_w < 0.05:
            print("  Wynik: roznica istotna statystycznie (p < 0.05)")
        else:
            print("  Wynik: brak istotnej roznicy (p >= 0.05)")
    else:
        stat_t, p_t = ttest_rel(vals_a['F1_raw'], vals_b['F1_raw'])
        print("Test t dla prob zaleznych (normalne roznice):")
        print(f"  {name_a} vs {name_b}: t={stat_t:.4f}, p={p_t:.4f}")
        if p_t < 0.05:
            print("  Wynik: roznica istotna statystycznie (p < 0.05)")
        else:
            print("  Wynik: brak istotnej roznicy (p >= 0.05)")


print("\n" + "=" * 60)
print("KRZYWE UCZENIA")
print("=" * 60)

def plot_learning_curve(pipeline, X, y, model_name,
                        cv_lc, figsize=(8, 5)):
    train_sizes_abs, train_scores, val_scores = learning_curve(
        pipeline, X, y,
        cv=cv_lc,
        scoring='f1_macro',
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train_sizes_abs, train_mean,
            'o-', color='blue', lw=2, label='Zbior treningowy')
    ax.fill_between(train_sizes_abs,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.15, color='blue')
    ax.plot(train_sizes_abs, val_mean,
            's-', color='red', lw=2, label='Zbior walidacyjny (CV)')
    ax.fill_between(train_sizes_abs,
                    val_mean - val_std,
                    val_mean + val_std,
                    alpha=0.15, color='red')

    ax.set_xlabel('Rozmiar zbioru treningowego', fontsize=12)
    ax.set_ylabel('F1 Macro', fontsize=12)
    ax.set_title(f'Krzywa uczenia\nModel: {model_name}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    gap = train_mean[-1] - val_mean[-1]
    ax.text(0.05, 0.05,
            f'Roznica train-val (ostatni punkt): {gap:.3f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    plt.tight_layout()
    fname = f'learning_curve_{model_name.replace(" ", "_")}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Zapisano: {fname}")

cv_lc = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=RANDOM_STATE)

for lc_model in ['Random Forest', 'XGBoost']:
    print(f"Obliczam krzywa uczenia dla: {lc_model}...")
    plot_learning_curve(
        model_pipelines[lc_model],
        X_train_feat.values,
        y_train.values,
        model_name=lc_model,
        cv_lc=cv_lc
    )


print("\n" + "=" * 60)
print("TRENOWANIE FINALNYCH MODELI (pelny pipeline)")
print("=" * 60)

trained_pipelines = {}
X_train_raw = X_train_feat.values
X_test_raw  = X_test_feat.values

for name, pipe in model_pipelines.items():
    print(f"Trenuję pipeline: {name}...")
    pipe.fit(X_train_raw, y_train.values)
    trained_pipelines[name] = pipe
    print(f"  Gotowy")

def evaluate_model(name, pipeline, X_test_data,
                   y_test_true, labels):
    y_pred  = pipeline.predict(X_test_data)
    y_proba = pipeline.predict_proba(X_test_data)

    acc      = accuracy_score(y_test_true, y_pred)
    bal_acc  = balanced_accuracy_score(y_test_true, y_pred)
    f1_macro = f1_score(y_test_true, y_pred,
                        average='macro', zero_division=0)
    auc_roc  = roc_auc_score(y_test_true, y_proba,
                              multi_class='ovr', average='macro')
    bs = bootstrap_metrics(y_test_true, y_pred, y_proba,
                           n_bootstrap=2000)

    print(f"\n{'='*60}")
    print(f"MODEL: {name}")
    print(f"{'='*60}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  F1-score (macro):  {f1_macro:.4f}  "
          f"[95% CI: {bs['F1_ci'][0]:.4f} - {bs['F1_ci'][1]:.4f}]")
    print(f"  AUC-ROC (macro):   {auc_roc:.4f}  "
          f"[95% CI: {bs['AUC_ci'][0]:.4f} - {bs['AUC_ci'][1]:.4f}]")
    print(f"\nRaport klasyfikacji:")
    print(classification_report(
        y_test_true, y_pred,
        target_names=[labels[i] for i in sorted(labels)],
        zero_division=0
    ))

    return {
        'Model':             name,
        'Accuracy':          acc,
        'Balanced Accuracy': bal_acc,
        'F1 Macro':          f1_macro,
        'F1 CI lower':       bs['F1_ci'][0],
        'F1 CI upper':       bs['F1_ci'][1],
        'AUC-ROC':           auc_roc,
        'AUC CI lower':      bs['AUC_ci'][0],
        'AUC CI upper':      bs['AUC_ci'][1],
        'y_pred':            y_pred,
        'y_proba':           y_proba
    }


print("\n" + "=" * 60)
print("EWALUACJA NA ZBIORZE TESTOWYM")
print("=" * 60)

test_results = {}
results_list = []
if STRICT_SINGLE_TEST_EVAL:
    print("wybor modelu tylko na podstawie CV.")
    best_name_cv = max(cv_results_summary, key=lambda x: cv_results_summary[x]['CV_F1_mean'])
    print(f"Najlepszy model wg CV F1: {best_name_cv}")
else:
    for name, pipe in trained_pipelines.items():
        res = evaluate_model(name, pipe, X_test_raw,
                             y_test, category_labels)
        test_results[name] = res
        results_list.append({
            'Model':             res['Model'],
            'Accuracy':          res['Accuracy'],
            'Balanced Accuracy': res['Balanced Accuracy'],
            'F1 Macro':          res['F1 Macro'],
            'F1 CI lower':       res['F1 CI lower'],
            'F1 CI upper':       res['F1 CI upper'],
            'AUC-ROC':           res['AUC-ROC'],
            'AUC CI lower':      res['AUC CI lower'],
            'AUC CI upper':      res['AUC CI upper']
        })

    results_df = pd.DataFrame(results_list).set_index('Model')
    results_df = results_df.sort_values('F1 Macro', ascending=False)

    print("\n" + "="*60)
    print("TABELA POROWNAWCZA MODELI BAZOWYCH")
    print("="*60)
    print(results_df.round(4).to_string())

print("\n" + "=" * 60)
print("GRID SEARCH - Random Forest")
print("=" * 60)

rf_pipeline_gs = ImbPipeline([
    ('imputer',   SimpleImputer(strategy='median')),
    ('scaler',    RobustScaler()),
    ('smote_enn', SMOTEENN(
        random_state=RANDOM_STATE,
        smote=SMOTE(k_neighbors=k_neighbors_smote,
                    random_state=RANDOM_STATE)
    )),
    ('clf', RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1
    ))
])

rf_param_grid = {
    'clf__n_estimators':      [100, 200],
    'clf__max_depth':         [5, 10, None],
    'clf__min_samples_split': [5, 10],
    'clf__min_samples_leaf':  [1, 2],
    'clf__class_weight':      [None, 'balanced']
}

cv_gs = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=RANDOM_STATE)

rf_gs = GridSearchCV(
    rf_pipeline_gs, rf_param_grid,
    cv=cv_gs, scoring='f1_macro',
    n_jobs=-1, verbose=1, refit=True
)

print("Uruchamiam Grid Search dla Random Forest...")
rf_gs.fit(X_train_raw, y_train.values)
pd.DataFrame(rf_gs.cv_results_).to_csv('rf_gridsearch_results.csv', index=False)
print("Zapisano: rf_gridsearch_results.csv")

print(f"\nNajlepsze parametry RF: {rf_gs.best_params_}")
print(f"Najlepszy F1 Macro(CV): {rf_gs.best_score_:.4f}")

rf_tuned = rf_gs.best_estimator_

print("\n" + "=" * 60)
print("GRID SEARCH - SVM")
print("=" * 60)

svm_pipeline_gs = ImbPipeline([
    ('imputer',   SimpleImputer(strategy='median')),
    ('scaler',    RobustScaler()),
    ('smote_enn', SMOTEENN(
        random_state=RANDOM_STATE,
        smote=SMOTE(k_neighbors=k_neighbors_smote,
                    random_state=RANDOM_STATE)
    )),
    ('clf', SVC(probability=True, random_state=RANDOM_STATE))
])

svm_param_grid = {
    'clf__C':            [0.1, 1, 10, 100],
    'clf__gamma':        ['scale', 'auto', 0.001, 0.01],
    'clf__kernel':       ['rbf', 'linear'],
    'clf__class_weight': [None, 'balanced']
}

svm_gs = GridSearchCV(
    svm_pipeline_gs, svm_param_grid,
    cv=cv_gs, scoring='f1_macro',
    n_jobs=-1, verbose=1, refit=True
)

print("Uruchamiam Grid Search dla SVM...")
svm_gs.fit(X_train_raw, y_train.values)
pd.DataFrame(svm_gs.cv_results_).to_csv('svm_gridsearch_results.csv', index=False)
print("Zapisano: svm_gridsearch_results.csv")

print(f"\nNajlepsze parametry SVM: {svm_gs.best_params_}")
print(f"Najlepszy F1 Macro(CV):  {svm_gs.best_score_:.4f}")

svm_tuned = svm_gs.best_estimator_

print("\n" + "=" * 60)
print("VOTING CLASSIFIER - Top 3 modele (wg CV F1)")
print("=" * 60)

cv_f1_sorted = sorted(
    cv_results_summary.items(),
    key=lambda x: x[1]['CV_F1_mean'],
    reverse=True
)
top3_names = [name for name, _ in cv_f1_sorted[:3]]
print(f"Top 3 modele (wg CV F1): {top3_names}")

voting_estimators = []
for name in top3_names:
    if name == 'Random Forest':
        pipe = rf_gs.best_estimator_
    elif name == 'SVM':
        pipe = svm_gs.best_estimator_
    else:
        pipe = make_pipeline(model_constructors[name])
    voting_estimators.append((name, pipe))

voting_clf = VotingClassifier(
    estimators=voting_estimators,
    voting='soft'
)

print("CV dla Voting Classifier (Top 3)...")
voting_cv_scores = cross_validate(
    voting_clf,
    X_train_raw, y_train.values,
    cv=cv_main,
    scoring={
        'f1_macro':          'f1_macro',
        'balanced_accuracy': 'balanced_accuracy',
        'roc_auc_ovr':       'roc_auc_ovr'
    },
    n_jobs=-1
)
cv_results_summary['Voting Classifier (Top 3)'] = {
    'CV_F1_mean':      voting_cv_scores['test_f1_macro'].mean(),
    'F1 Macro (mean)': voting_cv_scores['test_f1_macro'].mean(),
    'F1 Macro (std)':  voting_cv_scores['test_f1_macro'].std(),
    'Bal.Acc (mean)':  voting_cv_scores['test_balanced_accuracy'].mean(),
    'AUC-ROC (mean)':  voting_cv_scores['test_roc_auc_ovr'].mean(),
    'F1_raw':          voting_cv_scores['test_f1_macro']
}
print(f"  F1 Macro:  {cv_results_summary['Voting Classifier (Top 3)']['F1 Macro (mean)']:.4f}"
      f" +/- {cv_results_summary['Voting Classifier (Top 3)']['F1 Macro (std)']:.4f}")

voting_clf.fit(X_train_raw, y_train.values)

print("\n" + "=" * 60)
print("EWALUACJA MODELU BAZOWEGO (ARGMAX)")
print("=" * 60)

best_cv_name = best_name_cv if STRICT_SINGLE_TEST_EVAL else cv_f1_sorted[0][0]
print(f"Model bazowy: {best_cv_name}")

best_full_pipeline = trained_pipelines[best_cv_name]

y_pred_best = best_full_pipeline.predict(X_test_raw)
y_proba_best = best_full_pipeline.predict_proba(X_test_raw)

f1_best = f1_score(y_test, y_pred_best, average='macro', zero_division=0)
bal_best = balanced_accuracy_score(y_test, y_pred_best)
auc_best = roc_auc_score(y_test, y_proba_best,
                         multi_class='ovr', average='macro')
bs_best = bootstrap_metrics(y_test, y_pred_best, y_proba_best)

print(f"\n{best_cv_name} (argmax, bez strojenia progow):")
print(f"  Accuracy:          {accuracy_score(y_test, y_pred_best):.4f}")
print(f"  Balanced Accuracy: {bal_best:.4f}")
print(f"  F1 Macro:          {f1_best:.4f}  "
      f"[95% CI: {bs_best['F1_ci'][0]:.4f} - {bs_best['F1_ci'][1]:.4f}]")
print(f"  AUC-ROC:           {auc_best:.4f}  "
      f"[95% CI: {bs_best['AUC_ci'][0]:.4f} - {bs_best['AUC_ci'][1]:.4f}]")
print(classification_report(y_test, y_pred_best,
                             target_names=target_names,
                             zero_division=0))

test_results[f'{best_cv_name} (argmax)'] = {
    'Accuracy':          accuracy_score(y_test, y_pred_best),
    'Balanced Accuracy': bal_best,
    'F1 Macro':          f1_best,
    'F1 CI lower':       bs_best['F1_ci'][0],
    'F1 CI upper':       bs_best['F1_ci'][1],
    'AUC-ROC':           auc_best,
    'AUC CI lower':      bs_best['AUC_ci'][0],
    'AUC CI upper':      bs_best['AUC_ci'][1],
    'y_pred':            y_pred_best,
    'y_proba':           y_proba_best
}

print("\n" + "=" * 60)
print("TEST McNEMARA - POROWNANIE STATYSTYCZNE MODELI")
print("=" * 60)

try:
    from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test_fn

    def run_mcnemar(y_true, y_pred1, y_pred2,
                    name1='Model 1', name2='Model 2',
                    alpha=0.05, n_tests=1):
        correct1 = (np.array(y_pred1) == np.array(y_true))
        correct2 = (np.array(y_pred2) == np.array(y_true))
        b = np.sum(~correct1 &  correct2)
        c = np.sum( correct1 & ~correct2)
        table = np.array([
            [np.sum(~correct1 & ~correct2), b],
            [c, np.sum(correct1 & correct2)]
        ])
        result = mcnemar_test_fn(table, exact=False, correction=True)
        alpha_corr = alpha / max(1, n_tests)
        print(f"\n  {name1} vs {name2}:")
        print(f"    b={b}, c={c}  |  "
              f"Chi2={result.statistic:.4f}  "
              f"p={result.pvalue:.4f}  "
              f"(Bonferroni alpha={alpha_corr:.5f})", end="  ")
        if result.pvalue < alpha_corr:
            print("-> Istotna roznica")
        else:
            print("-> Brak istotnej roznicy")
        return result

    model_names_list = list(test_results.keys())
    n_models = len(model_names_list)
    if n_models < 2:
        print("Za malo modeli do testu McNemara (wymagane >=2).")
    else:
        n_pairwise_tests = n_models * (n_models - 1) // 2
        print(f"Liczba porownan: {n_pairwise_tests} (korekta Bonferroniego)")
        for i in range(len(model_names_list)):
            for j in range(i + 1, len(model_names_list)):
                n1 = model_names_list[i]
                n2 = model_names_list[j]
                run_mcnemar(y_test,
                            test_results[n1]['y_pred'],
                            test_results[n2]['y_pred'],
                            name1=n1, name2=n2,
                            n_tests=n_pairwise_tests)

except ImportError:
    print("UWAGA: statsmodels nie jest zainstalowany.")
    print("Zainstaluj: pip install statsmodels")

print("\n" + "=" * 60)
print("TABELA KONCOWA - WSZYSTKIE MODELE Z CI")
print("=" * 60)

final_rows = []
for name, res in test_results.items():
    final_rows.append({
        'Model':             name,
        'Accuracy':          res['Accuracy'],
        'Balanced Accuracy': res['Balanced Accuracy'],
        'F1 Macro':          res['F1 Macro'],
        'F1 CI 95%':         f"[{res['F1 CI lower']:.4f}, "
                             f"{res['F1 CI upper']:.4f}]",
        'AUC-ROC':           res['AUC-ROC'],
        'AUC CI 95%':        f"[{res['AUC CI lower']:.4f}, "
                             f"{res['AUC CI upper']:.4f}]"
    })

final_df = pd.DataFrame(final_rows).set_index('Model')
final_df = final_df.sort_values('F1 Macro', ascending=False)
print(final_df.round(4).to_string())

print("\n" + "=" * 60)
print("ANALIZA BLEDOW KLASYFIKACJI")
print("=" * 60)

def detailed_error_analysis(y_true, y_pred, class_labels):
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    print("\nANALIZA BLEDOW PER KLASA:")
    print("-" * 50)

    for true_class in sorted(class_labels.keys()):
        mask = (y_true_arr == true_class)
        total = int(mask.sum())
        if total == 0:
            continue

        y_true_sub = y_true_arr[mask]
        y_pred_sub = y_pred_arr[mask]
        correct = int((y_true_sub == y_pred_sub).sum())

        print(f"\nKlasa {true_class} ({class_labels[true_class]}):")
        print(f"  Poprawnie: {correct}/{total} ({correct / total * 100:.1f}%)")

        errors = y_pred_sub[y_true_sub != y_pred_sub]
        if len(errors) > 0:
            error_counts = pd.Series(errors).value_counts()
            print("  Bledy (przewidziano jako):")
            for pred_class, count in error_counts.items():
                print(f"    -> Klasa {pred_class} ({class_labels[pred_class]}): {count}x")
        else:
            print("  Brak bledow dla tej klasy.")

best_model_name_for_errors = final_df.index[0]
errors_df = pd.DataFrame({
    'y_true': y_test.values,
    'y_pred': test_results[best_model_name_for_errors]['y_pred']
})
errors_df = errors_df[errors_df['y_true'] != errors_df['y_pred']]
print(f"Liczba blednie sklasyfikowanych przypadkow: {len(errors_df)}")
if not errors_df.empty:
    print("Najczestsze pomylki (y_true -> y_pred):")
    print(
        errors_df.groupby(['y_true', 'y_pred'])
        .size()
        .sort_values(ascending=False)
        .to_string()
    )
else:
    print("Brak bledow klasyfikacji dla najlepszego modelu.")

detailed_error_analysis(
    y_test.values,
    test_results[best_model_name_for_errors]['y_pred'],
    category_labels
)

print("\n" + "=" * 60)
print("WIZUALIZACJA: MACIERZE POMYLEK")
print("=" * 60)

def plot_confusion_matrices(results_dict, y_true, labels,
                             figsize=(20, 14)):
    n_models = len(results_dict)
    ncols    = 3
    nrows    = int(np.ceil(n_models / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes      = axes.flatten()
    short_labels = [f'K{i}' for i in range(len(labels))]

    for idx, (name, res) in enumerate(results_dict.items()):
        if idx >= len(axes):
            break
        cm      = confusion_matrix(y_true, res['y_pred'])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_norm, annot=True, fmt='.2f',
                    cmap='Blues',
                    xticklabels=short_labels,
                    yticklabels=short_labels,
                    ax=axes[idx], vmin=0, vmax=1,
                    linewidths=0.5)
        f1_val = res['F1 Macro']
        axes[idx].set_title(f'{name}\nF1={f1_val:.3f}',
                            fontsize=10)
        axes[idx].set_xlabel('Przewidywana klasa', fontsize=9)
        axes[idx].set_ylabel('Rzeczywista klasa',  fontsize=9)

    for idx in range(len(results_dict), len(axes)):
        axes[idx].set_visible(False)

    legend_text = '\n'.join(
        [f'K{i}: {labels[i]}' for i in sorted(labels)]
    )
    fig.text(0.01, 0.01, legend_text, fontsize=8,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round',
                       facecolor='wheat', alpha=0.3))

    plt.suptitle('Znormalizowane macierze pomylek\n'
                 '(wartosci = recall per class)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('confusion_matrices.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Zapisano: confusion_matrices.png")

plot_confusion_matrices(test_results, y_test, category_labels)

print("\n" + "=" * 60)
print("WIZUALIZACJA: KRZYWE ROC (One-vs-Rest)")
print("=" * 60)

def plot_roc_curves_multiclass(y_true, y_proba, class_labels,
                                model_name, figsize=(10, 8)):
    n_cls  = len(class_labels)
    y_bin  = label_binarize(y_true, classes=list(range(n_cls)))
    colors = plt.cm.tab10(np.linspace(0, 1, n_cls))

    fig, ax = plt.subplots(figsize=figsize)
    fpr_list = []
    tpr_list = []

    for i, (label, color) in enumerate(
            zip(class_labels.values(), colors)):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc_i        = auc(fpr_i, tpr_i)
        fpr_list.append(fpr_i)
        tpr_list.append(tpr_i)
        ax.plot(fpr_i, tpr_i, color=color, lw=2,
                label=f'K{i}: {label} (AUC={roc_auc_i:.3f})')

    all_fpr  = np.unique(np.concatenate(fpr_list))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr_i, tpr_i in zip(fpr_list, tpr_list):
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= n_cls
    macro_auc = auc(all_fpr, mean_tpr)

    ax.plot(all_fpr, mean_tpr, 'k--', lw=2.5,
            label=f'Makro-srednia (AUC={macro_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'gray', lw=1,
            linestyle=':', label='Klasyfikator losowy')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Odsetek falszywie pozytywnych (FPR)', fontsize=12)
    ax.set_ylabel('Odsetek prawdziwie pozytywnych (TPR)', fontsize=12)
    ax.set_title(f'Krzywe ROC (One-vs-Rest)\nModel: {model_name}',
                 fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    fname = f'roc_{model_name.replace(" ", "_")}.png'
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Zapisano: {fname}")

best_model_name = final_df.index[0]
if best_model_name in test_results:
    plot_roc_curves_multiclass(
        y_test,
        test_results[best_model_name]['y_proba'],
        category_labels,
        model_name=best_model_name
    )

if 'Voting Classifier (Top 3)' in test_results:
    plot_roc_curves_multiclass(
        y_test,
        test_results['Voting Classifier (Top 3)']['y_proba'],
        category_labels,
        model_name='Voting Classifier (Top 3)'
    )

print("\n" + "=" * 60)
print("WIZUALIZACJA: KRZYWE PRECISION-RECALL")
print("=" * 60)

def plot_precision_recall_multiclass(y_true, y_proba,
                                      class_labels, model_name,
                                      figsize=(10, 8)):
    n_cls  = len(class_labels)
    y_bin  = label_binarize(y_true, classes=list(range(n_cls)))
    colors = plt.cm.tab10(np.linspace(0, 1, n_cls))

    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, color) in enumerate(
            zip(class_labels.values(), colors)):
        prec, rec, _ = precision_recall_curve(
            y_bin[:, i], y_proba[:, i]
        )
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, color=color, lw=2,
                label=f'K{i}: {label} (AUC={pr_auc:.3f})')

        # Baseline (rozklad klas)
        baseline = y_bin[:, i].mean()
        ax.axhline(y=baseline, color=color,
                   linestyle=':', alpha=0.4, lw=1)

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Krzywe Precision-Recall\nModel: {model_name}',
                 fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    ax.text(0.02, 0.02,
            'Linie przerywane = baseline (rozklad klas)',
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle='round',
                      facecolor='lightyellow', alpha=0.5))

    fname = f'pr_curve_{model_name.replace(" ", "_")}.png'
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Zapisano: {fname}")

if best_model_name in test_results:
    plot_precision_recall_multiclass(
        y_test,
        test_results[best_model_name]['y_proba'],
        category_labels,
        model_name=best_model_name
    )

if 'Voting Classifier (Top 3)' in test_results:
    plot_precision_recall_multiclass(
        y_test,
        test_results['Voting Classifier (Top 3)']['y_proba'],
        category_labels,
        model_name='Voting Classifier (Top 3)'
    )

print("\n" + "=" * 60)
print("WIZUALIZACJA: KALIBRACJA PRAWDOPODOBIENSTW")
print("=" * 60)

def plot_calibration_multiclass(y_true, y_proba, model_name,
                                n_classes=5, figsize=(8, 6)):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i in range(n_classes):
        fraction_pos, mean_pred = calibration_curve(
            y_bin[:, i], y_proba[:, i], n_bins=10, strategy='quantile'
        )
        ax.plot(mean_pred, fraction_pos, marker='o',
                color=colors[i], label=f'Klasa {i}')

    ax.plot([0, 1], [0, 1], 'k--', label='Idealna kalibracja')
    ax.set_xlabel('Srednie przewidywane prawdopodobienstwo')
    ax.set_ylabel('Frakcja pozytywnych')
    ax.set_title(f'Krzywa kalibracji\n{model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f'calibration_{model_name.replace(" ", "_")}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Zapisano: {fname}")

if best_model_name in test_results:
    plot_calibration_multiclass(
        y_test,
        test_results[best_model_name]['y_proba'],
        model_name=best_model_name,
        n_classes=len(category_labels)
    )

print("\n" + "=" * 60)
print("WIZUALIZACJA: WAZNOSC CECH")
print("=" * 60)

def plot_feature_importance(pipelines_dict, feat_names,
                             model_names=('Random Forest', 'XGBoost'),
                             figsize=(14, 7)):
    available = [n for n in model_names if n in pipelines_dict]
    if not available:
        print("Brak modeli drzewiastych.")
        return

    fig, axes = plt.subplots(1, len(available), figsize=figsize)
    if len(available) == 1:
        axes = [axes]

    for ax, name in zip(axes, available):
        clf = pipelines_dict[name].named_steps['clf']
        importances = clf.feature_importances_
        importances = importances / importances.sum()

        idx             = np.argsort(importances)[::-1]
        sorted_features = [feat_names[i] for i in idx]
        sorted_imp      = importances[idx]

        threshold = np.percentile(importances, 75)
        colors_fi = ['#d62728' if imp >= threshold
                     else '#1f77b4'
                     for imp in sorted_imp]

        ax.barh(range(len(sorted_features)),
                sorted_imp[::-1],
                color=colors_fi[::-1])
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features[::-1], fontsize=10)
        ax.set_xlabel('Wzgledna waznosc cechy', fontsize=11)
        ax.set_title(f'Waznosc cech\n{name}', fontsize=12)
        ax.axvline(x=1 / len(feat_names),
                   color='green', linestyle='--', alpha=0.6,
                   label=f'Rowna waznosc (1/{len(feat_names)})')
        ax.axvline(x=threshold,
                   color='red', linestyle=':', alpha=0.6,
                   label=f'75. percentyl ({threshold:.3f})')
        ax.legend(fontsize=9)
        ax.grid(True, axis='x', alpha=0.3)

    plt.suptitle('Waznosc cech - modele drzewiaste\n'
                 '(czerwone = top 25% najwazniejszych)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('feature_importance.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Zapisano: feature_importance.png")

plot_feature_importance(
    trained_pipelines,
    feature_cols_extended,
    model_names=('Random Forest', 'XGBoost')
)


print("\n" + "=" * 60)
print("WIZUALIZACJA: POROWNANIE MODELI Z CI")
print("=" * 60)

def plot_model_comparison_with_ci(results_dict, metric='F1 Macro',
                                   figsize=(14, 6)):
    names  = list(results_dict.keys())
    values = [results_dict[n][metric] for n in names]

    ci_key_map = {
        'F1 Macro': ('F1 CI lower', 'F1 CI upper'),
        'AUC-ROC':  ('AUC CI lower', 'AUC CI upper')
    }
    low_key, up_key = ci_key_map.get(metric, (None, None))

    if low_key and up_key:
        ci_low = [results_dict[n].get(low_key, values[i])
                  for i, n in enumerate(names)]
        ci_up  = [results_dict[n].get(up_key,  values[i])
                  for i, n in enumerate(names)]
    else:
        ci_low = values
        ci_up  = values

    order  = np.argsort(values)[::-1]
    names  = [names[i]  for i in order]
    values = [values[i] for i in order]
    ci_low = [ci_low[i] for i in order]
    ci_up  = [ci_up[i]  for i in order]

    yerr_low = [max(0, v - l) for v, l in zip(values, ci_low)]
    yerr_up  = [max(0, u - v) for v, u in zip(values, ci_up)]

    fig, ax = plt.subplots(figsize=figsize)
    x       = np.arange(len(names))

    bar_colors = ['#d62728' if i == 0 else '#1f77b4'
                  for i in range(len(names))]

    bars = ax.bar(
        x, values,
        yerr=[yerr_low, yerr_up],
        capsize=6,
        color=bar_colors,
        alpha=0.85,
        error_kw={'elinewidth': 2,
                  'ecolor': 'black',
                  'capthick': 2}
    )

    for bar, val, lo, hi in zip(bars, values, ci_low, ci_up):
        ax.text(bar.get_x() + bar.get_width() / 2,
                hi + 0.008,
                f'{val:.3f}',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35,
                       ha='right', fontsize=10)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(
        f'Porownanie modeli - {metric}\n'
        f'(slupki bledu = 95% CI bootstrap, n=2000)',
        fontsize=13
    )
    ax.set_ylim(0, min(1.15, max(values) + 0.18))
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=max(values), color='red',
               linestyle='--', alpha=0.4,
               label=f'Najlepszy: {max(values):.3f}')
    ax.legend(fontsize=10)

    plt.tight_layout()
    fname = f'model_comparison_{metric.replace(" ", "_")}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Zapisano: {fname}")

plot_model_comparison_with_ci(test_results, metric='F1 Macro')
plot_model_comparison_with_ci(test_results, metric='AUC-ROC')

print("\n" + "=" * 60)
print("WIZUALIZACJA: WYNIKI WALIDACJI KRZYZOWEJ")
print("=" * 60)

cv_plot_data = pd.DataFrame({
    name: {
        'F1 Macro':        vals['F1 Macro (mean)'],
        'Balanced Acc':    vals['Bal.Acc (mean)'],
        'AUC-ROC':         vals['AUC-ROC (mean)']
    }
    for name, vals in cv_results_summary.items()
}).T

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(cv_plot_data, annot=True, fmt='.4f',
            cmap='YlOrRd', ax=ax,
            linewidths=0.5,
            annot_kws={'size': 11})
ax.set_title('Wyniki walidacji krzyzowej (5-fold)\n'
             'srednie wartosci metryk',
             fontsize=13)
ax.set_xlabel('Metryka', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
plt.tight_layout()
plt.savefig('cv_results_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("Zapisano: cv_results_heatmap.png")

print("\n" + "=" * 60)
print("ANALIZA SHAP - INTERPRETOWALNOSC MODELU")
print("=" * 60)

try:
    import shap


    def shap_analysis(pipeline, X_test_data, feat_names,
                      model_name='Model', max_display=12):
        clf = pipeline.named_steps['clf']
        # Uzywamy tych samych, juz nauczonych transformacji co w pipeline.
        X_df = pd.DataFrame(X_test_data, columns=feat_names)
        X_imputed = pipeline.named_steps['imputer'].transform(X_df)
        X_transformed = pipeline.named_steps['scaler'].transform(X_imputed)
        X_transformed = np.array(X_transformed)

        try:
            explainer   = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_transformed)
            print("  Uzywam TreeExplainer")
        except Exception as e:
            print(f"  TreeExplainer niedostepny ({e})")
            background  = shap.sample(
                X_transformed, min(50, len(X_transformed))
            )
            explainer   = shap.KernelExplainer(
                clf.predict_proba, background
            )
            shap_values = explainer.shap_values(X_transformed)
            print("  Uzywam KernelExplainer")

        # Summary plot - bar (srednia absolutna)
        plt.figure(figsize=(10, 7))
        if isinstance(shap_values, list):
            shap_mean = np.mean(
                [np.abs(sv) for sv in shap_values], axis=0
            )
            shap.summary_plot(
                shap_mean, X_transformed,
                feature_names=feat_names,
                max_display=max_display,
                show=False, plot_type='bar'
            )
        else:
            shap.summary_plot(
                shap_values, X_transformed,
                feature_names=feat_names,
                max_display=max_display,
                show=False
            )

        plt.title(f'SHAP Summary Plot (bar)\nModel: {model_name}',
                  fontsize=13)
        plt.tight_layout()
        fname = f'shap_bar_{model_name.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"  Zapisano: {fname}")

        # Summary plot - beeswarm (dla pierwszej klasy)
        if isinstance(shap_values, list) and len(shap_values) > 0:
            plt.figure(figsize=(10, 7))
            shap.summary_plot(
                shap_values[0], X_transformed,
                feature_names=feat_names,
                max_display=max_display,
                show=False
            )
            plt.title(
                f'SHAP Beeswarm (Klasa 0: Zdrowy)\n'
                f'Model: {model_name}',
                fontsize=13
            )
            plt.tight_layout()
            fname2 = (f'shap_beeswarm_{model_name.replace(" ", "_")}'
                      f'_K0.png')
            plt.savefig(fname2, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"  Zapisano: {fname2}")

    for shap_model_name in ['Random Forest', 'XGBoost']:
        if shap_model_name in trained_pipelines:
            shap_analysis(
                trained_pipelines[shap_model_name],
                X_test_raw,
                feature_cols_extended,
                model_name=shap_model_name
            )

except ImportError:
    print("UWAGA: shap nie jest zainstalowany.")
    print("Zainstaluj: pip install shap")


print("\n" + "=" * 60)
print("ZAPIS MODELI I WYNIKOW")
print("=" * 60)

# Zapis najlepszego modelu
best_pipeline_name = final_df.index[0]
if best_pipeline_name in trained_pipelines:
    joblib.dump(trained_pipelines[best_pipeline_name],
                'best_model.pkl')
    print(f"Zapisano najlepszy model: best_model.pkl "
          f"({best_pipeline_name})")
elif best_pipeline_name == 'Random Forest (tuned)':
    joblib.dump(rf_tuned, 'best_model.pkl')
    print("Zapisano najlepszy model: best_model.pkl "
          "(Random Forest tuned)")
elif best_pipeline_name == 'SVM (tuned)':
    joblib.dump(svm_tuned, 'best_model.pkl')
    print("Zapisano najlepszy model: best_model.pkl (SVM tuned)")

# Zapis wszystkich pipeline'ow
for name, pipe in trained_pipelines.items():
    fname = f'model_{name.replace(" ", "_")}.pkl'
    joblib.dump(pipe, fname)
    print(f"Zapisano: {fname}")

# Zapis wynikow CSV
final_df.to_csv('wyniki_modeli.csv')
print("Zapisano: wyniki_modeli.csv")

cv_export = {}
for name, vals in cv_results_summary.items():
    cv_export[name] = {
        key: value for key, value in vals.items()
        if key != 'F1_raw'
    }
cv_df = pd.DataFrame(cv_export).T
cv_df.to_csv('wyniki_cv.csv')
print("Zapisano: wyniki_cv.csv")

# Zapis szczegolowych wynikow
detailed_rows = []
for name, res in test_results.items():
    detailed_rows.append({
        'Model':             name,
        'Accuracy':          res['Accuracy'],
        'Balanced Accuracy': res['Balanced Accuracy'],
        'F1 Macro':          res['F1 Macro'],
        'F1 CI lower':       res['F1 CI lower'],
        'F1 CI upper':       res['F1 CI upper'],
        'AUC-ROC':           res['AUC-ROC'],
        'AUC CI lower':      res['AUC CI lower'],
        'AUC CI upper':      res['AUC CI upper']
    })

detailed_df = pd.DataFrame(detailed_rows).set_index('Model')
detailed_df.to_csv('wyniki_szczegolowe.csv')
print("Zapisano: wyniki_szczegolowe.csv")

elapsed = time.time() - start_time

print("\n" + "=" * 60)
print("PODSUMOWANIE KONCOWE")
print("=" * 60)
print(f"\nCzas wykonania: {elapsed:.1f}s "
      f"({elapsed/60:.1f} min)")
print(f"\nNajlepszy model (wg F1 Macro): {final_df.index[0]}")
print(f"  F1 Macro:  {final_df['F1 Macro'].iloc[0]:.4f}  "
      f"{final_df['F1 CI 95%'].iloc[0]}")
print(f"  AUC-ROC:   {final_df['AUC-ROC'].iloc[0]:.4f}  "
      f"{final_df['AUC CI 95%'].iloc[0]}")
print(f"  Bal. Acc:  {final_df['Balanced Accuracy'].iloc[0]:.4f}")

print(f"\nRanking modeli (F1 Macro):")
for i, (model_name, row) in enumerate(final_df.iterrows(), 1):
    print(f"  {i}. {model_name}: "
          f"F1={row['F1 Macro']:.4f}  "
          f"AUC={row['AUC-ROC']:.4f}")

print(f"\nZapisane pliki:")
saved_files = [
    'eda_distributions.png',
    'eda_boxplots.png',
    'eda_class_distribution.png',
    'eda_engineered_features.png',
    'correlation_matrix.png',
    'learning_curve_Random_Forest.png',
    'learning_curve_XGBoost.png',
    'confusion_matrices.png',
    f'roc_{best_model_name.replace(" ", "_")}.png',
    'roc_Voting_Classifier_(Top_3).png',
    f'pr_curve_{best_model_name.replace(" ", "_")}.png',
    'pr_curve_Voting_Classifier_(Top_3).png',
    f'calibration_{best_model_name.replace(" ", "_")}.png',
    'feature_importance.png',
    'model_comparison_F1_Macro.png',
    'model_comparison_AUC-ROC.png',
    'cv_results_heatmap.png',
    'best_model.pkl',
    'wyniki_modeli.csv',
    'wyniki_cv.csv',
    'wyniki_szczegolowe.csv',
    'rf_gridsearch_results.csv',
    'svm_gridsearch_results.csv'
]
for f in saved_files:
    print(f"  - {f}")

print("\n" + "=" * 60)
print("KONIEC ANALIZY")
print("=" * 60)

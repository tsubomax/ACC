# -*- coding: utf-8 -*-
# python3 /share_win/tsubo/Satellite_Image/JGR/program/testClass251206.py > /share_win/tsubo/Satellite_Image/JGR/program/output/log2.txt
# データスケーリングヴぁーじょん
# 標準ライブラリ
import gc  # スクリプト冒頭に追加
import csv
import datetime
import os
import sys
import traceback
import warnings
import time
import math

# サードパーティライブラリ
# import lightgbm as lgb  # 一般的なエイリアスに変更
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb  # 一般的なエイリアスに変更
import itertools  # 組み合わせ生成用

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    TransformerMixin,  # TransformerMixinを追加
    clone,
)
from sklearn.datasets import make_blobs, make_classification
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,  # BaggingClassifierを追加
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    # HistGradientBoostingClassifier,  # HistGradientBoostingClassifierを追加
    RandomForestClassifier,
    # OneVsRestClassifierはmulticlassからインポートするため削除
)
from sklearn.linear_model import (
    LogisticRegression,  # LogisticRegressionを追加
    PassiveAggressiveClassifier,  # PassiveAggressiveClassifierを追加
    SGDClassifier,  # SGDClassifierを追加
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import (
    OneVsRestClassifier,
)  # OneVsRestClassifierはこちらでインポート
from sklearn.naive_bayes import GaussianNB  # GaussianNBを追加
from sklearn.neighbors import KNeighborsClassifier  # KNeighborsClassifierを追加
from sklearn.neural_network import MLPClassifier  # MLPClassifierを追加
from sklearn.preprocessing import StandardScaler  # StandardScalerを追加
from sklearn.svm import LinearSVC  # LinearSVCを追加
from sklearn.tree import DecisionTreeClassifier  # DecisionTreeClassifierを追加
from sklearn.utils import shuffle
from sklearn.utils.multiclass import unique_labels  # unique_labelsを追加
from sklearn.utils.validation import (  # validation関連をまとめる
    check_X_y,
    check_array,
    check_is_fitted,
)

# HGBとGBの両方を追加

# ローカルアプリケーション/ライブラリ
from TTmethod.classTTmethod import TTClassifier
from validation_lib.feature_expander2 import FeatureExpansionClassifier
from lightgbm import LGBMClassifier  # ★ LightGBM をインポート
from xgboost import XGBClassifier  # ★ XGBoost をインポート

# 250430
from sklearn.preprocessing import StandardScaler
import spectral
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.multiclass import unique_labels
import numpy as np
import time
import traceback
import warnings
from sklearn.base import clone  # メインループで使うのでグローバルが良い
import numpy as np
from pathlib import Path
import os
import spectral
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import os
import warnings
from scipy.linalg import (
    eigh,
)

# --- 以下はインポート文ではないため、インポートブロックの後や設定が必要な箇所に配置 ---
# 実行時の警告を抑制
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
import glob

# ★★★ verbose 変数をここで定義する ★★★
verbose = 1  # ログレベルを設定 (0: 静か, 1: 基本情報, 2: 詳細)

# --- 必要なライブラリのインポート ---
try:
    from osgeo import gdal

    # GDALの例外処理を有効にする（オプション）
    gdal.UseExceptions()
    GDAL_AVAILABLE = True
    print("GDAL library loaded successfully.")
except ImportError:
    print(
        "Error: GDAL library not found. Please install it (e.g., 'pip install GDAL' or use conda). Image loading will fail."
    )
    GDAL_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
    print("Pillow library loaded successfully.")
except ImportError:
    print(
        "Error: Pillow library not found. Please install it ('pip install Pillow'). Image saving will fail."
    )
    PIL_AVAILABLE = False


# --- 設定 ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
today_date_str = datetime.datetime.now().strftime("%Y%m%d")  # YYYYMMDD形式
start_image_time = time.time()
# --- データ/出力パス ---
# ★★★ パスは実際の環境に合わせてください ★★★
# Windowsパスの場合、'\' をエスケープするか、raw文字列(r"...")を使用
# ENVIデータがあるディレクトリ
feature_data_dir = r"/share_win/tsubo/Satellite_Image/2025July/Cuprite/MS_hisui"
# TIFラベルがあるディレクトリ
label_data_dir = r"/share_win/tsubo/Satellite_Image/2025July/Cuprite/HS_HISUI_cup"
base_output_dir = (
    r"/share_win/tsubo/Satellite_Image/2025July/Cuprite/output"  # 出力ベースディレクトリ
)

# --- 出力ディレクトリ作成 ---
output_date_dir = os.path.join(base_output_dir, today_date_str)  # YYYYMMDD
image_output_dir = os.path.join(
    output_date_dir, f"output_images_{timestamp}"
)  # HHMMSS_img の代わりにタイムスタンプ
log_output_dir = output_date_dir  # ログとCSVは日付ディレクトリ直下

os.makedirs(image_output_dir, exist_ok=True)
print(f"Created image output directory: {image_output_dir}")
os.makedirs(log_output_dir, exist_ok=True)  # ログ用ディレクトリも確認

# --- AdvancedCascadeClassifier のインポート ---
try:
    from validation_lib.advanced_cascade import (
        AdvancedCascadeClassifier,
    )  # validation_lib.py に保存されていると仮定

    ADVANCED_CASCADE_AVAILABLE = True
    print("Successfully imported AdvancedCascadeClassifier.")
except ImportError as e:
    print(
        f"Warning: Could not import AdvancedCascadeClassifier from validation_lib: {e}"
    )
    ADVANCED_CASCADE_AVAILABLE = False


# ファクトリ関数と設定辞書をインポート
try:
    from validation_lib.high_pref_configs import (
        get_high_performing_cascade,
        HIGH_PERF_CONFIGS,
    )

    HIGH_PERF_CASCADE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import high-performance cascade factory: {e}")
    HIGH_PERF_CASCADE_AVAILABLE = False

# --- 実行時警告の抑制 (オプション) ---
# warnings.filterwarnings('ignore', category=UserWarning)
# warnings.filterwarnings('ignore', category=RuntimeWarning) # NaN関連の警告抑制など

start = time.time()


def fl(P):  # floor
    Q = abs(math.floor(P * 10**4) / (10**4))
    return Q


# === カスタム特徴量生成器の定義 ===
class SelectFirstKFeaturesAndScale(BaseEstimator, TransformerMixin):
    """最初のK個の特徴量を選択し、StandardScalerを適用するTransformer"""

    def __init__(self, k=5):
        # k が 0 以下にならないようにチェック
        if k <= 0:
            warnings.warn(
                f"k={k} <= 0 specified. No features will be selected.", UserWarning
            )
            self.k = 0
        else:
            self.k = k
        self.scaler_ = None  # 学習済みスケーラー
        self._n_features_in = 0
        self._k_eff = 0

    def fit(self, X, y=None):
        self._n_features_in = X.shape[1]
        # 実際に選択可能な特徴量数
        self._k_eff = min(self.k, self._n_features_in)

        if self._k_eff > 0:
            self.scaler_ = StandardScaler()
            # 最初の k_eff 個の特徴量でスケーラーを学習
            self.scaler_.fit(X[:, : self._k_eff])
        else:
            self.scaler_ = None  # 特徴量が選択されない場合はスケーラー不要
        return self

    def transform(self, X):
        # 入力特徴量数が fit 時と一致するかチェック
        if X.shape[1] != self._n_features_in:
            raise ValueError(
                f"Input features mismatch: {X.shape[1]} != {self._n_features_in}"
            )

        # 特徴量を選択
        X_selected = X[:, : self._k_eff]

        if self.scaler_ is not None:
            # 学習済みスケーラーで変換
            return self.scaler_.transform(X_selected)
        else:
            # スケーラーがない場合 (k_eff=0 など) は選択結果のみ返す
            return X_selected

    # オプション: 出力特徴量名を返すメソッド
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return [f"feature_{i}" for i in range(self._k_eff)]
        elif len(input_features) != self._n_features_in:
            raise ValueError("input_features length mismatch")
        else:
            return input_features[: self._k_eff]


###########################################################################################################

# --- 設定 ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
text_log_filename = f"multi_model_validation_log_{timestamp}.txt"
csv_filename = f"multi_model_validation_results_{timestamp}.csv"
plot_dir = f"validation_plots_{timestamp}"

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

text_log_filepath = os.path.join(script_dir, text_log_filename)
csv_filepath = os.path.join(script_dir, csv_filename)
plot_dir_path = os.path.join(script_dir, plot_dir)

# --- モデルハイパーパラメータ定数 ---
MAX_DEPTH_PARAM = 100
N_ESTIMATORS_PARAM = 100
RANDOM_STATE = 0
N_JOBS = -1

##############################
##############################


# --- 検証対象モデル定義 ---
base_estimator_rf = RandomForestClassifier(
    max_depth=MAX_DEPTH_PARAM,
    n_estimators=N_ESTIMATORS_PARAM,
    random_state=RANDOM_STATE,
    # n_jobs=-1,
    # class_weight="balanced_subsample",
)
# base_hgb = HistGradientBoostingClassifier(max_iter=100, random_state=RANDOM_STATE)

base_rf = base_estimator_rf

##############################
##############################
# --- 検証対象モデル定義 ---
models = {
    # --- 既存 + コメントアウト解除など ---
    "RandomForest": RandomForestClassifier(
        max_depth=MAX_DEPTH_PARAM,
        n_estimators=N_ESTIMATORS_PARAM,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    ),
    "OneVsRest_RF": OneVsRestClassifier(clone(base_rf), n_jobs=N_JOBS),
    # "TTclassifier": TTClassifier(estimator=base_estimator_rf, verbose=1),
    # "FeatureExpanderRF": FeatureExpansionClassifier(  # 特徴量拡張 + RF (これが以前のもの)
    #        max_depth=MAX_DEPTH_PARAM,
    #        n_estimators=N_ESTIMATORS_PARAM,
    #        random_state=RANDOM_STATE,
    #        n_jobs=N_JOBS,
    #    ),
    #    use_slope=False,
    #    verbose=0,  # verbose は必要なら1以上に
    # ),
    "LogisticRegression": LogisticRegression(
        solver="saga", max_iter=1000, random_state=RANDOM_STATE, n_jobs=N_JOBS
    ),
    "SGD_LinearSVM": SGDClassifier(
        loss="hinge",
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        early_stopping=True,
        n_iter_no_change=10,
    ),
    # "LinearSVC": LinearSVC(
    #    dual=True, max_iter=2000, random_state=RANDOM_STATE
    # ),  # dual=True を確認
    "GaussianNB": GaussianNB(),
    "KNeighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS),
    # --- HistGradientBoostingClassifier を削除 ---
    # "HistGradientBoosting": HistGradientBoostingClassifier(...),
    # ★★★ LightGBM と XGBoost を追加 ★★★
    # "LightGBM": LGBMClassifier(
    #    random_state=RANDOM_STATE,
    #    n_jobs=N_JOBS,
    #    # verbosity=-1 # ログを抑制する場合
    # ),
    "XGBoost": XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        use_label_encoder=False,  # 警告回避のためFalse推奨
        eval_metric="mlogloss",  # 多クラス分類用の評価指標
    ),
    # --- 最適化ソルバーを変更 (SGD) ---
    "MLP_SGD_Momentum": MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=1000,  # SGDは収束が遅いことがあるためmax_iterを増やす
        activation="relu",
        solver="sgd",  # 最適化ソルバーにSGD(確率的勾配降下法)を使用
        alpha=0.0001,
        batch_size=64,  # SGDではバッチサイズ指定が重要
        learning_rate="adaptive",  # 学習率スケジュール (adaptive: 改善が続けば維持、停滞すれば下げる)
        learning_rate_init=0.01,  # SGD用の初期学習率 (Adamより大きめが良い場合も)
        momentum=0.9,  # モーメンタム (勾配更新を滑らかにする)
        nesterovs_momentum=True,  # ネステロフ・モーメンタムを使用
        random_state=RANDOM_STATE,
        early_stopping=True,
        n_iter_no_change=10,
    ),
    # ------------------------------------
}
# --- 高精度設定の AdvancedCascadeClassifier を追加 ---
# "\\nasu\share_win$\tsubo\Satellite_Image\JGR\program\validation_lib\high_pref_configs.py"
if HIGH_PERF_CASCADE_AVAILABLE:
    print("\nAdding high-performing AdvancedCascadeClassifier configurations...")
    # HIGH_PERF_CONFIGS 辞書にあるすべての設定を追加
    for config_key in HIGH_PERF_CONFIGS.keys():
        model_key = f"AdvCas_{config_key}"  # モデル名を生成
        try:
            models[model_key] = get_high_performing_cascade(
                base_estimator_rf=base_estimator_rf,
                config_name=config_key,
                random_state=RANDOM_STATE,
                verbose=1,  # または 0 に設定
            )
        except (ValueError, ImportError) as e:
            print(f"Could not create model for config '{config_key}': {e}")

print(
    f"\nFinal models for validation ({len(models)} total): {list(models.keys())}")

"""
"""
"""
    # --- 非常に深いモデルの例 ---
    "MLP_Very_Deep": MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32, 16),  # 5層の隠れ層
        max_iter=1500,
        activation="relu",
        solver="adam",
        alpha=0.0005,  # 少し正則化を強める
        batch_size=128,  # バッチサイズを少し大きめに
        random_state=RANDOM_STATE,
        early_stopping=True,
        n_iter_no_change=20,  # 深いモデルは収束に時間がかかる可能性があるため猶予を増やす
    ),
base_estimator_name = "RF"  # モデル名用
# --- パラメータの組み合わせを定義 ---
tolerances = [0, 0.01, 0.05, 0.1, 0.3]
updates = [0, 1, 2]
f1_thresholds = [0.8, 0.85, 0.9, 0.95, 1.0]
# val_size=0 はエラーになるため、最小値を 0.01 に変更
val_sizes = [0.1, 0.2, 0.5, 0.9]
# feature_generator の設定 (オブジェクトと名前のタプル)
feature_generators_list = [(None, "None"), ("scaler", "Scl")]
# カスタムクラスが利用可能な場合のみ追加
feature_generators_list.extend(
    [
        (SelectFirstKFeaturesAndScale(k=5), "Sel5"),
        (SelectFirstKFeaturesAndScale(k=10), "Sel10"),
        # (SelectFirstKFeaturesAndScale(k=15), "Sel15"),  # 5パターン目
    ]
)
# print("Skipping configurations using SelectFirstKFeaturesAndScale.")
# feature_generators_list は None と scaler のみのまま


# --- AdvancedCascadeClassifier の設定を大量生成 ---
models = {}  # models辞書を新規作成 (既存の他のモデルは別途追加が必要)

if ADVANCED_CASCADE_AVAILABLE:
    num_combinations = (
        len(tolerances)
        * len(updates)
        * len(f1_thresholds)
        * len(val_sizes)
        * len(feature_generators_list)
    )
    print(f"Generating {num_combinations} AdvancedCascadeClassifier configurations...")

    # 全パラメータの組み合わせを生成
    param_combinations = itertools.product(
        tolerances, updates, f1_thresholds, val_sizes, feature_generators_list
    )

    count = 0
    for tol, upd, f1t, val, (fg_obj, fg_name) in param_combinations:
        count += 1
        if count % 100 == 0:  # 100個ごとに進捗表示
            print(f"  Generating config {count}/{num_combinations}...")

        # --- モデル名の生成 ---
        # 浮動小数点誤差を考慮し、整数化してから文字列へ
        tol_int = int(round(tol * 100))
        f1_int = int(round(f1t * 100))
        val_int = int(round(val * 100))

        tol_str = str(tol_int).zfill(2) if tol > 0 else "0"  # 0 の場合は "0"
        f1_str = str(f1_int)
        val_str = str(val_int).zfill(2)

        # 例: AdvCas_RF_SclUpd1_F85_Tol05_Val20
        model_name = f"AdvCas_{base_estimator_name}_{fg_name}Upd{upd}_F{f1_str}_Tol{tol_str}_Val{val_str}"

        # --- モデルインスタンスの生成と辞書への追加 ---
        # feature_generator にはオブジェクトまたは識別子文字列を渡す
        # _get_feature_transformer が fg_obj (Select...) をそのまま返せるように修正済みと仮定
        models[model_name] = AdvancedCascadeClassifier(
            estimator=clone(base_rf),  # ベース推定器を毎回クローン
            unclassified_tolerance_p=tol,
            max_updates=upd,
            min_f1_threshold=f1t,
            val_size=val,
            feature_generator=fg_obj,  # オブジェクト or 文字列 or None
            verbose=0,  # 大量実行なので verbose は 0 推奨
            random_state=RANDOM_STATE,
        )
"""


# 最終的なモデル数を確認
# print(f"Total models defined in dictionary: {len(models)}")
"""
# 基本データ生成パラメータ
BASE_N_SAMPLES = 1000
BASE_N_FEATURES = 14  # > BASE_N_INFORMATIVE + BASE_N_REDUNDANT
BASE_N_INFORMATIVE = 9
BASE_N_REDUNDANT = 5
BASE_N_CLASSES = 9  # デフォルト値 (n_classesシナリオで上書き)
BASE_CLASS_SEP = 1.0
BASE_WEIGHTS = None
"""
BASE_RANDOM_STATE = 0


# --- ヘルパー関数 (変更なし) ---
def log_to_text(filepath, message):
    print(message)
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except IOError as e:
        print(f"Error writing to text log file {filepath}: {e}")


def calculate_f1av(report_dict):
    precision_values = []
    recall_values = []
    num_classes = 0
    for label, metrics in report_dict.items():
        if str(label).isdigit():
            precision_values.append(metrics["precision"])
            recall_values.append(metrics["recall"])
            num_classes += 1
    if not precision_values or not recall_values:
        return 0.0

    if not precision_values or not recall_values:
        return 0.0
    precision_macro_avg = sum(precision_values) / len(precision_values)
    recall_macro_avg = sum(recall_values) / len(recall_values)
    if precision_macro_avg + recall_macro_avg == 0:
        return 0.0
    return (
        2
        * precision_macro_avg
        * recall_macro_avg
        / (precision_macro_avg + recall_macro_avg)
    )


def calculate_macro_f1_excluding_zero(report):
    """
    classification_reportの辞書からラベル'0'を除外してMacro F1を計算する。

    Args:
        report (dict): sklearn.metrics.classification_report(output_dict=True) の出力。
                        キーはラベルの文字列 ('0', '1', '2', ...) と、
                        'accuracy', 'macro avg', 'weighted avg'。

    Returns:
        float: ラベル'0'を除外したMacro F1スコア。計算不能な場合は 0.0 を返す。
    """
    if not isinstance(report, dict):
        print("Error: Input 'report' must be a dictionary.")
        return 0.0

    precision_values = []
    recall_values = []

    for label_str, metrics in report.items():
        # 'accuracy', 'macro avg', 'weighted avg' はスキップ
        if label_str in ["accuracy", "macro avg", "weighted avg"]:
            continue

        # ラベル文字列が '0' でない場合のみ値を追加
        # 注意: reportのキーは通常文字列なので、数値の 0 ではなく文字列の '0' と比較
        if label_str != "0":
            if (
                isinstance(metrics, dict)
                and "precision" in metrics
                and "recall" in metrics
            ):
                precision_values.append(metrics["precision"])
                recall_values.append(metrics["recall"])
            else:
                # 必要なメトリクスがない場合は警告（デバッグ用）
                # print(f"Warning: Metrics for label '{label_str}' are missing or invalid. Skipping.")
                pass  # スキップ

    # 値が何も集まらなかった場合（ラベル'0'以外の有効なクラスがなかった）
    if not precision_values or not recall_values:
        # print("Warning: No valid precision/recall values found after excluding label '0'. Cannot calculate Macro F1.")
        return 0.0  # 計算不能

    # Macro AverageのPrecisionとRecallを計算
    # (サポート数0のクラスはclassification_reportのデフォルトで除外されているはず)
    precision_macro_avg = sum(precision_values) / len(precision_values)
    recall_macro_avg = sum(recall_values) / len(recall_values)

    # Macro F1を計算 (ゼロ除算を回避)
    if (precision_macro_avg + recall_macro_avg) == 0:
        f1_macro_rm0 = 0.0
    else:
        f1_macro_rm0 = (
            2
            * precision_macro_avg
            * recall_macro_avg
            / (precision_macro_avg + recall_macro_avg)
        )

    return f1_macro_rm0


"""
# --- カスタムデータ生成関数 (変更なし) ---
def generate_minority_high_sep_data(
    n_samples=BASE_N_SAMPLES,
    n_features=BASE_N_FEATURES,
    centers=5,
    minority_proportion=0.05,
    maj_std=1.5,
    min_std=0.2,
    center_box=(-10.0, 10.0),
    random_state=0,
):
    if centers < 2:
        raise ValueError("Number of centers must be at least 2.")
    rng = np.random.RandomState(random_state)
    n_minority = max(1, int(n_samples * minority_proportion))
    n_majority_total = n_samples - n_minority
    n_majority_classes = centers - 1
    if n_majority_classes <= 0:
        samples_per_maj_class = np.array([], dtype=int)
    else:
        samples_per_maj_class = np.full(
            n_majority_classes, n_majority_total // n_majority_classes, dtype=int
        )
        if n_majority_total % n_majority_classes > 0:
            samples_per_maj_class[: n_majority_total % n_majority_classes] += 1
    samples_per_class = np.concatenate(([n_minority], samples_per_maj_class))
    active_centers = centers
    non_zero_indices = np.where(samples_per_class > 0)[0]
    samples_per_class = samples_per_class[non_zero_indices]
    if len(non_zero_indices) < centers:
        active_centers = len(non_zero_indices)
        print(f"Warning: Reduced centers to {active_centers}.")
    X_list, y_list = [], []
    if active_centers == 0:
        return np.empty((0, n_features)), np.empty((0,), dtype=int)
    cluster_centers = rng.uniform(
        center_box[0], center_box[1], size=(active_centers, n_features)
    )
    original_class_labels = non_zero_indices
    for idx, original_label in enumerate(original_class_labels):
        n_samples_class = samples_per_class[idx]
        center = cluster_centers[idx : idx + 1, :]
        std = min_std if original_label == 0 else maj_std
        blob_seed = rng.randint(10**6)
        X_class, _ = make_blobs(
            n_samples=n_samples_class,
            n_features=n_features,
            centers=center,
            cluster_std=std,
            random_state=blob_seed,
        )
        y_class = np.full(n_samples_class, original_label, dtype=int)
        X_list.append(X_class)
        y_list.append(y_class)
    if not X_list:
        return np.empty((0, n_features)), np.empty((0,), dtype=int)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    X, y = shuffle(X, y, random_state=random_state)
    return X, y
"""


# --- モデル評価関数 (KeyError修正・レポート2パターン対応版) ---


def evaluate_model_v2(
    model_name,
    model_instance,
    X_train,
    y_train,
    X_test,
    y_test,
    n_classes_train=None,  # 現在は未使用
):
    """
    モデルを学習・評価し、結果辞書を返す (背景クラス0を除外した指標計算を含む)。
    Classification Report と混同行列、関連指標を
    「訓練時に存在した非背景クラス」基準 (Known) と
    「テスト時に出現した非背景クラス」基準 (All) で計算。

    Args:
        model_name (str): モデル名
        model_instance: 学習前のモデルインスタンス
        X_train, y_train: 訓練データ
        X_test, y_test: テストデータ
        n_classes_train (int, optional): 訓練データのクラス数（未使用）

    Returns:
        dict: 評価結果を含む辞書
    """
    start = time.time()  # 関数開始時間 (デバッグ用に追加)
    # --- 結果辞書の初期化 ---
    results = {
        "accuracy": 0.0,
        "mcc": 0.0,
        "f1av_known": 0.0,
        "macro_precision_known": 0.0,
        "macro_recall_known": 0.0,
        "macro_f1_known": 0.0,
        "weighted_precision_known": 0.0,
        "weighted_recall_known": 0.0,
        "weighted_f1_known": 0.0,
        "report_str_known": "N/A",
        "confusion_matrix_known": np.array([]),
        "f1av_all": 0.0,
        "macro_precision_all": 0.0,
        "macro_recall_all": 0.0,
        "macro_f1_all": 0.0,
        "weighted_precision_all": 0.0,
        "weighted_recall_all": 0.0,
        "weighted_f1_all": 0.0,
        "report_str_all": "N/A",
        "confusion_matrix_all": np.array([]),
        "unclassified_rate": 0.0,
        "unknown_rate": 0.0,
        "fit_time": -1.0,
        "pred_time": -1.0,
        "error_info": None,
    }

    fit_time = -1.0
    pred_time = -1.0
    y_pred = None

    try:
        # === モデル学習 ===
        if X_train.shape[0] == 0 or len(unique_labels(y_train)) < 1:
            raise ValueError("Training data is empty or has no valid labels.")
        start_fit = time.time()
        model_instance.fit(X_train, y_train)
        fit_time = time.time() - start_fit
        results["fit_time"] = fit_time
        # print(f"        {model_name} fit completed in {fit_time:.2f} sec") # デバッグ用

        # === 予測 ===
        if X_test.shape[0] == 0:
            print(
                "        Warning: Test data is empty. Skipping prediction and evaluation."
            )
            return results
        start_pred = time.time()
        y_pred = model_instance.predict(X_test)
        pred_time = time.time() - start_pred
        results["pred_time"] = pred_time
        # print(f"        {model_name} prediction completed in {pred_time:.2f} sec") # デバッグ用

        # --- ラベル情報の準備 (背景0を除外) ---
        all_unique_labels_train = unique_labels(y_train)
        # 訓練データに存在した非背景クラスのリスト (ソート済み)
        known_labels_train = sorted(
            [label for label in all_unique_labels_train if label != 0]
        )

        # テストデータと予測結果に出現した全ラベルを取得
        all_unique_labels_test_pred = unique_labels(y_test, y_pred)
        # テストデータに出現した非背景クラスのリスト (ソート済み)
        eval_labels_all = sorted(
            [label for label in all_unique_labels_test_pred if label != 0]
        )

        # --- 未知クラス率の計算 (テストデータ中に、訓練時(非背景)になかった非背景クラスが存在する割合) ---
        unknown_mask_test = ~np.isin(
            y_test, known_labels_train) & (y_test != 0)
        num_unknown = np.sum(unknown_mask_test)
        results["unknown_rate"] = num_unknown / \
            len(y_test) if len(y_test) > 0 else 0.0

        # --- 未分類率の計算 (ACCなど特殊モデル用) ---
        is_unclassified_mask = np.zeros(len(y_pred), dtype=bool)
        if hasattr(model_instance, "_is_unclassified") and callable(
            model_instance._is_unclassified
        ):
            try:
                is_unclassified_mask = model_instance._is_unclassified(y_pred)
            except Exception as e_uncl:
                print(
                    f"        Warning: Error calling _is_unclassified for {model_name}: {e_uncl}"
                )
        num_unclassified = np.sum(is_unclassified_mask)
        results["unclassified_rate"] = (
            num_unclassified / len(y_pred) if len(y_pred) > 0 else 0.0
        )
        # -------------------------------------------------

        # --- 評価対象サンプルの決定 (未分類を除外) ---
        classified_mask = ~is_unclassified_mask
        y_test_classified = y_test[classified_mask]
        y_pred_classified = y_pred[classified_mask]

        if len(y_test_classified) == 0:
            print("        Warning: No classified samples left for evaluation.")
            results["report_str_known"] = "No classified samples for known eval."
            results["report_str_all"] = "No classified samples for all eval."
            return results
        # ------------------------------------------

        # --- 全体精度とMCC (未分類除外後のサンプルで計算) ---
        results["accuracy"] = accuracy_score(
            y_test_classified, y_pred_classified)
        try:
            results["mcc"] = matthews_corrcoef(
                y_test_classified, y_pred_classified)
        except ValueError:
            results["mcc"] = 0.0  # または np.nan
        # ------------------------------------

        # --- 訓練クラス基準での評価 (背景0を除く) ---
        report_dict_known = {}
        report_str_known = "N/A"
        conf_matrix_known = np.array([])

        if not known_labels_train:  # 訓練データに背景以外のクラスがなかった場合
            print(
                "        Warning: No non-background labels found in training set. Skipping Known Class evaluation."
            )
            results["report_str_known"] = "No non-background train labels."
        else:
            try:
                # labels 引数に背景(0)を除いた訓練クラスを指定
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    report_dict_known = classification_report(
                        y_test_classified,
                        y_pred_classified,
                        labels=known_labels_train,  # ★ 背景除去済み訓練ラベル
                        zero_division=0,
                        output_dict=True,
                    )
                report_str_known = classification_report(
                    y_test_classified,
                    y_pred_classified,
                    labels=known_labels_train,  # ★ 背景除去済み訓練ラベル
                    zero_division=0,
                )
                report_str_known += f"\n(Evaluation based on {len(known_labels_train)} known non-background train classes: {known_labels_train})"
                report_str_known += (
                    f"\nUnclassified Rate: {results['unclassified_rate']:.4f}"
                )
                report_str_known += f"\nUnknown Rate (in test vs train known): {results['unknown_rate']:.4f}"

                # results 辞書に指標を格納
                results["f1av_known"] = calculate_macro_f1_excluding_zero(
                    report_dict_known
                )  # calculate_f1av は report_dict を受け取る
                if "macro avg" in report_dict_known:
                    results["macro_precision_known"] = report_dict_known[
                        "macro avg"
                    ].get("precision", 0.0)
                    results["macro_recall_known"] = report_dict_known["macro avg"].get(
                        "recall", 0.0
                    )
                    results["macro_f1_known"] = report_dict_known["macro avg"].get(
                        "f1-score", 0.0
                    )
                if "weighted avg" in report_dict_known:
                    results["weighted_precision_known"] = report_dict_known[
                        "weighted avg"
                    ].get("precision", 0.0)
                    results["weighted_recall_known"] = report_dict_known[
                        "weighted avg"
                    ].get("recall", 0.0)
                    results["weighted_f1_known"] = report_dict_known[
                        "weighted avg"
                    ].get("f1-score", 0.0)

                # 混同行列も背景(0)を除いた訓練クラス基準で作成
                conf_matrix_known = confusion_matrix(
                    y_test_classified,
                    y_pred_classified,
                    labels=known_labels_train,  # ★ 背景除去済み訓練ラベル
                )

            except Exception as report_err_known:
                error_msg = f"Known Class Report Error: {report_err_known}\n{traceback.format_exc()}"
                print(f"        {error_msg}")
                results["report_str_known"] = error_msg  # エラー情報をレポート文字列に

        # 最後に結果を格納 (エラーの場合も N/A や空配列が格納される)
        results["report_str_known"] = report_str_known
        results["confusion_matrix_known"] = conf_matrix_known
        # ------------------------------------

        # --- 全テスト出現クラス基準での評価 (背景0を除く) ---
        report_dict_all = {}
        report_str_all = "N/A"
        conf_matrix_all = np.array([])

        # eval_labels_all (背景除去済みテスト出現ラベル) が空でないことを確認
        if not eval_labels_all:
            print(
                "        Warning: No non-background labels found in test/pred set. Skipping All Class evaluation."
            )
            results["report_str_all"] = "No non-background test/pred labels."
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # labels 引数に背景(0)を除いたテスト出現クラスを指定
                    report_dict_all = classification_report(
                        y_test_classified,
                        y_pred_classified,
                        labels=eval_labels_all,  # ★ 背景除去済みテスト出現ラベル
                        zero_division=0,
                        output_dict=True,
                    )
                report_str_all = classification_report(
                    y_test_classified,
                    y_pred_classified,
                    labels=eval_labels_all,  # ★ 背景除去済みテスト出現ラベル
                    zero_division=0,
                )
                report_str_all += f"\n(Evaluation based on {len(eval_labels_all)} non-background classes present in test/pred data: {eval_labels_all})"
                report_str_all += (
                    f"\nUnclassified Rate: {results['unclassified_rate']:.4f}"
                )
                report_str_all += f"\nUnknown Rate (in test vs train known): {results['unknown_rate']:.4f}"

                # results 辞書に指標を格納
                results["f1av_all"] = calculate_f1av(
                    report_dict_all
                )  # ★ 背景除去クラス基準のF1AV
                if "macro avg" in report_dict_all:
                    results["macro_precision_all"] = report_dict_all["macro avg"].get(
                        "precision", 0.0
                    )
                    results["macro_recall_all"] = report_dict_all["macro avg"].get(
                        "recall", 0.0
                    )
                    results["macro_f1_all"] = report_dict_all["macro avg"].get(
                        "f1-score", 0.0
                    )
                if "weighted avg" in report_dict_all:
                    results["weighted_precision_all"] = report_dict_all[
                        "weighted avg"
                    ].get("precision", 0.0)
                    results["weighted_recall_all"] = report_dict_all[
                        "weighted avg"
                    ].get("recall", 0.0)
                    results["weighted_f1_all"] = report_dict_all["weighted avg"].get(
                        "f1-score", 0.0
                    )

                # 混同行列も背景(0)を除いたテスト出現クラス基準で作成
                conf_matrix_all = confusion_matrix(
                    y_test_classified,
                    y_pred_classified,
                    labels=eval_labels_all,  # ★ 背景除去済みテスト出現ラベル
                )

            except Exception as report_err_all:
                error_msg = f"All Class Report Error: {report_err_all}\n{traceback.format_exc()}"
                print(f"        {error_msg}")
                results["report_str_all"] = error_msg

        # 最後に結果を格納
        results["report_str_all"] = report_str_all
        results["confusion_matrix_all"] = conf_matrix_all
        # ----------------------------------

    except Exception as e:  # 学習・予測・評価全体のエラーキャッチ
        results["error_info"] = (
            f"Evaluation Error for {model_name}: {e}\n{traceback.format_exc()}"
        )
        # エラー発生時はレポート文字列にもエラー情報を入れる
        results["report_str_known"] = results["error_info"]
        results["report_str_all"] = results["error_info"]

    # 不要なデバッグプリントは削除
    # print("omeko")
    return results


# --- 新しいカスタムデータ生成関数 ---
def generate_imbalance_weights(
    n_classes, pattern="linear_decrease", min_prop=0.05, max_prop=0.8, random_state=None
):
    # クラス数に応じた不均衡weightsリストを生成する
    if n_classes <= 1:
        return None  # 1クラス以下は不均衡にできない

    rng = np.random.RandomState(random_state)
    weights = np.zeros(n_classes)

    if pattern == "linear_decrease":
        # 線形に減少するパターン (例: [0.5, 0.3, 0.2])
        base = np.linspace(1, 0.1, n_classes)  # 大きい方から減少
        weights = base / base.sum()
    elif pattern == "one_dominant":
        # 1クラスが支配的なパターン (例: [0.8, 0.1, 0.1])
        dominant_prop = max_prop
        other_prop = (1.0 - dominant_prop) / (n_classes - 1)
        # 他クラスの割合が min_prop を下回らないように調整
        if other_prop < min_prop and n_classes > 1:
            other_prop = min_prop
            dominant_prop = 1.0 - other_prop * (n_classes - 1)
            if dominant_prop < 0:  # クラス数が多すぎて調整不可の場合
                warnings.warn(
                    f"Cannot create 'one_dominant' weights for {n_classes} classes with min_prop={min_prop}. Using linear_decrease."
                )
                return generate_imbalance_weights(
                    n_classes, "linear_decrease", min_prop, max_prop, random_state
                )

        weights[0] = dominant_prop
        weights[1:] = other_prop
        rng.shuffle(weights)  # 支配クラスをランダムにする
    elif pattern == "two_dominant":
        # 2クラスが支配的なパターン
        if n_classes < 2:
            return None
        dom1_prop = rng.uniform(0.3, 0.6)
        dom2_prop = rng.uniform(0.2, 1.0 - dom1_prop -
                                (n_classes - 2) * min_prop)
        dom2_prop = max(min_prop, dom2_prop)  # 最小値保証
        remaining_prop = 1.0 - dom1_prop - dom2_prop
        if remaining_prop < (n_classes - 2) * min_prop and n_classes > 2:
            # 再調整が必要な場合 (簡略化のため linear_decrease にフォールバック)
            return generate_imbalance_weights(
                n_classes, "linear_decrease", min_prop, max_prop, random_state
            )
        other_prop = remaining_prop / (n_classes - 2) if n_classes > 2 else 0

        weights[0] = dom1_prop
        weights[1] = dom2_prop
        if n_classes > 2:
            weights[2:] = other_prop
        rng.shuffle(weights)
    else:  # デフォルトは線形減少
        base = np.linspace(1, 0.1, n_classes)
        weights = base / base.sum()

    # 最終チェックと正規化
    weights = np.maximum(weights, 0)  # 負の値を除去
    if weights.sum() <= 0:
        return None  # 合計が0以下ならNone
    weights /= weights.sum()
    return weights.tolist()


def generate_data_with_unknown_classes(
    X_orig, y_orig, unknown_fraction=0.1, n_unknown_classes=1, random_state=None
):
    """テストデータに未知クラスを混入させる"""
    rng = np.random.RandomState(random_state)
    X_train, X_test, y_train, y_test_orig = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=random_state, stratify=y_orig
    )
    y_test = y_test_orig.copy()  # コピーして変更
    n_test_samples = len(y_test)
    n_unknown_samples = int(n_test_samples * unknown_fraction)

    if n_unknown_samples > 0 and n_test_samples > 0:
        # 未知クラスのラベルを決定 (既存ラベルの最大値 + 1 から開始)
        max_known_label = np.max(y_train) if len(y_train) > 0 else -1
        unknown_labels = np.arange(
            max_known_label + 1, max_known_label + 1 + n_unknown_classes
        )

        # テストデータからランダムに未知クラスにするサンプルを選ぶ
        unknown_indices = rng.choice(
            n_test_samples, n_unknown_samples, replace=False)
        # 選ばれたサンプルに未知クラスラベルをランダムに割り当てる
        assigned_unknown_labels = rng.choice(unknown_labels, n_unknown_samples)
        y_test[unknown_indices] = assigned_unknown_labels

    # (X_train, y_train) と (X_test, y_test_with_unknown) を返す
    # この形式だと既存のループ構造に合わないため、構造の変更が必要
    # -> 代替案: シナリオ定義でこの関数を指定し、メインループで特別処理する
    return X_train, X_test, y_train, y_test  # この関数自体はデータを返す


def generate_data_with_outliers(
    X_orig, y_orig, outlier_fraction=0.05, magnitude=5.0, random_state=None
):
    """テストデータに外れ値を混入させる"""
    rng = np.random.RandomState(random_state)
    X_train, X_test_orig, y_train, y_test = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=random_state, stratify=y_orig
    )
    X_test = X_test_orig.copy()  # コピーして変更
    n_test_samples, n_features = X_test.shape
    n_outliers = int(n_test_samples * outlier_fraction)

    if n_outliers > 0 and n_test_samples > 0 and n_features > 0:
        outlier_indices = rng.choice(n_test_samples, n_outliers, replace=False)
        # 各外れ値サンプルに対して、ランダムな特徴量に外れ値を加える
        for i in outlier_indices:
            feature_idx = rng.randint(n_features)
            # その特徴量の標準偏差を計算（全体で or 各クラスで）
            # 簡単のため、ここでは全体の標準偏差を使う
            std_dev = np.std(X_train[:, feature_idx])
            if std_dev > 1e-6:  # 標準偏差が0でない場合
                # 平均から大きく離れた値にする (magnitude * std_dev)
                direction = rng.choice([-1, 1])
                outlier_value = (
                    np.mean(X_train[:, feature_idx]) +
                    direction * magnitude * std_dev
                )
                X_test[i, feature_idx] = outlier_value

    # -> 代替案: シナリオ定義でこの関数を指定し、メインループで特別処理する
    return X_train, X_test, y_train, y_test  # この関数自体はデータを返す


# --- 新しいラベル自動生成関数 ---
def generate_labels_from_rgb(rgb_image_array):
    """
    RGB画像配列からユニークな色を検出し、自動的に数値ラベルを割り当てる。

    Args:
        rgb_image_array (np.ndarray): (height, width, 3) のRGB画像配列。

    Returns:
        tuple: 以下の要素を含むタプル
            - label_image (np.ndarray): (height, width) の数値ラベル配列。
            - rgb_to_label_map (dict): 検出されたRGBタプルをキー、数値ラベルを値とする辞書。
            - label_to_rgb_map (dict): 数値ラベルをキー、RGBタプルを値とする辞書。
    """
    height, width, bands = rgb_image_array.shape
    if bands != 3:
        raise ValueError(f"Input image is not RGB (bands={bands})")

    # (H, W, 3) -> (H*W, 3) に変形してユニークな色を取得
    reshaped_rgb = rgb_image_array.reshape(-1, 3)
    unique_colors = np.unique(reshaped_rgb, axis=0)
    print(
        f"  Auto-detecting labels: Found {len(unique_colors)} unique RGB colors.")

    # RGBと数値ラベルのマッピングを作成 (0から連番)
    # 注意: np.unique はソートされた結果を返すので、ラベル割り当て順序は色に依存
    rgb_to_label_map = {tuple(color): i for i,
                        color in enumerate(unique_colors)}
    label_to_rgb_map = {i: tuple(color)
                        for i, color in enumerate(unique_colors)}

    # ラベル画像を生成 (効率的な方法)
    label_image_flat = np.zeros(height * width, dtype=np.uint32)  # 十分な大きさの型
    for rgb_tuple, label in rgb_to_label_map.items():
        # 元のreshapeされた配列で一致するピクセルのインデックスを探す
        match_indices = np.where(
            np.all(reshaped_rgb == np.array(rgb_tuple), axis=1))[0]
        label_image_flat[match_indices] = label

    label_image = label_image_flat.reshape(height, width)  # 元の形状に戻す

    print(f"  Generated labels: {np.unique(label_image)}")
    # print(f"  RGB to Label map created: {rgb_to_label_map}") # デバッグ用

    return label_image, rgb_to_label_map, label_to_rgb_map


# --- 予測結果画像生成関数 (label_to_rgb_map を使うように変更) ---
def labels_to_rgb_image(label_image_array, label_to_rgb_map, default_color=(0, 0, 0)):
    """数値ラベル配列をRGB画像配列に変換 (マッピング辞書を使用)"""
    height, width = label_image_array.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    unique_pred_labels = np.unique(label_image_array)

    for label_value in unique_pred_labels:
        mask = label_image_array == label_value
        # マッピング辞書から色を取得、なければデフォルト色
        rgb_color = label_to_rgb_map.get(label_value, default_color)
        rgb_image[mask] = np.array(rgb_color)
    return rgb_image


# =================================================


# --- CSV ファイルの準備 (変更なし) ---
# --- CSV ファイルの準備 (ヘッダー修正) ---
# --- RGBからラベルへの自動変換関数 ---
def generate_labels_from_rgb(rgb_image_array):
    height, width, bands = rgb_image_array.shape
    if bands != 3:
        raise ValueError("Input image is not RGB")
    reshaped_rgb = rgb_image_array.reshape(-1, 3)
    unique_colors, indices = np.unique(
        reshaped_rgb, axis=0, return_inverse=True)
    n_unique = len(unique_colors)
    print(f"  Auto-detecting labels: Found {n_unique} unique RGB colors.")
    # マッピング辞書を作成
    rgb_to_label_map = {tuple(color): i for i,
                        color in enumerate(unique_colors)}
    label_to_rgb_map = {i: tuple(color)
                        for i, color in enumerate(unique_colors)}
    # ラベル画像を生成 (uniqueの逆インデックスを利用)
    label_image = indices.reshape(height, width)
    print(f"  Generated labels: {np.unique(label_image)}")
    return label_image, rgb_to_label_map, label_to_rgb_map


# --- ラベルからRGBへの変換関数 ---
def labels_to_rgb_image(label_image_array, label_to_rgb_map, default_color=(0, 0, 0)):
    height, width = label_image_array.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for label_value in np.unique(label_image_array):
        mask = label_image_array == label_value
        rgb_color = label_to_rgb_map.get(label_value, default_color)
        if isinstance(rgb_color, (tuple, list)) and len(rgb_color) == 3:
            rgb_image[mask] = np.array(rgb_color)
        else:
            rgb_image[mask] = np.array(default_color)
            if label_value not in label_to_rgb_map:
                warnings.warn(
                    f"Label {label_value} not in map. Using default.", RuntimeWarning
                )
                label_to_rgb_map[label_value] = default_color  # 再警告抑制
    return rgb_image


# --- CSV ファイルの準備 ---
csv_header = [
    "Image_File",
    "Model_Name",
    "Accuracy",
    "MCC",
    "F1AV_Known",
    "Macro_F1_Known",
    "Weighted_F1_Known",
    "F1AV_All",
    "Macro_F1_All",
    "Weighted_F1_All",
    "Unclassified_Rate",
    "Unknown_Rate",  # Unknown はこのシナリオでは0
    "Fit_Time",
    "Pred_Time",
    "Error_Info",
]
csv_write_ok = False
try:
    with open(csv_filepath, "w", newline="", encoding="utf-8-sig") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_header)
    csv_write_ok = True
except IOError as e:
    print(f"Error creating CSV: {e}")

# --- メイン処理 (全組み合わせ対応版) ---
if not GDAL_AVAILABLE:
    print("Error: GDAL library not found. Exiting.")
    exit()


# --- メイン処理 (全組み合わせ対応版) ---
log_to_text(text_log_filepath,
            f"--- Image Data Validation Start ({timestamp}) ---")
log_to_text(
    text_log_filepath, f"Label Dir: {label_data_dir}, Feature Dir: {feature_data_dir}"
)
log_to_text(text_log_filepath, f"Output Dir: {output_date_dir}")
log_to_text(text_log_filepath,
            f"Models to test ({len(models)}): {list(models.keys())}")
log_to_text(text_log_filepath, "=" * 60)

start_total_time = time.time()

# === ラベル画像ファイル (.tif) を検索 ===
label_files = glob.glob(os.path.join(label_data_dir, "*.tif"))
if not label_files:
    log_to_text(text_log_filepath,
                f"Error: No .tif files found in {label_data_dir}")
    exit()
log_to_text(text_log_filepath, f"Found {len(label_files)} label files.")

# === 特徴量ファイル を検索 ===
feature_extensions = [
    "",
    ".bip",
    ".dat",
    ".img",
    # ".hdr",
]  # GDALが読める可能性のある拡張子
feature_files = []
potential_files = glob.glob(os.path.join(feature_data_dir, "*"))
for f in potential_files:
    if os.path.isfile(f):
        base, ext = os.path.splitext(f)
        # ENVIデータ(.hdrなし)や GeoTIFF など拡張子がない場合や、リストにある拡張子の場合
        if ext.lower() in feature_extensions or ext == "":
            # さらにGDALで開けるか軽くチェック (オプション、時間がかかる可能性あり)
            # try:
            #     ds_test = gdal.Open(f)
            #     if ds_test is not None: feature_files.append(f)
            #     ds_test = None
            # except: pass
            feature_files.append(f)  # ここでは拡張子だけで判断

if not feature_files:
    log_to_text(
        text_log_filepath,
        f"Error: No suitable feature files found in {feature_data_dir}",
    )
    exit()
log_to_text(text_log_filepath,
            f"Found {len(feature_files)} potential feature files.")

# === 全組み合わせでループ ===
combination_count = 0
total_combinations = len(feature_files) * len(label_files)
log_to_text(text_log_filepath,
            f"Total combinations to process: {total_combinations}")

# --- ループ開始 ---
for label_filepath in label_files:
    for current_feature_path in feature_files:
        combination_count += 1
        start_combination_time = time.time()
        label_filename = os.path.basename(label_filepath)
        feature_filename = os.path.basename(current_feature_path)
        feature_filename_base = os.path.splitext(label_filename)[0]

        log_to_text(
            text_log_filepath,
            f"\n===== Processing Combination {combination_count}/{total_combinations} =====",
        )
        log_to_text(text_log_filepath, f"    Label: {label_filename}")
        log_to_text(text_log_filepath, f"    Feature: {feature_filename}")

        # 変数初期化
        X_image, y_image, img_height, img_width, n_bands = None, None, 0, 0, 0
        label_to_rgb_map_current = None
        data_load_error = None
        X_train, X_test, y_train, y_test = None, None, None, None
        split_error = None
        effective_n_classes = 0

        # --- データ読み込みと前処理 ---
        try:
            # --- ラベル(TIF)読み込み ---
            ds_label = gdal.Open(label_filepath)
            if ds_label is None:
                raise IOError(f"Cannot open label TIF: {label_filename}")
            img_height = ds_label.RasterYSize
            img_width = ds_label.RasterXSize

            if ds_label.RasterCount == 3:
                if verbose > 0:
                    print("        Label is RGB TIF, auto-generating labels...")
                rgb_array = np.transpose(ds_label.ReadAsArray(), (1, 2, 0))
                y_image_2d, _, label_to_rgb_map_current = generate_labels_from_rgb(
                    rgb_array
                )
            elif ds_label.RasterCount == 1:
                if verbose > 0:
                    print("        Label is single-band TIF, using values as labels...")
                y_image_2d = ds_label.ReadAsArray()  # (H, W)
                # 必要であれば label_to_rgb_map を手動で定義するか、
                # グレースケール表示用のマップなどを作成
                unique_labels_in_tif = np.unique(y_image_2d)
                # 例: グレースケールマップ
                label_to_rgb_map_current = {
                    label: (
                        (
                            int(label * 255 / max(unique_labels_in_tif))
                            if max(unique_labels_in_tif) > 0
                            else 0
                        ),
                    )
                    * 3
                    for label in unique_labels_in_tif
                }
                print(
                    f"        Generated grayscale map for labels: {unique_labels_in_tif}"
                )
            else:
                raise ValueError(
                    f"Unsupported label band count: {ds_label.RasterCount}"
                )
            ds_label = None  # Close dataset

            # --- 特徴量(ENVI等)読み込み ---
            feature_file_to_open = None
            feature_path_obj = Path(current_feature_path)
            feature_filename = feature_path_obj.name  # ファイル名を取得

            if feature_path_obj.suffix.lower() == ".hdr":
                # .hdrファイルの場合、対応するデータファイルを探す
                base_name = (
                    feature_path_obj.stem
                )  # 拡張子なしのファイル名 (例: ASTER2006)
                parent_dir = feature_path_obj.parent  # ファイルがあるディレクトリ
                # データファイルの一般的な拡張子（拡張子なしも含む）
                possible_data_extensions = ["", ".img", ".dat", ".bip", ".bsq"]

                found_data_file = False
                for ext in possible_data_extensions:
                    # データファイルのフルパス候補を作成
                    data_file_path = parent_dir / (base_name + ext)
                    if data_file_path.is_file():
                        # データファイルが見つかった
                        if verbose > 0:
                            print(
                                f"  Found corresponding data file for '{feature_filename}': '{data_file_path.name}'"
                            )
                        feature_file_to_open = str(
                            data_file_path
                        )  # gdal.Open に渡すパス
                        found_data_file = True
                        break  # 見つかったらループを抜ける

                if not found_data_file:
                    # 対応するデータファイルが見つからなかった場合
                    # spectralライブラリを使うか、エラーにする
                    raise IOError(
                        f"Could not find corresponding data file for HDR: '{feature_filename}' in directory '{parent_dir}'. Searched for base name '{base_name}' with extensions {possible_data_extensions}."
                    )
                    # --- spectral を使う場合の代替案 ---
                    # try:
                    #         import spectral
                    #         print(f"  Attempting to open '{feature_filename}' with spectral library...")
                    #         img = spectral.open_image(str(feature_path_obj))
                    #         # spectral.Image は通常 (rows, cols, bands) または (lines, samples, bands)
                    #         X_image_raw = img.load() # メモリに読み込み
                    #         if verbose > 0:
                    #                 print(f"  Successfully loaded with spectral: shape={X_image_raw.shape}")
                    #         # spectral 用のデータ整形 (GDALと軸の順序が違う可能性)
                    #         img_height_sp, img_width_sp, n_bands_sp = X_image_raw.shape
                    #         if img_height_sp != img_height or img_width_sp != img_width:
                    #                 raise ValueError(f"Dimension mismatch (spectral): Label({img_width}x{img_height}) vs Feature({img_width_sp}x{img_height_sp})")
                    #         n_bands = n_bands_sp
                    #         X_image = X_image_raw.reshape(-1, n_bands) # (ピクセル数, バンド数)
                    #         feature_file_to_open = None # gdal.Open は使わないのでNoneのまま
                    # except ImportError:
                    #         raise ImportError("Spectral library not found. Cannot open HDR file without corresponding data file for GDAL. Please install spectral (`pip install spectral`).")
                    # except Exception as e_spec:
                    #         raise IOError(f"Failed to open '{feature_filename}' with spectral library: {e_spec}")
                    # --- spectral 代替案ここまで ---

            else:
                # .hdr 以外の場合はそのままファイルパスを使用
                feature_file_to_open = str(current_feature_path)  # 文字列に変換

            # --- gdal.Open を実行 ---
            X_image = None  # 初期化
            ds_feat = None
            if (
                feature_file_to_open
            ):  # feature_file_to_open に有効なパスが入っている場合
                ds_feat = gdal.Open(feature_file_to_open)
                if ds_feat is None:
                    # ファイルが開けなかった場合 (対応ファイルが見つからない、破損など)
                    raise IOError(
                        f"Cannot open feature file using GDAL: {os.path.basename(feature_file_to_open)}"
                    )

                # --- GDALでのデータ読み込みと整形 ---
                if (
                    ds_feat.RasterXSize != img_width
                    or ds_feat.RasterYSize != img_height
                ):
                    feat_width = ds_feat.RasterXSize
                    feat_height = ds_feat.RasterYSize
                    ds_feat = None  # エラー前に閉じる
                    raise ValueError(
                        f"Dimension mismatch: Label({img_width}x{img_height}) vs Feature({feat_width}x{feat_height})"
                    )
                n_bands = ds_feat.RasterCount
                if n_bands == 0:
                    ds_feat = None
                    raise ValueError("Feature file has no bands.")
                # ReadAsArray() は (bands, height, width) の順で返す
                X_image_raw = ds_feat.ReadAsArray()
                ds_feat = None  # Close dataset
                # モデル入力形式 (ピクセル数, バンド数) に変形
                # (bands, H, W) -> (H, W, bands) -> (H*W, bands)
                X_image = np.transpose(
                    X_image_raw, (1, 2, 0)).reshape(-1, n_bands)

            # X_image が None のまま (=spectral等他の方法でも読み込めなかった) 場合のエラー処理
            if X_image is None:
                # spectral代替案を使っていない場合は、feature_file_to_open is None のエラーハンドリングで捕捉されるはず
                # spectral代替案を使った場合は、そこでエラーが発生するか、X_image に値が入る
                raise IOError(
                    f"Failed to load feature data from {feature_filename} using available methods."
                )

            # --- 共通の後処理 ---
            y_image = y_image_2d.flatten()  # (ピクセル数,)

            # NaN/Inf チェック & 処理 (X_image がNoneでないことを確認してから)
            if X_image is not None and not np.all(np.isfinite(X_image)):
                warnings.warn(
                    "NaN/Inf detected in feature data, replacing with 0.")
                X_image = np.nan_to_num(
                    X_image, nan=0.0, posinf=0.0, neginf=0.0)

            # データスケーリング (StandardScaler)
            if X_image is not None:
                log_to_text(
                    text_log_filepath,
                    "        Applying StandardScaler to feature data...",
                )
                scaler = StandardScaler()
                X_image_scaled = scaler.fit_transform(X_image)
                X_image = X_image_scaled  # スケーリング済みデータで上書き
                log_to_text(text_log_filepath,
                            "        StandardScaler applied.")
            else:
                log_to_text(
                    text_log_filepath,
                    "        Skipping scaling because feature data (X_image) is None.",
                )

            log_to_text(
                text_log_filepath,
                f"        Data loaded & processed: X={'N/A' if X_image is None else X_image.shape}, y={y_image.shape}, Image=({img_height}x{img_width}), Bands={n_bands}",
            )
            unique_labels_found, label_counts = np.unique(
                y_image, return_counts=True)
            log_to_text(
                text_log_filepath,
                f"        Labels found in y_image: {unique_labels_found}",
            )

        except Exception as e:
            data_load_error = f"Error loading/preprocessing data for {label_filename} / {feature_filename}: {e}\n{traceback.format_exc()}"
            log_to_text(text_log_filepath, f"    {data_load_error}")
            # CSVにエラー記録
            if csv_write_ok:
                csv_row = (
                    [label_filename, feature_filename, "N/A"]
                    + ["N/A"] * 10
                    + [-1.0, -1.0, f"Data load error: {str(e)[:100]}"]
                )
                try:
                    with open(
                        csv_filepath, "a", newline="", encoding="utf-8-sig"
                    ) as cf:
                        csv.writer(cf).writerow(csv_row)
                except IOError as e_csv:
                    print(f"CSV write error: {e_csv}")
                    csv_write_ok = False
            log_to_text(text_log_filepath, "#" * 60 + "\n")
            continue  # 次の組み合わせへ

        # --- データ分割とモデル評価 ---
        if (
            X_image is not None
            and y_image is not None
            and X_image.shape[0] > 0
            and label_to_rgb_map_current is not None
        ):
            try:  # 訓練/テスト分割
                test_fraction = 0.2
                X_image_valid = X_image  # スケーリング済みを使用
                y_image_valid = y_image

                # 層化抽出のためのサンプル数チェック
                unique_y_valid = unique_labels(y_image_valid)
                if len(y_image_valid) > 0 and len(unique_y_valid) > 1:
                    min_samples_for_stratify = (
                        len(unique_y_valid) * 2
                    )  # 各クラス最低2サンプル必要
                    can_stratify = (
                        len(y_image_valid) *
                        test_fraction >= min_samples_for_stratify
                    ) and (
                        len(y_image_valid) * (1 - test_fraction)
                        >= min_samples_for_stratify
                    )

                    if can_stratify:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_image_valid,
                            y_image_valid,
                            test_size=test_fraction,
                            random_state=BASE_RANDOM_STATE,
                            stratify=y_image_valid,
                        )
                    else:
                        warnings.warn(
                            "Not enough samples per class for stratified split, using non-stratified split."
                        )
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_image_valid,
                            y_image_valid,
                            test_size=test_fraction,
                            random_state=BASE_RANDOM_STATE,
                        )
                elif len(y_image_valid) > 0:  # クラスが1種類しかない場合など
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_image_valid,
                        y_image_valid,
                        test_size=test_fraction,
                        random_state=BASE_RANDOM_STATE,
                    )
                else:
                    raise ValueError("No valid pixels found for splitting.")

                log_to_text(
                    text_log_filepath,
                    f"        Split data: Train={X_train.shape}, Test={X_test.shape}",
                )
                effective_n_classes = len(
                    unique_labels(y_train)
                )  # 訓練データのクラス数

            except Exception as e:
                split_error = f"Error splitting data: {e}"
                log_to_text(text_log_filepath, f"    {split_error}")
                # CSVエラー記録など (必要なら)
                if csv_write_ok:
                    csv_row = (
                        [label_filename, feature_filename, "N/A"]
                        + ["N/A"] * 10
                        + [-1.0, -1.0,
                            f"Data split failed: {str(split_error)[:100]}"]
                    )
                    try:
                        with open(
                            csv_filepath, "a", newline="", encoding="utf-8-sig"
                        ) as cf:
                            csv.writer(cf).writerow(csv_row)
                    except IOError as e_csv:
                        print(f"CSV write error: {e_csv}")
                        csv_write_ok = False
                X_train, X_test, y_train, y_test = (
                    None,
                    None,
                    None,
                    None,
                )  # エラー時はNoneにする

            # --- モデル評価ループ (分割成功時のみ) ---
            if (
                X_train is not None
                and X_test is not None
                and y_train is not None
                and y_test is not None
                and X_train.shape[0] > 0
            ):
                for model_name, model_proto in models.items():
                    start_model_time = time.time()
                    log_to_text(
                        text_log_filepath,
                        # インデント調整
                        f"\n                Evaluating Model: {model_name} ...",
                    )
                    # --- ここから evaluate_model_v2 呼び出し以降の処理 ---
                    results = {}  # 結果辞書を初期化
                    error_info = None  # エラー情報を初期化
                    current_model_instance = None  # インスタンスを初期化

                    try:
                        current_model_instance = clone(
                            model_proto
                        )  # 各モデルを複製して独立させる
                        results = evaluate_model_v2(  # evaluate_model_v2 は変更しない
                            model_name,
                            current_model_instance,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            effective_n_classes,  # 訓練データのクラス数を渡す
                        )
                        error_info = results.get(
                            "error_info"
                        )  # evaluate_model_v2内でエラーがあれば取得

                    except (
                        Exception
                    ) as eval_e:  # evaluate_model_v2呼び出し自体のエラーなど
                        error_info = (
                            f"Evaluation crashed: {eval_e}\n{traceback.format_exc()}"
                        )
                        results["error_info"] = error_info  # エラー情報を結果に含める

                    # --- 結果画像生成 (TIF形式で保存) ---
                    # この部分は変更なし (ただし、必要な変数がスコープ内にある前提)
                    if (
                        error_info is None
                        and PIL_AVAILABLE
                        and current_model_instance is not None
                    ):
                        if verbose > 0:
                            log_to_text(
                                text_log_filepath,
                                "                Generating prediction image (TIF)...",
                            )
                        try:
                            # X_image が None でないことを確認 (データ読み込み成功しているはずだが念のため)
                            if X_image is not None:
                                y_pred_full = current_model_instance.predict(
                                    X_image
                                )  # 画像全体で予測
                                pred_image_labels = y_pred_full.reshape(
                                    img_height, img_width
                                )

                                # label_to_rgb_map_current が None でないことを確認
                                if label_to_rgb_map_current is not None:
                                    pred_image_rgb = labels_to_rgb_image(
                                        pred_image_labels, label_to_rgb_map_current
                                    )
                                    pil_img = Image.fromarray(
                                        pred_image_rgb, "RGB")

                                    # ファイル名生成に必要な変数を確認
                                    if (
                                        "feature_filename_base" in locals()
                                        and feature_filename_base is not None
                                    ):
                                        current_time_str = (
                                            datetime.datetime.now().strftime("%H%M%S")
                                        )
                                        safe_feature_name = "".join(
                                            c if c.isalnum() else "_"
                                            for c in feature_filename_base
                                        )
                                        safe_model_name = "".join(
                                            c if c.isalnum() else "_"
                                            for c in model_name
                                        )
                                        image_filename = f"{current_time_str}_{safe_feature_name}_{safe_model_name}.tif"

                                        # image_output_dir が None でないことを確認
                                        if image_output_dir is not None:
                                            image_output_path = os.path.join(
                                                image_output_dir, image_filename
                                            )

                                            # TIF形式で保存
                                            pil_img.save(
                                                image_output_path,
                                                format="TIFF",
                                                compression="tiff_lzw",
                                            )

                                            if verbose > 0:
                                                log_to_text(
                                                    text_log_filepath,
                                                    f"                Prediction image saved (TIF): {os.path.basename(image_output_path)}",
                                                )
                                        else:
                                            log_to_text(
                                                text_log_filepath,
                                                "                Error saving image: image_output_dir is not defined.",
                                            )
                                    else:
                                        log_to_text(
                                            text_log_filepath,
                                            "                Error saving image: feature_filename_base is not defined.",
                                        )
                                else:
                                    log_to_text(
                                        text_log_filepath,
                                        "                Error generating image: label_to_rgb_map_current is None.",
                                    )
                            else:
                                log_to_text(
                                    text_log_filepath,
                                    "                Error generating image: X_image is None.",
                                )

                        except ImportError:
                            log_to_text(
                                text_log_filepath,
                                "                Error: Pillow does not have TIFF support enabled. Cannot save as TIF.",
                            )
                        except Exception as img_e:
                            log_to_text(
                                text_log_filepath,
                                f"                Error generating/saving prediction TIF image: {img_e}\n{traceback.format_exc()}",
                            )
                    elif error_info:
                        log_to_text(
                            text_log_filepath,
                            "                Skipping prediction image generation due to evaluation error.",
                        )
                    elif not PIL_AVAILABLE:
                        log_to_text(
                            text_log_filepath,
                            "                Skipping prediction image generation: Pillow library not available.",
                        )
                    elif current_model_instance is None:
                        log_to_text(
                            text_log_filepath,
                            "                Skipping prediction image generation: Model instance is None.",
                        )
                    # --- 画像保存ここまで ---

                    # ★★★ ここからログ・CSV出力の修正部分 (evaluate_model_v2 変更なし版) ★★★
                    if error_info:  # モデル評価でエラーが発生した場合
                        log_to_text(
                            text_log_filepath,
                            f"                Eval Error: {error_info}",
                        )
                        # --- CSVにもエラー情報を記録 ---
                        if csv_write_ok:
                            csv_row = (
                                [label_filename, feature_filename, model_name]
                                + ["N/A"] * 10  # 精度指標は N/A
                                + [
                                    -1.0,
                                    -1.0,
                                    str(error_info)[:200],
                                ]  # エラー情報を短縮して記録
                            )
                            try:
                                with open(
                                    csv_filepath, "a", newline="", encoding="utf-8-sig"
                                ) as cf:
                                    csv.writer(cf).writerow(csv_row)
                            except IOError as e_csv:
                                print(f"CSV write error: {e_csv}")
                                csv_write_ok = False
                        # --- エラー時の処理ここまで ---

                    else:  # モデル評価が正常に終了した場合
                        # --- 精度指標のログ出力 (より明確に) ---
                        log_to_text(
                            text_log_filepath,
                            f"                => Acc:{results.get('accuracy', 'N/A'):.4f}"
                            f" MCC:{results.get('mcc', 'N/A'):.4f}"
                            f" Uncl:{results.get('unclassified_rate', 'N/A'):.4f}"
                            f" Unk:{results.get('unknown_rate', 'N/A'):.4f}"
                            f" Fit:{results.get('fit_time', -1):.2f}s"
                            f" Pred:{results.get('pred_time', -1):.2f}s",
                        )
                        log_to_text(
                            text_log_filepath,
                            f"                   Known Class Metrics (excl. 0):"
                            f" F1AV:{results.get('f1av_known', 'N/A'):.4f}"
                            f" MacroF1:{results.get('macro_f1_known', 'N/A'):.4f}"
                            f" WgtF1:{results.get('weighted_f1_known', 'N/A'):.4f}",
                        )
                        log_to_text(
                            text_log_filepath,
                            f"                   All Test Class Metrics (excl. 0):"
                            f" F1AV:{results.get('f1av_all', 'N/A'):.4f}"
                            f" MacroF1:{results.get('macro_f1_all', 'N/A'):.4f}"
                            f" WgtF1:{results.get('weighted_f1_all', 'N/A'):.4f}",
                        )
                        # ---------------------------------------

                        # # --------------------------------------------------
                        # 混同行列 (Known) - ラベル 0 を除外して表示
                        # --------------------------------------------------
                        cm_known = results.get("confusion_matrix_known")
                        # labels_known は evaluate_model_v2 から受け取るか、ここで y_train から生成
                        labels_known = results.get(
                            "labels_known"
                        )  # 例: np.unique(y_train) など
                        if (
                            labels_known is None and y_train is not None
                        ):  # evaluate_model_v2が返さない場合の代替
                            labels_known = np.unique(y_train)

                        if (
                            cm_known is not None
                            and labels_known is not None
                            and cm_known.size > 0
                        ):
                            try:
                                labels_known_list = list(labels_known)
                                if 0 in labels_known_list:
                                    zero_index = labels_known_list.index(0)
                                    # 0 の行と列を除外
                                    cm_known_rm0 = np.delete(
                                        cm_known, zero_index, axis=0
                                    )
                                    cm_known_rm0 = np.delete(
                                        cm_known_rm0, zero_index, axis=1
                                    )
                                    labels_known_rm0 = [
                                        l for l in labels_known_list if l != 0
                                    ]

                                    # ログ出力用に文字列化
                                    cm_known_rm0_str = np.array2string(
                                        cm_known_rm0,
                                        separator=", ",
                                        threshold=np.inf,
                                        max_line_width=120,
                                    )
                                    labels_known_rm0_str = str(
                                        sorted(labels_known_rm0))

                                    log_to_text(
                                        text_log_filepath,
                                        f"                    Confusion Matrix (Known Train Classes excl. 0: {labels_known_rm0_str} - Shape: {cm_known_rm0.shape} - rows: true, cols: pred):\n{cm_known_rm0_str}",
                                    )
                                else:  # ラベル 0 がそもそも存在しない場合
                                    cm_known_str = np.array2string(
                                        cm_known,
                                        separator=", ",
                                        threshold=np.inf,
                                        max_line_width=120,
                                    )
                                    labels_known_str = str(
                                        sorted(list(labels_known))
                                    )  # 全ラベル表示
                                    log_to_text(
                                        text_log_filepath,
                                        f"                    Confusion Matrix (Known Train Classes, label 0 not present: {labels_known_str} - Shape: {cm_known.shape} - rows: true, cols: pred):\n{cm_known_str}",
                                    )
                            except Exception as e_cm_known:
                                log_to_text(
                                    text_log_filepath,
                                    f"                    Error processing/displaying cm_known_rm0: {e_cm_known}",
                                )
                        else:
                            log_to_text(
                                text_log_filepath,
                                "                    Confusion Matrix (Known Classes excl. 0): N/A or empty",
                            )

                        # ==============================================================
                        # 混同行列 (All) ← ★★★ここから修正★★★
                        # ==============================================================
                        cm_all = results.get("confusion_matrix_all")  # レベル6

                        # --- unique_labels の引数チェックと呼び出し (修正部分) ---
                        y_pred_val = results.get("y_pred")  # レベル6

                        labels_to_process = []  # レベル6
                        if y_test is not None:  # レベル6
                            labels_to_process.append(y_test)  # レベル7
                        else:  # レベル6
                            # y_test が None の場合のエラーログや処理
                            log_to_text(
                                text_log_filepath,
                                "                    Warning: y_test is None, cannot determine labels for cm_all.",
                            )  # レベル7

                        if y_pred_val is not None:  # レベル6
                            labels_to_process.append(y_pred_val)  # レベル7
                        else:  # レベル6
                            # y_pred が None の場合のエラーログや処理
                            log_to_text(
                                text_log_filepath,
                                "                    Warning: y_pred is None, cannot include predicted labels for cm_all.",
                            )  # レベル7

                        all_unique_labels = []  # レベル6
                        if (
                            labels_to_process
                        ):  # レベル6 -> 有効な配列が1つ以上ある場合のみ unique_labels を呼ぶ
                            try:  # レベル7
                                all_unique_labels = unique_labels(
                                    *labels_to_process
                                )  # レベル8 -> アスタリスクでリストを展開して渡す
                            except Exception as e_ul:  # レベル7
                                log_to_text(
                                    text_log_filepath,
                                    f"                    Error in unique_labels: {e_ul}",
                                )  # レベル8
                                # エラー発生時の処理（例：デフォルト値を設定）
                                all_unique_labels = []  # レベル8
                        # labels_to_process が空の場合、all_unique_labels は [] のまま

                        all_test_labels_list_str = str(
                            sorted([l for l in all_unique_labels if l != 0])
                        )  # レベル6
                        # --- unique_labels 関連の修正ここまで ---

                        # 混同行列 (All) のログ出力 (修正後)
                        if cm_all is not None and cm_all.size > 0:  # レベル6
                            cm_all_str = np.array2string(  # レベル7
                                cm_all,
                                separator=", ",
                                threshold=np.inf,
                                max_line_width=120,
                            )
                            log_to_text(  # レベル7
                                text_log_filepath,
                                f"                    Confusion Matrix (All Test Classes excl. 0: {all_test_labels_list_str} - Shape: {cm_all.shape} - rows: true, cols: pred):\n{cm_all_str}",
                            )
                        else:  # レベル6
                            # cm_all が None や空の場合もラベルリストは表示できるようにする
                            log_to_text(
                                text_log_filepath,
                                f"                    Confusion Matrix (All Test Classes excl. 0: {all_test_labels_list_str}): N/A or empty",
                            )  # レベル7
                        # ==============================================================
                        # ★★★ 混同行列 (All) 関連の修正ここまで ★★★
                        # ==============================================================

                        log_to_text(
                            text_log_filepath, "                " + "-" * 40  # 区切り線
                        )

                        # --- CSV出力 (正常終了時) ---
                        if csv_write_ok:
                            # 結果辞書から値を取得、なければ 'N/A'
                            csv_row_data = {
                                "Accuracy": results.get("accuracy", "N/A"),
                                "MCC": results.get("mcc", "N/A"),
                                "F1AV_Known": results.get("f1av_known", "N/A"),
                                "Macro_F1_Known": results.get("macro_f1_known", "N/A"),
                                "Weighted_F1_Known": results.get(
                                    "weighted_f1_known", "N/A"
                                ),
                                "F1AV_All": results.get("f1av_all", "N/A"),
                                "Macro_F1_All": results.get("macro_f1_all", "N/A"),
                                "Weighted_F1_All": results.get(
                                    "weighted_f1_all", "N/A"
                                ),
                                "Unclassified_Rate": results.get(
                                    "unclassified_rate", "N/A"
                                ),
                                "Unknown_Rate": results.get("unknown_rate", "N/A"),
                                "Fit_Time": results.get("fit_time", -1),
                                "Pred_Time": results.get("pred_time", -1),
                            }
                            # CSV行リストを作成 (csv_header はグローバルスコープにある想定)
                            csv_row = [
                                label_filename,
                                feature_filename,
                                model_name,
                            ]
                            # 精度指標をフォーマットして追加
                            # AccuracyからUnknown_Rateまで
                            for key in csv_header[3:13]:
                                value = csv_row_data.get(key)
                                # N/Aでない数値の場合のみフォーマット
                                if isinstance(value, (int, float)) and value != "N/A":
                                    try:
                                        csv_row.append(f"{float(value):.4f}")
                                    except (ValueError, TypeError):
                                        csv_row.append("N/A")  # フォーマット失敗時
                                else:
                                    csv_row.append("N/A")
                            # 時間を追加
                            fit_time = csv_row_data.get("Fit_Time")
                            pred_time = csv_row_data.get("Pred_Time")
                            csv_row.append(
                                f"{fit_time:.4f}"
                                if isinstance(fit_time, (int, float)) and fit_time >= 0
                                else "-1.0000"
                            )
                            csv_row.append(
                                f"{pred_time:.4f}"
                                if isinstance(pred_time, (int, float))
                                and pred_time >= 0
                                else "-1.0000"
                            )
                            # エラー情報 (正常時は空)
                            csv_row.append("")

                            # CSV書き込み
                            try:
                                with open(
                                    csv_filepath, "a", newline="", encoding="utf-8-sig"
                                ) as cf:
                                    csv.writer(cf).writerow(csv_row)
                            except IOError as e:
                                print(f"CSV write error: {e}")
                                csv_write_ok = False
                        # --- CSV出力ここまで ---
                    # ★★★ ログ・CSV出力の修正部分ここまで ★★★

                    # --- モデルごとの処理時間ログ ---
                    model_elapsed_time = time.time() - start_model_time
                    log_to_text(
                        text_log_filepath,
                        f"                Finished evaluating {model_name} in {model_elapsed_time:.2f} sec",
                    )

            elif split_error:  # 分割失敗のログは分割try-except内で出力済み
                pass  # モデル評価はスキップ
            else:  # データがないなどの理由で分割しなかった場合
                log_to_text(
                    text_log_filepath,
                    f"    Skipping model evaluations for {label_filename}/{feature_filename} due to lack of data for splitting.",
                )
                # 必要であればCSVに記録
                if csv_write_ok:
                    csv_row = (
                        [label_filename, feature_filename, "N/A"]
                        + ["N/A"] * 10
                        + [-1.0, -1.0, "No data for split"]
                    )
                    try:
                        with open(
                            csv_filepath, "a", newline="", encoding="utf-8-sig"
                        ) as cf:
                            csv.writer(cf).writerow(csv_row)
                    except IOError as e_csv:
                        print(f"CSV write error: {e_csv}")
                        csv_write_ok = False

        # --- 組み合わせごとの処理時間 ---
        combination_elapsed_time = time.time() - start_combination_time
        log_to_text(
            text_log_filepath,
            f"===== Finished Combination {combination_count}/{total_combinations} ({label_filename}/{feature_filename}) in {combination_elapsed_time:.2f} sec =====",
        )
        log_to_text(text_log_filepath, "#" * 60 + "\n")

# --- 全ての処理完了 ---
log_to_text(
    text_log_filepath, f"\n===== All {combination_count} combinations processed. ====="
)
log_to_text(
    text_log_filepath,
    f"--- Image Data Validation Complete ({datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}) ---",
)
# (終了ログ出力は省略)
print(f"\nValidation finished.")
# (ファイルパス出力は省略)

try:
    if "__file__" in globals():
        script_path = os.path.abspath(__file__)
        script_filename = os.path.basename(script_path)
    elif len(sys.argv) > 0 and os.path.exists(sys.argv[0]):
        script_path = os.path.abspath(sys.argv[0])
        script_filename = os.path.basename(script_path)
    else:
        print("エラー: スクリプトのパスを特定できませんでした。")
        sys.exit(1)
    print(
        f"デバッグ: 保存先ディレクトリパス: {plot_dir_path}"
    )  # パスを確認するための出力

    # 保存先ディレクトリが存在しない場合は作成
    try:
        os.makedirs(plot_dir_path, exist_ok=True)
        print(f"デバッグ: ディレクトリ '{plot_dir_path}' を作成または確認しました。")
    except OSError as e:
        print(f"エラー: ディレクトリ '{plot_dir_path}' の作成に失敗しました: {e}")
        sys.exit(1)  # ディレクトリ作成失敗で終了

    # コピー先のファイルパスを定義
    destination_path = os.path.join(plot_dir_path, script_filename)
    print(f"デバッグ: コピー元パス: {script_path}")
    print(f"デバッグ: コピー先パス: {destination_path}")

    # スクリプトファイルをバイナリモードで読み込み、別のファイルに書き込む
    buffer_size = 8192  # 読み書きする際のバッファサイズ (バイト単位)
    try:
        with open(script_path, "rb") as fsrc:  # バイナリ読み込みモード
            with open(destination_path, "wb") as fdst:  # バイナリ書き込みモード
                while True:
                    chunk = fsrc.read(buffer_size)
                    if not chunk:
                        break
                    fdst.write(chunk)
        print(f"スクリプト '{script_filename}' を '{plot_dir_path}' にコピーしました。")

    except IOError as e:
        # ファイルI/Oエラー（権限エラーなど）が発生した場合
        print(f"エラー: ファイルのコピー中にIOエラーが発生しました: {e}")
        print(f"詳細: コピー元='{script_path}', コピー先='{destination_path}'")
        # ここで権限関連のエラーメッセージが再度表示される可能性があります
    except Exception as e:
        # その他の予期せぬエラー
        print(f"エラー: ファイルのコピー中に予期せぬエラーが発生しました: {e}")


except Exception as e:
    # スクリプトパス取得やディレクトリ作成前の段階でのエラー
    print(f"スクリプトのコピー準備中にエラーが発生しました: {e}")

print(f"Total 実行時間: {fl(time.time() - start)} [sec]")

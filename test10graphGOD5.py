# -*- coding: utf-8 -*-
# python3 /share_win/tsubo/Satellite_Image/JGR/program/test10graphGOD5.py
# 標準ライブラリ
import csv
import datetime
import os
import sys
import traceback
import warnings

# サードパーティライブラリ
import lightgbm as lgb  # 一般的なエイリアスに変更
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb  # 一般的なエイリアスに変更

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

# ローカルアプリケーション/ライブラリ
from TTmethod.classTTmethod import TTClassifier
from validation_lib.feature_expander2 import FeatureExpansionClassifier
from lightgbm import LGBMClassifier  # ★ LightGBM をインポート
from xgboost import XGBClassifier  # ★ XGBoost をインポート

# --- 以下はインポート文ではないため、インポートブロックの後や設定が必要な箇所に配置 ---
# 実行時の警告を抑制
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
"""
import os
import datetime
import numpy as np
import csv
import traceback
import warnings
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)

# グラフ描画用ライブラリ
import matplotlib.pyplot as plt
import pandas as pd  # CSV読み込みとデータ操作用

# 実行時の警告を抑制
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
###########################################################################################################
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from validation_lib.feature_expander2 import FeatureExpansionClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)  # HGBとGBの両方を追加
import sys  # sysモジュールを追加
from lightgbm import LGBMClassifier  # ★ LightGBM をインポート
from xgboost import XGBClassifier  # ★ XGBoost をインポート
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,  # GradientBoosting は残す
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    OneVsRestClassifier,
)

# from sklearn.ensemble import HistGradientBoostingClassifier # ← 不要なので削除
from lightgbm import LGBMClassifier  # ★ LightGBM をインポート
from xgboost import XGBClassifier  # ★ XGBoost をインポート

from sklearn.linear_model import (  # 他のモデルのインポートも確認
    LogisticRegression,
    SGDClassifier,
    PassiveAggressiveClassifier,
)
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from TTmethod.classTTmethod import TTClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.multiclass import OneVsRestClassifier
# from feature_expander import FeatureExpansionClassifier  # 改造したクラスをインポート
"""
# --- AdvancedCascadeClassifier のインポート ---
try:
    from validation_lib.NewTT6 import (
        AdvancedCascadeClassifier,
    )  # validation_lib.py に保存されていると仮定

    ADVANCED_CASCADE_AVAILABLE = True
    print("Successfully imported AdvancedCascadeClassifier.")
except ImportError as e:
    print(
        f"Warning: Could not import AdvancedCascadeClassifier from validation_lib: {e}"
    )
    ADVANCED_CASCADE_AVAILABLE = False

# --- 実行時警告の抑制 (オプション) ---
# warnings.filterwarnings('ignore', category=UserWarning)
# warnings.filterwarnings('ignore', category=RuntimeWarning) # NaN関連の警告抑制など


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
N_JOBS = 1
##############################
##############################


# --- 検証対象モデル定義 ---
base_estimator_rf = RandomForestClassifier(
    max_depth=MAX_DEPTH_PARAM,
    n_estimators=N_ESTIMATORS_PARAM,
    random_state=0,
    # n_jobs=-1,
    # class_weight="balanced_subsample",
)
# base_hgb = HistGradientBoostingClassifier(max_iter=100, random_state=RANDOM_STATE)

base_rf = base_estimator_rf

##############################
##############################
# --- 検証対象モデル定義 ---
"""
models = {
    "RandomForest": RandomForestClassifier(
        max_depth=MAX_DEPTH_PARAM, n_estimators=N_ESTIMATORS_PARAM, random_state=0
    ),
    # "TTclassifier": TTClassifier(estimator=base_estimator_rf, verbose=1),
    "OneVsRest": OneVsRestClassifier(clone(base_estimator_rf)),
    # "GradientBoosting": GradientBoostingClassifier(        max_depth=MAX_DEPTH_PARAM, n_estimators=N_ESTIMATORS_PARAM, random_state=0),
    # "AdaBoost": AdaBoostClassifier(n_estimators=N_ESTIMATORS_PARAM, random_state=0),
    # "ExtraTrees": ExtraTreesClassifier(   max_depth=MAX_DEPTH_PARAM, n_estimators=N_ESTIMATORS_PARAM, random_state=0),
    # "DecisionTree": DecisionTreeClassifier(max_depth=MAX_DEPTH_PARAM, random_state=0),
    "HistGradientBoosting": HistGradientBoostingClassifier(  # 高速GBDT (通常特徴量)
        max_iter=100, random_state=RANDOM_STATE  # パラメータは適宜調整
    ),
    # --- FeatureExpansionClassifier を使用するモデル ---
    "FeatureExpanderRF": FeatureExpansionClassifier(  # 特徴量拡張 + RF (これが以前のもの)
        base_classifier=RandomForestClassifier(  # ベース分類器を指定
            n_estimators=150, max_depth=15, random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        use_slope=False,
        verbose=0,  # verbose は必要なら1以上に
    ),
    "FeatureExpanderHGB": FeatureExpansionClassifier(  # ★ 特徴量拡張 + 高速GBDT ★
        base_classifier=HistGradientBoostingClassifier(  # ベース分類器を指定
            max_iter=100, random_state=RANDOM_STATE  # HGBのパラメータ
        ),
        use_slope=False,
        verbose=0,
    ),
}
"""
base_dt = RandomForestClassifier(
    max_depth=MAX_DEPTH_PARAM,
    n_estimators=N_ESTIMATORS_PARAM,
    random_state=0,
    # n_jobs=-1,
    # class_weight="balanced_subsample",
)

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
# --- AdvancedCascadeClassifier の設定例 ---
if ADVANCED_CASCADE_AVAILABLE:

    """
    models["TT + (3)5"] = AdvancedCascadeClassifier(
        estimator=clone(base_dt),
        unclassified_tolerance_p=0,
        max_updates=0,  # 特徴量変換なし
        min_f1_threshold=0.01,  # 現実的なF1閾値
        val_size=1,
        feature_generator=SelectFirstKFeaturesAndScale(k=5),
        verbose=1,
        random_state=0,
    )
    models["TT + (3)10"] = AdvancedCascadeClassifier(
        estimator=clone(base_dt),
        unclassified_tolerance_p=0,
        max_updates=0,  # 特徴量変換なし
        min_f1_threshold=0.7,  # 現実的なF1閾値
        val_size=1,
        feature_generator=SelectFirstKFeaturesAndScale(k=10),
        verbose=1,
        random_state=0,
    )
    models["TT + (3)15"] = AdvancedCascadeClassifier(
        estimator=clone(base_dt),
        unclassified_tolerance_p=0,
        max_updates=0,  # 特徴量変換なし
        min_f1_threshold=0.01,  # 現実的なF1閾値
        val_size=1,
        feature_generator=SelectFirstKFeaturesAndScale(k=15),
        verbose=1,
        random_state=0,
    )
    """
    """
    models["AdvCascade_DT_None95"] = AdvancedCascadeClassifier(
        estimator=clone(base_dt),
        unclassified_tolerance_p=0,
        max_updates=0,  # 特徴量変換なし
        min_f1_threshold=0.95,  # 現実的なF1閾値
        val_size=0.2,
        feature_generator=None,
        verbose=1,
        random_state=0,
    )

    models["AdvCascade_DT_ScaleUpd"] = AdvancedCascadeClassifier(
        estimator=clone(base_dt),
        unclassified_tolerance_p=0,
        max_updates=1,  # 1回アップデート試行
        min_f1_threshold=0.85,
        val_size=0.2,
        feature_generator="scaler",  # アップデート時に StandardScaler を使用
        verbose=1,
        random_state=0,
    )
    """
    """
    models["TT + (1)"] = AdvancedCascadeClassifier(
        estimator=clone(base_dt),
        unclassified_tolerance_p=0,
        max_updates=1,  # 2回アップデート試行
        min_f1_threshold=0.95,
        val_size=1,
        feature_generator=None,  # アップデート時に StandardScaler を使用
        verbose=0.2,
        random_state=0,
        useOVR=False
    )

    models["TT"] = AdvancedCascadeClassifier(
        estimator=clone(base_dt),
        unclassified_tolerance_p=0,
        max_updates=0,  # 特徴量変換なし
        min_f1_threshold=0.01,  # 現実的なF1閾値
        val_size=1,
        feature_generator=None,
        verbose=1,
        random_state=0,
        useOVR=False
    )
    models["TT + (2)"] = AdvancedCascadeClassifier(
        estimator=clone(base_dt),
        unclassified_tolerance_p=0.05,
        max_updates=0,  # 特徴量変換なし
        min_f1_threshold=0.01,  # 現実的なF1閾値
        val_size=1,
        feature_generator="scaler",
        verbose=1,
        random_state=0,
        useOVR=True
    )
    """
    """
    models["TT + (1)"] = AdvancedCascadeClassifier(
        estimator=clone(base_dt),
        unclassified_tolerance_p=0.01,
        max_updates=0,  # 特徴量変換なし
        min_f1_threshold=0.85,  # 現実的なF1閾値
        val_size=1,
        feature_generator="scaler",
        verbose=1,
        random_state=0,
        useOVR=False
    )
    """
    models["ACC"] = AdvancedCascadeClassifier(
        estimator=clone(base_dt),
        unclassified_tolerance_p=0.05,
        max_updates=3,  # 1回アップデート試行
        min_f1_threshold=0.98,
        val_size=0.2,
        feature_generator=SelectFirstKFeaturesAndScale(
            k=5),  # アップデート時に StandardScaler を使用
        verbose=1,
        random_state=0,
    )
    """
    # カスタム特徴量生成器を使用する例
    models["AdvCascade_DT_SelectScaleUpd1"] = AdvancedCascadeClassifier(
        estimator=clone(base_dt),
        unclassified_tolerance_p=0.1,
        max_updates=1,
        min_f1_threshold=0.85,
        val_size=0.2,
        feature_generator=SelectFirstKFeaturesAndScale(k=10),  # k=10で選択・スケール
        verbose=1,
        random_state=0,
    )

    models["AdvCascade_DT_ScaleUpd2"] = AdvancedCascadeClassifier(
        estimator=clone(base_dt),
        unclassified_tolerance_p=0.01,
        max_updates=2,  # 1回アップデート試行
        min_f1_threshold=0.85,
        val_size=0.2,
        feature_generator="scaler",  # アップデート時に StandardScaler を使用
        verbose=1,
        random_state=0,
    )

    # カスタム特徴量生成器を使用する例
    models["AdvCascade_DT_SelectScaleUpd"] = AdvancedCascadeClassifier(
        estimator=clone(base_dt),
        unclassified_tolerance_p=0.05,
        max_updates=1,
        min_f1_threshold=0.85,
        val_size=0.2,
        feature_generator=SelectFirstKFeaturesAndScale(k=10),  # k=10で選択・スケール
        verbose=1,
        random_state=0,
    )

    # カスタム特徴量生成器を使用する例
    models["AdvCascade_DT_SelectScaleUpd"] = AdvancedCascadeClassifier(
        estimator=clone(base_dt),
        unclassified_tolerance_p=0,
        max_updates=1,
        min_f1_threshold=0.85,
        val_size=0.2,
        feature_generator=SelectFirstKFeaturesAndScale(k=10),  # k=10で選択・スケール
        verbose=1,
        random_state=0,
    )
    """


# 基本データ生成パラメータ
BASE_N_SAMPLES = 50000
BASE_N_FEATURES = 14  # クラス数が増える場合、これも増やすことを検討
BASE_N_INFORMATIVE = 10
BASE_N_REDUNDANT = 2
BASE_N_CLASSES = 9  # デフォルト値 (n_classesシナリオで上書き)
BASE_CLASS_SEP = 1
BASE_WEIGHTS = None
BASE_RANDOM_STATE = 42


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


# --- カスタムデータ生成関数 (変更なし) ---
def generate_minority_high_sep_data(
    n_samples=BASE_N_SAMPLES,
    n_features=BASE_N_FEATURES,
    centers=5,
    minority_proportion=0.05,
    maj_std=1.5,
    min_std=0.2,
    center_box=(-10.0, 10.0),
    random_state=None,
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
        center = cluster_centers[idx: idx + 1, :]
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


# --- モデル評価関数 (変更なし) ---
def evaluate_model(model_instance, X, y, n_classes, test_size=0.2):
    results = {
        "accuracy": 0.0,
        "mcc": 0.0,
        "f1av": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "weighted_precision": 0.0,
        "weighted_recall": 0.0,
        "weighted_f1": 0.0,
        "report_str": "N/A",
        "confusion_matrix": np.array([]),
        "error": None,
    }
    if X.shape[0] < 10 or y.shape[0] == 0 or len(np.unique(y)) < 1 or n_classes < 1:
        results["error"] = "Insufficient data"
        results["report_str"] = results["error"]
        return results
    unique_y = np.unique(y)
    stratify_option = y if len(unique_y) >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=BASE_RANDOM_STATE,
            stratify=stratify_option,
        )
        if y_test.shape[0] == 0:
            results["error"] = "Empty test set"
            results["report_str"] = results["error"]
            return results
    except ValueError as e:
        try:
            print(f"Stratify failed ({e}), falling back.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=BASE_RANDOM_STATE
            )
            if y_test.shape[0] == 0:
                results["error"] = "Empty test set (fallback)"
                results["report_str"] = results["error"]
                return results
        except Exception as split_err:
            results["error"] = f"Split error: {split_err}"
            results["report_str"] = results["error"]
            return results
    effective_n_classes = n_classes
    unique_test_classes = np.unique(y_test)
    if len(unique_test_classes) < effective_n_classes and stratify_option is not None:
        print(
            f"Warning: Test set has {len(unique_test_classes)}/{effective_n_classes} classes for {type(model_instance).__name__}."
        )
    try:
        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mcc = 0.0
        if len(np.unique(y_test)) >= 2 and len(np.unique(y_pred)) >= 2:
            try:
                mcc = matthews_corrcoef(y_test, y_pred)
            except Exception as mcc_err:
                print(f"Warn: MCC calc failed: {mcc_err}")
        else:
            print(f"Warn: MCC requires >= 2 classes.")
        report_labels = list(range(effective_n_classes))
        report_target_names = [f"C{i}" for i in report_labels]
        try:
            report_dict = classification_report(
                y_test,
                y_pred,
                zero_division=0,
                output_dict=True,
                labels=list(range(n_classes)),
                # target_names=report_target_names,
            )
            report_str = classification_report(
                y_test,
                y_pred,
                zero_division=0,
                labels=report_labels,
                target_names=report_target_names,
            )
        except Exception as report_err:
            report_dict = {}
            report_str = f"Report gen error: {report_err}"
            print(report_str)
        f1av = calculate_f1av(report_dict) if report_dict else 0.0
        try:
            conf_matrix = confusion_matrix(
                y_test, y_pred, labels=report_labels)
        except Exception as cm_err:
            print(f"Warn: CM gen error: {cm_err}")
            conf_matrix = np.array([])
        results["accuracy"] = accuracy
        results["mcc"] = mcc
        results["f1av"] = f1av
        if "macro avg" in report_dict:
            results["macro_precision"] = report_dict["macro avg"]["precision"]
            results["macro_recall"] = report_dict["macro avg"]["recall"]
            results["macro_f1"] = report_dict["macro avg"]["f1-score"]
        if "weighted avg" in report_dict:
            results["weighted_precision"] = report_dict["weighted avg"]["precision"]
            results["weighted_recall"] = report_dict["weighted avg"]["recall"]
            results["weighted_f1"] = report_dict["weighted avg"]["f1-score"]
        results["report_str"] = report_str
        results["confusion_matrix"] = conf_matrix
    except Exception as e:
        results["error"] = f"Eval error: {e}\n{traceback.format_exc()}"
        results["report_str"] = results["error"]
    return results

# --- グラフ描画 (サマリー形式に変更) ---


def generate_summary_plots(csv_filepath, plot_dir_path):
    """
    CSVファイルから結果を読み込み、3種類のサマリーグラフを生成する。
    1. モデル別の平均スコア(棒グラフ)
    2. 全結果の散布図(Accuracy vs F1)
    3. 全データセット別の性能比較(棒グラフ)
    """
    print("\n--- Generating Summary Plots ---")
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        print(
            f"Error: Plotting libraries not found. Please install pandas, matplotlib, seaborn. {e}")
        return

    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(
            f"Error: Results CSV file not found at {csv_filepath}. Cannot generate plots.")
        return

    # --- プロット対象の指標を数値に変換 ---
    metrics_to_plot = ["Accuracy", "MCC", "Macro_F1_Known",
                       "Weighted_F1_Known", "Unclassified_Rate"]
    for col in metrics_to_plot:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            print(f"Warning: Column '{col}' not found in CSV. Skipping.")
            metrics_to_plot.remove(col)

    # プロットできない行（エラー等で数値データがない）を削除
    df.dropna(subset=metrics_to_plot, inplace=True)
    if df.empty:
        print("No valid numeric data to plot.")
        return

    # --- 1. モデル別 平均スコア (棒グラフ) ---
    print("  Generating average score bar chart...")
    plt.figure(figsize=(12, 8))
    # ここでは Macro_F1_Known を代表的な指標として平均を計算
    avg_scores = df.groupby("Model_Name")[
        "Macro_F1_Known"].mean().sort_values(ascending=False)

    bars = plt.bar(avg_scores.index, avg_scores.values,
                   color=plt.cm.viridis(np.linspace(0.2, 0.8, len(avg_scores))))

    plt.title("Average Model Performance (Macro F1 Known)", fontsize=16)
    plt.ylabel("Average Macro F1 (Known Classes)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval,
                 f'{yval:.4f}', va='bottom', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir_path, "summary_average_f1_known.png"))
    plt.close()

    # --- 2. 性能散布図 ---
    print("  Generating performance scatter plot...")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x="Macro_F1_Known",
        y="Accuracy",
        hue="Model_Name",
        palette="tab20",
        s=60,
        alpha=0.7,
    )
    plt.title("Performance Scatter Plot (Accuracy vs. F1 Score)", fontsize=16)
    plt.xlabel("Macro F1 (Known Classes)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(plot_dir_path, "summary_scatter_plot.png"))
    plt.close()

    # --- 3. 全データセット別 性能比較 (棒グラフ) ---
    print("  Generating detailed performance bar chart for all datasets...")
    # ユニークなデータセットIDを作成
    df['dataset_id'] = df['Scenario'] + ' - ' + df['Setting_Name']

    plt.figure(figsize=(24, 10))
    sns.barplot(x="dataset_id", y="Accuracy",
                hue="Model_Name", data=df, palette="tab20")
    plt.title("Model Accuracy per Dataset Scenario",
              fontsize=18, weight='bold')
    plt.xlabel("Dataset Scenario", fontsize=14)
    plt.ylabel("Overall Accuracy", fontsize=14)
    plt.xticks(rotation=60, ha="right", fontsize=9)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(plot_dir_path,
                "summary_accuracy_by_all_datasets.png"))
    plt.close()
# --- モデル評価関数 (KeyError修正・レポート2パターン対応版) ---


def evaluate_model_v2(
    model_name, model_instance, X_train, y_train, X_test, y_test, n_classes_train=None
):
    """
    モデルを学習・評価し、結果辞書を返す(train/test 分離、未分類/未知クラス対応)。
    Classification Report と関連指標を訓練クラス基準と全クラス基準で計算。
    """
    # ★★★ results 辞書の初期化を再確認・強化 ★★★
    results = {
        # 基本評価指標
        "accuracy": 0.0,
        "mcc": 0.0,
        # 訓練クラス基準
        "f1av_known": 0.0,
        "macro_precision_known": 0.0,
        "macro_recall_known": 0.0,
        "macro_f1_known": 0.0,
        "weighted_precision_known": 0.0,
        "weighted_recall_known": 0.0,
        "weighted_f1_known": 0.0,
        "report_str_known": "N/A",
        "confusion_matrix_known": np.array([]),
        # 全クラス基準
        "f1av_all": 0.0,
        "macro_precision_all": 0.0,
        "macro_recall_all": 0.0,
        "macro_f1_all": 0.0,
        "weighted_precision_all": 0.0,
        "weighted_recall_all": 0.0,
        "weighted_f1_all": 0.0,
        "report_str_all": "N/A",
        "confusion_matrix_all": np.array([]),
        # 追加情報
        "unclassified_rate": 0.0,
        "unknown_rate": 0.0,
        "error": None,
    }
    # -----------------------------------------------

    try:
        # === モデル学習 ===
        if X_train.shape[0] == 0 or len(unique_labels(y_train)) < 1:
            raise ValueError("Training data is empty or has no valid labels.")
        model_instance.fit(X_train, y_train)
        # ==================

        # === 予測 ===
        if X_test.shape[0] == 0:
            print("Warning: Test data is empty. Skipping prediction and evaluation.")
            # results の初期値をそのまま返す
            return results
        y_pred = model_instance.predict(X_test)
        # ============

        # --- 未分類/未知クラス処理 ---
        known_labels_train = unique_labels(y_train)
        all_labels_test = unique_labels(
            y_test
        )  # テストデータに実際に存在するラベル全体
        unknown_mask_test = ~np.isin(y_test, known_labels_train)
        num_unknown = np.sum(unknown_mask_test)
        results["unknown_rate"] = num_unknown / \
            len(y_test) if len(y_test) > 0 else 0.0

        is_advanced_cascade = hasattr(model_instance, "unclassified_value_")
        unclassified_value = (
            model_instance.unclassified_value_ if is_advanced_cascade else None
        )
        is_unclassified_mask = np.zeros(len(y_pred), dtype=bool)
        if is_advanced_cascade and hasattr(model_instance, "_is_unclassified"):
            if hasattr(model_instance, "unclassified_value_"):
                is_unclassified_mask = model_instance._is_unclassified(y_pred)
            else:
                warnings.warn(
                    "AdvancedCascadeClassifier missing 'unclassified_value_'.",
                    RuntimeWarning,
                )
        num_unclassified = np.sum(is_unclassified_mask)
        results["unclassified_rate"] = (
            num_unclassified / len(y_pred) if len(y_pred) > 0 else 0.0
        )

        # --- 評価対象サンプルの決定 (未分類を除外) ---
        classified_mask = ~is_unclassified_mask
        y_test_classified = y_test[classified_mask]
        y_pred_classified = y_pred[classified_mask]

        if len(y_test_classified) == 0:
            print("Warning: No classified samples left for evaluation.")
            results["report_str_known"] = "No classified samples."
            results["report_str_all"] = "No classified samples."
            # accuracy なども 0 のまま返す
            return results
        # ------------------------------------------

        # --- 全体精度とMCC (未分類除外後) ---
        results["accuracy"] = accuracy_score(
            y_test_classified, y_pred_classified)
        try:
            results["mcc"] = matthews_corrcoef(
                y_test_classified, y_pred_classified)
        except ValueError:
            results["mcc"] = 0.0
        # ------------------------------------

        # --- 訓練クラス基準での評価 ---
        report_dict_known = {}
        report_str_known = "N/A"
        conf_matrix_known = np.array([])
        try:
            # labels に訓練クラスを指定 (存在しない予測ラベルは無視される)
            # output_dict=True で計算に必要な情報を取得
            with warnings.catch_warnings():  # 存在しないラベルに関する警告を抑制する場合
                warnings.simplefilter("ignore", category=UserWarning)
                report_dict_known = classification_report(
                    y_test_classified,
                    y_pred_classified,
                    labels=known_labels_train,
                    zero_division=0,
                    output_dict=True,
                )
            # 文字列レポートも生成
            report_str_known = classification_report(
                y_test_classified,
                y_pred_classified,
                labels=known_labels_train,
                zero_division=0,
            )
            report_str_known += (
                f"\n(Eval based on {len(known_labels_train)} known train classes)"
            )
            report_str_known += f"\nUnclassified: {results['unclassified_rate']:.4f}"
            report_str_known += f"\nUnknown in Test: {results['unknown_rate']:.4f}"

            # results 辞書に値を格納 (キーは初期化で存在保証)
            results["f1av_known"] = calculate_f1av(report_dict_known)
            if "macro avg" in report_dict_known:
                results["macro_precision_known"] = report_dict_known["macro avg"][
                    "precision"
                ]
                results["macro_recall_known"] = report_dict_known["macro avg"]["recall"]
                results["macro_f1_known"] = report_dict_known["macro avg"]["f1-score"]
            if "weighted avg" in report_dict_known:
                results["weighted_precision_known"] = report_dict_known["weighted avg"][
                    "precision"
                ]
                results["weighted_recall_known"] = report_dict_known["weighted avg"][
                    "recall"
                ]
                results["weighted_f1_known"] = report_dict_known["weighted avg"][
                    "f1-score"
                ]

            # 混同行列も訓練クラス基準で作成
            conf_matrix_known = confusion_matrix(
                y_test_classified, y_pred_classified, labels=known_labels_train
            )

        except Exception as report_err_known:
            error_msg = f"Known Class Report Error: {report_err_known}\n{traceback.format_exc()}"
            print(error_msg)
            results["report_str_known"] = (
                error_msg  # エラー情報をレポート文字列に入れる
            )
            # results内の他の *_known キーは初期値のまま
        # 最後に結果を格納
        results["report_str_known"] = report_str_known
        results["confusion_matrix_known"] = conf_matrix_known
        # ------------------------------------

        # --- 全クラス基準での評価 ---
        report_dict_all = {}
        report_str_all = "N/A"
        conf_matrix_all = np.array([])
        try:
            # labels にテストデータ中の全クラスを指定
            # (未知クラスの Recall/F1 は 0 になるはず)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                report_dict_all = classification_report(
                    y_test_classified,
                    y_pred_classified,
                    labels=all_labels_test,
                    zero_division=0,
                    output_dict=True,
                )
            report_str_all = classification_report(
                y_test_classified,
                y_pred_classified,
                labels=all_labels_test,
                zero_division=0,
            )
            report_str_all += (
                f"\n(Eval based on all {len(all_labels_test)} classes in test data)"
            )
            report_str_all += f"\nUnclassified: {results['unclassified_rate']:.4f}"
            report_str_all += f"\nUnknown in Test: {results['unknown_rate']:.4f}"

            # results 辞書に値を格納 (キーは初期化で存在保証)
            results["f1av_all"] = calculate_f1av(report_dict_all)
            if "macro avg" in report_dict_all:
                results["macro_precision_all"] = report_dict_all["macro avg"][
                    "precision"
                ]
                results["macro_recall_all"] = report_dict_all["macro avg"]["recall"]
                results["macro_f1_all"] = report_dict_all["macro avg"]["f1-score"]
            if "weighted avg" in report_dict_all:
                results["weighted_precision_all"] = report_dict_all["weighted avg"][
                    "precision"
                ]
                results["weighted_recall_all"] = report_dict_all["weighted avg"][
                    "recall"
                ]
                results["weighted_f1_all"] = report_dict_all["weighted avg"]["f1-score"]

            conf_matrix_all = confusion_matrix(
                y_test_classified, y_pred_classified, labels=all_labels_test
            )

        except Exception as report_err_all:
            error_msg = (
                f"All Class Report Error: {report_err_all}\n{traceback.format_exc()}"
            )
            print(error_msg)
            results["report_str_all"] = error_msg  # エラー情報をレポート文字列に入れる
            # results内の他の *_all キーは初期値のまま
        # 最後に結果を格納
        results["report_str_all"] = report_str_all
        results["confusion_matrix_all"] = conf_matrix_all
        # ----------------------------------

    except Exception as e:  # 学習・予測段階でのエラー
        results["error"] = f"Evaluation Error: {e}\n{traceback.format_exc()}"
        results["report_str_known"] = results["error"]  # 両レポートにエラー記録
        results["report_str_all"] = results["error"]
        # この場合も results 辞書自体は初期化されたキーを持つはず

    # デバッグ用: 返す直前のresultsの内容を確認
    # print("DEBUG: Final results dict keys:", list(results.keys()))
    return results


# --- 新しいカスタムデータ生成関数 ---
def generate_imbalance_weights(
    n_classes, pattern="linear_decrease", min_prop=0.05, max_prop=0.8, random_state=None
):
    """クラス数に応じた不均衡weightsリストを生成する"""
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

    # === 検証シナリオ定義 (新しいシナリオ追加) ===
    # validation_scenarios = [
    # --- 既存シナリオ (一部調整) ---
    """
    {
        "type": "single_param",
        "name": "n_classes",
        "range": [2, 5, 10, 20],  # rangeを絞る
        "setting_name_format": "({} classes)",
        "dynamic_feature_adjustment": True,
    },
    {
        "type": "single_param",
        "name": "class_sep",
        "options": {"Low (0.1)": 0.1, "Med (1.0)": 1.0, "High (2.0)": 2.0},
    },
    # {"type": "single_param", "name": "weights", ... } # ← dynamic_weights に置き換え
    {
        "type": "single_param",
        "name": "n_features",
        "options": {"Low (10)": 10, "Med (20)": 20, "High (50)": 50},
    },
    # --- 不均衡データ (動的生成) ---
    {
        "type": "dynamic_weights",
        "name": "Imbalance",
        # 試すパターン
        "patterns": ["linear_decrease", "one_dominant", "two_dominant"],
        # "base_n_classes": 5, # このクラス数で weights を生成する場合 (省略可)
    },
    # --- 未知クラス混入 ---
    {
        "type": "unknown_classes",
        "name": "Unknown Classes",
        "options": {
            "Low (5%, 1 cls)": (0.05, 1),
            "Med (10%, 2 cls)": (0.10, 2),
            "High (20%, 3 cls)": (0.20, 3),
        },
        # options の値は (unknown_fraction, n_unknown_classes)
    },

    """
    # --- 異常値混入 ---


validation_scenarios = [
    {
        "type": "outliers",
        "name": "Outliers",
        "options": {
            "Low (1%, 3mag)": (0.01, 3.0),
            "Med (5%, 5mag)": (0.05, 5.0),
            "High (10%, 7mag)": (0.10, 7.0),
        },
        # options の値は (outlier_fraction, magnitude)
    },
    # --- 組み合わせシナリオ (例) ---
    {
        "type": "combined_params",
        "name": "HighClass_LowSep_HighFeat",
        "params": {
            "n_classes": 15,
            "class_sep": 0.5,
            "n_features": 60,
            "n_informative": 40,
            "n_redundant": 10,
            "n_samples": 2000,
        },
    },
    {
        "type": "custom_data",
        "name": "Minority_HighSep",
        "data_generator": generate_minority_high_sep_data,
        "params": {
            "n_samples": 1500,
            "n_features": 20,
            "centers": 5,
            "minority_proportion": 0.03,
            "maj_std": 1.8,
            "min_std": 0.1,
            "center_box": (-15.0, 15.0),
        },
    },

    {
        "type": "single_param",
        "name": "class_sep",
        "options": {"Low (0.1)": 0.1, "Medium (1.0)": 1.0, "High (2.0)": 2.0},
    },
    {
        "type": "single_param",
        "name": "weights",
        "options": {
            "Balanced": None,
            "Slightly Imbalanced (5cls)": [0.5, 0.3, 0.1, 0.05, 0.05],
            "Highly Imbalanced (5cls)": [0.8, 0.05, 0.05, 0.05, 0.05],
        },
        "condition": lambda p: p.get("n_classes", BASE_N_CLASSES) == 5,
    },
    {
        "type": "single_param",
        "name": "n_samples",
        "options": {"Low (100)": 100, "Medium (1000)": 1000, "High (10000)": 10000},
    },
    {
        "type": "combined_params",
        "name": "HighClass_LowSep_HighFeat",
        "params": {
            "n_classes": 10,
            "class_sep": 0.5,
            "n_features": 50,
            "n_informative": 30,
            "n_redundant": 10,
            "n_samples": 2000,
        },
    },
    {
        "type": "combined_params",
        "name": "LowSample_HighImbalance",
        "params": {
            "n_samples": 200,
            "n_classes": 5,
            "weights": [0.8, 0.05, 0.05, 0.05, 0.05],
            "class_sep": 0.8,
            "flip_y": 0,
        },
    },
    {
        "type": "custom_data",
        "name": "Minority_HighSep",
        "data_generator": generate_minority_high_sep_data,
        "params": {
            "n_samples": 1500,
            "n_features": 20,
            "centers": 5,
            "minority_proportion": 0.03,
            "maj_std": 1.8,
            "min_std": 0.1,
            "center_box": (-15.0, 15.0),
        },
    },
]
# =================================================


# --- CSV ファイルの準備 (変更なし) ---
# --- CSV ファイルの準備 (ヘッダー修正) ---
csv_header = [
    "Model_Name",
    "Scenario",
    "Setting_Name",
    "Setting_Value",
    "Accuracy",
    "MCC",
    "F1AV_Known",
    "Macro_F1_Known",
    "Weighted_F1_Known",  # Known基準指標
    "F1AV_All",
    "Macro_F1_All",
    "Weighted_F1_All",  # All基準指標
    "Unclassified_Rate",
    "Unknown_Rate",  # 追加情報
    "Macro_Precision_Known",
    "Macro_Recall_Known",  # 詳細指標 (Known)
    "Weighted_Precision_Known",
    "Weighted_Recall_Known",
    "Macro_Precision_All",
    "Macro_Recall_All",  # 詳細指標 (All)
    "Weighted_Precision_All",
    "Weighted_Recall_All",
    "Error_Info",
]
try:
    with open(csv_filepath, "w", newline="", encoding="utf-8-sig") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_header)
    csv_write_ok = True
except IOError as e:
    print(f"Error creating CSV: {e}")
    csv_write_ok = False


# --- メイン処理 (ループ部分修正 - custom_data 対応追加) ---
log_to_text(text_log_filepath,
            f"--- Multi-Model Validation Start ({timestamp}) ---")
# (ログヘッダー出力は省略)
log_to_text(text_log_filepath, "=" * 60)

# === シナリオループ ===
for scenario in validation_scenarios:
    scenario_type = scenario.get("type", "single_param")
    scenario_name = scenario["name"]
    log_to_text(
        text_log_filepath,
        f"\n===== SCENARIO: {scenario_name} (Type: {scenario_type}) =====",
    )
    settings_list = []  # 各シナリオで実行する設定のリスト

    # --- 設定リストの準備 ---
    try:  # 設定リスト準備中のエラーをキャッチ
        if scenario_type == "single_param":
            param_name_to_change = scenario["name"]
            dynamic_feature_adjustment = scenario.get(
                "dynamic_feature_adjustment", False
            )
            condition_func = scenario.get("condition")
            value_source = scenario.get("range", scenario.get("options", {}))
            is_range = "range" in scenario

            source_iterator = value_source if is_range else value_source.items()
            for setting_info in source_iterator:
                setting_value = setting_info if is_range else setting_info[1]
                setting_name = (
                    scenario.get("setting_name_format",
                                 "({})").format(setting_value)
                    if is_range
                    else setting_info[0]
                )

                current_params = {  # ベースパラメータ
                    "n_samples": BASE_N_SAMPLES,
                    "n_features": BASE_N_FEATURES,
                    "n_informative": BASE_N_INFORMATIVE,
                    "n_redundant": BASE_N_REDUNDANT,
                    "n_classes": BASE_N_CLASSES,
                    "class_sep": BASE_CLASS_SEP,
                    "weights": BASE_WEIGHTS,
                    "random_state": BASE_RANDOM_STATE,
                    "n_repeated": 0,
                    "flip_y": 0.01,
                    "n_clusters_per_class": 1,
                }
                current_params[param_name_to_change] = setting_value

                # パラメータ調整と制約チェック (省略)
                run_this_setting = True
                # ... (調整・チェックロジック、 run_this_setting = False の可能性) ...
                if dynamic_feature_adjustment and param_name_to_change == "n_classes":
                    adjusted_n_features = max(20, setting_value * 2)
                    current_params["n_features"] = adjusted_n_features
                    current_params["n_informative"] = max(
                        setting_value, adjusted_n_features // 2
                    )
                    current_params["n_redundant"] = max(
                        0,
                        adjusted_n_features
                        - current_params["n_informative"]
                        - setting_value,
                    )
                    # ... (他の制約チェック) ...

                if run_this_setting:
                    settings_list.append(
                        {
                            "name": setting_name,
                            "value": setting_value,
                            "params": current_params,
                            "gen_func": make_classification,
                            "is_make_classification": True,
                        }
                    )

        elif scenario_type == "dynamic_weights":
            patterns = scenario.get("patterns", ["linear_decrease"])
            base_n_cls = scenario.get("base_n_classes", BASE_N_CLASSES)
            for pattern in patterns:
                setting_name = f"Pattern: {pattern} ({base_n_cls} cls)"
                weights = generate_imbalance_weights(
                    base_n_cls, pattern, random_state=BASE_RANDOM_STATE
                )
                if weights is None:
                    continue

                current_params = {  # ベースパラメータ + weights
                    "n_samples": BASE_N_SAMPLES,
                    "n_features": BASE_N_FEATURES,
                    "n_informative": BASE_N_INFORMATIVE,
                    "n_redundant": BASE_N_REDUNDANT,
                    "n_classes": base_n_cls,
                    "class_sep": BASE_CLASS_SEP,
                    "weights": weights,
                    "random_state": BASE_RANDOM_STATE,
                    "n_repeated": 0,
                    "flip_y": 0,
                    "n_clusters_per_class": 1,
                }
                # --- 制約チェック (省略) ---
                settings_list.append(
                    {
                        "name": setting_name,
                        "value": f"{base_n_cls}cls-{pattern}",
                        "params": current_params,
                        "gen_func": make_classification,
                        "is_make_classification": True,
                    }
                )

        elif scenario_type == "unknown_classes" or scenario_type == "outliers":
            options_to_test = scenario.get("options", {})
            for setting_name, setting_value in options_to_test.items():
                current_params = {  # ベースパラメータ
                    "n_samples": BASE_N_SAMPLES,
                    "n_features": BASE_N_FEATURES,
                    "n_informative": BASE_N_INFORMATIVE,
                    "n_redundant": BASE_N_REDUNDANT,
                    "n_classes": BASE_N_CLASSES,
                    "class_sep": BASE_CLASS_SEP,
                    "weights": BASE_WEIGHTS,
                    "random_state": BASE_RANDOM_STATE,
                    "n_repeated": 0,
                    "flip_y": 0.01,
                    "n_clusters_per_class": 1,
                    "scenario_setting_value": setting_value,  # シナリオ固有設定
                }
                # --- 制約チェック (省略) ---
                settings_list.append(
                    {
                        "name": setting_name,
                        "value": setting_value,
                        "params": current_params,
                        "gen_func": make_classification,  # 元データ生成用
                        "scenario_type": scenario_type,  # 後処理用
                        "is_make_classification": True,
                    }
                )

        elif scenario_type == "combined_params":
            setting_name = scenario["name"]
            params_to_set = scenario["params"]
            current_params = {  # ベースパラメータ
                "n_samples": BASE_N_SAMPLES,
                "n_features": BASE_N_FEATURES,
                "n_informative": BASE_N_INFORMATIVE,
                "n_redundant": BASE_N_REDUNDANT,
                "n_classes": BASE_N_CLASSES,
                "class_sep": BASE_CLASS_SEP,
                "weights": BASE_WEIGHTS,
                "random_state": BASE_RANDOM_STATE,
                "n_repeated": 0,
                "flip_y": 0.01,
                "n_clusters_per_class": 1,
            }
            current_params.update(params_to_set)
            # --- 制約チェック (省略) ---
            run_this_setting = True
            # if constraints_failed: run_this_setting = False
            if run_this_setting:
                settings_list.append(
                    {
                        "name": setting_name,
                        "value": "Combined",
                        "params": current_params,
                        "gen_func": make_classification,
                        "is_make_classification": True,
                    }
                )

        # ★★★ custom_data シナリオの処理を追加 ★★★
        elif scenario_type == "custom_data":
            setting_name = scenario["name"]
            data_gen_func = scenario.get("data_generator")  # 関数オブジェクトを取得
            data_gen_params = scenario.get("params", {})  # 関数の引数を取得
            if data_gen_func is None or not callable(data_gen_func):
                log_to_text(
                    text_log_filepath,
                    f" Skipping '{setting_name}': Invalid or missing 'data_generator'.",
                )
                continue  # 関数がない場合はスキップ
            # パラメータに random_state を追加
            params_with_state = data_gen_params.copy()
            params_with_state["random_state"] = BASE_RANDOM_STATE
            settings_list.append(
                {
                    "name": setting_name,
                    "value": "Custom",  # 値は "Custom" 固定
                    "params": params_with_state,  # カスタム関数のパラメータ
                    "gen_func": data_gen_func,  # カスタムデータ生成関数
                    "is_make_classification": False,  # make_classification ではない
                }
            )
        # -----------------------------------------
        else:
            log_to_text(
                text_log_filepath, f" Skipping unknown scenario type: {scenario_type}"
            )

    except Exception as e:  # 設定リスト準備中のエラー
        log_to_text(
            text_log_filepath,
            f"Error preparing settings for scenario '{scenario_name}': {e}\n{traceback.format_exc()}",
        )
        continue  # このシナリオをスキップ

    # === 設定値ループ (共通処理) ===
    for setting_details in settings_list:
        # --- 変数取り出し ---
        setting_name = setting_details["name"]
        setting_value = setting_details["value"]
        current_params = setting_details["params"]
        if not isinstance(current_params, dict):
            log_to_text(
                text_log_filepath,
                f" Error: 'params' is not dict for '{setting_name}'. Skipping.",
            )
            continue
        data_gen_func = setting_details.get("gen_func")
        is_make_classification_base = setting_details.get(
            "is_make_classification", False
        )
        special_scenario_type = setting_details.get("scenario_type")
        # --- ---

        if data_gen_func is None:
            continue
        log_to_text(text_log_filepath, f"\n--- Setting: {setting_name} ---")

        # --- データ生成 (custom_data 対応) ---
        X_orig, y_orig = None, None
        X_train, X_test, y_train, y_test = None, None, None, None
        data_gen_error = None
        effective_n_classes = BASE_N_CLASSES  # デフォルト

        try:
            # 1. 元データを生成
            if is_make_classification_base:
                gen_params_for_func = {
                    k: v
                    for k, v in current_params.items()
                    if k
                    in [
                        "n_samples",
                        "n_features",
                        "n_informative",
                        "n_redundant",
                        "n_repeated",
                        "n_classes",
                        "n_clusters_per_class",
                        "weights",
                        "flip_y",
                        "class_sep",
                        "random_state",
                    ]
                }
                if "weights" in gen_params_for_func and not isinstance(
                    gen_params_for_func["weights"], list
                ):
                    gen_params_for_func["weights"] = None
                X_orig, y_orig = make_classification(**gen_params_for_func)
                effective_n_classes = len(unique_labels(y_orig))
            # ★★★ custom_data のデータ生成呼び出し ★★★
            elif scenario_type == "custom_data":  # is_make_classification_base が False
                gen_params_for_func = current_params  # params をそのまま渡す
                log_to_text(
                    text_log_filepath,
                    f"    Calling custom generator: {data_gen_func.__name__}",
                )
                # カスタム関数は X, y を直接返すことを期待
                X_orig, y_orig = data_gen_func(**gen_params_for_func)
                if X_orig is None or y_orig is None:
                    raise ValueError("Custom generator returned None.")
                effective_n_classes = len(unique_labels(y_orig))
            else:
                raise ValueError(f"Unexpected scenario/gen_func combination.")

            # 2. シナリオに応じた後処理とデータ分割
            if special_scenario_type == "unknown_classes":
                # ... (unknown class 処理 - 変更なし) ...
                unknown_fraction, n_unknown_classes = current_params[
                    "scenario_setting_value"
                ]
                X_train, X_test, y_train, y_test = generate_data_with_unknown_classes(
                    X_orig,
                    y_orig,
                    unknown_fraction,
                    n_unknown_classes,
                    current_params["random_state"],
                )
                log_to_text(
                    text_log_filepath,
                    f"    Processed data: X_train={X_train.shape}, X_test={X_test.shape}. Added unknown classes.",
                )
                effective_n_classes = len(unique_labels(y_train))
            elif special_scenario_type == "outliers":
                # ... (outlier 処理 - 変更なし) ...
                outlier_fraction, magnitude = current_params["scenario_setting_value"]
                X_train, X_test, y_train, y_test = generate_data_with_outliers(
                    X_orig,
                    y_orig,
                    outlier_fraction,
                    magnitude,
                    current_params["random_state"],
                )
                log_to_text(
                    text_log_filepath,
                    f"    Processed data: X_train={X_train.shape}, X_test={X_test.shape}. Added outliers.",
                )
                effective_n_classes = len(unique_labels(y_train))
            else:  # 通常分割 (custom_data含む)
                if X_orig is None or y_orig is None or X_orig.shape[0] == 0:
                    raise ValueError("Cannot split empty data.")
                stratify_opt = y_orig if len(
                    unique_labels(y_orig)) > 1 else None
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_orig,
                        y_orig,
                        test_size=0.2,
                        random_state=current_params["random_state"],
                        stratify=stratify_opt,
                    )
                except ValueError:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_orig,
                        y_orig,
                        test_size=0.2,
                        random_state=current_params["random_state"],
                    )
                log_to_text(
                    text_log_filepath,
                    f"    Generated/Split data: X_train={X_train.shape}, X_test={X_test.shape}, Source Classes={unique_labels(y_orig)}",
                )
                effective_n_classes = len(unique_labels(y_train))

        except Exception as e:
            data_gen_error = (
                f"Data generation/processing error: {e}\n{traceback.format_exc()}"
            )
            log_to_text(text_log_filepath, f"    {data_gen_error}")

        # --- モデルループ & 評価 & 出力 (変更なし) ---
        if (
            X_train is not None
            and X_test is not None
            and y_train is not None
            and y_test is not None
            and X_train.shape[0] > 0
        ):
            for model_name, model_proto in models.items():
                log_to_text(
                    text_log_filepath, f"\n  Evaluating Model: {model_name} ..."
                )
                current_model_instance = clone(model_proto)
                results = evaluate_model_v2(
                    model_name,
                    current_model_instance,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    effective_n_classes,
                )
                error_info = results.get("error")
                # (ログ、CSV出力 - 変更なし、ただし新しい results キーに対応させる)
                if error_info:
                    log_to_text(
                        text_log_filepath, f"    Evaluation Error: {error_info}"
                    )
                else:
                    # --- 基本指標の表示 ---
                    log_to_text(
                        text_log_filepath,
                        f"    => Accuracy: {results.get('accuracy', 0.0):.4f}",
                    )
                    log_to_text(
                        text_log_filepath, f"    => MCC: {results.get('mcc', 0.0):.4f}"
                    )
                    log_to_text(
                        text_log_filepath,
                        f"    => Unclassified Rate: {results.get('unclassified_rate', 0.0):.4f}",
                    )
                    log_to_text(
                        text_log_filepath,
                        f"    => Unknown Rate in Test: {results.get('unknown_rate', 0.0):.4f}",
                    )

                    # --- 訓練クラス基準のレポートと混同行列 ---
                    log_to_text(
                        text_log_filepath,
                        "\n    --- Evaluation (Known Train Classes) ---",
                    )
                    log_to_text(
                        text_log_filepath,
                        f"    => F1AV (Known): {results.get('f1av_known', 0.0):.4f}",
                    )
                    log_to_text(
                        text_log_filepath,
                        f"    => Macro F1 (Known): {results.get('macro_f1_known', 0.0):.4f}",
                    )
                    log_to_text(
                        text_log_filepath,
                        f"    => Weighted F1 (Known): {results.get('weighted_f1_known', 0.0):.4f}",
                    )
                    log_to_text(
                        text_log_filepath,
                        "    Classification Report (Known Classes):\n"
                        + results.get("report_str_known", "N/A"),
                    )
                    cm_known = results.get("confusion_matrix_known")
                    if cm_known is not None and cm_known.size > 0:
                        log_to_text(
                            text_log_filepath,
                            f"    Confusion Matrix (Known Classes - rows: true, cols: pred):\n{cm_known}",
                        )
                    else:
                        log_to_text(
                            text_log_filepath,
                            "    Confusion Matrix (Known Classes): N/A",
                        )

                    # --- 全クラス基準のレポートと混同行列 ---
                    # 未知クラスが存在する場合のみ表示する意義が大きい
                    if results.get("unknown_rate", 0.0) > 0 or results.get(
                        "report_str_all", "N/A"
                    ) != results.get("report_str_known", "N/A"):
                        log_to_text(
                            text_log_filepath,
                            "\n    --- Evaluation (All Test Classes) ---",
                        )
                        log_to_text(
                            text_log_filepath,
                            f"    => F1AV (All): {results.get('f1av_all', 0.0):.4f}",
                        )
                        log_to_text(
                            text_log_filepath,
                            f"    => Macro F1 (All): {results.get('macro_f1_all', 0.0):.4f}",
                        )
                        log_to_text(
                            text_log_filepath,
                            f"    => Weighted F1 (All): {results.get('weighted_f1_all', 0.0):.4f}",
                        )
                        log_to_text(
                            text_log_filepath,
                            "    Classification Report (All Test Classes):\n"
                            + results.get("report_str_all", "N/A"),
                        )
                        cm_all = results.get("confusion_matrix_all")
                        if cm_all is not None and cm_all.size > 0:
                            log_to_text(
                                text_log_filepath,
                                f"    Confusion Matrix (All Test Classes - rows: true, cols: pred):\n{cm_all}",
                            )
                        else:
                            log_to_text(
                                text_log_filepath,
                                "    Confusion Matrix (All Test Classes): N/A",
                            )

                log_to_text(text_log_filepath, "-" * 40)  # モデル間の区切り

                if csv_write_ok:
                    setting_value_str = str(setting_value)[:50]
                    csv_row = [
                        model_name,
                        scenario_name,
                        setting_name,
                        setting_value_str,
                        f"{results.get('accuracy', 0.0):.4f}",
                        f"{results.get('mcc', 0.0):.4f}",
                        f"{results.get('f1av_known', 0.0):.4f}",
                        f"{results.get('macro_f1_known', 0.0):.4f}",
                        f"{results.get('weighted_f1_known', 0.0):.4f}",
                        f"{results.get('f1av_all', 0.0):.4f}",
                        f"{results.get('macro_f1_all', 0.0):.4f}",
                        f"{results.get('weighted_f1_all', 0.0):.4f}",
                        f"{results.get('unclassified_rate', 0.0):.4f}",
                        f"{results.get('unknown_rate', 0.0):.4f}",
                        f"{results.get('macro_precision_known', 0.0):.4f}",
                        f"{results.get('macro_recall_known', 0.0):.4f}",
                        f"{results.get('weighted_precision_known', 0.0):.4f}",
                        f"{results.get('weighted_recall_known', 0.0):.4f}",
                        f"{results.get('macro_precision_all', 0.0):.4f}",
                        f"{results.get('macro_recall_all', 0.0):.4f}",
                        f"{results.get('weighted_precision_all', 0.0):.4f}",
                        f"{results.get('weighted_recall_all', 0.0):.4f}",
                        error_info if error_info else "",
                    ]
                    try:
                        with open(
                            csv_filepath, "a", newline="", encoding="utf-8-sig"
                        ) as cf:
                            csv.writer(cf).writerow(csv_row)
                    except IOError as e:
                        print(f"CSV write error: {e}")
                        csv_write_ok = False
            log_to_text(text_log_filepath, "#" * 60 + "\n")
        else:  # データ生成失敗時
            # (失敗時のログ、CSV記録 - 変更なし)
            log_to_text(
                text_log_filepath,
                f"    Skipping evals for '{setting_name}': {data_gen_error}",
            )
            if csv_write_ok:
                setting_value_str = str(setting_value)[:50]
                csv_row = [
                    model_name,
                    scenario_name,
                    setting_name,
                    setting_value_str,
                    "N/A",
                    ...,
                ]  # エラー行
                try:
                    with open(
                        csv_filepath, "a", newline="", encoding="utf-8-sig"
                    ) as cf:
                        csv.writer(cf).writerow(csv_row)
                except IOError as e:
                    print(f"CSV write error: {e}")
                    csv_write_ok = False
            log_to_text(text_log_filepath, "#" * 60 + "\n")
# --- シナリオループ終了 ---
# --- シナリオループ終了 ---

# --- グラフ描画とスクリプトのコピー ---
if csv_write_ok and os.path.exists(csv_filepath):
    # プロット保存用ディレクトリを作成
    plot_dir_path = os.path.join(script_dir, f"plots_{timestamp}")
    os.makedirs(plot_dir_path, exist_ok=True)

    # ★★★ 新しいサマリーグラフ生成関数を呼び出す ★★★
    generate_summary_plots(csv_filepath, plot_dir_path)

    # --- 実行したスクリプト自体のコピー処理 ---
    try:
        if "__file__" in globals():
            script_path = os.path.abspath(__file__)
        else:
            script_path = os.path.abspath(sys.argv[0])

        script_filename = os.path.basename(script_path)
        destination_path = os.path.join(plot_dir_path, script_filename)

        with open(script_path, "rb") as fsrc, open(destination_path, "wb") as fdst:
            fdst.write(fsrc.read())
        print(f"\nScript '{script_filename}' copied to '{plot_dir_path}'")
    except Exception as e:
        print(f"\nError copying script: {e}")

# --- 終了処理 ---
log_to_text(text_log_filepath, "\n" + "=" * 60)
log_to_text(text_log_filepath, f"--- Multi-Model Validation Complete ---")
print(f"\nValidation finished.")
print(f"Log file: {text_log_filepath}")
print(f"CSV results: {csv_filepath}")
print(f"Plots directory: {plot_dir_path}")
"""
# --- グラフ描画 (修正版) ---
if csv_write_ok and os.path.exists(csv_filepath):
    print("\n--- Generating Plots ---")
    plot_dir_path = os.path.join(
        script_dir, f"plots_{timestamp}"
    )  # プロット保存ディレクトリ
    try:
        # 必要なライブラリをインポート
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns  # スタイル設定のためにインポート推奨

        if not os.path.exists(plot_dir_path):
            os.makedirs(plot_dir_path)
            print(f"Created plot directory: {plot_dir_path}")

        # --- スタイル設定 (Seabornを使うのが推奨) ---
        try:
            # スタイルを設定 (例: seaborn-v0_8-darkgrid)
            plt.style.use("seaborn-v0_8-darkgrid")
        except OSError:
            try:
                plt.style.use("seaborn-darkgrid")  # フォールバック
            except OSError:
                try:
                    plt.style.use("ggplot")  # 更なるフォールバック
                except OSError:
                    print("Warning: Could not set preferred plot style. Using default.")
        # ------------------------------------------

        df = pd.read_csv(csv_filepath)

        # --- プロット対象とする指標 (新しい列名に合わせる) ---
        metrics_to_plot = [
            "Accuracy",
            "MCC",
            "F1AV_Known",
            "Macro_F1_Known",
            "Weighted_F1_Known",
            "F1AV_All",
            "Macro_F1_All",
            "Weighted_F1_All",
            "Unclassified_Rate",
            "Unknown_Rate",  # これらのプロットも有用
        ]
        # --------------------------------------------------

        # 数値に変換できないデータをNaNにする
        for col in metrics_to_plot:
            if col in df.columns:  # 列が存在するか確認
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                print(
                    f"Warning: Column '{col}' not found in CSV for plotting.")
                # metrics_to_plot から削除するか、エラー処理をする
                if col in metrics_to_plot:
                    metrics_to_plot.remove(col)

        # NaNを含む行を削除 (プロット可能なデータのみ残す)
        df_clean = df.dropna(subset=metrics_to_plot)

        if df_clean.empty:
            print("No valid numeric data found in CSV for plotting.")
        else:
            for scenario_name in df_clean["Scenario"].unique():
                print(f"  Generating Plots for Scenario: {scenario_name}...")
                scenario_df = df_clean[df_clean["Scenario"]
                                       == scenario_name].copy()

                # Setting_Value を数値に変換 (n_classes プロット用)
                scenario_df["Setting_Value_Num"] = pd.to_numeric(
                    scenario_df["Setting_Value"], errors="coerce"
                )

                for metric in metrics_to_plot:
                    if metric not in scenario_df.columns:
                        continue  # データがない場合はスキップ

                    plt.figure(figsize=(12, 7))  # サイズ調整

                    # --- n_classes シナリオは折れ線グラフ ---
                    if scenario_name == "n_classes":
                        # Setting_Value_Num でソート
                        scenario_df_sorted = scenario_df.sort_values(
                            "Setting_Value_Num"
                        )
                        # 数値データがない場合はスキップ
                        if scenario_df_sorted["Setting_Value_Num"].isna().all():
                            print(
                                f"   Skipping line plot for {metric} in {scenario_name}: No numeric Setting_Value."
                            )
                            plt.close()
                            continue
                        # NaNを除外してピボット
                        pivot_df = scenario_df_sorted.dropna(
                            subset=["Setting_Value_Num"]
                        ).pivot(
                            index="Setting_Value_Num",
                            columns="Model_Name",
                            values=metric,
                        )
                        if pivot_df.empty:
                            print(
                                f"   No data to plot line graph for {metric} in {scenario_name}"
                            )
                            plt.close()
                            continue

                        pivot_df.plot(kind="line", marker="o", ax=plt.gca())
                        # ★★★ タイトルを修正 ★★★
                        title = f"{metric} vs. Number of Classes"
                        if "_Known" in metric:
                            title += " (Known Classes)"
                        elif "_All" in metric:
                            title += " (All Test Classes)"
                        plt.title(title)
                        # --------------------------
                        plt.xlabel("Number of Classes")
                        plt.ylabel(metric)
                        # x軸の目盛りを整数にする (可能な場合)
                        try:
                            plt.xticks(ticks=pivot_df.index.astype(int))
                        except:
                            pass  # 変換失敗は無視
                        plt.grid(True, which="both",
                                 linestyle="--", linewidth=0.5)
                        plt.legend(
                            title="Model", bbox_to_anchor=(1.05, 1), loc="upper left"
                        )
                        plt.tight_layout(rect=[0, 0, 0.85, 1])  # 凡例スペース確保

                    # --- 他のシナリオは棒グラフ ---
                    else:
                        # Setting_Name をカテゴリカルにして元の順序を維持
                        setting_order = scenario_df["Setting_Name"].unique()
                        scenario_df["Setting_Name_Cat"] = pd.Categorical(
                            scenario_df["Setting_Name"],
                            categories=setting_order,
                            ordered=True,
                        )
                        grouped = (
                            scenario_df.groupby(["Setting_Name_Cat", "Model_Name"])[
                                metric
                            ]
                            .mean()
                            .unstack()
                        )

                        if grouped.empty:
                            print(
                                f"   No data to plot bar graph for {metric} in {scenario_name}"
                            )
                            plt.close()
                            continue

                        grouped.plot(kind="bar", rot=15,
                                     width=0.8, ax=plt.gca())
                        # ★★★ タイトルを修正 ★★★
                        title = f"{metric} Comparison for Scenario: {scenario_name}"
                        if "_Known" in metric:
                            title += " (Known Classes)"
                        elif "_All" in metric:
                            title += " (All Test Classes)"
                        plt.title(title)
                        # --------------------------
                        plt.ylabel(metric)
                        plt.xlabel("Setting")
                        plt.xticks(
                            ticks=range(len(grouped.index)),
                            labels=grouped.index,
                            rotation=15,
                            ha="right",
                        )
                        plt.grid(
                            axis="y", linestyle="--", linewidth=0.5
                        )  # y軸のみグリッド
                        plt.legend(
                            title="Model", bbox_to_anchor=(1.05, 1), loc="upper left"
                        )
                        plt.tight_layout(rect=[0, 0, 0.85, 1])  # 凡例スペース確保

                    # グラフを保存
                    plot_filename = f"{scenario_name}_{metric}_comparison.png".replace(
                        " ", "_"
                    ).replace(
                        "/", "_"
                    )  # ファイル名整形
                    plot_filepath = os.path.join(plot_dir_path, plot_filename)
                    try:
                        plt.savefig(
                            plot_filepath, bbox_inches="tight"
                        )  # bbox_inches='tight' を追加
                        # print(f"   Saved plot: {plot_filepath}") # verboseに応じて出力
                    except Exception as e:
                        print(f"   Error saving plot {plot_filepath}: {e}")
                    plt.close()  # メモリ解放のため閉じる
            print(f"  Finished plots for {scenario_name}.")
    except FileNotFoundError:
        print(
            f"Error: CSV file not found at {csv_filepath}. Cannot generate plots.")
    except ImportError:
        print("Error: pandas and/or matplotlib not installed. Cannot generate plots.")
        print("Please install them: pip install pandas matplotlib seaborn")
    except Exception as e:
        print(f"An error occurred during plot generation: {e}")
        traceback.print_exc()
# --- 終了処理 ---
log_to_text(text_log_filepath, "\n" + "=" * 60)
log_to_text(text_log_filepath, f"--- Multi-Model Validation Complete ---")
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
"""

# validation_lib/config.py
import os
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeClassifier

# カスタムデータ生成関数をインポートするため (後で data_generator.py からインポート)
# from .data_generator import generate_minority_high_sep_data


from TTmethod.classTTmethod import TTClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.multiclass import OneVsRestClassifier

# --- モデルハイパーパラメータ定数 ---
MAX_DEPTH_PARAM = 100
N_ESTIMATORS_PARAM = 100

# --- 検証対象モデル定義 ---
base_estimator_rf = RandomForestClassifier(
    max_depth=MAX_DEPTH_PARAM,
    n_estimators=N_ESTIMATORS_PARAM,
    random_state=0,
    # n_jobs=-1,
    # class_weight="balanced_subsample",
)


class Config:
    """検証の設定を管理するクラス"""

    def __init__(self, base_dir=None):
        # --- 基本設定 ---
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = base_dir or os.getcwd()  # 基本ディレクトリ
        self.random_state = 42

        # --- ファイルパス設定 ---
        self.text_log_filename = f"multi_model_validation_log_{self.timestamp}.txt"
        self.csv_filename = f"multi_model_validation_results_{self.timestamp}.csv"
        self.plot_dir_name = f"validation_plots_{self.timestamp}"
        self.text_log_filepath = os.path.join(self.base_dir, self.text_log_filename)
        self.csv_filepath = os.path.join(self.base_dir, self.csv_filename)
        self.plot_dir_path = os.path.join(self.base_dir, self.plot_dir_name)

        # --- モデルハイパーパラメータ ---
        self.max_depth_param = 10
        self.n_estimators_param = 100

        # --- 基本データ生成パラメータ ---
        self.base_data_params = {
            "n_samples": 10000,
            "n_features": 20,
            "n_informative": 15,
            "n_redundant": 3,
            "n_classes": 5,  # デフォルトクラス数
            "class_sep": 1.0,
            "weights": None,
            "n_repeated": 0,
            "flip_y": 0.01,
            "n_clusters_per_class": 1,
            "random_state": self.random_state,  # Configのrandom_stateを使用
        }

        # --- 検証対象モデル定義 ---
        self.models = self._define_models()

        # --- 検証シナリオ定義 ---
        self.scenarios = self._define_scenarios()

    def _define_models(self):
        """検証対象のモデル辞書を定義"""
        return {
            "RandomForest": RandomForestClassifier(
                max_depth=self.max_depth_param,
                n_estimators=self.n_estimators_param,
                random_state=self.random_state,
            ),
            "RandomForest": RandomForestClassifier(
                max_depth=MAX_DEPTH_PARAM,
                n_estimators=N_ESTIMATORS_PARAM,
                random_state=0,
            ),
            "TTclassifier": TTClassifier(estimator=base_estimator_rf, verbose=1),
            "OneVsRest": OneVsRestClassifier(clone(base_estimator_rf)),
            # "GradientBoosting": GradientBoostingClassifier(        max_depth=MAX_DEPTH_PARAM, n_estimators=N_ESTIMATORS_PARAM, random_state=0),
            # "AdaBoost": AdaBoostClassifier(n_estimators=N_ESTIMATORS_PARAM, random_state=0),
            # "ExtraTrees": ExtraTreesClassifier(   max_depth=MAX_DEPTH_PARAM, n_estimators=N_ESTIMATORS_PARAM, random_state=0),
            # "DecisionTree": DecisionTreeClassifier(max_depth=MAX_DEPTH_PARAM, random_state=0),
        }

    def _define_scenarios(self):
        """検証シナリオのリストを定義"""
        # (lambda は削除し、単純な値で条件を保持)
        return [
            {
                "type": "single_param",
                "name": "n_classes",
                "range": range(2, 31),
                "setting_name_format": "({} classes)",
                "dynamic_feature_adjustment": True,
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
                "condition_n_classes": 5,
            },  # どのクラス数で有効か
            {
                "type": "single_param",
                "name": "n_samples",
                "options": {
                    "Low (100)": 100,
                    "Medium (1000)": 1000,
                    "High (10000)": 10000,
                },
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
                "data_generator_id": "minority_high_sep",  # <<< 関数名の代わりにIDを保持
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

    # --- アクセサメソッド (変更なし) ---
    def get_models(self):
        return self.models

    def get_scenarios(self):
        return self.scenarios

    def get_base_params(self):
        return self.base_data_params.copy()

    def get_paths(self):
        return {
            "text_log": self.text_log_filepath,
            "csv": self.csv_filepath,
            "plots": self.plot_dir_path,
        }

    def get_model_params(self):
        return {
            "max_depth": self.max_depth_param,
            "n_estimators": self.n_estimators_param,
        }

    def get_random_state(self):
        return self.random_state


# Configクラス内で datetime を使うようになったのでここでインポート
import datetime

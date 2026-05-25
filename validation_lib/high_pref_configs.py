# validation_lib/high_perf_configs.py (絞り込み版)

from sklearn.base import clone
import warnings

# 必要なクラスをインポート (場所は環境に合わせて調整)
try:
    from .advanced_cascade import AdvancedCascadeClassifier

    ADVANCED_CASCADE_AVAILABLE = True
except ImportError:
    warnings.warn("Could not import AdvancedCascadeClassifier.", ImportWarning)
    ADVANCED_CASCADE_AVAILABLE = False

    class AdvancedCascadeClassifier:
        pass  # ダミー


try:
    from feature_generator_lib.custom_transformers import SelectFirstKFeaturesAndScale

    CUSTOM_FEATURE_GEN_AVAILABLE = True
except ImportError:
    warnings.warn(
        "Could not import SelectFirstKFeaturesAndScale. Related configs might fail.",
        ImportWarning,
    )
    CUSTOM_FEATURE_GEN_AVAILABLE = False

    class SelectFirstKFeaturesAndScale:  # ダミー
        def __init__(self, k=0):
            self.k = k

        def __repr__(self):
            return f"SelectFirstKFeaturesAndScale(k={self.k}) # (Import Failed)"


# --- ユーザー指定の高精度モデル設定のみを定義 ---
HIGH_PERF_CONFIGS = {}  # 辞書を初期化


# 設定を追加するヘルパー関数 (エラーチェック付き)
def add_config(name, params):
    fg_requested = params.get("feature_generator")
    if (
        isinstance(fg_requested, SelectFirstKFeaturesAndScale)
        and not CUSTOM_FEATURE_GEN_AVAILABLE
    ):
        print(
            f"Warning: Skipping config '{name}' because SelectFirstKFeaturesAndScale is not available."
        )
    else:
        HIGH_PERF_CONFIGS[name] = params


"""
# 4. AdvCas_NoneUpd0_F85_Tol10_Val20
add_config(
    "NoneUpd0_F85_Tol10_Val20",
    {
        "unclassified_tolerance_p": 0.1,
        "max_updates": 0,
        "min_f1_threshold": 0.85,
        "val_size": 0.2,
        "feature_generator": None,
    },
)

# 4. AdvCas_NoneUpd0_F85_Tol10_Val20
add_config(
    "Upd0",
    {
        "unclassified_tolerance_p": 0.1,
        "max_updates": 0,
        "min_f1_threshold": 0.80,
        "val_size": 0.2,
        "feature_generator": None,
    },
)
"""
add_config(
    "Upd2",
    {
        "unclassified_tolerance_p": 0.1,
        "max_updates": 2,
        "min_f1_threshold": 0.85,
        "val_size": 0.2,
        "feature_generator": "scaler",
    },
)
"""
# 1. AdvCas_SclUpd0_F80_Tol10_Val20
add_config(
    "Upd99",
    {
        "unclassified_tolerance_p": 0.1,
        "max_updates": 2,
        "min_f1_threshold": 0.99,
        "val_size": 0.2,
        "feature_generator":  "scaler",
    },
)
add_config(
    "Upd99None",
    {
        "unclassified_tolerance_p": 0.1,
        "max_updates": 2,
        "min_f1_threshold": 0.99,
        "val_size": 0.2,
        "feature_generator":  None,
    },
)

    models["AdvCascade_DT_ScaleUpd99"] = AdvancedCascadeClassifier(
        estimator=clone(base_dt),
        unclassified_tolerance_p=0.01,
        max_updates=1,  # 1回アップデート試行
        min_f1_threshold=0.99,
        val_size=0.2,
        feature_generator="scaler",  # アップデート時に StandardScaler を使用
        verbose=1,
        random_state=0,
    )
#
# 2. AdvCas_NoneUpd0_F80_Tol10_Val20
add_config(
    "NoneUpd0_F80_Tol10_Val20",
    {
        "unclassified_tolerance_p": 0.1,
        "max_updates": 0,
        "min_f1_threshold": 0.80,
        "val_size": 0.2,
        "feature_generator": None,
    },
)

# 3. AdvCas_NoneUpd0_F90_Tol10_Val20
add_config(
    "NoneUpd0_F90_Tol10_Val20",
    {
        "unclassified_tolerance_p": 0.1,
        "max_updates": 0,
        "min_f1_threshold": 0.90,
        "val_size": 0.2,
        "feature_generator": None,
    },
)
"""
"""

# 5. AdvCas_Sel5Upd0_F80_Tol10_Val20
add_config(
    "Sel5Upd1_F80_Tol10_Val20",
    {
        "unclassified_tolerance_p": 0.1,
        "max_updates": 1,
        "min_f1_threshold": 0.80,
        "val_size": 0.2,
        "feature_generator": (
            SelectFirstKFeaturesAndScale(
                k=5) if CUSTOM_FEATURE_GEN_AVAILABLE else None
        ),
    },
)
"""
"""
# 6. AdvCas_Sel10Upd0_F80_Tol10_Val20
add_config(
    "Sel10Upd2_F80_Tol10_Val20",
    {
        "unclassified_tolerance_p": 0.1,
        "max_updates": 2,
        "min_f1_threshold": 0.80,
        "val_size": 0.2,
        "feature_generator": (
            SelectFirstKFeaturesAndScale(
                k=10) if CUSTOM_FEATURE_GEN_AVAILABLE else None
        ),
    },
)
"""
# -----------------------------------------------

# デフォルト設定名（リストの最初のキーを使用）
DEFAULT_CONFIG_NAME = next(iter(HIGH_PERF_CONFIGS)
                           ) if HIGH_PERF_CONFIGS else None


def get_high_performing_cascade(
    base_estimator_rf,  # ベース推定器は必須
    config_name=None,  # 設定名を指定（Noneならデフォルト）
    random_state=0,
    verbose=1,
):
    """
    事前定義された高精度パラメータ設定に基づいて
    AdvancedCascadeClassifier のインスタンスを生成して返すファクトリ関数。
    """
    if not ADVANCED_CASCADE_AVAILABLE:
        raise ImportError("AdvancedCascadeClassifier could not be imported.")

    if not HIGH_PERF_CONFIGS:
        raise ValueError("No high-performance configurations are defined.")

    if config_name is None:
        if DEFAULT_CONFIG_NAME is None:
            raise ValueError("No default configuration name set.")
        config_name = DEFAULT_CONFIG_NAME
        print(f"Using default high-performance config: {config_name}")

    if config_name not in HIGH_PERF_CONFIGS:
        # 利用可能なキーを表示してエラーを分かりやすく
        available_keys = list(HIGH_PERF_CONFIGS.keys())
        raise ValueError(
            f"Unknown configuration name: '{config_name}'. Available: {available_keys}"
        )

    params = HIGH_PERF_CONFIGS[config_name]

    # feature_generator がカスタムクラスで利用不可の場合のエラー処理
    fg_requested = params.get("feature_generator")
    if (
        isinstance(fg_requested, SelectFirstKFeaturesAndScale)
        and not CUSTOM_FEATURE_GEN_AVAILABLE
    ):
        raise ImportError(
            f"Configuration '{config_name}' requires SelectFirstKFeaturesAndScale, but it was not imported successfully."
        )

    # AdvancedCascadeClassifier のインスタンス化
    try:
        model_instance = AdvancedCascadeClassifier(
            estimator=clone(base_estimator_rf),  # 必ずクローンを使う
            unclassified_tolerance_p=params["unclassified_tolerance_p"],
            max_updates=params["max_updates"],
            min_f1_threshold=params["min_f1_threshold"],
            val_size=params["val_size"],
            feature_generator=params[
                "feature_generator"
            ],  # オブジェクト or 文字列 or None
            verbose=verbose,  # 引数の verbose を使用
            random_state=random_state,  # 引数の random_state を使用
        )
    except Exception as e:
        print(
            f"Error instantiating AdvancedCascadeClassifier for config '{config_name}': {e}"
        )
        raise  # エラーを再発生させる

    if verbose > 0:
        print(
            f"Created AdvancedCascadeClassifier instance with config: '{config_name}'"
        )
    return model_instance

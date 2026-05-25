# feature_expander.py (分類器指定可能バージョン)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone  # clone をインポート
from sklearn.ensemble import RandomForestClassifier  # デフォルト分類器としてインポート
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import export_text  # Tree系のモデルなら使える
import warnings
import traceback


class FeatureExpansionClassifier(BaseEstimator, ClassifierMixin):
    """
    特徴量エンジニアリングを行い、指定されたベース分類器で分類を行うクラス。

    Parameters
    ----------
    base_classifier : estimator object, default=RandomForestClassifier()
        特徴量拡張後に使用するベースとなる分類器インスタンス。
        fit, predict, predict_proba メソッドを持つ scikit-learn 互換の
        分類器である必要がある。パラメータは事前に設定しておくこと。
    use_diff : bool, default=True
    use_slope : bool, default=False
    use_area : bool, default=True
    verbose : int, default=0
    """

    def __init__(
        self,
        base_classifier=None,  # ベース分類器を受け取る
        use_diff=True,
        use_slope=False,
        use_area=True,
        verbose=0,
    ):
        # ★★★ ベース分類器を受け取り、デフォルトを設定 ★★★
        self.base_classifier = (
            base_classifier
            if base_classifier is not None
            else RandomForestClassifier(random_state=None)
        )  # デフォルトはRF
        # -------------------------------------------------
        self.use_diff = use_diff
        self.use_slope = use_slope
        self.use_area = use_area
        self.verbose = verbose
        # --- rf_params 関連は削除 ---
        # self.n_estimators = n_estimators # 削除 (base_classifier に含まれる)
        # self.max_depth = max_depth # 削除
        # self.random_state = random_state # 削除 (base_classifier に含まれる)
        # self.n_jobs = n_jobs # 削除
        # self.rf_params = rf_params if rf_params is not None else {} # 削除
        # --------------------------

        # 内部状態
        self.model_ = None
        self.classes_ = None
        self.feature_names_in_ = None
        self.feature_names_expanded_ = None
        self.feature_importances_ = None  # ベース分類器が対応していれば設定される
        self.feature_importances_dict_ = None
        self.is_fitted_ = False
        self._n_features_in = 0

    def _generate_features(self, X):
        # (このメソッドは変更なし - コード省略)
        if self.verbose > 1:
            print(" Starting feature generation...")
        input_is_dataframe = isinstance(X, pd.DataFrame)
        if not input_is_dataframe:
            X_df = pd.DataFrame(X, columns=[f"orig_{i}" for i in range(X.shape[1])])
        else:
            X_df = X.copy()
        n_features_orig = X_df.shape[1]
        feature_names = list(X_df.columns)
        X_expanded_df = X_df.copy()
        if self.use_diff and n_features_orig >= 2:
            if self.verbose > 1:
                print("  Calculating difference features...")
            for i in range(n_features_orig - 1):
                col1_name = X_df.columns[i]
                col2_name = X_df.columns[i + 1]
                diff_name = f"diff_{i}_{i+1}"
                try:
                    X_expanded_df[diff_name] = X_df.iloc[:, i + 1] - X_df.iloc[:, i]
                    feature_names.append(diff_name)
                except Exception as e:
                    warnings.warn(f"Could not calc diff {diff_name}: {e}")
        if self.use_slope and n_features_orig >= 2:
            if self.verbose > 1:
                print("  Calculating slope features...")
            for i in range(n_features_orig - 1):
                col1_name = X_df.columns[i]
                col2_name = X_df.columns[i + 1]
                slope_name = f"slope_{i}_{i+1}"
                try:
                    X_expanded_df[slope_name] = X_df.iloc[:, i + 1] - X_df.iloc[:, i]
                    feature_names.append(slope_name)
                except Exception as e:
                    warnings.warn(f"Could not calc slope {slope_name}: {e}")
        if self.use_area and n_features_orig >= 3:
            if self.verbose > 1:
                print("  Calculating area features...")
            for i in range(n_features_orig - 2):
                col1_name = X_df.columns[i]
                col2_name = X_df.columns[i + 1]
                col3_name = X_df.columns[i + 2]
                area_name = f"area_{i}_{i+1}_{i+2}"
                try:
                    y1 = X_df.iloc[:, i]
                    y2 = X_df.iloc[:, i + 1]
                    y3 = X_df.iloc[:, i + 2]
                    area = 0.5 * np.abs(
                        i * (y2 - y3) + (i + 1) * (y3 - y1) + (i + 2) * (y1 - y2)
                    )
                    X_expanded_df[area_name] = area
                    feature_names.append(area_name)
                except Exception as e:
                    warnings.warn(f"Could not calc area {area_name}: {e}")
        self.feature_names_expanded_ = feature_names
        if self.verbose > 1:
            print(
                f"  Finished feature generation. Total features: {len(feature_names)}"
            )
        return X_expanded_df.values

    def fit(self, X, y):
        """特徴量を拡張し、指定されたベース分類器を学習させる。"""
        X, y = check_X_y(X, y, accept_sparse=False)
        self.classes_ = unique_labels(y)
        self._n_features_in = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"orig_{i}" for i in range(X.shape[1])]

        if self.verbose > 0:
            print("Generating expanded features...")
        X_df_input = pd.DataFrame(X, columns=self.feature_names_in_)
        X_expanded = self._generate_features(X_df_input)
        if self.verbose > 0:
            print(f"Expanded features shape: {X_expanded.shape}")

        # ★★★ 指定されたベース分類器をクローンして使用 ★★★
        self.model_ = clone(self.base_classifier)
        # -------------------------------------------

        if self.verbose > 0:
            print(f"Fitting {self.model_.__class__.__name__}...")
        try:
            self.model_.fit(X_expanded, y)
        except Exception as e:
            print(f"Error during fitting: {e}\n{traceback.format_exc()}")
            raise

        # 特徴量重要度を保存 (属性があれば)
        if hasattr(self.model_, "feature_importances_"):
            self.feature_importances_ = self.model_.feature_importances_
            if (
                hasattr(self, "feature_names_expanded_")
                and self.feature_names_expanded_ is not None
                and len(self.feature_names_expanded_) == len(self.feature_importances_)
            ):
                self.feature_importances_dict_ = dict(
                    zip(self.feature_names_expanded_, self.feature_importances_)
                )
            else:
                warnings.warn("Could not create feature importance dictionary.")
                self.feature_importances_dict_ = None
        else:
            self.feature_importances_ = None
            self.feature_importances_dict_ = None

        self.is_fitted_ = True
        if self.verbose > 0:
            print("Fitting complete.")
        return self

    def predict(self, X):
        # (特徴量生成部分は変更なし - コード省略)
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        if X.shape[1] != self._n_features_in:
            raise ValueError(f"Feature count mismatch")
        if self.verbose > 1:
            print("Generating features for prediction...")
        X_df_input = pd.DataFrame(X, columns=self.feature_names_in_)
        X_expanded = self._generate_features(X_df_input)
        if self.verbose > 1:
            print("Predicting...")
        return self.model_.predict(X_expanded)

    def predict_proba(self, X):
        # (特徴量生成部分は変更なし - コード省略)
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        if X.shape[1] != self._n_features_in:
            raise ValueError(f"Feature count mismatch")
        if not hasattr(self.model_, "predict_proba"):
            raise AttributeError(
                f"{self.model_.__class__.__name__} does not support predict_proba."
            )
        if self.verbose > 1:
            print("Generating features for predict_proba...")
        X_df_input = pd.DataFrame(X, columns=self.feature_names_in_)
        X_expanded = self._generate_features(X_df_input)
        if self.verbose > 1:
            print("Predicting probabilities...")
        return self.model_.predict_proba(X_expanded)

    def get_feature_importances(self, sort=True):
        # (変更なし - コード省略)
        check_is_fitted(self)
        if self.feature_importances_dict_ is None:
            warnings.warn("Importances not available.")
            return None if not sort else []
        if sort:
            return sorted(
                self.feature_importances_dict_.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        else:
            return self.feature_importances_dict_.copy()

    def export_tree_structure(self, filepath, tree_index=0):
        """指定されたインデックスの決定木(ベースが木の場合)の構造を出力"""
        check_is_fitted(self)
        # ベース分類器がアンサンブルか、または単一の木かチェック
        tree_model = None
        if (
            hasattr(self.model_, "estimators_") and self.model_.estimators_
        ):  # RandomForest, Baggingなど
            if not (0 <= tree_index < len(self.model_.estimators_)):
                raise ValueError(f"tree_index out of range")
            tree_model = self.model_.estimators_[tree_index]
        elif hasattr(self.model_, "tree_"):  # DecisionTreeなど
            if tree_index != 0:
                warnings.warn("tree_index ignored for single tree model.")
            tree_model = self.model_
        else:
            print(
                f"Error: Base classifier {self.model_.__class__.__name__} does not seem to be tree-based."
            )
            return

        if tree_model is None:
            print("Error: Could not get tree model.")
            return

        feature_names = getattr(self, "feature_names_expanded_", None)
        if feature_names is None:
            warnings.warn("Feature names not found, exporting tree without names.")

        if self.verbose > 0:
            print(f"Exporting tree {tree_index} to {filepath}...")
        try:
            tree_rules = export_text(tree_model, feature_names=feature_names)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Tree {tree_index}\n{'-'*20}\n{tree_rules}")
            if self.verbose > 0:
                print("Export complete.")
        except Exception as e:
            print(f"Error exporting tree: {e}\n{traceback.format_exc()}")

    def _more_tags(self):
        # ベース分類器が多クラス対応かどうかに依存するが、基本的には対応と仮定
        return {"multiclass": True}

# validation_lib.py (インデント再々修正版)

# --- 必要なインポート ---
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import traceback
import warnings
from numpy import __version__ as np_version
from packaging import version

# import inspect # 不要

# NumPy 1.17 未満かどうかをチェック
NP_LT_117 = version.parse(np_version) < version.parse("1.17")
# if NP_LT_117:
#     warnings.warn(f"NumPy version {np_version} < 1.17.", UserWarning)

# --- デフォルトパラメータ ---
DEFAULT_UNCLASSIFIED_TOLERANCE = 0.03
DEFAULT_MAX_UPDATES = 2
DEFAULT_MIN_F1_THRESHOLD = 0.85
DEFAULT_VAL_SIZE = 0.3
DEFAULT_ESTIMATOR = DecisionTreeClassifier(random_state=0)


# --- 特徴量変換器取得ヘルパー ---
def _get_feature_transformer(identifier, random_state=None):
    """指定された識別子に基づいて特徴量変換器のインスタンスまたは関数を返す。"""
    if identifier is None:
        return None
    elif callable(identifier):
        if identifier is callable:
            warnings.warn(
                "Built-in function 'callable' passed. Use None or a specific generator.",
                UserWarning,
            )
            return None
        try:
            transformer = clone(identifier)
            # if verbose > 1: print(f"Using custom feature generator (cloned): {identifier}")
            return transformer
        except TypeError:
            # if verbose > 1: print(f"Using custom feature generator function: {identifier}")
            return identifier
    elif isinstance(identifier, str):
        id_lower = identifier.lower()
        if id_lower == "scaler":
            # if verbose > 1: print("Selected feature transformer: StandardScaler")
            return StandardScaler()
        elif id_lower == "poly2":
            # if verbose > 1: print("Selected feature transformer: PolynomialFeatures(degree=2)")
            return PolynomialFeatures(degree=2, include_bias=False)
        elif id_lower == "poly3int":
            # if verbose > 1: print("Selected feature transformer: PolynomialFeatures(degree=3, interaction_only=True)")
            return PolynomialFeatures(
                degree=3, include_bias=False, interaction_only=True
            )
        else:
            warnings.warn(
                f"Unknown feature generator string '{identifier}'. No transformation applied.",
                UserWarning,
            )
            return None
    elif isinstance(identifier, int):
        if identifier == 1:
            # if verbose > 1: print("Selected feature transformer (id=1): StandardScaler")
            return StandardScaler()
        elif identifier == 2:
            # if verbose > 1: print("Selected feature transformer (id=2): PolynomialFeatures(degree=2)")
            return PolynomialFeatures(degree=2, include_bias=False)
        elif identifier == 3:
            # if verbose > 1: print("Selected feature transformer (id=3): PolynomialFeatures(degree=3, interaction_only=True)")
            return PolynomialFeatures(
                degree=3, include_bias=False, interaction_only=True
            )
        else:
            warnings.warn(
                f"Unknown feature generator int '{identifier}'. No transformation applied.",
                UserWarning,
            )
            return None
    else:
        warnings.warn(
            f"Invalid feature generator type '{type(identifier)}'. No transformation applied.",
            UserWarning,
        )
        return None


# --- AdvancedCascadeClassifier クラス (インデント再々修正版) ---
class AdvancedCascadeClassifier(BaseEstimator, ClassifierMixin):
    """カスケード分類器 (Docstring省略)"""

    def __init__(
        self,
        estimator=None,
        unclassified_tolerance_p=DEFAULT_UNCLASSIFIED_TOLERANCE,
        max_updates=DEFAULT_MAX_UPDATES,
        min_f1_threshold=DEFAULT_MIN_F1_THRESHOLD,
        val_size=DEFAULT_VAL_SIZE,
        feature_generator=None,
        verbose=0,
        random_state=0,
    ):
        self.estimator = (
            estimator if estimator is not None else clone(DEFAULT_ESTIMATOR)
        )
        self.unclassified_tolerance_p = unclassified_tolerance_p
        self.max_updates = max_updates
        self.min_f1_threshold = min_f1_threshold
        self.val_size = val_size
        self.feature_generator = feature_generator
        self.verbose = verbose
        self.random_state = random_state
        # --- 内部状態変数 ---
        self.classes_ = None
        self.classifier_sets_ = []
        self.feature_transformers_ = []
        self.ovr_classifier_ = None
        self.ovr_proba_threshold_ = 0.5  # 自動決定前のデフォルト
        self.unclassified_value_ = -1
        self.is_fitted_ = False

    def fit(self, X, y):
        """モデルを学習させます。"""
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        n_classes = len(self.classes_)
        self.classifier_sets_ = []
        self.feature_transformers_ = []
        n_samples_total = X.shape[0]
        current_random_state = self.random_state

        if self.verbose > 0:
            print(
                f"Starting fitting for {n_classes} classes, {n_samples_total} samples."
            )
            print(
                f"Params: tolerance={self.unclassified_tolerance_p}, max_updates={self.max_updates}, f1_thr={self.min_f1_threshold}, val_size={self.val_size}, feature_gen={self.feature_generator}"
            )

        # 1. OVR補完分類器の学習
        if self.verbose > 0:
            print("Training fallback OneVsRestClassifier...")
        try:
            base_est_clone = clone(self.estimator)
            try:
                base_est_clone.set_params(random_state=current_random_state)
            except ValueError:
                pass
            self.ovr_classifier_ = OneVsRestClassifier(base_est_clone, n_jobs=-1)
            self.ovr_classifier_.fit(X, y)
            if self.verbose > 0:
                print("Fallback OVR trained.")
        except Exception as e:
            print(f"Error training OVR: {e}\n{traceback.format_exc()}")
            self.ovr_classifier_ = None

        # 2. 適応的なカスケード分類器セットの構築ループ
        last_X_val, last_y_val = None, None
        for update_attempt in range(self.max_updates + 1):
            loop_random_state = current_random_state + update_attempt
            if self.verbose > 0:
                print(
                    f"\n--- Cascade Set Attempt {update_attempt + 1} / {self.max_updates + 1} (r_state={loop_random_state}) ---"
                )

            # 2.1 訓練/検証データ分割
            try:
                X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                    X,
                    y,
                    test_size=self.val_size,
                    random_state=loop_random_state,
                    stratify=y,
                )
            except ValueError:
                X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                    X, y, test_size=self.val_size, random_state=loop_random_state
                )
            last_X_val, last_y_val = X_val, y_val
            if self.verbose > 1:
                print(f"Split: train={X_train_sub.shape}, val={X_val.shape}")

            # 2.2 特徴量変換/生成
            current_transformer_fitted = None
            X_train_sub_processed = X_train_sub  # 初期化

            if update_attempt > 0 and self.feature_generator is not None:
                if self.verbose > 0:
                    print(
                        f"Applying feature transformation (gen={self.feature_generator})..."
                    )
                try:
                    transformer_proto = _get_feature_transformer(
                        self.feature_generator, loop_random_state
                    )
                    if transformer_proto is not None:
                        # Case 1: scikit-learn 互換変換器
                        if hasattr(transformer_proto, "fit") and hasattr(
                            transformer_proto, "transform"
                        ):
                            current_transformer_fitted = transformer_proto.fit(
                                X_train_sub
                            )
                            X_train_sub_processed = (
                                current_transformer_fitted.transform(X_train_sub)
                            )
                            if self.verbose > 1:
                                print(
                                    f" Applied sklearn-like transformer: {current_transformer_fitted}"
                                )
                        # Case 2: 関数形式 (X -> (X_transformed, transformer))
                        elif callable(transformer_proto):
                            try:
                                # 関数呼び出し (引数はXのみと仮定)
                                result = transformer_proto(X_train_sub)
                                # 返り値チェック
                                if isinstance(result, tuple) and len(result) == 2:
                                    (
                                        X_train_sub_processed,
                                        current_transformer_fitted,
                                    ) = result
                                    if self.verbose > 1:
                                        print(
                                            f" Applied function transformer: {transformer_proto.__name__ if hasattr(transformer_proto,'__name__') else 'custom function'}"
                                        )
                                    # 返された transformer が適切か軽くチェック
                                    if current_transformer_fitted is not None and not (
                                        hasattr(current_transformer_fitted, "fit")
                                        and hasattr(
                                            current_transformer_fitted, "transform"
                                        )
                                    ):
                                        warnings.warn(
                                            f"Transformer returned by callable does not seem valid.",
                                            UserWarning,
                                        )
                                else:
                                    warnings.warn(
                                        f"Callable feature_generator did not return (X_transformed, transformer). Skipping.",
                                        UserWarning,
                                    )
                                    current_transformer_fitted = None
                            except Exception as call_e:  # 関数呼び出し自体のエラー
                                warnings.warn(
                                    f"Callable feature_generator errored: {call_e}. Skipping.",
                                    UserWarning,
                                )
                                current_transformer_fitted = None
                        else:  # _get_feature_transformer が予期せぬものを返した場合
                            warnings.warn(
                                f"Invalid object from _get_feature_transformer: {type(transformer_proto)}",
                                UserWarning,
                            )
                            current_transformer_fitted = None

                    if current_transformer_fitted is not None and self.verbose > 1:
                        print(
                            f" Features transformed. Shape: {X_train_sub_processed.shape}."
                        )

                except Exception as e:  # fit/transform でのエラーなど
                    print(
                        f"Warning: Feature transformation process failed: {e}\n{traceback.format_exc()}"
                    )
                    X_train_sub_processed = X_train_sub  # 元データに戻す
                    current_transformer_fitted = None

            # 2.3 内部分類器セットの訓練
            if self.verbose > 0:
                print(f"Training classifier set {update_attempt + 1}...")
            try:
                classifiers, class_order, class_scores = self._fit_classifier_set(
                    X_train_sub_processed, y_train_sub, loop_random_state
                )
                if not class_order:
                    print(
                        f"Warning: No classifier met F1 threshold in attempt {update_attempt + 1}."
                    )
                    if self.classifier_sets_:
                        break  # 前のセットがあればそこで終了
                    else:
                        raise RuntimeError("Failed initial set.")
                # 成功したら結果を追加
                self.classifier_sets_.append(
                    {
                        "classifiers": classifiers,
                        "class_order": class_order,
                        "class_scores": class_scores,
                    }
                )
                self.feature_transformers_.append(current_transformer_fitted)
                if self.verbose > 0:
                    print(
                        f" Classifier set {update_attempt + 1} trained. Order: {class_order}"
                    )
            except Exception as e:
                print(
                    f"Error training classifier set {update_attempt + 1}: {e}\n{traceback.format_exc()}"
                )
                if self.classifier_sets_:
                    break  # 前のセットがあればそこで終了
                else:
                    raise RuntimeError(f"Failed set {update_attempt+1}: {e}")

            # 2.4 検証データでの未分類率評価
            if X_val.shape[0] == 0:
                if self.verbose > 0:
                    print("Validation set empty, stopping.")
                    break
            if self.verbose > 0:
                print(f"Evaluating on validation set...")
            try:
                y_pred_val = self._predict_internal(X_val)
                pred_dtype_val = self._get_prediction_array_dtype()
                self.unclassified_value_ = self._set_unclassified_value(
                    pred_dtype_val
                )  # 評価前に設定
                unclassified_mask_val = self._is_unclassified(y_pred_val)
                unclassified_rate = (
                    np.mean(unclassified_mask_val)
                    if len(unclassified_mask_val) > 0
                    else 0.0
                )
                if self.verbose > 0:
                    print(
                        f" Val unclassified rate: {unclassified_rate:.4f} (target: {self.unclassified_tolerance_p})"
                    )

                if (
                    unclassified_rate <= self.unclassified_tolerance_p + 1e-6
                ):  # 誤差許容
                    if self.verbose > 0:
                        print(" Rate within tolerance. Finalizing.")
                        break
                else:  # 目標未達の場合
                    if update_attempt < self.max_updates:
                        print(" Rate exceeds tolerance. Next attempt.")
                    else:
                        print(" Rate exceeds tolerance, max updates reached.")
            except Exception as e:
                print(f"Error validation: {e}")
                break

        # --- ループ終了後 ---
        if not self.classifier_sets_:
            raise RuntimeError("No valid classifier sets trained.")
        final_pred_dtype = self._get_prediction_array_dtype()
        self.unclassified_value_ = self._set_unclassified_value(final_pred_dtype)

        # 3. OVR 確率閾値の自動決定
        determined_threshold = 0.5  # デフォルト
        if (
            self.ovr_classifier_ is not None
            and last_X_val is not None
            and last_X_val.shape[0] > 0
        ):
            if self.verbose > 0:
                print("\nDetermining OVR threshold...")
            try:
                y_pred_val_cascade = self._predict_internal(last_X_val)
                unclassified_mask_val = self._is_unclassified(y_pred_val_cascade)
                unclassified_idx_val = np.where(unclassified_mask_val)[0]
                n_uncl_val = len(unclassified_idx_val)
                n_val = len(last_X_val)
                current_uncl_rate = n_uncl_val / n_val if n_val > 0 else 0.0
                if self.verbose > 1:
                    print(
                        f" Cascade unclassified on val: {n_uncl_val}/{n_val} ({current_uncl_rate:.4f})"
                    )

                if current_uncl_rate <= self.unclassified_tolerance_p + 1e-6:
                    determined_threshold = 1.0
                    if self.verbose > 0:
                        print(
                            f" Rate ({current_uncl_rate:.4f}) <= tolerance. Setting OVR threshold to {determined_threshold:.4f}."
                        )
                elif n_uncl_val > 0:
                    X_uncl_val = last_X_val[unclassified_idx_val]
                    # NaN/Inf 処理 (入力)
                    input_needs_cleaning = False
                    if NP_LT_117:
                        if np.any(np.isnan(X_uncl_val)) or np.any(np.isinf(X_uncl_val)):
                            input_needs_cleaning = True
                    else:
                        if np.isnan(X_uncl_val).any() or np.isinf(X_uncl_val).any():
                            input_needs_cleaning = True
                    if input_needs_cleaning:
                        if self.verbose > 1:
                            print(
                                "  NaN/Inf in OVR input (validation). Applying nan_to_num."
                            )
                        if NP_LT_117:
                            X_uncl_val = np.nan_to_num(X_uncl_val)
                        else:
                            X_uncl_val = np.nan_to_num(
                                X_uncl_val,
                                nan=0.0,
                                posinf=np.finfo(X_uncl_val.dtype).max,
                                neginf=np.finfo(X_uncl_val.dtype).min,
                            )

                    ovr_probas_val = self.ovr_classifier_.predict_proba(X_uncl_val)
                    # NaN 処理 (確率)
                    proba_needs_cleaning = False
                    if NP_LT_117:
                        if np.any(np.isnan(ovr_probas_val)):
                            proba_needs_cleaning = True
                    else:
                        if np.isnan(ovr_probas_val).any():
                            proba_needs_cleaning = True
                    if proba_needs_cleaning:
                        if self.verbose > 1:
                            print(
                                "  NaN in OVR probas (validation). Applying nan_to_num (NaN -> 0)."
                            )
                        if NP_LT_117:
                            ovr_probas_val = np.nan_to_num(ovr_probas_val)
                        else:
                            ovr_probas_val = np.nan_to_num(ovr_probas_val, nan=0.0)

                    max_probas_val = np.max(ovr_probas_val, axis=1)
                    nan_mask_val = np.isnan(max_probas_val)
                    max_probas_val_safe = max_probas_val.copy()
                    max_probas_val_safe[nan_mask_val] = -np.inf

                    target_classify_frac = (
                        max(0, (current_uncl_rate - self.unclassified_tolerance_p))
                        / current_uncl_rate
                        if current_uncl_rate > 0
                        else 0
                    )
                    percentile_q = np.clip(100 * (1.0 - target_classify_frac), 0, 100)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        if len(max_probas_val_safe) > 0:
                            calculated_thr = np.percentile(
                                max_probas_val_safe, percentile_q
                            )
                        else:
                            calculated_thr = 0.5
                    determined_threshold = np.clip(calculated_thr, 0.01, 0.99)
                    if self.verbose > 0:
                        print(
                            f" Target OVR fraction: {target_classify_frac:.4f}. Percentile q: {percentile_q:.2f}. Determined OVR threshold: {determined_threshold:.4f}"
                        )
                else:
                    determined_threshold = 1.0  # 未分類なし
                self.ovr_proba_threshold_ = determined_threshold
            except Exception as e:
                print(f"Error OVR threshold: {e}")
                self.ovr_proba_threshold_ = 0.5
                print(f"Using fallback OVR threshold: {self.ovr_proba_threshold_}")

        # --- 学習完了 ---
        self.is_fitted_ = True
        if self.verbose > 0:
            print(
                f"\nFitting complete. Trained {len(self.classifier_sets_)} set(s). Final OVR threshold: {self.ovr_proba_threshold_:.4f}"
            )
        if self.min_f1_threshold >= 0.95 and self.verbose >= 0:
            print(f"Note: min_f1_threshold ({self.min_f1_threshold}) is high.")
        return self

    def _fit_classifier_set(self, X, y, loop_random_state):
        """単一のカスケード分類器セットを学習する内部メソッド。"""
        n_classes = len(self.classes_)
        classifiers = {}
        class_scores = {}
        if self.verbose > 1:
            print(f"  Fitting internal classifier set for {n_classes} classes...")

        for cls in self.classes_:
            if self.verbose > 2:
                print(f"    Training OvR for class '{cls}'...")

            y_binary = np.where(y == cls, 1, 0)
            unique_labels_binary = np.unique(y_binary)

            if len(unique_labels_binary) < 2 or X.shape[0] < 2:
                if self.verbose > 2:
                    print(
                        f"    Skip cls '{cls}': samples={X.shape[0]}, unique_bin_labels={len(unique_labels_binary)}"
                    )
                continue

            estimator = clone(self.estimator)
            try:
                estimator.set_params(random_state=loop_random_state)
            except ValueError:
                pass

            try:
                estimator.fit(X, y_binary)
            except Exception as e:
                print(f"    Fit fail cls '{cls}': {e}")
                continue

            try:
                y_pred_train = estimator.predict(X)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    score = f1_score(
                        y_binary, y_pred_train, pos_label=1, zero_division=0
                    )
                if self.verbose > 2:
                    print(f"      Train F1 score for class '{cls}': {score:.4f}")
            except Exception as e:
                print(f"    F1 calc fail cls '{cls}': {e}")
                score = 0.0

            if score > self.min_f1_threshold:
                classifiers[cls] = estimator
                class_scores[cls] = score
                if self.verbose > 2:
                    print(
                        f"      Classifier for '{cls}' accepted (F1 > {self.min_f1_threshold})."
                    )
            elif self.verbose > 2:
                print(
                    f"      Classifier for '{cls}' rejected (F1 = {score:.4f} <= {self.min_f1_threshold})."
                )

        if not class_scores:
            class_order = []
        else:
            class_order = list(
                sorted(
                    class_scores.keys(),
                    key=lambda c: class_scores.get(c, -1.0),
                    reverse=True,
                )
            )

        if self.verbose > 1:
            print(
                f"  Internal set fitting done. {len(class_order)} classifiers accepted. Order: {class_order}"
            )
        return classifiers, class_order, class_scores

    def _set_unclassified_value(self, dtype):
        """予測結果配列の未分類値を決定する"""
        if np.issubdtype(dtype, np.number):
            pv = -1
            if (
                hasattr(self, "classes_")
                and self.classes_ is not None
                and np.isin(pv, self.classes_)
            ):
                pl = self.classes_.astype(float)
                pv = np.min(pl) - 1
                while np.isin(pv, pl):
                    pv -= 1
                if np.issubdtype(dtype, np.integer):
                    return int(pv)
                else:
                    return float(pv)
            else:
                return pv
        elif np.issubdtype(dtype, np.bool_):
            return object()
        else:
            return f"__UNCLASSIFIED_{id(self)}__"

    def _get_prediction_array_dtype(self):
        """予測結果配列の適切なdtypeを取得する"""
        if not hasattr(self, "classes_") or self.classes_ is None:
            return object
        dt = self.classes_.dtype
        if np.issubdtype(dt, np.bool_):
            return object
        return dt

    def _is_unclassified(self, y_pred):
        """予測結果が未分類値かどうかを判定するマスクを返す"""
        if not hasattr(self, "unclassified_value_"):
            warnings.warn(
                "Checking unclassified status before value is set.", RuntimeWarning
            )
            temp_val = self._set_unclassified_value(
                y_pred.dtype if hasattr(y_pred, "dtype") else object
            )
            try:
                return y_pred == temp_val
            except (TypeError, ValueError):
                return np.array([x == temp_val for x in y_pred])
        try:
            return y_pred == self.unclassified_value_
        except (TypeError, ValueError):
            return np.array([x == self.unclassified_value_ for x in y_pred])

    def predict(self, X):
        """学習済みの分類器を用いて予測を行います。"""
        check_is_fitted(self)
        X = check_array(X)
        n_samples = X.shape[0]
        if n_samples == 0:
            return np.array([], dtype=self._get_prediction_array_dtype())
        pred_dtype = self._get_prediction_array_dtype()
        self.unclassified_value_ = self._set_unclassified_value(pred_dtype)
        try:
            y_pred_final = np.full(
                n_samples, self.unclassified_value_, dtype=pred_dtype
            )
        except TypeError:
            y_pred_final = np.full(n_samples, self.unclassified_value_, dtype=object)
        unclassified_indices = np.arange(n_samples)

        # --- カスケード予測 ---
        if self.verbose > 0:
            print(
                f"\nPredicting with {len(self.classifier_sets_)} sets for {n_samples} samples..."
            )
        for i, set_info in enumerate(self.classifier_sets_):
            if len(unclassified_indices) == 0:
                break
            classifiers = set_info["classifiers"]
            class_order = set_info["class_order"]
            transformer = self.feature_transformers_[i]
            if not class_order:
                continue

            X_curr = X[unclassified_indices]
            if X_curr.shape[0] == 0:
                continue

            X_proc = X_curr
            if transformer:
                try:
                    X_proc = transformer.transform(X_curr)
                except Exception as e:
                    print(f" Predict transform fail set {i+1}: {e}")
                    X_proc = X_curr

            mask_local = np.zeros(X_proc.shape[0], dtype=bool)
            for cls in class_order:
                if cls not in classifiers:
                    continue
                clf = classifiers[cls]
                idx_mask = ~mask_local
                if not np.any(idx_mask):
                    break
                X_pred_subset = X_proc[idx_mask]
                if X_pred_subset.shape[0] == 0:
                    continue
                try:
                    y_bin = clf.predict(X_pred_subset)
                except Exception as e:
                    print(f" Predict fail cls '{cls}' set {i+1}: {e}")
                    continue
                pred_as_cls_in_subset = np.where(y_bin == 1)[0]
                if len(pred_as_cls_in_subset) > 0:
                    local_indices_classified = np.where(idx_mask)[0][
                        pred_as_cls_in_subset
                    ]
                    global_indices_classified = unclassified_indices[
                        local_indices_classified
                    ]
                    y_pred_final[global_indices_classified] = cls
                    mask_local[local_indices_classified] = True
            unclassified_indices = unclassified_indices[~mask_local]

        # --- OVR補完 (tolerance=0 対応追加) ---
        n_remain = len(unclassified_indices)
        final_num_unclassified = n_remain
        if n_remain > 0 and self.ovr_classifier_ is not None:
            # tolerance=0 なら閾値無視、それ以外なら自動決定閾値を使用
            ignore_threshold = self.unclassified_tolerance_p == 0
            threshold_to_use = (
                0.0 if ignore_threshold else self.ovr_proba_threshold_
            )  # 使う閾値
            log_threshold = (
                "-inf (classify all)" if ignore_threshold else f"{threshold_to_use:.4f}"
            )

            if self.verbose > 0:
                print(
                    f"\n{n_remain} samples remain. Applying OVR fallback (threshold={log_threshold})..."
                )

            X_ovr = X[unclassified_indices]
            if X_ovr.shape[0] > 0:
                try:
                    # NaN/Inf 処理 (入力)
                    input_needs_cleaning = False
                    if NP_LT_117:
                        if np.any(np.isnan(X_ovr)) or np.any(np.isinf(X_ovr)):
                            input_needs_cleaning = True
                    else:
                        if np.isnan(X_ovr).any() or np.isinf(X_ovr).any():
                            input_needs_cleaning = True
                    if input_needs_cleaning:
                        if self.verbose > 1:
                            print("  NaN/Inf in OVR input. Applying nan_to_num.")
                        if NP_LT_117:
                            X_ovr = np.nan_to_num(X_ovr)
                        else:
                            X_ovr = np.nan_to_num(
                                X_ovr,
                                nan=0.0,
                                posinf=np.finfo(X_ovr.dtype).max,
                                neginf=np.finfo(X_ovr.dtype).min,
                            )

                    ovr_probas = self.ovr_classifier_.predict_proba(X_ovr)

                    # NaN 処理 (確率)
                    proba_needs_cleaning = False
                    if NP_LT_117:
                        if np.any(np.isnan(ovr_probas)):
                            proba_needs_cleaning = True
                    else:
                        if np.isnan(ovr_probas).any():
                            proba_needs_cleaning = True
                    if proba_needs_cleaning:
                        if self.verbose > 1:
                            print(
                                "  NaN in OVR probas. Applying nan_to_num (NaN -> 0)."
                            )
                        if NP_LT_117:
                            ovr_probas = np.nan_to_num(ovr_probas)
                        else:
                            ovr_probas = np.nan_to_num(ovr_probas, nan=0.0)

                    # tolerance=0 なら全サンプルを強制分類
                    if ignore_threshold:
                        pred_indices = np.argmax(ovr_probas, axis=1)
                        pred_classes = self.ovr_classifier_.classes_[pred_indices]
                        indices_to_update = unclassified_indices  # 全て更新
                        classes_to_assign = pred_classes
                    else:
                        # 通常通り閾値で判定
                        max_probas = np.max(ovr_probas, axis=1)
                        pred_indices = np.argmax(ovr_probas, axis=1)
                        pred_classes = self.ovr_classifier_.classes_[pred_indices]
                        # NaN対策済みの比較
                        nan_mask = np.isnan(max_probas)
                        max_probas_safe = max_probas.copy()
                        max_probas_safe[nan_mask] = -np.inf
                        # 使用する閾値で比較
                        classify_mask_ovr = max_probas_safe >= threshold_to_use
                        indices_to_update = unclassified_indices[classify_mask_ovr]
                        classes_to_assign = pred_classes[classify_mask_ovr]

                    if len(indices_to_update) > 0:
                        y_pred_final[indices_to_update] = classes_to_assign
                        if self.verbose > 1:
                            if ignore_threshold:
                                print(
                                    f"  OVR classified all {len(indices_to_update)} remaining samples (tolerance=0)."
                                )
                            else:
                                print(
                                    f"  OVR classified {len(indices_to_update)} samples (threshold={threshold_to_use:.4f})."
                                )

                    # 最終的な未分類数を再計算
                    final_unclassified_mask = self._is_unclassified(y_pred_final)
                    final_num_unclassified = np.sum(final_unclassified_mask)

                except Exception as e:
                    print(f"  Error during OVR fallback: {e}\n{traceback.format_exc()}")
                    final_num_unclassified = n_remain  # エラー時はOVR前の数
            else:  # OVR対象データが0件の場合
                final_num_unclassified = 0

        elif n_remain > 0:  # OVRがない or 不要
            if self.verbose > 0:
                print(f"\n{n_remain} samples remain (No OVR fallback or needed).")
            final_num_unclassified = n_remain
        else:  # カスケードで全て分類
            if self.verbose > 0:
                print("\nAll samples classified by cascade.")
            final_num_unclassified = 0

        # 最終的な未分類サンプルに関する警告
        if final_num_unclassified > 0:
            final_unclassified_rate = (
                final_num_unclassified / n_samples if n_samples > 0 else 0.0
            )
            if self.verbose >= 0:  # verbose=0でも警告は出す
                print(
                    f"Warning: Prediction finished with {final_num_unclassified} ({final_unclassified_rate:.2%}) unclassified samples remaining."
                )

        return y_pred_final

    def _more_tags(self):
        """scikit-learn 互換性のためのタグ"""
        return {"multiclass": True, "poor_score": True}

    def _predict_internal(self, X):
        """fit内の検証で使用するための予測メソッド (OVR補完なし)"""
        check_is_fitted(self)
        X = check_array(X)
        n_samples = X.shape[0]
        if n_samples == 0:
            return np.array([], dtype=self._get_prediction_array_dtype())

        pred_dtype = self._get_prediction_array_dtype()
        # unclassified_value を設定
        self.unclassified_value_ = self._set_unclassified_value(pred_dtype)

        try:
            y_pred_final = np.full(
                n_samples, self.unclassified_value_, dtype=pred_dtype
            )
        except TypeError:
            y_pred_final = np.full(n_samples, self.unclassified_value_, dtype=object)

        unclassified_indices = np.arange(n_samples)

        # カスケード予測実行
        for i, set_info in enumerate(self.classifier_sets_):
            if len(unclassified_indices) == 0:
                break
            classifiers = set_info["classifiers"]
            class_order = set_info["class_order"]
            transformer = self.feature_transformers_[i]
            if not class_order:
                continue

            X_curr = X[unclassified_indices]
            if X_curr.shape[0] == 0:
                continue

            X_proc = X_curr
            if transformer:
                try:
                    X_proc = transformer.transform(X_curr)
                except Exception:
                    X_proc = X_curr  # 失敗時は元データ

            mask_local = np.zeros(X_proc.shape[0], dtype=bool)
            for cls in class_order:
                if cls not in classifiers:
                    continue
                clf = classifiers[cls]
                idx_mask = ~mask_local
                if not np.any(idx_mask):
                    break
                X_pred_subset = X_proc[idx_mask]
                if X_pred_subset.shape[0] == 0:
                    continue
                try:
                    y_bin = clf.predict(X_pred_subset)
                except Exception:
                    continue  # 予測失敗
                pred_as_cls_in_subset = np.where(y_bin == 1)[0]
                if len(pred_as_cls_in_subset) > 0:
                    local_indices_classified = np.where(idx_mask)[0][
                        pred_as_cls_in_subset
                    ]
                    global_indices_classified = unclassified_indices[
                        local_indices_classified
                    ]
                    y_pred_final[global_indices_classified] = cls
                    mask_local[local_indices_classified] = True
            unclassified_indices = unclassified_indices[~mask_local]

        return y_pred_final

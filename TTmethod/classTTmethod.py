# TT法用のクラス
# from sklearn.multiclass import OneVsRestClassifier

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    f1_score,
)


# --- TTClassifier クラス (fitメソッドの順序決定ロジックを修正) ---
class TTClassifier(BaseEstimator, ClassifierMixin):
    """
    TT法 (Greedy iterative OvR Method / Tsubomatsu & Tonooka法) による多クラス分類器。
    予測時のクラス決定順序は、訓練データにおける各OvR分類器のクラス固有F1スコア
    に基づいて決定されます。

    Parameters
    ----------
    estimator : estimator object
        ベースとなる2クラス分類器。
    verbose : int, default=0
        ログ出力レベル。
    """

    def __init__(self, estimator, verbose=0):
        """分類器の初期化"""
        self.estimator = estimator
        self.verbose = verbose
        self.classifiers_ = {}
        self.classes_ = None
        self.class_order_ = None

    def fit(self, X, y):
        """
        TT法分類器を訓練データ (X, y) で学習させます。
        クラスの処理順序は、訓練データに対する各OvR分類器の
        F1スコアに基づいて決定されます。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            トレーニングデータ。
        y : array-like of shape (n_samples,)
            トレーニングデータのターゲットラベル。

        Returns
        -------
        self : object
            学習済みのTTClassifierインスタンス。
        """
        # 入力データのチェックとクラスラベルの取得
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        n_classes = len(self.classes_)
        self.classifiers_ = {}

        if self.verbose > 0:
            print(f"Fitting TTClassifier for {n_classes} classes: {self.classes_}")

        # --- ステップ1: 各クラスに対するOne-vs-Rest (OvR) 分類器を学習 ---
        for cls in self.classes_:
            if self.verbose > 1:
                print(f"  Training OvR classifier for class '{cls}'...")
            y_binary = np.where(y == cls, 1, 0)
            estimator = clone(self.estimator)
            estimator.fit(X, y_binary)
            self.classifiers_[cls] = estimator

        # --- ステップ2: 訓練データでの評価に基づき、クラス決定順序を決定 ---
        if self.verbose > 0:
            print("Determining class order using F1 score on training data...")

        # 各クラスの評価スコアを格納する辞書（デバッグ用）
        performance_scores = {}
        # まだ順序が決定していないクラスのセット (コピーを作成)
        remaining_classes = set(self.classes_)
        # 決定されたクラス順序を格納するリスト
        determined_order = []

        # カスケード順序決定ループ: 残りのクラスがなくなるまで繰り返す
        while len(remaining_classes) > 0:
            best_score = -1.0  # 現在のステップでの最高スコアを初期化
            best_class = None  # 現在のステップで最も性能が良いクラス
            current_step_scores = {}  # このステップでの各クラスのスコア（ログ用）

            # 残っているクラス候補それぞれについて性能を評価
            for current_class in remaining_classes:
                # 対応する学習済みOvR分類器を取得
                clf = self.classifiers_[current_class]
                # *** 訓練データに対する予測 *** (クラス current_class かどうか -> 1 or 0)
                y_pred_binary_train = clf.predict(X)
                # 訓練データの正解ラベルを2値化 (クラス current_class かどうか -> 1 or 0)
                y_true_binary_train = np.where(y == current_class, 1, 0)

                # 現在のクラスに対するF1スコアを計算 (クラス1 = current_class)
                score = f1_score(
                    y_true_binary_train,
                    y_pred_binary_train,
                    pos_label=1,
                    zero_division=0,
                )
                current_step_scores[current_class] = score  # スコアを記録（ログ用）

                # これまでの最高スコアより高ければ、ベストクラスとスコアを更新
                if score > best_score:
                    best_score = score
                    best_class = current_class

            # ベストクラスが見つかった場合 (通常はこちら)
            if best_class is not None:
                if self.verbose > 1:
                    print(
                        f"  Evaluated training F1 scores for remaining classes: {current_step_scores}"
                    )
                    print(
                        f"  Selected class '{best_class}' with training F1 score: {best_score:.4f}"
                    )
                # 決定したクラスを順序リストに追加
                determined_order.append(best_class)
                # 残りのクラスセットから削除
                remaining_classes.remove(best_class)
            # 例外的状況: スコアがすべて0以下などでベストクラスが見つからない場合
            elif remaining_classes:  # まだクラスが残っている場合
                if self.verbose > 0:
                    print(
                        f"Warning: Could not determine the best class among {remaining_classes} based on training F1. Adding remaining classes in default order."
                    )
                # 残っているクラスをデフォルトの順序（ソート順）で追加してループを終了
                determined_order.extend(sorted(list(remaining_classes)))
                remaining_classes.clear()  # ループを抜けるために空にする
            else:  # すべてのクラスが処理された場合
                break  # ループを正常に終了

        # 最終的に決定されたクラス順序をインスタンス変数に格納
        self.class_order_ = determined_order
        if self.verbose > 0:
            print(f"Determined class order based on training F1: {self.class_order_}")

        # 学習完了フラグ
        self.is_fitted_ = True
        return self

    # predict メソッドは変更なし (前回と同じ)
    def predict(self, X):
        """学習済みのTT法分類器を用いて予測を行います。"""
        check_is_fitted(self)
        X = check_array(X)
        n_samples = X.shape[0]
        if not self.class_order_:
            raise ValueError("Class order is not determined. Fit the model first.")
        if not self.classifiers_:
            raise ValueError("Classifiers are not trained. Fit the model first.")
        final_pred_dtype = self.classes_.dtype
        if np.issubdtype(final_pred_dtype, np.number):
            unclassified_value = -1
        else:
            unclassified_value = "UNCLASSIFIED_TEMP"
        y_pred_final = np.full(n_samples, unclassified_value, dtype=final_pred_dtype)
        unclassified_indices = np.arange(n_samples)
        if self.verbose > 0:
            print(f"Predicting with TTClassifier using order: {self.class_order_}")
        num_classes_to_process = len(self.class_order_)
        for i, current_class in enumerate(self.class_order_):
            if len(unclassified_indices) == 0:
                if self.verbose > 1:
                    print("  All samples classified, stopping early.")
                break
            clf = self.classifiers_[current_class]
            X_unclassified = X[unclassified_indices]
            if self.verbose > 1:
                print(
                    f"  Processing class '{current_class}' ({i+1}/{num_classes_to_process}) for {len(unclassified_indices)} samples."
                )
            if i == num_classes_to_process - 1:
                if self.verbose > 1:
                    print(
                        f"    Assigning remaining {len(unclassified_indices)} samples to the last class '{current_class}'."
                    )
                y_pred_final[unclassified_indices] = current_class
                unclassified_indices = np.array([], dtype=int)
                break
            if X_unclassified.shape[0] > 0:
                y_pred_binary = clf.predict(X_unclassified)
                predicted_as_current_indices_local = np.where(y_pred_binary == 1)[0]
                if len(predicted_as_current_indices_local) > 0:
                    predicted_as_current_indices_global = unclassified_indices[
                        predicted_as_current_indices_local
                    ]
                    y_pred_final[predicted_as_current_indices_global] = current_class
                    if self.verbose > 1:
                        print(
                            f"    Classified {len(predicted_as_current_indices_global)} samples as '{current_class}'."
                        )
                    mask_classified_in_this_step = np.zeros(
                        len(unclassified_indices), dtype=bool
                    )
                    mask_classified_in_this_step[predicted_as_current_indices_local] = (
                        True
                    )
                    unclassified_indices = unclassified_indices[
                        ~mask_classified_in_this_step
                    ]
                elif self.verbose > 1:
                    print(
                        f"    No samples classified as '{current_class}' in this step."
                    )
            elif self.verbose > 1:
                print("    No unclassified samples remaining for this step.")
        if np.any(y_pred_final == unclassified_value):
            print(
                f"Warning: {np.sum(y_pred_final == unclassified_value)} samples remained unclassified after the process."
            )
        return y_pred_final

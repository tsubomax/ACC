from sklearn.preprocessing import StandardScaler  # StandardScalerを追加
import warnings
from sklearn.base import BaseEstimator, TransformerMixin, clone


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

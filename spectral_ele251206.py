# # --- グローバル定数の例 (スクリプトのトップレベルに記述) ---
"""
pkill -u tsubo -f loky.backend.popen_loky_posix
python3 /share_win/tsubo/Satellite_Image/JGR/program/spectral_ele251206.py
"""
import os
import csv
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import spectral
from scipy.linalg import eigh
from sklearn.decomposition import PCA
import warnings

# --- 数値計算関連の設定 ---
# warnings.filterwarnings('ignore', category=RuntimeWarning) # 除算エラーなどを無視する場合
np.seterr(divide="ignore", invalid="ignore")  # 0除算、NaN発生時の警告を抑制

# --- 定数 ---
BACKGROUND_RGB = (0, 0, 0)  # 背景とみなすRGB値
PCA_N_COMPONENTS = 2  # PCAの主成分数
COV_REGULARIZATION = 1e-6  # 共分散行列の正則化のための微小値
CSV_FILENAME_SUFFIX = "_analysis_summary.csv"  # 出力CSVファイル名の接尾辞
BACKGROUND_CLASS_ID = 0  # 背景とみなすクラスID

# --- プロットスタイル設定 ---
FT = 22
FLa = 30
FLe = 25
Ftr = 30
(Fyoko, Ftate) = 14, 7

# --- クラスごとの色と名前の凡例 ---
CLASS_LEGEND = {
    (0, 255, 0):   {'name': 'Calcite',         'color': 'lime'},
    (255, 0, 0):   {'name': 'Alunite',         'color': 'red'},
    (255, 127, 64): {'name': 'Montmorillonite', 'color': 'lightcoral'},
    # (255, 128, 64): {'name': 'Unidentified', 'color': 'lightcoral'},
    (255, 127, 80): {'name': 'Montmorillonite', 'color': 'lightcoral'},
    (255, 128, 128): {'name': 'Montmorillonite', 'color': 'lightcoral'},
    (0, 128, 0):   {'name': 'Chlorite',        'color': 'darkgreen'},
    (0, 128, 64):   {'name': 'Chlorite',        'color': 'darkgreen'},
    (0, 255, 255): {'name': 'Opal',            'color': 'cyan'},
    (128, 255, 255): {'name': 'Opal',            'color': 'cyan'},
    (0, 0, 255):   {'name': 'Kaolinite',       'color': 'blue'},
    (255, 0, 255): {'name': 'Muscovite',       'color': 'magenta'},
    (255, 255, 0): {'name': 'Buddingtonite',   'color': 'yellow'},
    (255, 128, 255): {'name': 'Nontronite',      'color': 'violet'},
    (235, 223, 235): {'name': 'Nontronite',      'color': 'violet'},
    (0, 0, 0):     {'name': 'Unidentified',    'color': 'black'},
    (255, 255, 255):     {'name': 'Unidentified',    'color': 'black'}
}


# --- ヘルパー関数 (analyze_dataset の前に定義) ---
def calculate_bhattacharyya_distance(mean1, cov1, mean2, cov2):
    """
    2つの多変量正規分布間のバタチャリヤ距離を計算します。
    共分散行列の正則性や数値安定性を考慮します。
    """
    n_bands = mean1.shape[0]
    cov_mean = (cov1 + cov2) / 2

    try:
        cov_mean_inv = np.linalg.inv(cov_mean)
    except np.linalg.LinAlgError:
        cov_mean_reg = cov_mean + np.eye(n_bands) * COV_REGULARIZATION
        try:
            cov_mean_inv = np.linalg.inv(cov_mean_reg)
        except np.linalg.LinAlgError:
            try:
                cov_mean_inv = np.linalg.pinv(cov_mean_reg)
            except np.linalg.LinAlgError:
                return np.nan

    try:
        diff_mean = mean1 - mean2
        term1 = 0.125 * diff_mean.T @ cov_mean_inv @ diff_mean

        det_cov1 = np.linalg.det(cov1)
        if det_cov1 <= 0:
            cov1_reg = cov1 + np.eye(n_bands) * COV_REGULARIZATION
            det_cov1 = np.linalg.det(cov1_reg)
            if det_cov1 <= 0:
                raise ValueError(
                    "det(cov1) non-positive after regularization.")

        det_cov2 = np.linalg.det(cov2)
        if det_cov2 <= 0:
            cov2_reg = cov2 + np.eye(n_bands) * COV_REGULARIZATION
            det_cov2 = np.linalg.det(cov2_reg)
            if det_cov2 <= 0:
                raise ValueError(
                    "det(cov2) non-positive after regularization.")

        det_cov_mean = np.linalg.det(cov_mean)
        if det_cov_mean <= 0:
            cov_mean_reg = cov_mean + np.eye(n_bands) * COV_REGULARIZATION
            det_cov_mean = np.linalg.det(cov_mean_reg)
            if det_cov_mean <= 0:
                raise ValueError(
                    "det(cov_mean) non-positive after regularization.")

        denominator = np.sqrt(det_cov1 * det_cov2)
        if denominator <= 0:
            raise ValueError("sqrt(det(cov1)*det(cov2)) is non-positive.")

        ratio = det_cov_mean / denominator
        if ratio <= 1e-15:
            return np.nan

        term2 = 0.5 * np.log(ratio)
        b_distance = term1 + term2
        return max(0.0, b_distance)

    except (np.linalg.LinAlgError, ValueError, RuntimeWarning, FloatingPointError):
        return np.nan


def calculate_jm_distance(b_distance):
    """バタチャリヤ距離からJeffries-Matusita (JM) 距離を計算する (0-2スケール)"""
    if b_distance is None or np.isnan(b_distance) or b_distance < 0:
        return np.nan
    try:
        exp_term = np.exp(-min(b_distance, 700))
    except FloatingPointError:
        exp_term = 0.0
    return 2.0 * (1.0 - exp_term)


def calculate_td(b_distance):
    """バタチャリヤ距離からTransformed Divergence (TD) を計算する (ここではJMと同じ式を使用)"""
    if b_distance is None or np.isnan(b_distance) or b_distance < 0:
        return np.nan
    try:
        exp_term = np.exp(-min(b_distance, 700))
    except FloatingPointError:
        exp_term = 0.0
    return 2.0 * (1.0 - exp_term)


def analyze_dataset(feature_file_path, label_file_path, output_dir):
    """
    ENVI特徴量ファイルとTIFラベルファイルを用いて分析を行う。
    TIFファイルはインデックスカラー形式とRGB形式の両方に対応。
    """
    # ★★★ 修正箇所: エラー発生時も変数を使えるよう、関数冒頭でファイル名プレフィックスを定義
    base_name_prefix = f"{Path(feature_file_path).stem}_vs_{Path(label_file_path).stem}"

    print(f"\n--- Analyzing Dataset ---")
    print(f"Feature file: {feature_file_path}")
    print(f"Label file: {label_file_path}")
    print(f"Output directory: {output_dir}")

    if not label_file_path:
        print("!!! Error: Label TIF file is mandatory. Skipping. !!!")
        return

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    class_stats_list = []
    separability_list = []

    try:
        # --- 1. データ読み込み ---
        print("Loading feature data (ENVI)...")
        feature_img = spectral.open_image(feature_file_path)
        n_rows, n_cols, n_bands = feature_img.shape
        X_img = feature_img.load(dtype=np.float64)

        # --- TIFラベルの読み込みとクラス名の特定 (RGB/インデックス両対応) ---
        print("Loading label data (TIF) and identifying class names...")
        class_map = {}  # {クラスID: {'name': 物質名, 'color': 色}}
        y_img_label = np.zeros((n_rows, n_cols), dtype=np.uint8)

        with rasterio.open(label_file_path) as src:
            if (n_rows, n_cols) != (src.height, src.width):
                raise ValueError(
                    f"Shape mismatch: ENVI({(n_rows, n_cols)}) != TIF({(src.height, src.width)}).")

            try:
                # --- A. インデックスカラー画像の場合 ---
                colormap = src.colormap(1)
                print("TIF type: Indexed Color Image (Colormap found).")
                y_img_label = src.read(1)
                unique_ids = np.unique(y_img_label)

                for class_id in unique_ids:
                    if class_id == BACKGROUND_CLASS_ID:
                        continue
                    if class_id in colormap:
                        r, g, b, _ = colormap[class_id]
                        rgb_tuple = (r, g, b)
                        if rgb_tuple in CLASS_LEGEND:
                            class_map[class_id] = CLASS_LEGEND[rgb_tuple]
                        else:
                            class_map[class_id] = {
                                'name': f'Unknown(ID:{class_id})', 'color': 'grey'}
            except ValueError:
                # --- B. RGB画像の場合 ---
                print("TIF type: RGB Image (No Colormap).")
                rgb_image_np = src.read((1, 2, 3))
                rgb_image_np = np.moveaxis(rgb_image_np, 0, -1)
                pixels = rgb_image_np.reshape(-1, 3)
                unique_colors = np.unique(pixels, axis=0)

                for i, color_rgb in enumerate(unique_colors):
                    class_id = i + 1
                    rgb_tuple = tuple(color_rgb)

                    if rgb_tuple == BACKGROUND_RGB:
                        continue

                    mask = np.all(rgb_image_np == color_rgb, axis=-1)
                    y_img_label[mask] = class_id

                    if rgb_tuple in CLASS_LEGEND:
                        class_map[class_id] = CLASS_LEGEND[rgb_tuple]
                    else:
                        class_map[class_id] = {
                            'name': f'UnknownRGB{rgb_tuple}', 'color': 'grey'}

        # --- クラス数決定 ---
# --- クラス数決定と分析対象のフィルタリング ---
        # TIFファイル内で見つかった全クラスをログに表示
        all_found_ids = sorted(class_map.keys())
        print(f"Found {len(all_found_ids)} non-BG classes in TIF file.")
        """
        # 分析対象とするクラスIDをフィルタリング
        classes_to_analyze = []
        for class_id, class_info in class_map.items():
            class_name = class_info.get('name', '')
            # 'Unidentified' または 'Unknown' で始まるクラスは除外
            if class_name != 'Unidentified' and not class_name.startswith('Unknown'):
                classes_to_analyze.append(class_id)

        # これ以降の分析で使用するクラスリスト
        unique_classes = sorted(classes_to_analyze)
        n_classes = len(unique_classes)

        print(
            f"\nExcluding 'Unidentified'/'Unknown'. Analyzing {n_classes} classes:")
        if not unique_classes:
            print("  -> No classes left to analyze.")
        for cid in unique_classes:
            print(
                f"  -> ID: {cid}, Name: {class_map[cid]['name']}")

        # --- 2. データ整形 ---
        print("Preprocessing data...")
        y_label_flat = y_img_label.flatten()
        valid_mask = y_label_flat != BACKGROUND_CLASS_ID
        X_flat = X_img.reshape(-1, n_bands)

        X = X_flat[valid_mask]
        y = y_label_flat[valid_mask]

        valid_pixels_count = X.shape[0]
        if valid_pixels_count == 0:
            raise ValueError("No valid pixels found.")
        print(f"Number of valid pixels: {valid_pixels_count}")
        """
        # --- クラスのフィルタリング ---
        # 背景(ID=0)と'Unidentified'/'Unknown'クラスを除外リストに追加
        ids_to_exclude = {BACKGROUND_CLASS_ID}

        print("Filtering classes to exclude from analysis...")
        for class_id, class_info in class_map.items():
            class_name = class_info.get('name', '')
            if class_name == 'Unidentified' or class_name.startswith('Unknown'):
                ids_to_exclude.add(class_id)
                print(
                    f"  -> Excluding Class ID: {class_id} (Name: {class_name})")

        # --- 2. データ整形 ---
        print("Preprocessing data...")
        y_label_flat = y_img_label.flatten()

        # 除外リストに含まれていないピクセルだけを対象とするマスクを作成
        valid_mask = ~np.isin(y_label_flat, list(ids_to_exclude))

        X_flat = X_img.reshape(-1, n_bands)
        X = X_flat[valid_mask]
        y = y_label_flat[valid_mask]

        if X.shape[0] == 0:
            raise ValueError("No valid pixels found after filtering.")

        # 分析対象となるクラスのリストを、フィルタリング後のデータから再作成
        unique_classes = sorted(np.unique(y))
        n_classes = len(unique_classes)

        print(f"\nPerforming analysis on {n_classes} final classes:")
        if not unique_classes:
            print("  -> No classes left to analyze.")
        else:
            for cid in unique_classes:
                # class_mapに存在しないIDは万が一のため'N/A'とする
                name = class_map.get(cid, {}).get('name', 'N/A')
                print(f"  -> ID: {cid}, Name: {name}")

        print(f"Number of valid pixels for analysis: {X.shape[0]}")

        # --- 3. 統計的分析 ---
        print("\n--- Statistical Analysis ---")
        class_means = {}
        class_stds = {}
        class_data = {}

        for cls_id in unique_classes:
            cls_mask = y == cls_id
            cls_pixels = X[cls_mask]
            n_pixels = cls_pixels.shape[0]

            if n_pixels == 0:
                continue

            class_data[cls_id] = cls_pixels
            if np.any(~np.isfinite(cls_pixels)):
                print(
                    f"Warning: Non-finite values found for Class ID {cls_id}. Stats calculated using finite values.")
                cls_pixels_finite = cls_pixels[np.all(
                    np.isfinite(cls_pixels), axis=1)]
                if cls_pixels_finite.shape[0] == 0:
                    print(
                        f"  -> No finite pixels left for Class ID {cls_id}. Skipping stats.")
                    continue
                mean_spectrum = np.mean(cls_pixels_finite, axis=0)
                std_spectrum = np.std(cls_pixels_finite, axis=0)
            else:
                mean_spectrum = np.mean(cls_pixels, axis=0)
                std_spectrum = np.std(cls_pixels, axis=0)

            class_means[cls_id] = mean_spectrum
            class_stds[cls_id] = std_spectrum
            mean_of_means = np.mean(mean_spectrum)
            mean_of_stds = np.mean(std_spectrum)

            # ★★★ 修正箇所: 未定義の`classid_to_rgb`の代わりに`class_map`からクラス名を取得 ★★★
            class_info = class_map.get(cls_id, {'name': f'ID:{cls_id}'})
            class_name = class_info.get('name', f'ID:{cls_id}')
            print(
                f"Class ID {cls_id} (Name: {class_name}): N_pixels={n_pixels}, Mean(avg)={mean_of_means:.4f}, StdDev(avg)={mean_of_stds:.4f}"
            )

            # ★★★ 修正箇所: 辞書のキーを "RGB" から "Class_Name" に変更 ★★★
            stats_entry = {
                "Class_ID": cls_id,
                "Class_Name": class_name,
                "N_Pixels": n_pixels,
                "Mean_Overall": mean_of_means,
                "StdDev_Overall": mean_of_stds,
            }
            class_stats_list.append(stats_entry)

        # --- 4. 可視化 ---
        print("\n--- Visualization ---")
        plot_suffix = base_name_prefix

        # 4.1 平均スペクトルプロット
        if n_bands > 1:
            plt.figure(figsize=(Fyoko, Ftate))
            bands = np.arange(n_bands) + 1
            plot_success = False
            for cls_id in unique_classes:
                if cls_id in class_means:
                    # ★★★ 修正箇所: `classid_to_rgb` の代わりに `class_map` から名前と色を取得 ★★★
                    class_info = class_map.get(
                        cls_id, {'name': f'ID:{cls_id}', 'color': 'grey'})
                    label_name = class_info.get('name')
                    plot_color = class_info.get('color')
                    plt.plot(
                        bands, class_means[cls_id], label=label_name, color=plot_color
                    )
                    plot_success = True

            if plot_success:
                ax = plt.gca()
                plt.xlabel("Band Number", fontsize=FLa)
                plt.ylabel("Mean Value", fontsize=FLa)
                plt.legend(
                    bbox_to_anchor=(1.02, 1),
                    loc='upper left',
                    borderaxespad=0,
                    fontsize=FLe,
                    title="Classes",
                    title_fontsize=FLe + 2,
                )
                ax.grid(False)
                ax.xaxis.set_major_locator(
                    plt.MaxNLocator(nbins=5, integer=True))
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(np.floor(ymin), np.ceil(ymax))
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                plt.tight_layout()
                plot_path = output_dir_path / f"{plot_suffix}_mean_spectra.png"
                try:
                    plt.savefig(plot_path)
                    print(f"Mean spectra plot saved to: {plot_path}")
                except Exception as e:
                    print(f"Error saving mean spectra plot: {e}")
            else:
                print("No valid class means to plot.")
            plt.close()
        else:
            print("Skipping mean spectra plot (n_bands <= 1).")

        # 4.2 PCAによるクラス分布プロット
        if n_bands >= PCA_N_COMPONENTS:
            try:
                print("Performing PCA...")
                pca = PCA(n_components=PCA_N_COMPONENTS)
                if not np.all(np.isfinite(X)):
                    print(
                        "Warning: Non-finite values still present before PCA. Using finite subset.")
                    finite_mask = np.all(np.isfinite(X), axis=1)
                    if np.sum(finite_mask) < PCA_N_COMPONENTS:
                        raise ValueError("Not enough finite samples for PCA.")
                    X_pca = pca.fit_transform(X[finite_mask])
                    y_pca = y[finite_mask]
                else:
                    X_pca = pca.fit_transform(X)
                    y_pca = y

                print(
                    f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
                plt.figure(figsize=(Fyoko, Ftate))

                plot_success = False
                max_points_per_class = 1000
                unique_classes_pca_plot = np.unique(y_pca)

                for i, cls_id in enumerate(unique_classes_pca_plot):
                    cls_mask_pca = y_pca == cls_id
                    n_points_in_class = np.sum(cls_mask_pca)
                    if n_points_in_class > 0:
                        # ★★★ 修正箇所: 欠落していたサンプリングロジックを追記 ★★★
                        if n_points_in_class > max_points_per_class:
                            indices_in_class = np.where(cls_mask_pca)[0]
                            sampled_indices = np.random.choice(
                                indices_in_class, max_points_per_class, replace=False)
                        else:
                            sampled_indices = np.where(cls_mask_pca)[0]

                        if len(sampled_indices) > 0:
                            style_info = class_map.get(
                                cls_id, {'name': f'ID:{cls_id}', 'color': 'grey'})
                            label_name = style_info['name']
                            plot_color = style_info['color']
                            plt.scatter(
                                X_pca[sampled_indices, 0],
                                X_pca[sampled_indices, 1],
                                color=plot_color,
                                marker='o',
                                label=label_name,
                                alpha=0.7,
                                s=20,
                                edgecolors="none",
                            )
                            plot_success = True

                if plot_success:
                    ax = plt.gca()
                    plt.xlabel(
                        f"PC 1 ({pca.explained_variance_ratio_[0]:.2f})", fontsize=FLa)
                    plt.ylabel(
                        f"PC 2 ({pca.explained_variance_ratio_[1]:.2f})", fontsize=FLa)
                    plt.xticks(fontsize=Ftr)
                    plt.yticks(fontsize=Ftr)
                    plt.legend(
                        bbox_to_anchor=(1.02, 1),
                        loc='upper left',
                        borderaxespad=0,
                        fontsize=FLe,
                        title="Classes",
                        title_fontsize=FLe + 2,
                        markerscale=4,
                    )
                    ax.grid(False)
                    xmin, xmax = ax.get_xlim()
                    ymin, ymax = ax.get_ylim()
                    ax.set_xlim(np.floor(xmin), np.ceil(xmax))
                    ax.set_ylim(np.floor(ymin), np.ceil(ymax))
                    ax.xaxis.set_major_locator(
                        plt.MaxNLocator(nbins=5, integer=True))
                    ax.yaxis.set_major_locator(
                        plt.MaxNLocator(nbins=5, integer=True))

                    for spine in ax.spines.values():
                        spine.set_edgecolor('black')
                    plt.tight_layout()

                    plot_path = output_dir_path / \
                        f"{plot_suffix}_pca_scatter.png"
                    plt.savefig(plot_path)
                    print(f"PCA scatter plot saved to: {plot_path}")
                else:
                    print("No data points to plot for PCA.")
                plt.close()

            except Exception as e:
                print(f"Error during PCA visualization: {e}")
                traceback.print_exc()
        else:
            print(
                f"Skipping PCA plot (n_bands={n_bands} < n_components={PCA_N_COMPONENTS}).")

        # --- 5. 分離可能性指標 ---
        print("\n--- Separability Analysis (JM Distance & TD) ---")
        class_ids_list = unique_classes
        class_covariances = {}
        valid_classes_for_sep = []
        min_samples_needed = n_bands + 1 if n_bands > 0 else 2

        if n_bands <= 0:
            print("Warning: Skipping separability analysis (no bands).")
        else:
            for cls_id in unique_classes:
                if cls_id in class_data:
                    cls_pixels = class_data[cls_id]
                    if np.any(~np.isfinite(cls_pixels)):
                        cls_pixels_finite = cls_pixels[
                            np.all(np.isfinite(cls_pixels), axis=1)
                        ]
                    else:
                        cls_pixels_finite = cls_pixels
                    n_pixels = cls_pixels_finite.shape[0]

                    if n_pixels >= min_samples_needed:
                        try:
                            cov = np.cov(cls_pixels_finite,
                                         rowvar=False, bias=True)
                            cov += np.eye(n_bands) * COV_REGULARIZATION
                            if np.all(np.isfinite(cov)):
                                class_covariances[cls_id] = cov
                                valid_classes_for_sep.append(cls_id)
                            else:
                                print(
                                    f"Warning: Cov matrix for Class {cls_id} has non-finite values. Skipping.")
                        except Exception as e:
                            print(
                                f"Error calculating cov for Class {cls_id}: {e}. Skipping.")
                    else:
                        print(
                            f"Warning: Class {cls_id} insufficient finite samples ({n_pixels} < {min_samples_needed}). Skipping.")

            print(
                f"Calculating separability for {len(valid_classes_for_sep)} valid classes: {valid_classes_for_sep}")
            for i in range(len(valid_classes_for_sep)):
                for j in range(i + 1, len(valid_classes_for_sep)):
                    cls_i_id = valid_classes_for_sep[i]
                    cls_j_id = valid_classes_for_sep[j]
                    mean_i = class_means.get(cls_i_id)
                    cov_i = class_covariances.get(cls_i_id)
                    mean_j = class_means.get(cls_j_id)
                    cov_j = class_covariances.get(cls_j_id)

                    if any(v is None for v in [mean_i, cov_i, mean_j, cov_j]):
                        continue

                    b_dist, jm_dist_val, td_val_val = np.nan, np.nan, np.nan
                    try:
                        b_dist = calculate_bhattacharyya_distance(
                            mean_i, cov_i, mean_j, cov_j)
                        if np.isfinite(b_dist):
                            jm_dist_val = calculate_jm_distance(b_dist)
                            td_val_val = calculate_td(b_dist)
                    except Exception as e:
                        print(
                            f"Error calculating distance between {cls_i_id} and {cls_j_id}: {e}")

                    # ★★★ 修正箇所: `class_map` からクラス名を取得し、キーを "Class_Name" に変更 ★★★
                    class_name_i = class_map.get(
                        cls_i_id, {'name': f'ID:{cls_i_id}'}).get('name')
                    class_name_j = class_map.get(
                        cls_j_id, {'name': f'ID:{cls_j_id}'}).get('name')

                    sep_entry = {
                        "Class_ID_1": cls_i_id,
                        "Class_Name_1": class_name_i,
                        "Class_ID_2": cls_j_id,
                        "Class_Name_2": class_name_j,
                        "Bhattacharyya_Distance": f"{b_dist:.4f}" if np.isfinite(b_dist) else "Error/NaN",
                        "JM_Distance_0_2": f"{jm_dist_val:.4f}" if np.isfinite(jm_dist_val) else "Error/NaN",
                        "TD_0_2": f"{td_val_val:.4f}" if np.isfinite(td_val_val) else "Error/NaN",
                    }
                    separability_list.append(sep_entry)

            print("\nJM Distances (0-2 scale, between valid classes):")
            output_lines = 0
            for entry in separability_list:
                if entry["JM_Distance_0_2"] != "Error/NaN":
                    # ★★★ 修正箇所: ログにクラス名を表示するように変更 ★★★
                    print(
                        f"  JM({entry['Class_Name_1']}, {entry['Class_Name_2']}) = {entry['JM_Distance_0_2']}")
                    output_lines += 1
            if output_lines == 0:
                print("  No valid JM distances were calculated.")

            # JM距離ヒストグラム
            valid_jm_values = [
                float(s["JM_Distance_0_2"])
                for s in separability_list if s["JM_Distance_0_2"] != "Error/NaN"
            ]
            if valid_jm_values:
                try:
                    plt.figure(figsize=(10, 6))
                    plt.hist(valid_jm_values, bins=20,
                             range=(0, 2), color='gray')

                    ax = plt.gca()
                    plt.xlabel("JM Distance", fontsize=FLa)
                    plt.ylabel("Frequency", fontsize=FLa)
                    plt.xticks(fontsize=Ftr)
                    plt.yticks(fontsize=Ftr)
                    ax.grid(False)
                    ax.set_xticks(np.linspace(0, 2, 5))
                    ymin, ymax = ax.get_ylim()
                    ax.set_ylim(0, np.ceil(ymax))
                    ax.yaxis.set_major_locator(
                        plt.MaxNLocator(nbins=5, integer=True))

                    for spine in ax.spines.values():
                        spine.set_edgecolor('black')
                    plt.tight_layout()
                    plot_path = output_dir_path / \
                        f"{plot_suffix}_jm_histogram.png"
                    plt.savefig(plot_path)
                    print(f"JM distance histogram saved to: {plot_path}")
                    plt.close()
                except Exception as e:
                    print(f"Error saving JM histogram plot: {e}")
            else:
                print("No valid JM distances to generate histogram.")

            # 分離可能性の要約
            print("\n--- Separability Summary ---")
            if valid_jm_values:
                avg_jm, min_jm, max_jm, std_jm = np.mean(valid_jm_values), np.min(
                    valid_jm_values), np.max(valid_jm_values), np.std(valid_jm_values)
                print(f"Average JM Distance (calculated pairs): {avg_jm:.4f}")
                print(f"Min/Max JM Distance: {min_jm:.4f} / {max_jm:.4f}")
                print(f"Std Dev of JM Distance: {std_jm:.4f}")
            else:
                print("Separability analysis could not be completed.")

        # --- 6. CSVファイルへの出力 ---
        csv_filepath = output_dir_path / \
            f"{base_name_prefix}{CSV_FILENAME_SUFFIX}"
        print(f"\n--- Writing Analysis Summary to CSV ---")
        print(f"CSV file path: {csv_filepath}")
        try:
            with open(csv_filepath, "w", newline="", encoding="utf-8-sig") as csvfile:
                if class_stats_list:
                    stats_writer = csv.DictWriter(
                        csvfile, fieldnames=class_stats_list[0].keys())
                    csvfile.write("# Class Statistics\n")
                    stats_writer.writeheader()
                    stats_writer.writerows(class_stats_list)
                else:
                    csvfile.write("# Class Statistics\nNo data available.\n")

                if separability_list:
                    sep_writer = csv.DictWriter(
                        csvfile, fieldnames=separability_list[0].keys())
                    csvfile.write("\n# Class Pair Separability\n")
                    sep_writer.writeheader()
                    sep_writer.writerows(separability_list)
                else:
                    csvfile.write(
                        "\n# Class Pair Separability\nNo data available.\n")
            print("Successfully wrote summary to CSV.")
        except Exception as e:
            print(f"Error writing summary to CSV: {e}")
            traceback.print_exc()

    except Exception as e:
        print(
            f"!!! An unexpected error occurred during the analysis for {base_name_prefix} !!!")
        print(f"Error message: {e}")
        traceback.print_exc()
    finally:
        plt.close("all")
        print(f"--- Analysis Complete for {base_name_prefix} ---")


# --- メイン実行ブロック ---
if __name__ == "__main__":
    # --- 1. パス設定 ---
    base_dir_str = "/share_win/tsubo/Satellite_Image/2025July"
    target_area = "Cuprite/spectral_ele1206"
    feature_subdir = "MS"
    label_subdir = "HS_HISUI_cup"
    output_dirname = f"analysis_results_{target_area}_v9_fixed"

    # --- 2. パスの組み立てとディレクトリ作成 ---
    try:
        base_dir = Path(base_dir_str)
        feature_data_dir = base_dir / target_area / feature_subdir
        label_data_dir = base_dir / target_area / label_subdir
        output_base_dir = base_dir / target_area / output_dirname
        output_base_dir.mkdir(parents=True, exist_ok=True)

        print("--- Directory Settings ---")
        print(f"Feature data from: {feature_data_dir}")
        print(f"Label data from:   {label_data_dir}")
        print(f"Output results to: {output_base_dir}")
        print("--------------------------")

    except Exception as e:
        print(f"!!! Error setting up directories: {e}")
        exit()

    # --- 3. ファイルの検索 ---
    feature_hdr_files = list(feature_data_dir.glob("*.[hH][dD][rR]"))
    label_tif_files = list(label_data_dir.glob("*.[tT][iI][fF]"))

    print(
        f"\nFound {len(feature_hdr_files)} ENVI header files in '{feature_data_dir}':")
    for f in feature_hdr_files:
        print(f"  - {f.name}")

    print(
        f"\nFound {len(label_tif_files)} Label TIF files in '{label_data_dir}':")
    for f in label_tif_files:
        print(f"  - {f.name}")

    # --- 4. 全組み合わせでの分析実行 ---
    if not feature_hdr_files or not label_tif_files:
        print("\n!!! Error: No feature (.hdr) or label (.tif) files found.")
    else:
        total_combinations = len(feature_hdr_files) * len(label_tif_files)
        print(
            f"\nStarting analysis for all {total_combinations} feature-label combinations...")
        processed_count = 0
        error_count = 0

        for feature_hdr_path in feature_hdr_files:
            for label_tif_path in label_tif_files:
                processed_count += 1
                feature_base_name = feature_hdr_path.stem
                label_base_name = label_tif_path.stem

                print(
                    f"\n===== Combination {processed_count}/{total_combinations} =====")
                print(f"  Feature: {feature_hdr_path.name}")
                print(f"  Label:   {label_tif_path.name}")

                combination_output_dir = (
                    output_base_dir /
                    f"{feature_base_name}_vs_{label_base_name}"
                )

                try:
                    analyze_dataset(
                        str(feature_hdr_path),
                        str(label_tif_path),
                        str(combination_output_dir),
                    )
                except Exception as e:
                    print(
                        f"!!! Error processing combination: Feature='{feature_hdr_path.name}', Label='{label_tif_path.name}'")
                    print(f"!!! Error message: {e}")
                    error_count += 1

        print(f"\n===== Analysis Summary =====")
        print(f"Finished processing {processed_count} combinations.")
        if error_count > 0:
            print(
                f"Encountered errors in {error_count} combination(s). Check logs above.")
        print(f"Results are saved in subdirectories under: {output_base_dir}")

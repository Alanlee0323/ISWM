import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict
import colorsys
import os

def get_distinct_colors(n: int, saturation: float = 0.7, value: float = 0.9) -> List[str]:
    """生成n個具有高度可區分性的顏色
    
    Args:
        n: 需要的顏色數量
        saturation: HSV色彩空間中的飽和度 (0-1)
        value: HSV色彩空間中的明度 (0-1)
    
    Returns:
        List[str]: 十六進制顏色代碼列表
    """
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append('#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        ))
    return colors

# 為不同的比較組定義專門的顏色方案
COLOR_SCHEMES = {
    'base': [
        '#3277FF',
        '#403F4C',
        '#E84855',
        '#35FF0C',
        '#9813F7',
        '#FFB359',
        '#2A9D8F'
    ]
}

def set_plot_style():
    """設置基本的繪圖樣式"""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['lines.markersize'] = 4

def compare_mixed_experiments(experiment_configs: List[dict],
                            save_filename: str = 'mixed_experiments_comparison.png',
                            title_prefix: str = '',
                            color_scheme: str = 'base',
                            output_dir: str = './plots'):
    """比較來自不同實驗ID的運行記錄
    
    Args:
        experiment_configs: 實驗配置列表
        save_filename: 保存的文件名
        title_prefix: 圖表標題前綴
        color_scheme: 使用的顏色方案 ('base', 'hidden', 或 'ablation')
        output_dir: 輸出目錄
    """
    try:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['lines.markersize'] = 4
        plt.rcParams['font.size'] = 14          # 整體字體大小
        plt.rcParams['axes.titlesize'] = 16     # 子圖標題
        plt.rcParams['axes.labelsize'] = 14     # 軸標籤
        plt.rcParams['legend.fontsize'] = 12    # 圖例
        plt.rcParams['xtick.labelsize'] = 12    # x軸刻度
        plt.rcParams['ytick.labelsize'] = 12    # y軸刻度
        
        client = mlflow.tracking.MlflowClient()
        
        # 獲取顏色方案
        colors = COLOR_SCHEMES.get(color_scheme, [])
        if not colors or len(colors) < len(experiment_configs):
            # 如果顏色不夠或沒有預定義方案，則動態生成
            colors = get_distinct_colors(len(experiment_configs))
        
        # 將指標分為三組
        segmentation_metrics = {
            'val_miou': 'Mean IoU (MIoU)',
            'val_foreground_iou': 'Foreground IoU Score',
            'val_foreground_f1': 'Foreground F1 Score'
        }
        
        temporal_metrics = {
            'val_temporal_consistency': 'Temporal Consistency',
            'val_front_tracking_error': 'Front Tracking Error',
            'val_region_continuity': 'Region Continuity'
        }
        
        loss_metric = {
            'train_loss': 'training Loss'
        }
        
        score_metric = {
            'weighted_score': 'Model Scores'
        }

        # 根據需要創建三個獨立的圖表
        figure_configs = [
            {'metrics': segmentation_metrics, 'title': 'Segmentation accuracy analysis', 'filename': save_filename.replace('.png', '_segmentation.png')},
            {'metrics': temporal_metrics, 'title': 'Internal wave assessment analysis', 'filename': save_filename.replace('.png', '_temporal.png')},
            {'metrics': loss_metric, 'title': 'Loss Function Analysis', 'filename': save_filename.replace('.png', '_loss.png')},
            {'metrics': score_metric, 'title': 'Model Scores', 'filename': save_filename.replace('.png', '_scores.png')}

        ]
        
        for fig_config in figure_configs:
            metrics = fig_config['metrics']
            fig_title = title_prefix + fig_config['title']
            save_path = os.path.join(output_dir, fig_config['filename'])
            
            # 計算子圖佈局 (盡量保持方形佈局)
            n_metrics = len(metrics)
            if n_metrics <= 1:
                n_rows, n_cols = 1, 1
            elif n_metrics <= 2:
                n_rows, n_cols = 1, 2
            elif n_metrics <= 4:
                n_rows, n_cols = 1, 3
            else:
                n_rows = int(np.ceil(np.sqrt(n_metrics)))
                n_cols = int(np.ceil(n_metrics / n_rows))
            
            # 根據子圖數量調整大小
            if n_cols == 1:
                figsize = (8, 7)
            elif n_cols == 2:
                figsize = (14, 7)
            elif n_cols == 3:
                figsize = (18, 7)  # 3個子圖橫排時用更寬的尺寸
            else:
                figsize = (6*n_cols, 5*n_rows)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

            # 添加子圖間距調整
            plt.subplots_adjust(
                left=0.08,      # 左邊距
                bottom=0.15,    # 底部邊距（為軸標籤留更多空間）
                right=0.95,     # 右邊距
                top=0.85,       # 頂部邊距
                wspace=0.3,     # 子圖間的寬度間距
                hspace=0.3      # 子圖間的高度間距
            )

            # 確保axes始終是一個數組，即使只有一個子圖
            if n_metrics == 1:
                axes = np.array([axes])
            axes = axes.ravel()
            
            # 為每個配置繪製當前組的所有指標
            for idx, config in enumerate(experiment_configs):
                exp_id = config['experiment_id']
                if not exp_id:  # 跳過空的實驗ID
                    continue
                    
                run_name = config.get('run_name', 'latest')
                display_name = config.get('display_name', run_name)
                color = colors[idx % len(colors)]
                
                # 獲取運行記錄
                runs = mlflow.search_runs(experiment_ids=[exp_id])
                if len(runs) == 0:
                    print(f"未找到實驗 {exp_id} 的運行記錄")
                    continue
                
                # 選擇運行
                if run_name == 'latest':
                    run = runs.iloc[0]
                else:
                    matching_runs = runs[runs['tags.mlflow.runName'] == run_name]
                    if len(matching_runs) == 0:
                        print(f"在實驗 {exp_id} 中未找到運行 {run_name}")
                        continue
                    run = matching_runs.iloc[0]
                
                run_id = run.run_id
                
                # 繪製每個指標
                for ax_idx, (metric_name, metric_label) in enumerate(metrics.items()):
                    if ax_idx < len(axes):
                        metric_data = client.get_metric_history(run_id, metric_name)
                        if metric_data:
                            df = pd.DataFrame([(m.step, m.value) for m in metric_data], 
                                            columns=['step', 'value']).sort_values('step')
                            axes[ax_idx].plot(df['step'], df['value'], '-o', 
                                            color=color, 
                                            label=display_name, 
                                            markersize=4,
                                            alpha=0.8)  # 添加透明度以提高可讀性
                            axes[ax_idx].set_title(f'{metric_label}')
                            axes[ax_idx].set_xlabel('Steps')
                            axes[ax_idx].set_ylabel(metric_label)
                            axes[ax_idx].grid(True, alpha=0.3)  # 降低網格線的顯著性
                            axes[ax_idx].set_xlim(0, 30000)

            # 移除未使用的子圖
            for idx in range(len(metrics), len(axes)):
                fig.delaxes(axes[idx])
            
            # 添加共用圖例
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels,
                      loc='lower center',
                      bbox_to_anchor=(0.5, 0),
                      ncol=len(experiment_configs),
                      fontsize=16)
            
            # 添加整體標題
            fig.suptitle(fig_title, fontsize=20, y=0.98)
            
            plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # 為底部的圖例和頂部的標題留出空間
            plt.savefig(save_path,
                       dpi=300,
                       bbox_inches='tight',
                       pad_inches=0.3)
            plt.close()
            
            print(f"已成功保存 {fig_title} 圖表到 {save_path}")
        
        # 打印最終結果比較
        print(f"\n{title_prefix}Final Results Comparison:")
        print("=" * 50)
        
        # 合併所有指標以用於結果輸出
        all_metrics = {}
        all_metrics.update(segmentation_metrics)
        all_metrics.update(temporal_metrics)
        all_metrics.update(loss_metric)
        all_metrics.update(score_metric)
        
        for config in experiment_configs:
            exp_id = config['experiment_id']
            if not exp_id:  # 跳過空的實驗ID
                continue
                
            run_name = config.get('run_name', 'latest')
            display_name = config.get('display_name', run_name)
            
            runs = mlflow.search_runs(experiment_ids=[exp_id])
            if len(runs) > 0:
                if run_name == 'latest':
                    run = runs.iloc[0]
                else:
                    matching_runs = runs[runs['tags.mlflow.runName'] == run_name]
                    if len(matching_runs) == 0:
                        continue
                    run = matching_runs.iloc[0]
                
                run_id = run.run_id
                print(f"\n{display_name}:")
                
                for metric_name, metric_label in all_metrics.items():
                    metric_data = client.get_metric_history(run_id, metric_name)
                    if not metric_data:
                        print(f"  Best {metric_label}: N/A")
                        continue

                    # 將指標數據轉換為列表便於處理
                    values = [m.value for m in metric_data]
    
                    # 針對 Front Tracking Error 找最小值，其他指標找最大值
                    if metric_name == 'val_front_tracking_error':
                        best_value = min(values)
                    else:
                        best_value = max(values)
                    
                    if isinstance(best_value, (int, float)):
                        print(f"  Best {metric_label}: {best_value:.4f}")
                    else:
                        print(f"  Best {metric_label}: {best_value}")
    
    except Exception as e:
        print(f"比較實驗時發生錯誤: {str(e)}")

if __name__ == "__main__":
    # 設置MLflow追蹤服務器URI
    mlflow.set_tracking_uri("http://localhost:5002")
    
    try:
        # 創建輸出目錄
        output_dir = './comparison_complex_plots'
        os.makedirs(output_dir, exist_ok=True)
 
        # Model_configs = [
        #     {
        #         'experiment_id': '798',
        #         'display_name': 'Deeplabv3Plus + IWCE',
        #     },
        #     {
        #         'experiment_id': '812',
        #         'display_name': 'Unet + IWCE ',
        #     },
        #     {
        #         'experiment_id': '799',
        #         'display_name': 'PSPNet + IWCE ',
        #     }
        # ]
        # compare_mixed_experiments(Model_configs,
        #                         'Model_Compare.png',
        #                         'IWCE Loss - ',
        #                         color_scheme='base',
        #                         output_dir=output_dir)
        
        Model_JPGU_configs = [
            {
                'experiment_id': '834',
                'display_name': 'Deeplabv3Plus',
            },
            {
                'experiment_id': '812',
                'display_name': 'U-net',
            },
            {
                'experiment_id': '799',
                'display_name': 'PSPNet',
            },
            {
                'experiment_id': '832',
                'display_name': 'ISWM',
            },
        ]
        compare_mixed_experiments(Model_JPGU_configs,
                                'JPGU_Model_IWCE.png',
                                'IWCE Loss - ',
                                color_scheme='base',
                                output_dir=output_dir)
        
        Model_JPGU_CE_configs = [
            {
                'experiment_id': '817',
                'display_name': 'Deeplabv3Plus',
            },
            {
                'experiment_id': '810',
                'display_name': 'U-net',
            },
            {
                'experiment_id': '803',
                'display_name': 'PSPNet',
            },
            {
                'experiment_id': '833',
                'display_name': 'ISWM',
            },
        ]
        compare_mixed_experiments(Model_JPGU_CE_configs,
                                'JPGU_Model_CE.png',
                                'CE Loss - ',
                                color_scheme='base',
                                output_dir=output_dir)
        
        AllCEcompare_configs = [
            {
                'experiment_id': '817',
                'display_name': 'Deeplabv3Plus + CE',
            },
            {
                'experiment_id': '803',
                'display_name': 'PSPNet + CE ',
            },
            {
                'experiment_id': '810',
                'display_name': 'Unet + CE ',
            }
        ]

        compare_mixed_experiments(AllCEcompare_configs,
                                'CE_Compare.png',
                                'CE Loss - ',
                                color_scheme='base',
                                output_dir=output_dir)
        
        PSPcompare_configs = [
            {
                'experiment_id': '803',
                'display_name': 'PSPNet + CE ',
            },
            {
                'experiment_id': '799',
                'display_name': 'PSPNet + IWCE ',
            }
        ]

        compare_mixed_experiments(PSPcompare_configs,
                                'PSPNet_Compare.png',
                                'PSPNet Seg - ',
                                color_scheme='base',
                                output_dir=output_dir)
        
        Unetcompare_configs = [
            {
                'experiment_id': '810',
                'display_name': 'Unet + CE ',
            },
            {
                'experiment_id': '812',
                'display_name': 'Unet + IWCE ',
            }
        ]

        compare_mixed_experiments(Unetcompare_configs,
                                'UNet_Compare.png',
                                'UNet Seg - ',
                                color_scheme='base',
                                output_dir=output_dir)
        
        Deeplabv3compare_configs = [
            {
                'experiment_id': '798',
                'display_name': 'Deeplabv3Plus + IWCE',
            },
            {
                'experiment_id': '817',
                'display_name': 'Deeplabv3Plus + CE',
            }
        ]

        compare_mixed_experiments(Deeplabv3compare_configs,
                                'Deeplabv3+.png',
                                'Deeplabv3+ - ',
                                color_scheme='base',
                                output_dir=output_dir)
        
        Augcompare_configs = [
            {
                'experiment_id': '815',
                'display_name': 'No-Aug',
            },
            {
                'experiment_id': '827', #亮度對比度
                'display_name': 'AugA',
            },
            {
                'experiment_id': '828', #msr_clahe
                'display_name': 'AugB',
            },
            {
                'experiment_id': '830', #高斯模糊
                'display_name': 'AugC',
            },
            {
                'experiment_id': '822', #亮度對比度+msr_clahe
                'display_name': 'AugD',
            },
            {
                'experiment_id': '829', #msr_clahe+高斯模糊
                'display_name': 'AugE',
            },
            {
                'experiment_id': '823', #亮度對比度+msr_clahe+高斯模糊
                'display_name': 'AugF',
            },
            {
                'experiment_id': '826', #亮度對比度+msr_clahe+模糊+隨機擦除
                'display_name': 'AugG',
            },
            {
                'experiment_id': '825', #亮度對比度+msr_clahe+高斯模糊+旋轉+裁剪 (Crop)+CutMix
                'display_name': 'AugH',
            },
            {
                'experiment_id': '824', #亮度對比度+msr_clahe+高斯模糊+旋轉+裁剪 (Crop)
                'display_name': 'AugI',
            }
            
        ]

        compare_mixed_experiments(Augcompare_configs,
                                'Aug.png',
                                'Aug Compare - ',
                                color_scheme='base',
                                output_dir=output_dir)

        
        Augcompare_single_configs = [
            {
                'experiment_id': '815',
                'display_name': 'No-Aug',
            },
            {
                'experiment_id': '827', #亮度對比度
                'display_name': 'AugA',
            },
            {
                'experiment_id': '828', #msr_clahe
                'display_name': 'AugB',
            },
            {
                'experiment_id': '830', #高斯模糊
                'display_name': 'AugC',
            }
        ]

        compare_mixed_experiments(Augcompare_single_configs,
                                'Aug_compare_single.png',
                                'Aug Compare - ',
                                color_scheme='base',
                                output_dir=output_dir)
        
        Augcompare_msr_clahe_configs = [
            {
                'experiment_id': '815',
                'display_name': 'No-Aug',
            },
            
            {
                'experiment_id': '828',
                'display_name': 'AugB',
            },
            {
                'experiment_id': '822',
                'display_name': 'AugD',
            },
            {
                'experiment_id': '829',
                'display_name': 'AugE',
            },
            {
                'experiment_id': '823',
                'display_name': 'AugF',
            },
            {
                'experiment_id': '826',
                'display_name': 'AugG',
            }
            
        ]

        compare_mixed_experiments(Augcompare_msr_clahe_configs,
                                'Aug_compare_msr_clahe_basic.png',
                                'Aug Compare - ',
                                color_scheme='base',
                                output_dir=output_dir)
        
        Augcompare_addgeo_configs = [
            {
                'experiment_id': '815',
                'display_name': 'No-Aug',
            },
            {
                'experiment_id': '828',
                'display_name': 'AugB',
            },
            {
                'experiment_id': '823',
                'display_name': 'AugF',
            },
            {
                'experiment_id': '825', #亮度對比度+msr_clahe+高斯模糊+旋轉+裁剪 (Crop)+CutMix
                'display_name': 'AugH',
            },
            {
                'experiment_id': '824', #亮度對比度+msr_clahe+高斯模糊+旋轉+裁剪 (Crop)
                'display_name': 'AugI',
            }         
        ]

        compare_mixed_experiments(Augcompare_addgeo_configs,
                                'Aug_compare_addgeo.png',
                                'Aug Compare - ',
                                color_scheme='base',
                                output_dir=output_dir)
        
        compare_Decoder33_configs = [
            {
                'experiment_id': '835',
                'display_name': 'AugF + Decoder3*3',
            },
            {
                'experiment_id': '823',
                'display_name': 'AugF',
            }
        ]

        compare_mixed_experiments(compare_Decoder33_configs,
                                'Decoder33.png',
                                'Decoder 3*3 - ',
                                color_scheme='base',
                                output_dir=output_dir)
        
    except Exception as e:
        print(f"程序執行時發生錯誤: {str(e)}")
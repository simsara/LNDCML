为了毕业
# 环境变量
Key|Value|Desc
|:----:|:----:|:----:|
|luna_segment|/your-path-of-luna/|存放mask数据集的地方|
|luna_data|your-path-of-luna/|存放原始数据的地方，里面有subset0~9个文件夹|
|luna_label|your-path-of-luna/|存放标记的地方|
|preprocess_result_path|your-path-of-luna/|保存预处理文件的路径|
|prepare_cover_data|1|是否覆盖之前生成的预处理文件|

## 示例
luna_segment=G:\workspace\LNDCML\LUNA16\seg-lungs-LUNA16
preprocess_result_path=G:\workspace\LNDCML\LUNA16\data
luna_data=G:\workspace\LNDCML\LUNA16
luna_label=G:\workspace\LNDCML\LUNA16\CSVFILES\annotations.csv
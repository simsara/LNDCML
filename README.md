为了毕业
# 执行方式
python3 run.py --job prepare
## 可用参数
Key|Default|Desc
|:----:|:----:|:----:|
|--job|train|prepare/train/val/test/eval|
|--model|res18|模型名称 res18 / dpn3d26|
|--id|noid|模型id|
|--workers|32|读数据的worker数量|
|--epochs|100|跑的epoch总数|
|--batch-size|16|mini-batch size|
|--learning-rate|0.01|lr|
|--momentum|0.9|momentum|
|--weight-decay|1e-4|weight decay|
|--save-freq|1|保存频率|
|--resume|1|是否从断点继续，不继续会删除之前的记录|
|--start-epoch|-1|从哪个epoch继续，-1代表最新|
|--testthresh|-3|threshold for get pbb|
|--split|8|In the test phase, split the image to 8 parts|
|--gpu|all|用哪个gpu|
|--n_test|4|用几个gpu去测试|
|--nd-train|8|用多少个文件夹去训练|
|--nd-test|1|用多少个文件夹去测试|
|--multi-process|0|是否多线程执行 可用阶段 eval|

# 环境变量
## 数据预处理部分
Key|Value|Desc
|:----:|:----:|:----:|
|luna_segment|your-path-of-luna|存放mask数据集的地方|
|luna_data|your-path-of-luna|存放原始数据的地方，里面有subset0~9个文件夹|
|luna_csv|your-path-of-luna|存放csv的地方|
|preprocess_result_path|your-path-of-luna|保存预处理文件的路径|

### 示例
luna_segment=G:\workspace\LNDCML\LUNA16\seg-lungs-LUNA16
preprocess_result_path=G:\workspace\LNDCML\LUNA16\data
luna_data=G:\workspace\LNDCML\LUNA16
luna_label=G:\workspace\LNDCML\LUNA16\CSVFILES\annotations.csv
prepare_cover_data=1

## 检测部分
Key|Value|Desc
|:----:|:----:|:----:|
|net_save_dir|your-path-of-save-net|网络模型保存路径|

### 示例
net_save_dir=G:\workspace\LNDCML\detect\netsave


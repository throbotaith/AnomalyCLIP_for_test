# Anomaly CLIP 検証用リポジトリ

# Dockerを用いたAnomalyCLIPの推論

以下の手順でDocker環境上で検証を行います。

1. ワークスペースのディレクトリでリポジトリを取得
```bash

git clone https://github.com/throbotaith/AnomalyCLIP_for_test.git

cd AnomalyCLIP_for_test/


```

2. Dockerイメージを取得

```bash

docker build -t anomalyclip .

```

GPUが無い環境で実行する場合は `Dockerfile.cpu` を使用して
```bash
docker build -f Dockerfile.cpu -t anomalyclip-cpu .
```


3. 現在のディレクトリをワークスペースとしてコンテナを起動
4. 
**Ubuntuの場合**
```bash

docker run --gpus all -it -v $(pwd):/workspace/AnomalyCLIP anomalyclip

```
**Windowsの場合**
```bash

docker run --gpus all -it -v "${PWD}:/workspace/AnomalyCLIP" anomalyclip bash

```
4. AnomalyCLIPディレクトリ内で画像フォルダを作成し、画像をそこに配置


5. 学習済み推論モデルを検証するにはコンテナ内で `test_folder.py` を実行

```bash

python test_folder.py --folder_path /path/to/images \
--checkpoint_path checkpoints/9_12_4_multiscale/epoch_15.pth \
--features_list 6 12 18 24 --image_size 518 --depth 9 \
--n_ctx 12 --t_n_ctx 4

```
**画像フォルダのパスは適宜変更してください**

```/workspace/AnomalyCLIP/imgs```としていた場合、コマンドライン引数は

```bash

--folder_path imgs/

```
となります。

推論結果は `anomaly_detection_results/<タイムスタンプ>_<モデル名>/` に保存されます。また、`inference_settings.txt` に実行設定が出力されます。

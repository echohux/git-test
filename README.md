# git-test
第一步：制作数据集
1. 生成文本描述
python create_ascii_captions.py \
  --dataset /data/way/segment_generate/datasets/level_windows_with_pacing.json \
  --tileset /data/way/segment_generate/datasets/smb.json \
  --output /data/way/segment_generate/datasets/level_windows_with_captions.json
2. 生成tokennizer
python tokenizer.py save \
  --json_file /data/way/segment_generate/datasets/level_windows_with_captions.json \
  --pkl_file /data/way/segment_generate/datasets/level_windows_tokenizer.pkl
3. 随机测试caption
python create_random_test_captions.py \
  --dataset /data/way/segment_generate/datasets/level_windows_with_captions.json \
  --output /data/way/segment_generate/datasets/random_test_captions.json
4. 划分训练集测试集验证集
python split_data.py \
  --input /data/way/segment_generate/datasets/level_windows_with_captions.json \
  --train /data/way/segment_generate/datasets/level_windows_train.json \
  --val /data/way/segment_generate/datasets/level_windows_val.json \
  --test /data/way/segment_generate/datasets/level_windows_test.json
   ![Uploading 037faa2e-8a66-4155-adfc-0293b257d5dc.png…]()

第二步：训练文本编码器
1. 训练文本编码器:
  python train_mlm.py   --epochs 300   --save_checkpoints   --json /data/way/segment_generate/datasets/level_windows_with_captions.json   --val_json /data/way/segment_generate/datasets/level_windows_with_captions-validate.json   --test_json /data/way/segment_generate/datasets/level_windows_with_captions-test.json   --pkl /data/way/segment_generate/datasets/level_windows_tokenizer.pkl   --output_dir /data/way/segment_generate/Mar1and2-MLM-regular0   --seed 0
2. 测试文本编码器训练效果：
   python evaluate_masked_token_prediction.py --model_path Mar1and2-MLM-regular0 --json datasets\level_windows_with_captions-train.json
   
第三步：训练文本扩散模型
python train_diffusion.py --save_image_epochs 20 --augment --text_conditional --output_dir Mar1and2-conditional-regular0 --num_epochs 500 --json datasets\level_windows_with_captions-train.json --val_json datasets\level_windows_with_captions-validate.json --pkl datasets\level_windows_tokenizer.pkl --mlm_model_dir Mar1and2-MLM-regular0 --plot_validation_caption_score --seed 0

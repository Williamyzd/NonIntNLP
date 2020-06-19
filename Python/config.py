# -*- ecoding: utf-8 -*-

"""
xlnet_server : config
@copyright: William Yang <yang.zedong@pku.edu.cn>
@Time: 2020/6/19 2:57 下午
"""
#数据根地址
data_dir= '/Users/williamyang/Documents/server/datas/xl_net'
#data_dir= '/home/ubuntu/MyFiles/datas/xl_net'
#原始输入数据
data_file='/home/ubuntu/MyFiles/datas/input.json'
#预训练模型
# pre_model_name='hfl/chinese-xlnet-base'
wandb_token='456b5ffce999b13f19a43e456293e9e0bc6828d0'
project_name='xlnet_sum'

#训练参数
seq_sz=256
epochs = 10
max_predict_sz=50
batch_sz=3
aggregate_batch_sz=3
start_lr=5e-6
device='cpu'
number_to_generate=30
# is_fp16= 0

#储存切割数据
input_data_dir = data_dir+"/input_datas"
test_data_path=input_data_dir+"/test.pt"
pre_model_name = data_dir+"/chinese_xlnet_base_pytorch/"

#存储模型数据
out_put_dir=data_dir + "/output_datas"
final_model_dir=out_put_dir+ '/xlnet_trainer_checkpoints/final'

#prepare
#pip uninstall apex
# git clone https://www.github.com/nvidia/apex
# cd apex
# python3 setup.py install
# pip install torch torchvision wandb
# git clone https://github.com/neonbjb/transformers.git
# git clone https://github.com/Williamyzd/NonIntNLP.git
# cd transformers
# pip install .
# cd ../NonIntNLP
# git reset --hard remotes/origin/xlnet_xsum_train
#cd 2020-NoIntNLP-william/Python
#cd xlnet-sum/Python
#wandb登录
# # echo '----------初始化wandb----------'
# # wandb login $wandb_token
# # #数据切割
# # echo '----------数据处理----------'
# # #python processors/process_xsum.py --input_file $data_file  --model_name_or_path $pre_model_name  --data_dir $input_data_dir
# # #数据训练
# # echo '----------开始训练----------'
# # python train_xlnet_lang_modeler_pytorch.py --input_folder $input_data_path --epochs 1 --output_dir $out_put_dir  --seq_sz $seq_sz --max_predict_sz $max_predict_sz --batch_sz $batch_sz --aggregate_batch_sz $aggregate_batch_sz  --start_lr $start_lr --device $device --project_name $project_name --model_name $pre_model_name
# # #生成摘要
# # if [ $? -eq 0 ]; then
# # 	echo '----------tain succeed----------'
# # 	echo '----------开始生成----------'
# # 	python generate_with_transformers_chunked.py --model_dir=$final_model_dir --data_file=$test_data_path --number_to_generate=$number_to_generate --device=$device
# # else
# # 	echo '----------train failed----------'
# fi


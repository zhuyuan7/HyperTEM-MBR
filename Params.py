import argparse


def parse_args():
	parser = argparse.ArgumentParser(description='Model Params')


	parser.add_argument('--cuda_num', default=3, type=str, help='cuda_num')  
	parser.add_argument('--wandb', default="Tmall", type=str, help='wandb run name')  #retail_rocket
	# parser.add_argument('--wandb', default="IJCAI_15", type=str, help='wandb run name')  #retail_rocket
# 	#for this model
	parser.add_argument('--hidden_dim', default=16, type=int, help='embedding size')  
	parser.add_argument('--gnn_layer', default="[16,16,16]", type=str, help='gnn layers: number + dim')  
	parser.add_argument('--dataset', default='Tmall', type=str, help='name of dataset')   # Tmall  /home/joo/CML/data/retail_rocket
	# parser.add_argument('--dataset', default='IJCAI_15', type=str, help='name of dataset') 
	parser.add_argument('--point', default='for_meta_hidden_dim', type=str, help='')
	parser.add_argument('--title', default='dim__8', type=str, help='title of model')  
	parser.add_argument('--sampNum', default=10, type=int, help='batch size for sampling') 
	parser.add_argument('--hyperNum', default=128, type=int, help='number of hyperedges')
	parser.add_argument('--gcn_hops', default=2, type=int, help='number of hops in gcn precessing')
	# parser.add_argument('--gnn_layer', default=2, type=int, help='number of hops in gcn precessing')
	parser.add_argument('--mult', default=1, type=int, help='')
	parser.add_argument('--num_layers', default=3, type=int, help='')
	
# 	#for train
	parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
	parser.add_argument('--opt_base_lr', default=0.002, type=float, help='learning rate')# 0.004
	parser.add_argument('--opt_max_lr', default=0.05, type=float, help='learning rate')#0.005  =5e-3
	parser.add_argument('--opt_weight_decay', default=0.0001, type=float, help='weight decay regularizer')
	parser.add_argument('--meta_opt_base_lr', default=0.0001, type=float, help='learning rate') # 0.0004  1e-4
	parser.add_argument('--meta_opt_max_lr', default=0.005, type=float, help='learning rate') # 0.005
	parser.add_argument('--meta_opt_weight_decay', default=0.0001, type=float, help='weight decay regularizer')
	parser.add_argument('--meta_lr', default=0.003, type=float, help='_meta_learning rate')   # 1e-3
	parser.add_argument('--hyper_lr', default=0.005, type=float, help='hypergraph learning rate') #1e-3
	parser.add_argument('--batch', default=8192, type=int, help='batch size')   #8192        
	parser.add_argument('--meta_batch', default=32, type=int, help='batch size') # 32  
	parser.add_argument('--SSL_batch', default=60, type=int, help='batch size')   #30
	parser.add_argument('--reg', default=0.001, type=float, help='weight decay regularizer') #1e-3
	parser.add_argument('--beta', default=0.005, type=float, help='scale of infoNCELoss')
	parser.add_argument('--epoch', default=400, type=int, help='number of epochs')  #300
	# parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')  
	parser.add_argument('--shoot', default=10, type=int, help='K of top k')
	parser.add_argument('--inner_product_mult', default=1, type=float, help='multiplier for the result')  
	parser.add_argument('--inner_product_mult_last', default=3, type=float, help='multiplier for the result') 
	parser.add_argument('--keepRate', default=0.5, type=float, help='ratio of edges to keep')   
	parser.add_argument('--drop_rate', default=0.8, type=float, help='drop_rate')  
	parser.add_argument('--drop_rate1', default=0.8, type=float, help='drop_rate')  
	parser.add_argument('--seed', type=int, default=6)  
	parser.add_argument('--slope', type=float, default=0.1)  
	parser.add_argument('--patience', type=int, default=50)
	#for save and read
	parser.add_argument('--path', default='/home/joo/JOOCML/data/', type=str, help='data path')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
	parser.add_argument('--isload', default=False , type=bool, help='whether load model')  
	parser.add_argument('--isJustTest', default=False , type=bool, help='whether load model')
	parser.add_argument('--loadModelPath', default='"/home/ww/Code/work3/BSTRec/Model/Tmall/for_meta_hidden_dim_dim__8_Tmall_2021_07_08__01_35_54_lr_0.0003_reg_0.001_batch_size_4096_gnn_layer_[16,16,16].pth', type=str, help='loadModelPath')
	parser.add_argument('--isJustbeh', default=True , type=bool, help='GRU')
	parser.add_argument('--is_Meta_Path', default=True , type=bool, help='not GRU')
	parser.add_argument('--History_path', default='/home/joo/new_HY_CML/History/', type=str, help='data path')
	parser.add_argument('--Model_path', default='/home/joo/new_HY_CML/Model/', type=str, help='data path')
	# #Tmall: # loadPath_SSL_meta = "/home/ww/Code/work3/BSTRec/Model/Tmall/for_meta_hidden_dim_dim__8_Tmall_2021_07_08__01_35_54_lr_0.0003_reg_0.001_batch_size_4096_gnn_layer_[16,16,16].pth"
	# #IJCAI_15: # loadPath_SSL_meta = "/home/ww/Code/work3/BSTRec/Model/IJCAI_15/for_meta_hidden_dim_dim__8_IJCAI_15_2021_07_10__14_11_55_lr_0.0003_reg_0.001_batch_size_4096_gnn_layer_[16,16,16].pth"
	# #retailrocket: # loadPath_SSL_meta = "/home/ww/Code/work3/BSTRec/Model/retailrocket/for_meta_hidden_dim_dim__8_retailrocket_2021_07_10__18_35_32_lr_0.0003_reg_0.01_batch_size_1024_gnn_layer_[16,16,16].pth"


# 	#use less
	# parser.add_argument('--memosize', default=2, type=int, help='memory size') 
	parser.add_argument('--head_num', default=4, type=int, help='head_num_of_multihead_attention')  
	parser.add_argument('--beta_multi_behavior', default=0.005, type=float, help='scale of infoNCELoss') 
	parser.add_argument('--sampNum_slot', default=30, type=int, help='SSL_step')
	parser.add_argument('--SSL_slot', default=1, type=int, help='SSL_step')
	parser.add_argument('--k', default=2, type=float, help='MFB')
	parser.add_argument('--meta_time_rate', default=0.8, type=float, help='gating rate')
	parser.add_argument('--meta_behavior_rate', default=0.8, type=float, help='gating rate')  
	parser.add_argument('--meta_slot', default=2, type=int, help='epoch number for each SSL')
	parser.add_argument('--time_slot', default=60*60*24*360, type=float, help='length of time slots')  
	parser.add_argument('--hidden_dim_meta', default=16, type=int, help='embedding size')
	parser.add_argument('--predir', default="/home/joo/JOOCML/data/Tmall/", type=str, help='dataset')
	# parser.add_argument('--predir', default="/home/joo/JOOCML/data/IJCAI_15/", type=str, help='dataset')


	# parser.add_argument('--att_head', default=2, type=int, help='number of attention heads')  
	# parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
	# parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')  
	# parser.add_argument('--deep_layer', default=0, type=int, help='number of deep layers to make the final prediction')  
	# parser.add_argument('--iiweight', default=0.3, type=float, help='weight for ii')  
	# parser.add_argument('--graphSampleN', default=10000, type=int, help='use 25000 for training and 200000 for testing, empirically')  
	# parser.add_argument('--divSize', default=1000, type=int, help='div size for smallTestEpoch')
	# parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
	# parser.add_argument('--subUsrSize', default=10, type=int, help='number of item for each sub-user')
	# parser.add_argument('--subUsrDcy', default=0.9, type=float, help='decay factor for sub-users over time')  
	# parser.add_argument('--slot', default=0.5, type=float, help='length of time slots')  

	
	return parser.parse_args()
args = parse_args()


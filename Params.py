import argparse


def parse_args():
	parser = argparse.ArgumentParser(description='Model Params')


	parser.add_argument('--cuda_num', default=1, type=str, help='cuda_num')  
	parser.add_argument('--wandb', default="Tmall", type=str, help='wandb run name')  
# 	#for this model
	parser.add_argument('--hidden_dim', default=16, type=int, help='embedding size')  
	parser.add_argument('--gnn_layer', default="[16,16,16]", type=str, help='gnn layers: number + dim')  
	parser.add_argument('--dataset', default='Tmall', type=str, help='name of dataset')  
	parser.add_argument('--point', default='for_meta_hidden_dim', type=str, help='')
	parser.add_argument('--title', default='dim__8', type=str, help='title of model')  
	parser.add_argument('--sampNum', default=10, type=int, help='batch size for sampling') 
	parser.add_argument('--hyperNum', default=128, type=int, help='number of hyperedges')
	parser.add_argument('--gcn_hops', default=2, type=int, help='number of hops in gcn precessing')
	parser.add_argument('--mult', default=1, type=int, help='')
	parser.add_argument('--num_layers', default=3, type=int, help='')
	
# 	#for train
	parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
	parser.add_argument('--opt_base_lr', default=0.002, type=float, help='learning rate')
	parser.add_argument('--opt_max_lr', default=0.05, type=float, help='learning rate')
	parser.add_argument('--opt_weight_decay', default=0.0001, type=float, help='weight decay regularizer')
	parser.add_argument('--meta_opt_base_lr', default=0.0001, type=float, help='learning rate') 
	parser.add_argument('--meta_opt_max_lr', default=0.001, type=float, help='learning rate') 
	parser.add_argument('--meta_opt_weight_decay', default=0.001, type=float, help='weight decay regularizer')
	parser.add_argument('--meta_lr', default=1e-3, type=float, help='_meta_learning rate')   
	parser.add_argument('--hyper_lr', default=0.003, type=float, help='hypergraph learning rate') 
	parser.add_argument('--batch', default=8192, type=int, help='batch size')           
	parser.add_argument('--meta_batch', default=32, type=int, help='batch size')   
	parser.add_argument('--SSL_batch', default=30, type=int, help='batch size')  
	parser.add_argument('--reg', default=0.003, type=float, help='weight decay regularizer') 
	parser.add_argument('--beta', default=0.005, type=float, help='scale of infoNCELoss')
	parser.add_argument('--epoch', default=10, type=int, help='number of epochsb')  
	parser.add_argument('--leaky', default=0.5, type=float, help='slope of leaky relu')
	parser.add_argument('--shoot', default=10, type=int, help='K of top k')
	parser.add_argument('--inner_product_mult', default=1, type=float, help='multiplier for the result')  
	parser.add_argument('--inner_product_mult_last', default=3, type=float, help='multiplier for the result') 
	parser.add_argument('--keepRate', default=0.5, type=float, help='ratio of edges to keep')   
	parser.add_argument('--drop_rate', default=0.8, type=float, help='drop_rate')  
	parser.add_argument('--drop_rate1', default=0.8, type=float, help='drop_rate')  
	parser.add_argument('--seed', type=int, default=6)  
	parser.add_argument('--slope', type=float, default=0.1)  
	parser.add_argument('--patience', type=int, default=30)
	
	
	#for save and read
	parser.add_argument('--path', default='/home/joo/JOOCML/data/', type=str, help='data path')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
	parser.add_argument('--isload', default=False , type=bool, help='whether load model')  
	parser.add_argument('--isJustTest', default=False , type=bool, help='whether load model')
	parser.add_argument('--loadModelPath', type=str, help='loadModelPath')
	parser.add_argument('--isJustbeh', default=True , type=bool, help='GRU')
	parser.add_argument('--is_Meta_Path', default=True , type=bool, help='not GRU')
	parser.add_argument('--History_path', default='/nas_homes/joo/Recommendation/Try_hy_cml/History/', type=str, help='data path')
	parser.add_argument('--Model_path', default='/nas_homes/joo/Recommendation/Try_hy_cml/Model/', type=str, help='data path')


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

	return parser.parse_args()
args = parse_args()


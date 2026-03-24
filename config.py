import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--model', type=str, default='resnet18_gn', help='Neural network for training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Training dataset')
    parser.add_argument('--partition', type=str, default='noniid', help='Data partitioning strategy')
    parser.add_argument('--num_local_iterations', type=int, default=400, help='Number of local iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--personalized_learning_rate', type=float, default=0.01, help="Learning rate for personalized training")
    parser.add_argument('--epochs', type=int, default=10, help='Local training epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='Number of distributed clients')
    parser.add_argument('--n_domain_parties', type=int, default=2, help='Number of domain clients')
    parser.add_argument('--comm_round', type=int, default=55, help='Max communication rounds')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, default=0.0, help="Dropout probability")
    parser.add_argument('--datadir', type=str, default="./data/", help="Data directory")
    parser.add_argument('--beta', type=float, default=0.5, help='Dirichlet distribution parameter')
    parser.add_argument('--skew_class', type=int, default=2, help='Class skew parameter for non-IID')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--log_file_name', type=str, default=None, help='Log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer type')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='Client sampling fraction')
    parser.add_argument('--lambda_1', type=float, default=0.01, help='FedProx hyperparameter')
    parser.add_argument('--dump_partition_assets', action='store_true', help='Save partition statistics')
    parser.add_argument('--partition_assets_dir', type=str, default='./partition_assets', help='Directory for partition files')

    parser.add_argument('--femnist_sample_top', type=int, default=1, help='Sample top clients from FEMNIST')
    parser.add_argument('--femnist_train_num', type=int, default=20, help='Training clients for FEMNIST')
    parser.add_argument('--femnist_test_num', type=int, default=20, help='Testing clients for FEMNIST')

    parser.add_argument('--adv', default=0.1, type=float, help='Scale factor for adversarial loss')
    parser.add_argument('--bn', default=0.0, type=float, help='Scale factor for BN loss')
    parser.add_argument('--oh', default=1.0, type=float, help='Scale factor for cross-entropy loss')
    parser.add_argument('--generator', action="store_true", help="Use generator for synthetic data")
    parser.add_argument('--gen_aug', action="store_true", help="Use augmentation during generation")
    parser.add_argument('--gen_downsample', action="store_true", help="Enable downsampling during generation")
    parser.add_argument('--lr_g', default=1e-2, type=float, help='Learning rate for generator')
    parser.add_argument('--js_T', default=1, type=float, help='Temperature for JS divergence')
    parser.add_argument('--g_steps', default=500, type=int, help='Iterations for image generation')
    parser.add_argument('--nz', default=256, type=int, help='Latent noise dimension')
    parser.add_argument('--synthesis_batch_size', default=256, type=int, help='Batch size for synthesis')
    parser.add_argument('--T', default=1, type=float, help='Temperature for knowledge distillation')
    parser.add_argument('--start_round', default=50, type=int, help='Start generation round')
    parser.add_argument('--double_gen', action="store_true", help="Generate twice per round")
    parser.add_argument('--load_path', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--save_model', action="store_true", help="Save global model")
    parser.add_argument('--gen_complement', action="store_true", help="Generate data by class distribution")

    parser.add_argument('--server_momentum', type=float, default=0.0, help='Server momentum for FedAvgM')
    parser.add_argument('--mu', type=float, default=0.01, help='FedProx/MOON hyperparameter')
    parser.add_argument('--decorr_beta', type=float, default=0.1, help='FedDecorr loss coefficient')
    parser.add_argument('--exp_eps', type=float, default=1e-3, help='FedEXP aggregation parameter')
    parser.add_argument('--dyn_alpha', type=float, default=0.01, help='FedDyn regularization coefficient')

    parser.add_argument('--adam_server_momentum_1', type=float, default=0.9, help='First momentum for FedAdam')
    parser.add_argument('--adam_server_momentum_2', type=float, default=0.99, help='Second momentum for FedAdam')
    parser.add_argument('--adam_server_lr', type=float, default=1.0, help='Server learning rate for FedAdam')
    parser.add_argument('--adam_tau', type=float, default=0.001, help='Epsilon for FedAdam')
    parser.add_argument('--sam_rho', type=float, default=0.05, help='Rho parameter for FedSAM')

    parser.add_argument('--VHL_alpha', default=1.0, type=float, help='VHL loss weight')
    parser.add_argument('--VHL_feat_align', action="store_true", help='Enable feature alignment in VHL')
    parser.add_argument('--VHL_generative_dataset_root_path', default='/GPFS/data/yaxindu/FedHomo/VHL/data_preprocessing/generative/dataset/', type=str)
    parser.add_argument('--VHL_dataset_batch_size', default=128, type=int, help='Batch size for VHL dataset')
    parser.add_argument('--VHL_dataset_list', default="Gaussian_Noise", type=str, help="Dataset initialization type")
    parser.add_argument('--VHL_align_local_epoch', default=5, type=int, help='Local epochs for feature alignment')

    parser.add_argument('--reg_gamma', default=0.5, type=float, help='FedReg gamma')
    parser.add_argument('--reg_iter', default=10, type=int, help='FedReg iteration steps')
    parser.add_argument('--reg_eta', default=1e-3, type=float, help='FedReg learning rate')

    parser.add_argument('--gen_lr', default=1e-2, type=float, help='Generator learning rate for DisTrans')
    parser.add_argument('--distrans_alpha', default=0.3, type=float, help='DisTrans loss weight')
    parser.add_argument('--nn_agg', default=True, action="store_false", help='Enable nearest neighbor aggregation')
    parser.add_argument('--offset_eval', default=False, action="store_true", help='Enable offset evaluation')

    parser.add_argument('--channel', type=int, default=None, help='Input image channels (override dataset default)')
    parser.add_argument('--proto_reg', type=float, default=0.0, help='Weight for prototype regularization')
    parser.add_argument('--proto_mode', type=str, default='fc_weight', choices=['fc_weight'], help='Prototype construction mode')
    parser.add_argument("--proto_noise_std", type=float, default=0.0, help="Noise std for prototype perturbation")
    parser.add_argument("--proto_noise_mode", type=str, default="add", choices=["add", "replace"], help="Prototype noise application mode")
    parser.add_argument("--proto_noise_detachW", action="store_true", help="Detach FC weights from computation graph")

    parser.add_argument('--dump_syn', action='store_true', help='Save synthetic image data')
    parser.add_argument('--dump_syn_dir', type=str, default='./synthetic_dump/tmp', help='Directory for synthetic data')
    parser.add_argument('--dump_syn_batches', type=int, default=1, help='Number of batches to save')

    args = parser.parse_args()
    cfg = dict()

    # Configure dataset parameters: classes, channels, image size
    if args.dataset == 'mnist':
        cfg['classes_size'] = 10
        cfg["channel"] = 1
        cfg["image_size"] = 28
    elif args.dataset in {'cifar10', 'svhn', 'yahoo_answers'}:
        cfg['classes_size'] = 10
        cfg["channel"] = 3
        cfg["image_size"] = 32
    elif args.dataset == 'cifar100':
        cfg['classes_size'] = 100
        cfg["channel"] = 3
        cfg["image_size"] = 32
    elif args.dataset == 'tinyimagenet':
        cfg['classes_size'] = 200
        cfg["channel"] = 3
        cfg["image_size"] = 64
    elif args.dataset == 'pacs':
        cfg['classes_size'] = 7
        cfg["channel"] = 3
        cfg["image_size"] = 64
    elif args.dataset == 'femnist':
        cfg['classes_size'] = 62
        cfg["channel"] = 1
        cfg["image_size"] = 28
    elif args.dataset == 'flair':
        cfg['classes_size'] = 17
        cfg["channel"] = 3
    elif args.dataset ==  'fashionmnist':
        cfg['classes_size'] = 10
        cfg["channel"] = 1
        cfg["image_size"] = 28
    else:
        args.image_size = cfg["image_size"]
    
    # Use dataset default channels if not explicitly provided
    if args.channel is None:
        args.channel = cfg["channel"]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f">> Dataset default channel: {cfg['channel']}, final channel: {args.channel}")

    return args, cfg
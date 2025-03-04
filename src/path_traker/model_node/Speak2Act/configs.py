import argparse
#!/usr/bin/env python3

def get_parser():
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)

    # Directories
    parser.add_argument('--model_name', type=str,
                        default="DP1", metavar='N',
                        help='.')
    parser.add_argument('--root_train', default=r'/home/roblab20/PycharmProjects/Go_Teleop/clean_data/ordered_data',
                        metavar='DIR', help='path to dataset')
    parser.add_argument('--saveM_path', default=r'//home/roblab20/PycharmProjects/Speak2Act/checkpoint/BERT',
                        metavar='DIR', help='path for save the weights in optimizer of the model')
    parser.add_argument('--path4summary', default=r'/home/roblab20/PycharmProjects/Speak2Act/checkpoint/BERT',
                        metavar='DIR', help='')
    parser.add_argument("--checkpoint",
                        type=str, default="checkpoint/BERT/DP1/02_13_2025/2_net_Thu_Feb_13_17_50_11_2025.pt",
                        help="Path to pre-trained model weights")

    # Data loading configurations
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--max_length', type=int, default=50, metavar='N',
                        help='maximum length for padding the input.')
    parser.add_argument('--out_size', type=int, default=200, metavar='N',
                        help='What size will the output be 200.')
    parser.add_argument("--drop_last", default=True, type=str)
    parser.add_argument('-j', '--workers', type=int, default=2, metavar='N',
                        help='how many training processes to use (default: 0)')

    # Tokenization
    parser.add_argument('--tokenizer_type', type=str,
                        default="BERT", help="The model's name to use for tokenizing.")
    parser.add_argument('--vocab_size', type=int, default=30522, metavar='N',
                        help='For BERT: 30522 ; for GPT: 50257')

    # Model creation configuration
    parser.add_argument('--input_size', type=int, default=1, metavar='N',
                        help='.')
    parser.add_argument('--embedding_dim', type=int, default=512, metavar='N',
                        help='GPT-2 base model hidden size is 768.')
    parser.add_argument('--hidden_size', type=int, default=1024, metavar='N',
                        help="Model's hidden size")
    parser.add_argument('--num_heads', type=int, default=8, metavar='N',
                        help="Model's number of layers")
    parser.add_argument('--num_layers', type=int, default=6, metavar='N',
                        help="Model's number of layers (default: 6)")
    parser.add_argument('--seq_length', type=int, default=512, metavar='N',
                        help='Default: 512')
    parser.add_argument('--max_seq_len_out', type=int, default=200, metavar='N',
                        help='Output sequence length (default:200)')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout')

    # Optimizer parameters
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='Optimizer beta1 default=0.9')
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='Optimizer beta2 default=0.999')
    parser.add_argument('--weight_decay', type=float, default=1.25e-06,
                        help="weight decay default: 1e-2")
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate default=0.0001')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train')


    # Device
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='type "cpu" if there is no gpu')

    return parser
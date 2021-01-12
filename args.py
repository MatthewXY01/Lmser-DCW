import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Lmser-DCW SJTU 2020')
    parser.add_argument('--model', type=str, choices=['AE', 'DCW', 'DCW_woConstraint'], default='DCW_woConstraint')
    parser.add_argument('--dataset', type=str, choices=['minist', 'f-mnist'], default='mnist')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--result_path', type=str, default='./results/test.png')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.003)

    return parser
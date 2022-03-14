from argparse import ArgumentParser
import pandas as pd
import sys


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default='isic_usecase.csv')
    parser.add_argument("--model_save", type=str, default='isic_ctgan.pkl',
                            help='path to save trained model')
    parser.add_argument("--data_save", type=str, default='isic_ctgan.csv',
                            help='path to save generated data')
    parser.add_argument("--model", type=str, default='ctgan')
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--num_samples", type=int, default=100000)
    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    print(data.columns)

    categorical_features = ['admission_type',	'insurance',	'marital_status',	'ethnicity','blood',
                            'circulatory', 'congenital', 'digestive', 'endocrine',
                            'genitourinary', 'infectious', 'injury', 'mental', 'misc', 'muscular',
                            'neoplasms', 'nervous', 'pregnancy', 'prenatal', 'respiratory', 'skin',
                            'ICU', 'NICU']
    categorical_features = ['sex', 'anatom_site_general_challenge', 'target', 'hair_dense', 'hair_short', 'hair_medium', 'black_frame', 'ruler_mark', 'other']
    continuous_columns = ['LOS']
    continuous_columns = ['age_approx']

    epoch = args.epoch
    num_samples = args.num_samples

    if args.model == 'ctgan':
       from ctgan import CTGANSynthesizer
       model = CTGANSynthesizer(verbose=True)
       model.fit(data, categorical_features, epochs = epoch)
    elif args.model == 'tgan':
       from tgan.model import TGANModel
       model = TGANModel(
       continuous_columns,
       max_epoch=epoch,          # Number of epochs to use during training.
       steps_per_epoch=10000,    # Number of steps to run on each epoch.
       save_checkpoints=True,    # Weather the algorithm will save weights at each step
       restore_session=True,     # Option to continue training using previously saved weights
       batch_size=200,           # Batch Size when Training the Neural Network
       z_dim=200,                # Number of dimensions in the noise input for the generator.
       l2norm=0.00001,           # L2 reguralization coefficient when computing losses
       learning_rate=0.001,      # Learning rate for the optimizer
       num_gen_rnn=100,          # Number of units in rnn cell in generator.
       num_gen_feature=100,      # Number of units in fully connected layer in generator.
       num_dis_layers=1,         # Number of layers in discriminator.
       num_dis_hidden=100,       # Number of units per layer in discriminator.
       optimizer='AdamOptimizer' # Optimizer (str, default=AdamOptimizer): Name of the optimizer to use during fit, possible values are: [GradientDescentOptimizer, AdamOptimizer, AdadeltaOptimizer].
       )
       model.fit(data)
    else:
       sys.exit("Unsupported model!")

    model.save(args.model_save)
    samples = model.sample(num_samples)
    samples.to_csv(args.data_save, index = False)

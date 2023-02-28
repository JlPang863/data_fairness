import numpy as np
import pandas as pd
import jax
import flax
import optax
from transformers import BertTokenizer

from src.models import TextClassifier
from src.train_state import create_train_state, get_train_step
from src.utils import make_dir, print_args, save_args, set_global_seed

def load_jigsaw_raw_data(args, split='train', max_length=80):
    DATA_PATH = 'data/Jigsaw'

    if split == 'train':
        file_name = os.path.join(DATA_PATH, 'train.csv')
    else:
        file_name = os.path.join(DATA_PATH, 'test_public_expanded.csv')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encode = lambda text: tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='np')

    data = pd.read_csv(file_name)
    d0 = data.loc[data.male > 0]
    d1 = data.loc[data.female > 0]
    
    X, Y, A = [], [], []
    for _, row in tqdm(d0.iterrows()):
        X.append(encode(row['comment_text']))
        Y.append(int(row['toxicity']>=0.5))
        A.append(1)

    for _, row in tqdm(d1.iterrows()):
        X.append(encode(row['comment_text']))
        Y.append(int(row['toxicity']>=0.5))
        A.append(0)

    X = np.stack(X).squeeze()
    Y = np.stack(Y)
    A = np.stack(A)

    dataset = tf.data.Dataset.from_tensor_slices({'feature': X, 'label': Y, 'group': A})
    
    if split == 'train':
        dataset.shuffle(100).batch(args.train_batch_size)
    
    args.num_classes = 2
    args.input_shape = next(iter(dataset))['feature'].get_shape().as_list()[1:]
    return tfds.as_numpy(dataset)

def _make_dirs(args):
    make_dir(args.save_dir)
    make_dir(args.save_dir + '/ckpts')

def train(args):

    # setup
    set_global_seed()
    _make_dirs(args)

    model = TextClassifier(features=[256, 64, 16], num_classes=2)
    state = create_train_state(model)
    train_step = get_train_step('plain')
    # TODO: recorder

    T = 0
    for epoch_i in range(args.num_epochs):
        t = 0
        for batch in ds_train:
            t += 1
            T += 1

             state, train_metric = train_step(state, batch)

            if T % args.log_steps == 0:
                test_metric = test(state, ds_test)

import datetime

import yaml

from func.one.model.networklstm import create_model
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import os


class AmpData():
    def __init__(self, df, tokenizer_name='./func/one/prot_bert_bfd', max_len=300):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.max_len = max_len
        self.df = df
        self.seqs = self.get_seqs()

    def get_seqs(self):
        # isolate the amino acid sequences and their respective AMP labels
        seqs = list(self.df['sequence'])
        return seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = " ".join("".join(self.seqs[idx].split()))
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_len)
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        return sample


class PeptideBERTDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

        self.length = len(self.input_ids)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]

        return {
            'input_ids': torch.tensor(input_id, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }


def test(model, dataloader, device):
    model.eval()

    predictions = []
    predvalues = []
    allsequences = []
    print('dataloader')
    for batch in tqdm(dataloader):

        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.inference_mode():
            logits = model(inputs, attention_mask).squeeze()

        predvalues.extend(logits.cpu().tolist())
        preds = torch.where(logits > 0.5, 1, 0)
        predictions.extend(preds.cpu().tolist())

    return predvalues, predictions


def for_one(fa_path):
    # print(f'{"=" * 30}{"BEGAIN":^20}{"=" * 3}')



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: {}\n'.format(device))

    config = yaml.load(open('./config/config-test.yaml', 'r'), Loader=yaml.FullLoader)
    config['device'] = device

    # filepath = '/data3/lyr/project_AMP_pre/PeptideBERT-master-copy/test.csv'
    # filepath = sys.argv[1]
    filepath = fa_path
    basefile = os.path.basename(filepath)
    dirname = os.path.splitext(basefile)[0]

    # input csv
    # testdf = pd.read_csv(filepath)
    # testdf.columns = ['sequence']

    # input fasta
    print('read file')
    print(datetime.datetime.now())
    with open(filepath) as fa:
        fa_dict = {}
        for line in fa:
            # Remove line breaks at the end
            line = line.replace('\n', '')
            if line.startswith('>'):
                # remove >
                seq_name = line[1:]
                fa_dict[seq_name] = ''
            else:
                # Remove the end line break and concatenate the multi-line sequence
                fa_dict[seq_name] += line.replace('\n', '')
    print('read file end')
    print(datetime.datetime.now())
    # Calculate the length of each sequence and save it to the list
    fa_df1 = []
    fa_df2 = []
    # Calculate the length of each sequence
    for name, seq in fa_dict.items():
        fa_df1.append(name)
        fa_df2.append(seq)
    print(datetime.datetime.now())
    print('pandas data')
    testdf = pd.DataFrame(fa_df1, columns=['label'])
    testdf = pd.concat([testdf, pd.DataFrame(fa_df2, columns=['sequence'])], axis=1)
    print('AmpData', datetime.datetime.now())
    test_dataset = AmpData(testdf)
    print('AmpData end', datetime.datetime.now())
    sequences = []
    attention_mask_test = []
    test_inputs = []
    for index, content in enumerate(test_dataset):
        attention_mask_test.append(content['attention_mask'])
        test_inputs.append(content['input_ids'])

    # print(attention_mask_test)
    print('np.array', datetime.datetime.now())
    attention_mask_test = np.array(attention_mask_test)
    test_inputs = np.array(test_inputs)
    print('attention_mask_test', datetime.datetime.now())
    sequences = test_dataset.get_seqs()
    sequences = np.array(sequences)

    test_dataset = PeptideBERTDataset(input_ids=test_inputs, attention_masks=attention_mask_test)

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    print('Test dataset samples: ', len(test_dataset))
    print('Test dataset batches: ', len(test_data_loader))

    model = create_model(config)
    save_dir = config["model_path"]

    ################################################################################
    if not config['debug']:
        map_l = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load('{}/model.pt'.format(save_dir), map_location=map_l)['model_state_dict'],
                              strict=False)
        # model.load_state_dict(torch.load('{}/model.pt'.format(save_dir), map_location='cpu')['model_state_dict'], strict=False)
    print('load 1')
    print(datetime.datetime.now())
    predvalues, predictions = test(model, test_data_loader, device)
    print(datetime.datetime.now())
    print('load 2')
    # print(fa_df2)
    # print(predvalues)
    data = []
    for k, i in enumerate(fa_df2):
        data.append({
            'sequence': i,
            'AMP_Prediction_Score': '{:.4f}'.format(predvalues[k]),
            'type': predictions[k],
        })

    testdf['predvalue'] = predvalues
    testdf['predictions'] = predictions
    testdf.to_csv('./result/{}_pred_result.csv'.format(dirname), index=False)
    print('Model saved in: ', './result/{}_pred_result.csv'.format(dirname))
    return data

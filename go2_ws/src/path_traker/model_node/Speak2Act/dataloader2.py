import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from utils import new_load_files_and_chunk_data, quaternion_to_yaw
#!/usr/bin/env python3

def tokenizer(args):
    if args.tokenizer_type == 'GPT':
        # GPT TOKENIZER
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    if args.tokenizer_type == 'BERT':
        # BERT TOKENIZER
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer


class ClassDataset(Dataset):
    def __init__(self, args, input_com, out_pos, out_or, tokenizer, out_time=None):
        self.max_length = args.max_length
        self.out_size = args.out_size
        self.input_com = input_com
        self.tokenizer = tokenizer
        self.eos = torch.tensor([[1]])

        # Filter valid inputs
        valid_data = [
            (com, pos, or_)
            for com, pos, or_ in zip(input_com, out_pos, out_or)
            if com.strip() != "" and len(pos) > 0 and len(or_) > 0
        ]
        if len(valid_data) == 0:
            raise ValueError("No valid data found. Please check your dataset.")

        self.input_com, self.out_pos, self.out_or = zip(*valid_data)

        if out_time:
            self.out_time = out_time

    def __len__(self):
        return len(self.input_com)

    def __getitem__(self, index):
        command = self.input_com[index]
        ncoded_command = self.tokenizer(command, padding='max_length', truncation=True,
                                        return_tensors='pt', max_length=self.max_length)

        ncoded_command_ids = ncoded_command['input_ids'].squeeze()
        ncoded_command_att = ncoded_command['attention_mask'].squeeze()

        if ncoded_command_ids.nelement() == 0:
            raise ValueError(f"Empty input sequence at index {index}")

        # Output positions and orientations
        label_pos = torch.tensor(np.array(self.out_pos[index]))
        label_or = torch.tensor(np.array(self.out_or[index]))

        q_z = label_or[:, 0]  # Extract q_z
        q_w = label_or[:, 1]  # Extract q_w
        # q_log = torch.atan2(q_z, q_w).view(-1, 1)
        yaw = quaternion_to_yaw(q_z, q_w).view(-1, 1)

        if label_pos.nelement() == 0 or label_or.nelement() == 0:
            raise ValueError(f"Empty label at index {index}")

        label_combined = torch.cat((label_pos, yaw), dim=1)

        eos_vector = torch.ones((label_combined.shape[0], 1))

        padding_rows = self.out_size - len(label_combined)
        final_label_combined = F.pad(label_combined, (0, 0, 0, padding_rows), 'constant', 0)

        final_eos_vector = F.pad(eos_vector, (0, 0, 0, padding_rows), 'constant', 0)

        mask = ~(final_label_combined == 0).all(dim=-1)

        return ncoded_command_ids, ncoded_command_att, final_label_combined, mask, final_eos_vector


def data_loaders(args):

    train_path = args.root_train

    input_list, out_pos_list, out_or_list, out_time_list = new_load_files_and_chunk_data(train_path)

    input_train, input_test, pos_train, pos_test, or_train, or_test, time_train, time_test = train_test_split(
        input_list, out_pos_list, out_or_list, out_time_list, test_size=0.2, random_state=42
    )

    train_dataset = ClassDataset(args, input_train, pos_train, or_train, tokenizer=tokenizer(args))
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, drop_last=args.drop_last)

    test_dataset = ClassDataset(args, input_test, pos_test, or_test, tokenizer=tokenizer(args))
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, drop_last=args.drop_last)

    return trainloader, testloader


print("")

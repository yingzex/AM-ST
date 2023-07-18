from torch.autograd import Variable
import os
import torch
import numpy as np
import torch.nn.functional as F
from utils import read_data, clean_str, load_cls, load_vocab
from functools import cmp_to_key
import sys
import json
with open("run.config", 'rb') as f:
    configs_dict = json.load(f)

task_name = configs_dict.get("task_name")
model_name = configs_dict.get("model_name")
modified = configs_dict.get("modified")
batch_size = 16
max_line = 50000000
operation=sys.argv[2]
def cmp(a, b):
    return (a>b)-(a<b)

cls = load_cls("{}".format(task_name), "attn.{}".format(model_name)).cuda()
for i in cls.parameters():
	i.requires_grad = False
cls.eval()

fr=open(sys.argv[1],'r')
save_path = os.path.join(os.curdir, sys.argv[5], sys.argv[4])
if not os.path.exists(save_path):
	os.mkdir(save_path)
fwname = os.path.join(save_path, sys.argv[3]+'.data.'+operation)
fw=open(fwname,'w')
lines = []
for l in fr:
	lines.append(l)
fr.close()

line_num = min(len(lines), max_line)
for i in range(0, line_num, batch_size):
	batch_range = min(batch_size, line_num - i)
	batch_lines = lines[i:i + batch_range]
	batch_x = [clean_str(sent) for sent in batch_lines]
	pred, attn = cls(batch_x)
	pred = np.argmax(pred.cpu().data.numpy(), axis=1)
	for line, x, pre, att in zip(batch_lines, batch_x, pred, attn):
		if len(x) > 0:
			att = att[:len(x)]
			if task_name == 'yelp':
				avg = torch.mean(att)
			elif task_name == 'amazon':
				avg = 0.4
			mask = att.gt(avg)
			if sum(mask).item() == 0:
				mask = torch.argmax(att).unsqueeze(0)
			else:
				mask = torch.nonzero(mask.squeeze()).squeeze(1)
			idx = mask.cpu().numpy()
			idx = [int(ix) for ix in idx]
			contents = []
			for i in range(0, len(x)):
				if i not in idx:
					contents.append(x[i])
			wl = {"content": ' '.join(contents), "line": line.strip(), "masks": list(idx), "label": str(sys.argv[3][-1])}
			#print(wl)
			wl_str = json.dumps(wl)
			fw.write(wl_str)
			fw.write("\n")

fw.close()

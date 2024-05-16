import torch
import torchvision
import json

import time


class DependentDnn:
    def __init__(self, num_blocks: int, base_name: str):
        self.num_blocks = num_blocks
        self.next_input = None
        self.last_run = -1
        self.fblocks = []
        self.qblocks = []
        for i in range(num_blocks):
            self.fblocks.append(torch.jit.load(f"pylot/perception/detection/{base_name}block{i}.pt"))
            self.qblocks.append(torch.jit.load(f"pylot/perception/detection/{base_name}block{i}q.pt"))
        #with open('time_log.csv', 'w') as fp:
        #    fp.write('Start,Context Switch,End,Block,Location\n')

    def run_next_block(self, location: str):
        self.last_run += 1
        start = time.time()
        next_block = self.fblocks[self.last_run] if location == "f" else self.qblocks[self.last_run]
        self.next_input = self.next_input.to(torch.device('cuda' if location == "f" else 'cpu'))
        cntxtsw = time.time()
        if self.last_run < 3:
            bin, inst, nxt = next_block(self.next_input)
            self.next_input = nxt
        else:
            bin, inst = next_block(self.next_input)
        end = time.time()
        with open('time_log.csv', 'a') as fp:
            fp.write(f'{start},{cntxtsw},{end},{self.last_run},{location}\n')
        self.last_result = (bin.detach().cpu(), inst.detach().cpu())
        
    def get_last_result(self):
        return self.last_result

class IndependentDnn:
    def __init__(self, num_blocks: int, base_name: str):
        self.num_blocks = num_blocks

        self.next_input = None
        self.last_run = -1

    def run_next_block(self, location: str):
        pass

    def get_last_result(self):
        return 0


def main_loop(input_data: torch.Tensor, sched_file: str):
    model_output = None
    with open(sched_file, 'r') as fp:
        sched = json.loads(fp.read())
    
    dc = DependentDnn(num_blocks=4, base_name='lanenet')
    ic = IndependentDnn(num_blocks=5, base_name='yolo')

    while len(sched) > 0:
        ic.next_input = input_data
        dc.next_input = input_data
        for nb in sched['dc_list']:
            dc.run_next_block(nb)
        model_output = dc.get_last_result()

        for nb in sched['ic_list']:
            ic.run_next_block(nb)
        dec_var = ic.get_last_result()

        if dec_var == 1:
            sched = sched['dc+']
        else:
            sched = sched['dc-']


    return model_output
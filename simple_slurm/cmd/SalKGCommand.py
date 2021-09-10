from functools import reduce
from itertools import product
from operator import mul
from typing import Iterator
from pathlib import Path
import os


class TuneCommand:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.text = self._getParam("text", "must_have")
        self.graph = self._getParam("graph", "must_have")
        self.dataset = self._getParam("dataset", "must_have")
        self.split_type = self._getParam("split_type", "must_have")

        self.text_lr = self._getParam("tlr", 1e-5)
        self.graph_lr = self._getParam("glr", 1e-3)
        self.weight_decay = self._getParam("wd", 0)
        self.freeze_epochs = self._getParam("freeze", -1)
        self.pos_weight = self._getParam("pos_w", -1)

        self.python = self._getParam("py", "/home/jiashu/.conda/envs/fact/bin/python")
        self.main_file = self._getParam("main", "main.py")

        self.train_batch_size = self.eval_batch_size = self._getParam("bsz", 4)
        self.accumulate_grad_batches = self._getParam("acc", 2)

        self.save_saliency = self._getParam("save_saliency", False)
        self.save_checkpoint = self._getParam("save_checkpoint", False)
        self.seed = self._getParam("seed", 0)
        
        self.train_percentage = self._getParam("train_percentage", 100)

    def _config_path(self, dataset, arch, graph_encoder) -> str:
        return "NOCONFIG"

    def _getLoopParams(self) -> list:
        """
        params to be looped, group by param
        :return:
        [
          [ (p1, v1), (p1, v2)...]
          [ (p2, v1), ....]
        ]
        """
        return [
            [(param, val) for val in vals] if type(vals) is list else [(param, vals)]
            for param, vals in self.__dict__.items() if param != "kwargs"
        ]

    def __iter__(self) -> Iterator[str]:
        for config in product(*self._getLoopParams()):
            config_dict = {
                param: val for (param, val) in config
            }
            yield self.command(config_dict)

    def __len__(self):
        if "len" not in self.__dict__:
            loop_params = self._getLoopParams()
            self.len = reduce(mul, [len(p) for p in loop_params], 1)
        return self.len

    def mkString(self, delim: str="\n"+"+"*100+"\n", echo=False) -> str:
        """
        for 1. debug purpose
            2. in slurm class merge multiple run command into one slurm sbatch call (setting echo for printout in log)
        """
        s = ""
        for i, cmd in enumerate(self):
            if echo:
                s += f"echo '{i}/{len(self)}  {cmd}'\n"
            s += f"{cmd} {delim}\n"
        return s

    def command(self, d: dict) -> str:
        return (
            f'{d["python"]} {d["main_file"]} --config {self._config_path(d["dataset"], d["text"], d["graph"])} '
            f'--split_type {d["split_type"]} --pos_weight {d["pos_weight"]} '
            f'--text_lr {d["text_lr"]} --graph_lr {d["graph_lr"]} '
            f'--freeze_epochs {d["freeze_epochs"]} --weight_decay {d["weight_decay"]} '
            f'--train_batch_size {d["train_batch_size"]} --eval_batch_size {d["train_batch_size"]} --accumulate_grad_batches {d["accumulate_grad_batches"]} '
            f'--seed {d["seed"]} --train_percentage {d["train_percentage"]} '
            f'{"--save_checkpoint" if d["save_checkpoint"] else ""} {"--save_saliency" if d["save_saliency"] else ""} '
        )

    def _getParam(self, key: str, default):
        param_val = self.kwargs.get(key, default)
        assert param_val != "must_have", f"parameter {repr(key)} must be set manually"
        return param_val

class QACommand(TuneCommand):
    def __init__(self, **kwargs):
        # use no_kg for graph to do nokg qa training
        super().__init__(**kwargs)
        self.pooler = self._getParam("pooler", "cls")
        self.encoder_head = self._getParam("enc_head", "bos_token_mlp")

    def _config_path(self, dataset, arch, graph_encoder) -> str:
        config = f"configs/qa/{dataset}/{graph_encoder}/{arch}__quadro-rtx-8000.ini"
        assert Path(config).exists(), f"path {config} not exist"
        return config

    def command(self, d) -> str:
        base = super().command(d)
        cls_lr = f"--cls_lr {d['graph_lr']} " if self.graph == "no_kg" else ""
        qa_command = (
            f"{cls_lr} --encoder_pooler {d['pooler']} --text_encoder_head {d['encoder_head']} "
        )
        return base + qa_command


class CoarseCommand(TuneCommand):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sal_loss_weight = self._getParam("sal_loss", 1)
        self.threshold = self._getParam("threshold", None)
        self.no_kg_exp = self._getParam("no_kg_exp", "must_have")
        self.kg_exp = self._getParam("kg_exp", "must_have")
        self.qa_no_kg_dir = f"/home/jiashu/FactSelection/save2/FS-{self.no_kg_exp}/saliency/"
        assert Path(self.qa_no_kg_dir).exists()
        self.qa_kg_dir = f"/home/jiashu/FactSelection/save2/FS-{self.kg_exp}/saliency/"
        assert Path(self.qa_kg_dir).exists()
        self.coarse_model = self._getParam("coarse_model", "ensemble")
        self.no_kg_emb = self._getParam("no_kg_emb", "learned")
        self.criterion = self._getParam("criterion", "ce_loss")
        # if occl, SalKG-Coarse; if qa, Coarse-Heuristic
        self.saliency_method = self._getParam("saliency_method", "occl")
    
    def _config_path(self, dataset, arch, graph_encoder) -> str:
        config = f"configs/saliency/{dataset}/{graph_encoder}/coarse/occl/pred/{arch}__quadro-rtx-8000__{graph_encoder}_pqa.ini"
        assert Path(config).exists(), f"path {config} not exist"
        return config

    def command(self, d) -> str:
        base = super().command(d)
        threshold = "" if d["threshold"] is None else f"--threshold {d['threshold']}"
        coarse_command = (
            f'--sal_loss_weight {d["sal_loss_weight"]} '
            f'--coarse_model {d["coarse_model"]} --no_kg_emb {d["no_kg_emb"]} '
            f'--criterion {d["criterion"]} '
            f'{threshold} '
            f'--saliency_method {d["saliency_method"]} '
            f'--no_kg_exp {d["no_kg_exp"]} --qa_no_kg_dir {d["qa_no_kg_dir"]} '
            f'--kg_exp {d["kg_exp"]} --qa_kg_dir {d["qa_kg_dir"]} '
        )
        return base + coarse_command

class FineCommand(TuneCommand):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sal_loss_weight = self._getParam("sal_loss", 1)
        # saved fine saliency (from QA)
        self.sal_exp = self._getParam("sal_exp", "must_have")
        self.attn_head_agg = self._getParam("attn_head_agg", "avg_head")
        self.attn_head = self._getParam("attn_head", 2)
        self.attn_agg_k = self._getParam("attn_agg_k", "none")
        self.criterion = self._getParam("criterion", "KL_loss")
        self.attn_bound = self._getParam("attn_bound", 10)
        self.saliency_method = self._getParam("sal_method", "occl")
        self.saliency_heuristic = self._getParam("sal_heu", "ratio")
        self.saliency_value = self._getParam("sal_value", 10)
        
    def _config_path(self, dataset, arch, graph_encoder) -> str:
        config = f"configs/saliency/{dataset}/{graph_encoder}/fine/grad/target/{arch}__quadro-rtx-8000__{graph_encoder}__SALKG_FINE.ini"
        assert Path(config).exists(), f"path {config} not exist"
        return config

    def command(self, d) -> str:
        base = super().command(d)
        fine_command = (
            f'--criterion {d["criterion"]} --attn_bound {d["attn_bound"]} --sal_loss_weight {d["sal_loss_weight"]} '
            f'--saliency_exp {d["sal_exp"]} --saliency_method {d["saliency_method"]} '
            f'--attn_head_agg {d["attn_head_agg"]} --graph_att_head_num {d["attn_head"]} --attn_agg_k {d["attn_agg_k"]} '
            f'--saliency_heuristic {d["saliency_heuristic"]} --saliency_value {d["saliency_value"]} '
        )
        return base + fine_command


class HybridCommand(TuneCommand):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # coarse threshold
        self.threshold = self._getParam("threshold", 0.01)
        # fine saliency model's gnn lr
        self.fine_graph_lr = self._getParam("fine_graph_lr", 1e-4)

        # loss weight for coarse loss
        self.sal_loss_weight = self._getParam("sal_loss", 1)
        # fine loss = fine_qa_loss + fine_sal_loss_weight * fine_sal_loss
        self.fine_sal_loss_weight = self._getParam("fine_sal_loss_weight", 1)
        # loss weight for fine loss
        self.fine_loss_weight = self._getParam("fine_loss_weight", 1)

        # load coarse built dataset identified by no_kg_exp / kg_exp
        # also load no_kg_exp saliency
        self.kg_exp = self._getParam("kg_exp", "must_have")
        self.qa_kg_dir = f"/home/jiashu/FactSelection/save2/FS-{self.kg_exp}/saliency/"
        assert Path(self.qa_kg_dir).exists()
        self.no_kg_exp = self._getParam("no_kg_exp", "must_have")
        self.qa_no_kg_dir = f"/home/jiashu/FactSelection/save2/FS-{self.no_kg_exp}/saliency/"
        assert Path(self.qa_no_kg_dir).exists()

        # if not loading fine model from ckpt, provide QA model's built fine saliency dataset's exp id
        self.sal_exp = self._getParam("sal_exp", "NOEXP")
        # if loading from ckpt, eg. 21052
        self.fine_checkpoint_path = self._getParam("fine_ckpt", "NOCKPT") 
        if self.fine_checkpoint_path != "NOCKPT":
            if type(self.fine_checkpoint_path) is not list:
                self.fine_checkpoint_path = [self.fine_checkpoint_path]
            print(self.fine_checkpoint_path)
            for i, ckpt_path in enumerate(self.fine_checkpoint_path[:]):
                if ckpt_path == "NOCKPT": continue
                fine_checkpoint_path = Path(f"/mnt/nfs1/aaron/FactSelection/save/FS-{ckpt_path}/checkpoints/")
                print(fine_checkpoint_path)
                assert fine_checkpoint_path.exists(), "must ensure fine ckpt does exist"
                ckpt_path = os.listdir(fine_checkpoint_path)[0]
                self.fine_checkpoint_path[i] = str(fine_checkpoint_path / ckpt_path)

        # must provide one of them
        assert self.sal_exp != "NOEXP" or self.fine_checkpoint_path != "NOCKPT"
        self.saliency_heuristic = self._getParam("sal_heu", "ratio")
        
        self.attn_bound = self._getParam("attn_bound", 100)
        self.criterion = self._getParam("criterion", "KL_loss")
        self.saliency_method = self._getParam("sal_method", "occl")
        self.hybrid_mode = self._getParam("mode", None)
    
    def _config_path(self, dataset, arch, graph_encoder) -> str:
        config = f"configs/saliency/{dataset}/{graph_encoder}/hybrid/occl/pred/{arch}__quadro-rtx-8000__{graph_encoder}_pqa.ini"
        assert Path(config).exists(), f"config {config} not exist"
        return config

    def command(self, d) -> str:
        base = super().command(d)
        ckpt_str = f"--fine_checkpoint_path {d['fine_checkpoint_path']}" if d["fine_checkpoint_path"] != "NOCKPT" else ""
        mode  = f'--hybrid_mode {d["hybrid_mode"]}' if d["hybrid_mode"] is not None else ""
        fine_command = (
            f'--criterion {d["criterion"]} --attn_bound {d["attn_bound"]}  '
            f'--saliency_exp {d["sal_exp"]} --saliency_method {d["saliency_method"]} '
            f'--fine_graph_lr {d["fine_graph_lr"]} --fine_sal_loss_weight {d["fine_sal_loss_weight"]} '
            f'--fine_loss_weight {d["fine_loss_weight"]} --sal_loss_weight {d["sal_loss_weight"]} '
            f'{ckpt_str} '
            f"--threshold {d['threshold']} "
            f'--saliency_heuristic {d["saliency_heuristic"]} '
            f'{mode} '
            f'--no_kg_exp {d["no_kg_exp"]} --kg_exp {d["kg_exp"]} '
            f'--qa_no_kg_dir {d["qa_no_kg_dir"]} --qa_kg_dir {d["qa_kg_dir"]} '
        )
        return base + fine_command

if __name__ == '__main__':
    tc = FineCommand(tlr=[1e-5, 1e-3], glr=[3e-3, 2e-3], wd=0.02, pos_w=10, seed=[0, 1, 2], save_checkpoint=True,
                       save_saliency=True)
    for idx, i in enumerate(tc):
        print(idx, ": ", i)

# {"id": "8-343", 
#  "question": 
#      {"stem": 
#          "A person wants to start saving money so that they can afford a nice vacation at the end of the year. After looking over their budget and expenses, they decide the best way to save money is to", 
#          "choices": [
#              {"text": "make more phone calls", "label": "A"}, 
#              {"text": "quit eating lunch out", "label": "B"}, 
#              {"text": "buy less with monopoly money", "label": "C"}, 
#              {"text": "have lunch with friends", "label": "D"}
#             ]}, 
#  "answerKey": "B", 
#  "statements": [{"label": false, 
#                  "statement": "A person wants to start saving money so that they can afford a nice vacation at the end of the year. After looking over their budget and expenses, they decide the best way to save money is to make more phone calls"}, 
#                 {"label": true, 
#                  "statement": "A person wants to start saving money so that they can afford a nice vacation at the end of the year. After looking over their budget and expenses, they decide the best way to save money is to quit eating lunch out"}, 
#                 {"label": false, 
#                  "statement": "A person wants to start saving money so that they can afford a nice vacation at the end of the year. After looking over their budget and expenses, they decide the best way to save money is to buy less with monopoly money"}, 
#                 {"label": false, 
#                  "statement": "A person wants to start saving money so that they can afford a nice vacation at the end of the year. After looking over their budget and expenses, they decide the best way to save money is to have lunch with friends"}
#     ]
# }

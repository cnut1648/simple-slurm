from functools import reduce
from itertools import product
from operator import mul
from typing import Iterator


class TuneCommand:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.config = "NOCONFIG"

        self.text = self._getParam("text", "must_have")
        self.graph = self._getParam("graph", "must_have")
        self.dataset = self._getParam("dataset", "must_have")

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

    def mkString(self, delim: str="+"*100) -> str:
        """debug purpose"""
        return (delim + "\n\n").join(cmd for cmd in self)


    def command(self, d: dict) -> str:
        return (
            f'{d["python"]} {d["main_file"]} --config {d["config"]} '
            f'--text_lr {d["text_lr"]} --graph_lr {d["graph_lr"]} '
            f'--freeze_epochs {d["freeze_epochs"]} --weight_decay {d["weight_decay"]} '
            f'--train_batch_size {d["train_batch_size"]} --eval_batch_size {d["train_batch_size"]} --accumulate_grad_batches {d["accumulate_grad_batches"]} '
            f'--seed {d["seed"]} '
            f'{"--save_checkpoint" if d["save_checkpoint"] else ""} {"--save_saliency" if d["save_saliency"] else ""} '
        )

    def _getParam(self, key: str, default):
        param_val = self.kwargs.get(key, default)
        assert param_val != "must_have", f"parameter {repr(key)} must be set manually"
        return param_val

class QACommand(TuneCommand):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = f"configs/qa/{self.dataset}/{self.graph}/{self.text}__quadro-rtx-8000.ini"

class CoarseCommand(TuneCommand):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = f"configs/saliency/{self.dataset}/{self.graph}/coarse/pred/{self.text}__quadro-rtx-8000__{self.graph}_pqa.ini"
        self.sal_loss_weight = self._getParam("sal_loss", 1)
        self.no_kg_exp = self._getParam("no_kg_exp", "must_have")
        self.kg_exp = self._getParam("kg_exp", "must_have")
        self.qa_no_kg_dir = f"/home/jiashu/FactSelection/save/FS-{self.no_kg_exp}/saliency/"
        self.qa_kg_dir = f"/home/jiashu/FactSelection/save/FS-{self.kg_exp}/saliency/"
        self.coarse_model = self._getParam("coarse_model", "ensemble")
        self.no_kg_emb = self._getParam("no_kg_emb", "learned")
        self.criterion = self._getParam("criterion", "ce_loss")
        # if occl, SalKG-Coarse; if qa, Coarse-Heuristic
        self.saliency_method = self._getParam("saliency_method", "occl")

    def command(self, d) -> str:
        base = super().command(d)
        coarse_command = (
            f'--sal_loss_weight {d["sal_loss_weight"]} '
            f'--coarse_model {d["coarse_model"]} --no_kg_emb {d["no_kg_emb"]} '
            f'--criterion {d["criterion"]} '
            f'--saliency_method {d["saliency_method"]} '
            f'--no_kg_exp {d["no_kg_exp"]} --qa_no_kg_dir {d["qa_no_kg_dir"]} '
            f'--kg_exp {d["kg_exp"]} --qa_kg_dir {d["qa_kg_dir"]} '
        )
        return base + coarse_command

class FineCommand(TuneCommand):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = f"configs/saliency/{self.dataset}/{self.graph}/fine/grad/target/{self.text}__quadro-rtx-8000__{self.graph}__SALKG_FINE.ini"
        self.sal_loss_weight = self._getParam("sal_loss", 1)
        # saved fine saliency (from QA)
        self.sal_exp = self._getParam("sal_exp", "must_have")
        self.criterion = self._getParam("criterion", "KL_loss")
        self.attn_bound = self._getParam("attn_bound", 10)
        self.saliency_method = self._getParam("sal_method", "occl")
        self.saliency_heuristic = self._getParam("sal_heu", "ratio")
        self.saliency_value = self._getParam("sal_value", 10)

    def command(self, d) -> str:
        base = super().command(d)
        fine_command = (
            f'--criterion {d["criterion"]} --attn_bound {d["attn_bound"]} --sal_loss_weight {d["sal_loss_weight"]} '
            f'--saliency_exp {d["sal_exp"]} --saliency_method {d["saliency_method"]} '
            f'--saliency_heuristic {d["saliency_heuristic"]} --saliency_value {d["saliency_value"]} '
        )
        return base + fine_command


class HybridCommand(TuneCommand):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = f"configs/saliency/{self.dataset}/{self.graph}/hybrid/occl/pred/{self.text}__quadro-rtx-8000__{self.graph}_pqa.ini"

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
        self.qa_kg_dir = f"/home/jiashu/FactSelection/save/FS-{self.kg_exp}/saliency/"
        self.no_kg_exp = self._getParam("no_kg_exp", "must_have")
        self.qa_no_kg_dir = f"/home/jiashu/FactSelection/save/FS-{self.no_kg_exp}/saliency/"

        # if not loading fine model from ckpt, provide QA model's built fine saliency dataset's exp id
        self.sal_exp = self._getParam("sal_exp", "NOEXP")
        # if loading from ckpt
        self.fine_checkpoint_path = self._getParam("fine_ckpt", "NOCKPT")
        # must provide one of them
        assert self.sal_exp != "NOEXP" or self.fine_checkpoint_path != "NOCKPT"
        
        self.attn_bound = self._getParam("attn_bound", 100)
        self.criterion = self._getParam("criterion", "KL_loss")
        self.saliency_method = self._getParam("sal_method", "occl")

    def command(self, d) -> str:
        base = super().command(d)
        ckpt_str = f"--fine_checkpoint_path {d['fine_checkpoint_path']}" if d["fine_checkpoint_path"] != "NOCKPT" else ""
        fine_command = (
            f'--criterion {d["criterion"]} --attn_bound {d["attn_bound"]}  '
            f'--saliency_exp {d["sal_exp"]} --saliency_method {d["saliency_method"]} '
            f'--fine_graph_lr {d["fine_graph_lr"]} --fine_sal_loss_weight {d["fine_sal_loss_weight"]} '
            f'--fine_loss_weight {d["fine_loss_weight"]} --sal_loss_weight {d["sal_loss_weight"]} '
            f'{ckpt_str} '
            f'--no_kg_exp {d["no_kg_exp"]} --kg_exp {d["kg_exp"]} '
            f'--qa_no_kg_dir {d["qa_no_kg_dir"]} --qa_kg_dir {d["qa_kg_dir"]} '
        )
        return base + fine_command

if __name__ == '__main__':
    tc = FineCommand(tlr=[1e-5, 1e-3], glr=[3e-3, 2e-3], wd=0.02, pos_w=10, seed=[0, 1, 2], save_checkpoint=True,
                       save_saliency=True)
    for idx, i in enumerate(tc):
        print(idx, ": ", i)

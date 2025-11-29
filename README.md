## Wav address:
```
share/workspace/wangkuang/workspace/git/HSP/speech_dataset/filter_pipeline/filtered_result/batch_0000
```
```
share/workspace/wangkuang/workspace/git/HSP/speech_dataset/filter_pipeline/filtered_result/batch_0001
```
## how to run the code
- create cosyvoice
- inside cosyvoice root addr, do ```bash https://github.com/SteveThokaZhuang/cv_hlt.git```, and follow the env create guidance of **cosyvoice vllm**
- install swanlab ```bash pip install swanlab```
- modify vllm feats, open ./cosyvoice/cli/model.py and find
```python
def load_vllm(self, model_dir):
        export_cosyvoice2_vllm(self.llm, model_dir, self.device)
        from vllm import EngineArgs, LLMEngine
        engine_args = EngineArgs(model=model_dir,
                                 skip_tokenizer_init=True,
                                 enable_prompt_embeds=True,
                                 gpu_memory_utilization=0.95) # stevez
        self.llm.vllm = LLMEngine.from_engine_args(engine_args)
        self.llm.lock = threading.Lock()
        del self.llm.llm.model.model.layers
```
then change gpu_memory_utilization to `0.95`
```bash
mkdir data # place your jsonl data here
# create your text index, e.g.
```
- modify hlt_cv_inference.sh and hlt_cv_inference.sh under your needs
- run ```bash htl_inference_251126.sh```

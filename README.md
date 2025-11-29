## File structure
![Uploading image.png…]()
- Wav address:
```
share/workspace/wangkuang/workspace/git/HSP/speech_dataset/filter_pipeline/filtered_result/batch_0000
```
```
share/workspace/wangkuang/workspace/git/HSP/speech_dataset/filter_pipeline/filtered_result/batch_0001
```
## how to run the code
- create cosyvoice
- inside cosyvoice root addr, do ```bash https://github.com/SteveThokaZhuang/cv_hlt.git```, and follow the env create guidance of **cosyvoice vllm**
- create text index
```bash
mkdir data # place your jsonl data here
# create your text index, e.g.
```
- modify hlt_cv_inference.sh and hlt_cv_inference.sh under your needs
- run ```bash htl_inference_251126.sh```

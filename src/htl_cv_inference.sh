# python cosyvoice_htl.py \
#   --model-dir ../pretrained_models/CosyVoice2-0.5B \
#   --input text0.txt \
#   --outdir ref_out \
#   --prompt-wav ../asset/zero_shot_prompt.wav \
#   --prompt-text "希望你以后能够做的比我还好呦。" \
#   --use-spkinfo true \
#   --spk-id speaker0_spk

# python - <<'PY'
# import torch, os, sys
# md = "../pretrained_models/CosyVoice2-0.5B"
# d = torch.load(os.path.join(md,"spk2info.pt"), map_location="cpu")
# print("keys:", list(d.keys())[:10])
# print("has speaker0_spk? ->", "speaker0_spk" in d)
# PY

CUDA_VISIBLE_DEVICES=0 python cosyvoice_htl.py \
  --model-dir ../pretrained_models/CosyVoice2-0.5B \
  --input texts/text.txt \
  --outdir spk_out_0/ \
  --use-spkinfo true \
  --spk-id speaker0_spk &
# CUDA_VISIBLE_DEVICES=1 python cosyvoice_htl.py \
#   --model-dir ../pretrained_models/CosyVoice2-0.5B \
#   --input text.txt \
#   --outdir spk_out_1/ \
#   --use-spkinfo true \
#   --spk-id speaker0_spk &
# CUDA_VISIBLE_DEVICES=2 python cosyvoice_htl.py \
#   --model-dir ../pretrained_models/CosyVoice2-0.5B \
#   --input text.txt \
#   --outdir spk_out_2/ \
#   --use-spkinfo true \
#   --spk-id speaker0_spk &
# CUDA_VISIBLE_DEVICES=3 python cosyvoice_htl.py \
#   --model-dir ../pretrained_models/CosyVoice2-0.5B \
#   --input text.txt \
#   --outdir spk_out_3/ \
#   --use-spkinfo true \
#   --spk-id speaker0_spk &
# nvidia-smi -l 1 &
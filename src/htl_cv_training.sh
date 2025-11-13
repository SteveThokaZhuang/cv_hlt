python cosyvoice_htl.py \
  --model-dir ../pretrained_models/CosyVoice2-0.5B \
  --input texts/text0.txt \
  --outdir ref_out \
  --prompt-wav ../asset/zero_shot_prompt.wav \
  --prompt-text "希望你以后能够做的比我还好呦。" \
  --use-spkinfo true \
  --spk-id speaker0_spk
# --spk-id: assign speaker tag
# --prompt wav: provide a prompt audio assigned with speaker
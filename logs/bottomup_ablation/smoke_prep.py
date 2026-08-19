import sys
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, 'src')
from transformers import AutoTokenizer
md = Path('/workspace/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/snapshots/47e169305d2e8376be1d31e765533382721b2cc1')
tok = AutoTokenizer.from_pretrained(md)
tok.pad_token = tok.eos_token
from src.sandbox.ext_steerability.ablate_readdir_labeltokens import prep_task_nshot
for task in ['antonym', 'ag_news', 'iso_date_to_month', 'next_number_digits', 'word_polarity']:
    for n in (1, 6):
        items = prep_task_nshot(task, Path('dataset_files/isolation_prompts_ext'), tok, n)
        npos = sorted({len(it['label_pos']) for it in items})
        print(task, f'n={n}', 'ok', 'label_pos counts', npos[:4],
              'maxlen', max(len(it['ids']) for it in items), flush=True)
print('ALL OK')

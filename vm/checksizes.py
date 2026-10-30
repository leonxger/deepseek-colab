"""Check that every linked shard has its published byte size. usage: python3 -I checksizes.py MODEL_DIR"""
import os, sys
RUNG = 'IQ2_XXS'
SIZES = {1: 5780704, 2: 39535781184, 3: 39736712000, 4: 39036908384, 5: 19923784832, 6: 110595290816}
bad = 0
for i, n in SIZES.items():
    p = os.path.join(sys.argv[1], 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-%s-%05d-of-00006.gguf' % (RUNG, i))
    s = os.path.getsize(p) if os.path.exists(p) else -1
    print(i, os.path.realpath(p), s, 'ok' if s == n else 'BAD')
    bad += s != n
sys.exit(1 if bad else 0)

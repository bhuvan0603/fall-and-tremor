"""Quick test: run tremor inference on all demo files and check accuracy."""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import tremor_inference
import pandas as pd
import numpy as np

tremor_inference.load_model('models/tremor_model.joblib')
print('Model loaded:', tremor_inference.is_loaded())

demo_dir = 'test_data/tremor_samples'
files = sorted(glob.glob(os.path.join(demo_dir, '*.csv')))
print(f'Found {len(files)} demo files\n')

wrong_list = []
correct_count = 0
no_tremor_total = 0
no_tremor_correct = 0
tremor_total = 0
tremor_correct = 0

for f in files:
    fname = os.path.basename(f)
    true_label = 'no_tremor' if '_no_tremor' in fname else 'tremor'
    
    df = pd.read_csv(f)
    # find signal column
    col = None
    for c in df.columns:
        if 'velocity' in c.lower() or 'signal' in c.lower() or 'value' in c.lower():
            col = c
            break
    if col is None:
        col = df.columns[-1]
    
    signal = df[col].values.astype(float)
    result = tremor_inference.analyze_signal(signal, sample_rate=100)
    predicted = 'tremor' if result['tremor_detected'] else 'no_tremor'
    correct = (predicted == true_label)
    
    if true_label == 'no_tremor':
        no_tremor_total += 1
        if correct:
            no_tremor_correct += 1
    else:
        tremor_total += 1
        if correct:
            tremor_correct += 1
    
    if correct:
        correct_count += 1
    
    mark = 'OK' if correct else '** WRONG **'
    print(f'{fname:35s} true={true_label:12s} pred={predicted:12s} conf={result["confidence"]:.3f} t%={result["tremor_window_pct"]:5.1f}  {mark}')
    
    if not correct:
        wrong_list.append((fname, true_label, predicted, result['confidence'], result['tremor_window_pct']))

print('\n' + '='*80)
print(f'No-tremor files:  {no_tremor_correct}/{no_tremor_total} correct  ({100*no_tremor_correct/max(1,no_tremor_total):.0f}%)')
print(f'Tremor files:     {tremor_correct}/{tremor_total} correct  ({100*tremor_correct/max(1,tremor_total):.0f}%)')
print(f'Overall:          {correct_count}/{len(files)} correct  ({100*correct_count/max(1,len(files)):.0f}%)')

if wrong_list:
    print(f'\nWRONG predictions ({len(wrong_list)}):')
    for fname, tl, pr, conf, tpct in wrong_list:
        print(f'  {fname}: true={tl}, predicted={pr}, confidence={conf:.3f}, tremor%={tpct:.1f}')

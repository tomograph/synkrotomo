import json
import subprocess

ORIGINAL_FILE = '../../src/intersections.fut'
with open(ORIGINAL_FILE) as f:
  bench_src = f.read()

with open('benches.json') as f:
  configs = json.load(f)

for key in configs:
  config = configs[key]

  # Create benching file
  subprocess.check_call(['touch', 'temp.fut'])
  with open('temp.fut', 'w') as f:
    f.write('-- ==\n' + '\n'.join(config['inputs']))
    f.write(bench_src)

  # Run benching file
  print('Running ' + key + ' with ' + config['compiler'])
  subprocess.check_call(['futhark-bench', '--compiler=' + config['compiler'], '--json=results/' + key + '.json', 'temp.fut'])
  subprocess.check_call(['rm', './temp', './temp.c', 'temp.fut'])
  print('\n')
  
  
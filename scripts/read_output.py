fname = "logs/beta_0.0.4.0_svamp_test_e1_eval.log"
with open(fname) as fd:
    lines = fd.readlines()
for l in lines:
    if('ratio' in l): print(l.split('ratio ')[1].split()[0])
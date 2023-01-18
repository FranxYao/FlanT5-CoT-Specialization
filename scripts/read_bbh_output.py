filename = "logs/beta_0.0.2.8_bbh_e0_cot_eval.log"
lines = open(filename).readlines()
for l in lines:
    if('All average' in l): print(l.split('All average ')[1].split()[0])
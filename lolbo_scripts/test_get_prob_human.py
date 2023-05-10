import sys 
sys.path.append("../")
from oracle.get_prob_human import load_human_classier_model, get_prob_human

human_classifier_tokenizer, human_classifier_model = load_human_classier_model()


seq = "GAMAARAVAEQAAELLVLDDRLMAHMAEDKLSVAQALTNAAAGDTATTEMLQTFAKGLDMPAAERRRSRRATQEAWMQRHGGEELARVQAGLSAILARYLA"
xs_batch = [seq]

min_prob_human = 0.9 

c_vals = []
probs_h = []
for x in xs_batch:
    probh = get_prob_human(
        seq=x, 
        human_tokenizer=human_classifier_tokenizer, 
        human_model=human_classifier_model, 
    )
    c_val = (probh*-1) + min_prob_human
    c_vals.append(c_val)
    probs_h.append(probh)

import pdb 
pdb.set_trace() 
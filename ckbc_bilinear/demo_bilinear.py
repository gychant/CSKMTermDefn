"""
Adapted from the ckbc-demo at http://ttic.uchicago.edu/~kgimpel/commonsense.html
"""

# arg1 is term1

# arg2 is term2

# arg3 is the way to get the score, valid input including
# all, max, topfive, sum
# [SymbolOf, CreatedBy, MadeOf, PartOf, HasLastSubevent, HasFirstSubevent, Desires, CausesDesire,
# DefinedAs, HasA, ReceivesAction, MotivatedByGoal, Causes, HasProperty, HasPrerequisite,
# HasSubevent, AtLocation, IsA, CapableOf, UsedFor]
# case insensitive for the third argument

import pickle
import numpy as np
import sys
import math


def getVec(We, words, t, verbose=True):
    t = t.strip()
    array = t.split('_')
    if array[0] in words:
        vec = We[words[array[0]],:]
    else:
        vec = We[words['UUUNKKK'],:]
        if verbose:
            print('can not find corresponding vector:',array[0].lower())
    for i in range(len(array)-1):
        if array[i+1] in words:
            vec = vec + We[words[array[i+1]],:]
        else:
            if verbose:
                print('can not find corresponding vector:',array[i+1].lower())
            vec = vec + We[words['UUUNKKK'],:]
    vec = vec/len(array)
    return vec


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def score(term1, term2, words, We, rel, Rel, Weight, Offset, evaType, verbose=True):
    v1 = getVec(We, words, term1, verbose=verbose)
    v2 = getVec(We, words, term2, verbose=verbose)
    result = {}

    del_rels = [
        'HasPainIntensity', 'HasPainCharacter', 'LocationOfAction', 'LocatedNear',
        'DesireOf', 'NotMadeOf', 'InheritsFrom', 'InstanceOf', 'RelatedTo', 'NotDesires',
        'NotHasA', 'NotIsA', 'NotHasProperty', 'NotCapableOf']
    del_rels = set([r.lower() for r in del_rels])

    for k, v in rel.items():
        # skip relations that should be excluded
        if k in del_rels:
            continue

        v_r = Rel[rel[k], :]
        gv1 = np.tanh(np.dot(v1, Weight) + Offset)
        gv2 = np.tanh(np.dot(v2, Weight) + Offset)
    
        temp1 = np.dot(gv1, v_r)
        score = np.inner(temp1, gv2)
        result[k] = (sigmoid(score))

    if evaType.lower() == 'max':
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        if verbose:
            for k, v in result[:1]:
                print(k, 'score:', v)
        return result[:1]

    if evaType.lower() == 'topfive':
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        if verbose:
            for k, v in result[:5]:
                print(k, 'score:', v)
        return result[:5]

    if evaType.lower() == 'sum':
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        total = 0
        for i in result:
            total = total + i[1]
        if verbose:
            print('total score is:', total)
        return total

    if evaType.lower() == 'all':
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        if verbose:
            for k, v in result[:]:
                print(k, 'score:', v)
        return result
    else:
        tar_rel = evaType.lower()
        if result.get(tar_rel) is None:
            print('illegal relation, please re-enter a valid relation')
            return 'None'
        else:
            if verbose:
                print(tar_rel, 'relation score:', result.get(tar_rel))
            return result.get(tar_rel)


if __name__ == "__main__":
    model_file_path = "data/CKBC/ckbc-demo/Bilinear.pickle"
    model = pickle.load(open(model_file_path, "rb"), encoding="latin1")

    Rel = model['rel']
    We = model['embeddings']
    Weight = model['weight']
    Offset = model['bias']
    words = model['words_name']
    rel = model['rel_name']
    
    result = score(str(sys.argv[1]), str(sys.argv[2]), words, We, rel, Rel, Weight, Offset, str(sys.argv[3]))


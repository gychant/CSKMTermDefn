"""
Plot score distributions of models
"""
import os
from tqdm import tqdm
import numpy as np
from relation import rel2text

separate_plot = True


def plot_score_dist(models):
    for model in models:
        input_dir = model["input_dir"]

        test_files = [f for f in os.listdir(input_dir)
                      if os.path.isfile(os.path.join(input_dir, f))
                      and os.path.splitext(os.path.basename(f))[0] in rel2text]

        rel_scores = []
        relations = []
        for test_file in test_files:
            print("Reading file {} ...".format(test_file))
            relation = os.path.splitext(os.path.basename(test_file))[0]
            relations.append(relation)
            scores = []

            with open(os.path.join(input_dir, test_file)) as f:
                for line in tqdm(f):
                    line = line.strip()
                    score = float(line.split("\t")[3])
                    scores.append(score)
            rel_scores.append(scores)

        model["rel_scores"] = rel_scores
        model["relations"] = relations

    matplot_draw(models)
    print("DONE.")


def matplot_draw(models):
    import matplotlib.pyplot as plt

    if not os.path.exists("./data/plots"):
        os.makedirs("./data/plots")

    if not separate_plot:
        fig, ax = plt.subplots(1, 3, figsize=(50, 10))

    for i, model in enumerate(models):
        if separate_plot:
            fig, ax = plt.subplots()
        else:
            ax = axs[i]

        rel_scores = model["rel_scores"]
        bins = model["bins"]
        relations = model["relations"]
        data_name = model["data_name"]

        ax.hist(rel_scores, bins, histtype='bar')
        if isinstance(bins, list):
            plt.xticks(bins, fontsize=15)
            plt.yticks(fontsize=15)

        if data_name == "PMI":
            ax.set_xlabel("PMI Score", labelpad=5, fontsize=20)
        else:
            ax.set_xlabel("Plausibility Score", labelpad=5, fontsize=20)
        ax.set_ylabel("Number of Triples", labelpad=5, fontsize=20)
        # ax.set_title("{} Score Distribution".format(data_name), fontsize=20)
        ax.legend(relations, fontsize=12)
        plt.tight_layout()

        if separate_plot:
            plt.savefig("./data/plots/{}.svg".format(data_name))

    if not separate_plot:
        fig.savefig("./data/plots/score_dist.svg")


if __name__ == "__main__":
    bilinear_avg = dict(
        input_dir="./data/bilearavg_wiktionary_core",
        bins=list(np.arange(0, 1.1, 0.1)),
        data_name="Bilinear AVG")
    kg_bert = dict(
        input_dir="./data/kgbert_wiktionary_core",
        bins=list(np.arange(0, 1.1, 0.1)),
        data_name="KG-BERT")
    pmi = dict(
        input_dir="./data/pmi_coherency_wiktionary_core",
        bins=list(np.arange(-15, 18, 3)),
        data_name="PMI")
    plot_score_dist([bilinear_avg, kg_bert, pmi])


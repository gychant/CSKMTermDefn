
import os
import json
from tqdm import tqdm
from wiktionaryparser import WiktionaryParser

conceptnet_vocab_pos_tag_file = "data/conceptnet/conceptnet_vocab_pos_tag_core.tsv"
conceptnet_vocab_definition_file = "data/conceptnet/conceptnet_vocab_definition.jsonl"


def get_definition_entries(defn_file_path):
    print("\nLoading term definitions entries...")
    entries = set()

    if os.path.exists(defn_file_path):
        with open(defn_file_path) as f:
            for _line in tqdm(f):
                data = json.loads(_line.strip())
                entries.add(data["text"])
        print("Done loading ...")
    print("Number of existing definition entries:", len(entries))
    return entries


# Get entries already crawled in previous runs.
entries = get_definition_entries(conceptnet_vocab_definition_file)

parser = WiktionaryParser()
with open(conceptnet_vocab_pos_tag_file) as fin, \
        open(conceptnet_vocab_definition_file, "a+") as fout:
    cnt_success = 0
    cnt_failure = 0
    for line in tqdm(fin):
        if len(line.strip()) == 0:
            continue

        text, pos_list = line.split("\t")
        if text.replace("_", " ") in entries:
            continue

        pos_tags = set(pos_list.split(","))
        try:
            print("crawling [{}] ...".format(text))
            text = text.replace("_", " ")
            reponse = parser.fetch(text)
            rec = dict()
            rec["text"] = text
            rec["defn"] = dict()
            for meaning in reponse:
                for dfn in meaning["definitions"]:
                    pos = dfn["partOfSpeech"]
                    if pos in ["noun", "verb", "adjective"]:
                        if len(dfn["text"]) >= 2:
                            rec["defn"][pos] = dfn["text"][1]
                        elif len(dfn["text"]) == 1:
                            rec["defn"][pos] = dfn["text"][0]
            fout.write(json.dumps(rec) + "\n")
            cnt_success += 1
        except KeyboardInterrupt:
            raise
        except:
            print(reponse)
            cnt_failure += 1
            print("Failed to query text <{}>, so far {} failed".format(text, cnt_failure))

print("cnt_success:", cnt_success)
print("cnt_failure:", cnt_failure)
print("DONE")


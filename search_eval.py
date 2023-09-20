import math
import sys
import time

import metapy
import pytoml
# import numpy as np
# import scipy.stats as stats
import random

random.seed(100000)

class InL2Ranker(metapy.index.RankingFunction):
    """
    Create a new ranking function in Python that can be used in MeTA.
    """
    def __init__(self, some_param=1.0):
        self.param = some_param
        # You *must* call the base class constructor here!
        super(InL2Ranker, self).__init__()

    def score_one(self, sd):
        """
        You need to override this function to return a score for a single term.
        For fields available in the score_data sd object,
        @see https://meta-toolkit.org/doxygen/structmeta_1_1index_1_1score__data.html
        """
        tfn = sd.doc_term_count * math.log(1.0 + sd.avg_dl / sd.doc_size, 2)
        mid = tfn / (tfn + self.param)
        sub = (sd.num_docs + 1) / (sd.corpus_term_count + 0.5)
        return sd.query_term_weight * mid * math.log(sub, 2)


def load_ranker(cfg_file, some_param):
    """
    Use this function to return the Ranker object to evaluate, e.g. return InL2Ranker(some_param=1.0) 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index. You can ignore this for MP2.
    """
    return InL2Ranker(some_param)

if __name__ == '__main__':
    """
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)
    """

    cfg = "config.toml"
    print('Building or loading index...')
    idx = metapy.index.make_inverted_index(cfg)

    # === INL2 MAP ===
    print("=== INL2 MAP===\n\n")
    f = open("inl2.avg_p.txt", "w")
    for x in range(1, 15):
        ranker = load_ranker(cfg, x)
        ev = metapy.index.IREval(cfg)

        with open(cfg, 'r') as fin:
            cfg_d = pytoml.load(fin)

        query_cfg = cfg_d['query-runner']
        if query_cfg is None:
            print("query-runner table needed in {}".format(cfg))
            sys.exit(1)

        start_time = time.time()
        top_k = 10
        query_path = query_cfg.get('query-path', 'queries.txt')
        query_start = query_cfg.get('query-id-start', 0)

        query = metapy.index.Document()
        print('Running queries')
        with open(query_path) as query_file:
            for query_num, line in enumerate(query_file):
                query.content(line.strip())
                results = ranker.score(idx, query, top_k)
                avg_p = ev.avg_p(results, query_start + query_num, top_k)
                # print("Query {} average precision: {}".format(query_num + 1, avg_p))
                if x == 6:
                    f.write(str(avg_p) + "\n")

        print("Mean average precision: {}".format(ev.map()))
        print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))
        if x == 6:
            f.close()

    # === DirichletPrior MAP ===
    print("=== DirichletPrior MAP===\n\n")
    idx = metapy.index.make_inverted_index("config.toml")

    # Build the query object and initialize a ranker
    query = metapy.index.Document()
    for x in range(1, 15):
        ranker = metapy.index.DirichletPrior(x)
        # To do an IR evaluation, we need to use the queries file and relevance judgements.
        ev = metapy.index.IREval('config.toml')
        # Load the query_start from config.toml or default to zero if not found
        with open('config.toml', 'r') as fin:
            cfg_d = pytoml.load(fin)
        query_cfg = cfg_d['query-runner']
        query_start = query_cfg.get('query-id-start', 0)
        # We will loop over the queries file and add each result to the IREval object ev.
        num_results = 10
        with open('cranfield-queries.txt') as query_file:
            for query_num, line in enumerate(query_file):
                query.content(line.strip())
                results = ranker.score(idx, query, num_results)
                avg_p = ev.avg_p(results, query_start + query_num, num_results)
                # print("Query {} average precision: {}".format(query_num + 1, avg_p))
        print("Mean average precision: {} with mu = {}".format(ev.map(), x))

    # === OKAPI BM25 MAP ===
    # Build the query object and initialize a ranker
    query = metapy.index.Document()
    k1 = 1.2
    b = 0.75
    k3 = 500
    ranker = metapy.index.OkapiBM25(k1, b, k3)
    # To do an IR evaluation, we need to use the queries file and relevance judgements.
    ev = metapy.index.IREval('config.toml')
    # Load the query_start from config.toml or default to zero if not found
    with open('config.toml', 'r') as fin:
        cfg_d = pytoml.load(fin)
    query_cfg = cfg_d['query-runner']
    query_start = query_cfg.get('query-id-start', 0)
    # We will loop over the queries file and add each result to the IREval object ev.
    num_results = 10
    f = open("bm25.avg_p.txt", "w")
    with open('cranfield-queries.txt') as query_file:
        for query_num, line in enumerate(query_file):
            query.content(line.strip())
            results = ranker.score(idx, query, num_results)
            avg_p = ev.avg_p(results, query_start + query_num, num_results)
            f.write(str(avg_p) + "\n")
            # print("Query {} average precision: {}".format(query_num + 1, avg_p))

        print("Mean average precision: {} with k1 = {}, k3 = {}, b = {}".format(ev.map(), k1, k3, b))
        f.close()

    # === p value calculation ===
    """
    print("=== p value calculation ===\n\n")
    sig = open("significance.txt", "w")

    bm25 = np.loadtxt("bm25.avg_p.txt")
    inl2 = np.loadtxt("inl2.avg_p.txt")

    p_val = stats.ttest_rel(bm25, inl2)
    sig.write(str(p_val))
    sig.close()
    """


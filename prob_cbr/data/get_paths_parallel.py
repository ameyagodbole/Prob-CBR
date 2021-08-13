from tqdm import tqdm
from collections import defaultdict
import os
from prob_cbr.data.data_utils import get_unique_entities, create_adj_list, get_inv_relation
import time
import pickle
import argparse
import json
import wandb
import logging
import sys
from numpy.random import default_rng

rng = default_rng()

logger = logging.getLogger('get_paths')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s \t %(message)s]",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_paths(args, train_adj_list, start_node, max_len=3):
    """
    :param start_node:
    :param K:
    :param max_len:
    :return:
    """
    all_paths = set()
    for k in range(args.num_paths_to_collect):
        path = []
        curr_node = start_node
        entities_on_path = set([start_node])
        for l in range(max_len):
            outgoing_edges = train_adj_list[curr_node]
            if args.prevent_loops:
                # Prevent loops
                temp = []
                for oe in outgoing_edges:
                    if oe[1] in entities_on_path:
                        continue
                    else:
                        temp.append(oe)
                outgoing_edges = temp
            if len(outgoing_edges) == 0:
                break
            # pick one at random
            out_edge_idx = rng.integers(0, len(outgoing_edges), size=1)
            out_edge = outgoing_edges[out_edge_idx[0]]

            path.append(out_edge)
            curr_node = out_edge[1]  # assign curr_node as the node of the selected edge
            entities_on_path.add(out_edge[1])
        all_paths.add(tuple(path))

    return all_paths


def main(args):
    logger.info("============={}================".format(args.dataset_name))
    data_dir = os.path.join(args.data_dir, "data", args.dataset_name)
    out_dir = os.path.join(args.data_dir, "subgraphs", args.dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.prevent_loops = (args.prevent_loops == 1)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    kg_file = os.path.join(data_dir, "graph.txt")
    if args.get_unique_nodes:
        unique_entities = get_unique_entities(kg_file)
        args.unique_nodes_file = os.path.join(data_dir, "unique_entities.pkl")
        logger.info("Writing unique nodes in the KG to {}".format(args.unique_nodes_file))
        with open(args.unique_nodes_file, "wb") as fout:
            pickle.dump(unique_entities, fout)
        sys.exit(0)
    else:
        if args.unique_nodes_file is None:
            args.unique_nodes_file = os.path.join(data_dir, "unique_entities.pkl")
        with open(args.unique_nodes_file, "rb") as fin:
            unique_entities = pickle.load(fin)

    assert unique_entities is not None
    num_entities = len(unique_entities)
    num_entities_in_partition = num_entities / args.total_jobs
    st = args.job_id * num_entities_in_partition
    en = min(st + num_entities_in_partition, num_entities)
    logger.info("Starting a job with st ind {} and end ind {}".format(st, en))
    train_adj_list = create_adj_list(kg_file, args.add_inv_edges)
    st_time = time.time()
    paths_map = defaultdict(list)
    for ctr, e1 in enumerate(tqdm(unique_entities)):
        if st <= ctr < en:
            paths = get_paths(args, train_adj_list, e1, args.max_len)
            if paths is None:
                continue
            paths_map[e1] = paths
            if args.use_wandb and (ctr-st) % 100 == 0:
                wandb.log({"progress": (ctr-st) / num_entities_in_partition})

    logger.info("Took {} seconds to collect paths for {} entities".format(time.time() - st_time, len(paths_map)))

    out_file_name = "paths_" + str(args.num_paths_to_collect) + "_path_len_" + str(args.max_len) + "_"+ str(args.job_id)
    if args.prevent_loops:
        out_file_name += "_no_loops"
    out_file_name += ".pkl"
    fout = open(os.path.join(out_dir, out_file_name), "wb")
    logger.info("Saving at {}".format(os.path.join(out_dir, out_file_name)))
    pickle.dump(paths_map, fout)
    fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect subgraphs around entities")
    parser.add_argument("--dataset_name", type=str, default="nell")
    parser.add_argument("--data_dir", type=str, default="/mnt/nfs/scratch1/rajarshi/deep_case_based_reasoning/")
    parser.add_argument("--num_paths_to_collect", type=int, default=1000)
    parser.add_argument("--max_len", type=int, default=3)
    parser.add_argument("--prevent_loops", type=int, choices=[0, 1], default=1, help="prevent sampling of looped paths")
    parser.add_argument("--add_inv_edges", action="store_true")
    parser.add_argument("--use_wandb", type=int, choices=[0, 1], default=0, help="Set to 1 if using W&B")
    parser.add_argument("--get_unique_nodes", action="store_true")
    parser.add_argument("--unique_nodes_file", type=str,
                        help="File containing the list of unique entities, run the program with --get_unique_nodes to generate the file for the first time")
    parser.add_argument("--total_jobs", type=int, default=10, help="total_number of jobs to run parallely")
    parser.add_argument("--job_id", type=int, default=0, help="a job num in [0, total_jobs)")

    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(project='collect-paths')

    main(args)

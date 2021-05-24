import argparse
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
import pickle
import torch
import uuid
from prob_cbr.data.data_utils import create_vocab, load_data, get_unique_entities, \
    read_graph, get_entities_group_by_relation, get_inv_relation, load_data_all_triples, create_adj_list
from prob_cbr.data.get_paths import get_paths
from prob_cbr.clustering.entity_clustering import cluster_entities
from typing import *
import logging
import json
import sys
import wandb

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s \t %(message)s]",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


class ProbCBR(object):
    def __init__(self, args, train_map, eval_map, entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, eval_vocab,
                 eval_rev_vocab, all_paths, rel_ent_map):
        self.args = args
        self.eval_map = eval_map
        self.train_map = train_map
        self.all_zero_ctr = []
        self.all_num_ret_nn = []
        self.entity_vocab, self.rev_entity_vocab, self.rel_vocab, self.rev_rel_vocab = entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab
        self.eval_vocab, self.eval_rev_vocab = eval_vocab, eval_rev_vocab
        self.all_paths = all_paths
        self.rel_ent_map = rel_ent_map
        self.num_non_executable_programs = []
        self.nearest_neighbor_1_hop = None

    def set_nearest_neighbor_1_hop(self, nearest_neighbor_1_hop):
        self.nearest_neighbor_1_hop = nearest_neighbor_1_hop

    @staticmethod
    def calc_sim(adj_mat: torch.Tensor, query_entities: torch.LongTensor) -> torch.Tensor:
        """
        :param adj_mat: N X R
        :param query_entities: b is a batch of indices of query entities
        :return:
        """
        query_entities_vec = torch.index_select(adj_mat, dim=0, index=query_entities)
        sim = torch.matmul(query_entities_vec, torch.t(adj_mat))
        return sim

    def get_nearest_neighbor_inner_product(self, e1: str, r: str, k: Optional[int] = 5) -> Union[List[str], None]:
        try:
            nearest_entities = [self.rev_entity_vocab[e] for e in
                                self.nearest_neighbor_1_hop[self.eval_vocab[e1]].tolist()]
            # remove e1 from the set of k-nearest neighbors if it is there.
            nearest_entities = [nn for nn in nearest_entities if nn != e1]
            # making sure, that the similar entities also have the query relation
            ctr = 0
            temp = []
            for nn in nearest_entities:
                if ctr == k:
                    break
                if len(self.train_map[nn, r]) > 0:
                    temp.append(nn)
                    ctr += 1
            nearest_entities = temp
        except KeyError:
            return None
        return nearest_entities

    def get_nearest_neighbor_naive(self, e1: str, r: str, k: Optional[int] = 5) -> List[str]:
        """
        Return entities which have the query relation
        :param e1:
        :param r:
        :param k:
        :return:
        """
        entities_with_r = self.rel_ent_map[r]
        # choose randomly from this set
        # pick (k+1), if e1 is there then remove it otherwise return first k
        nearest_entities = np.random.choice(entities_with_r, k + 1)
        if e1 in nearest_entities:
            nearest_entities = [i for i in nearest_entities if i != e1]
        else:
            nearest_entities = nearest_entities[:k]

        return nearest_entities

    @staticmethod
    def get_programs(e: str, ans: str, all_paths_around_e: List[List[str]]):
        """
        Given an entity and answer, get all paths which end at that ans node in the subgraph surrounding e
        """
        all_programs = []
        for path in all_paths_around_e:
            for l, (r, e_dash) in enumerate(path):
                if e_dash == ans:
                    # get the path till this point
                    all_programs.append([x for (x, _) in path[:l + 1]])  # we only need to keep the relations
        return all_programs

    def get_programs_from_nearest_neighbors(self, e1: str, r: str, nn_func: Callable, num_nn: Optional[int] = 5):
        all_programs = []
        nearest_entities = nn_func(e1, r, k=num_nn)
        if nearest_entities is None:
            self.all_num_ret_nn.append(0)
            return None
        self.all_num_ret_nn.append(len(nearest_entities))
        zero_ctr = 0
        for e in nearest_entities:
            if len(self.train_map[(e, r)]) > 0:
                paths_e = self.all_paths[e]  # get the collected 3 hop paths around e
                nn_answers = self.train_map[(e, r)]
                for nn_ans in nn_answers:
                    all_programs += self.get_programs(e, nn_ans, paths_e)
            elif len(self.train_map[(e, r)]) == 0:
                zero_ctr += 1
        self.all_zero_ctr.append(zero_ctr)
        return all_programs

    def rank_programs(self, list_programs: List[List[str]], r: str) -> List[List[str]]:
        """
        Rank programs.
        """
        # sort it by the path score
        unique_programs = set()
        for p in list_programs:
            unique_programs.add(tuple(p))
        # now get the score of each path
        path_and_scores = []
        for p in unique_programs:
            try:
                path_and_scores.append((p, self.args.path_prior_map_per_relation[self.c][r][p] * self.args.precision_map[self.c][r][p]))
            except KeyError:
                # TODO: Fix key error
                if len(p) == 1 and p[0] == r:
                    continue  # ignore query relation
                else:
                    # use the fall back score
                    try:
                        c = 0
                        score = self.args.path_prior_map_per_relation_fallback[c][r][p] * self.args.precision_map_fallback[c][r][p]
                        path_and_scores.append((p, score))
                    except KeyError:
                        # still a path or rel is missing.
                        path_and_scores.append((p, 0))

        # sort wrt counts
        sorted_programs = [k for k, v in sorted(path_and_scores, key=lambda item: -item[1])]

        return sorted_programs

    def execute_one_program(self, e: str, path: List[str], depth: int, max_branch: int):
        """
        starts from an entity and executes the path by doing depth first search. If there are multiple edges with the same label, we consider
        max_branch number.
        """
        if depth == len(path):
            # reached end, return node
            return [e]
        next_rel = path[depth]
        next_entities = self.train_map[(e, path[depth])]
        # next_entities = list(set(self.train_map[(e, path[depth])] + self.args.rotate_edges[(e, path[depth])][:5]))
        if len(next_entities) == 0:
            # edge not present
            return []
        if len(next_entities) > max_branch:
            # select max_branch random entities
            next_entities = np.random.choice(next_entities, max_branch, replace=False).tolist()
        answers = []
        for e_next in next_entities:
            answers += self.execute_one_program(e_next, path, depth + 1, max_branch)
        return answers

    def execute_programs(self, e: str, r: str, path_list: List[List[str]], max_branch: Optional[int] = 1000) \
            -> Tuple[List[Tuple[str, float, List[str]]], List[List[str]]]:

        def _fall_back(r, p):
            """
            When a cluster does not have a query relation (because it was not seen during counting)
            or if a path is not found, then fall back to no cluster statistics
            :param r:
            :param p:
            :return:
            """
            c = 0  # one cluster for all entity
            try:
                score = self.args.path_prior_map_per_relation_fallback[c][r][p] * self.args.precision_map_fallback[c][r][p]
            except KeyError:
                # either the path or relation is missing from the fall back map as well
                score = 0
            return score

        all_answers = []
        not_executed_paths = []
        execution_fail_counter = 0
        executed_path_counter = 0
        for path in path_list:
            if executed_path_counter == self.args.max_num_programs:
                break
            ans = self.execute_one_program(e, path, depth=0, max_branch=max_branch)
            temp = []
            if self.args.use_path_counts:
                try:
                    if path in self.args.path_prior_map_per_relation[self.c][r] and path in \
                            self.args.precision_map[self.c][r]:
                        path_score = self.args.path_prior_map_per_relation[self.c][r][path] * self.args.precision_map[self.c][r][path]
                    else:
                        # logger.info("This path was not there in the cluster for the relation.")
                        path_score = _fall_back(r, path)
                except KeyError:
                    # logger.info("Looks like the relation was not found in the cluster, have to fall back")
                    # fallback to the global scores
                    path_score = _fall_back(r, path)
            else:
                path_score = 1
            for a in ans:
                path = tuple(path)
                temp.append((a, path_score, path))
            ans = temp
            if ans == []:
                not_executed_paths.append(path)
                execution_fail_counter += 1
            else:
                executed_path_counter += 1
            all_answers += ans
        self.num_non_executable_programs.append(execution_fail_counter)
        return all_answers, not_executed_paths

    @staticmethod
    def rank_answers(list_answers: List[Tuple[str, float, List[str]]]) -> List[str]:
        """
        Different ways to re-rank answers
        """
        count_map = {}
        uniq_entities = set()
        for e, e_score, path in list_answers:
            if e not in count_map:
                count_map[e] = {}
            if path not in count_map[e]:
                count_map[e][path] = e_score  # just count once for a path type.
            uniq_entities.add(e)
        score_map = defaultdict(int)
        for e, path_scores_map in count_map.items():
            sum_path_score = 0
            for p, p_score in path_scores_map.items():
                sum_path_score += p_score
            score_map[e] = sum_path_score

        sorted_entities_by_val = sorted(score_map.items(), key=lambda kv: -kv[1])
        return sorted_entities_by_val

    @staticmethod
    def get_rank_in_list(e, predicted_answers):
        for i, e_to_check in enumerate(predicted_answers):
            if e == e_to_check:
                return i + 1
        return -1

    def get_hits(self, list_answers: List[str], gold_answers: List[str], query: Tuple[str, str]) \
            -> Tuple[float, float, float, float, float]:
        hits_1 = 0.0
        hits_3 = 0.0
        hits_5 = 0.0
        hits_10 = 0.0
        rr = 0.0
        (e1, r) = query
        all_gold_answers = self.args.all_kg_map[(e1, r)]
        for gold_answer in gold_answers:
            # remove all other gold answers from prediction
            filtered_answers = []
            for pred in list_answers:
                if pred in all_gold_answers and pred != gold_answer:
                    continue
                else:
                    filtered_answers.append(pred)

            rank = ProbCBR.get_rank_in_list(gold_answer, filtered_answers)
            if rank > 0:
                if rank <= 10:
                    hits_10 += 1
                    if rank <= 5:
                        hits_5 += 1
                        if rank <= 3:
                            hits_3 += 1
                            if rank <= 1:
                                hits_1 += 1
                rr += 1.0 / rank
        return hits_10, hits_5, hits_3, hits_1, rr

    @staticmethod
    def get_accuracy(gold_answers: List[str], list_answers: List[str]) -> List[float]:
        all_acc = []
        for gold_ans in gold_answers:
            if gold_ans in list_answers:
                all_acc.append(1.0)
            else:
                all_acc.append(0.0)
        return all_acc

    def do_symbolic_case_based_reasoning(self):
        num_programs = []
        num_answers = []
        all_acc = []
        non_zero_ctr = 0
        hits_10, hits_5, hits_3, hits_1, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        per_relation_scores = {}  # map of performance per relation
        per_relation_query_count = {}
        total_examples = 0
        learnt_programs = defaultdict(lambda: defaultdict(int))  # for each query relation, a map of programs to count
        for ex_ctr, ((e1, r), e2_list) in enumerate(tqdm(self.eval_map.items())):
            # if e2_list is in train list then remove them
            # Normally, this shouldnt happen at all, but this happens for Nell-995.
            orig_train_e2_list = self.train_map[(e1, r)]
            temp_train_e2_list = []
            for e2 in orig_train_e2_list:
                if e2 in e2_list:
                    continue
                temp_train_e2_list.append(e2)
            self.train_map[(e1, r)] = temp_train_e2_list
            # also remove (e2, r^-1, e1)
            r_inv = get_inv_relation(r, args.dataset_name)
            temp_map = {}  # map from (e2, r_inv) -> outgoing nodes
            for e2 in e2_list:
                temp_map[(e2, r_inv)] = self.train_map[e2, r_inv]
                temp_list = []
                for e1_dash in self.train_map[e2, r_inv]:
                    if e1_dash == e1:
                        continue
                    else:
                        temp_list.append(e1_dash)
                self.train_map[e2, r_inv] = temp_list

            total_examples += len(e2_list)
            if e1 not in self.entity_vocab:
                all_acc += [0.0] * len(e2_list)
                # put it back
                self.train_map[(e1, r)] = orig_train_e2_list
                for e2 in e2_list:
                    self.train_map[(e2, r_inv)] = temp_map[(e2, r_inv)]
                continue  # this entity was not seen during train; skip?

            self.c = self.args.cluster_assignments[self.entity_vocab[e1]]
            all_programs = self.get_programs_from_nearest_neighbors(e1, r, self.get_nearest_neighbor_inner_product,
                                                                    num_nn=self.args.k_adj)
            if all_programs is None or len(all_programs) == 0:
                all_acc += [0.0] * len(e2_list)
                # put it back
                self.train_map[(e1, r)] = orig_train_e2_list
                for e2 in e2_list:
                    self.train_map[(e2, r_inv)] = temp_map[(e2, r_inv)]
                continue
            for p in all_programs:
                if p[0] == r:
                    continue
                if r not in learnt_programs:
                    learnt_programs[r] = {}
                p = tuple(p)
                if p not in learnt_programs[r]:
                    learnt_programs[r][p] = 0
                learnt_programs[r][p] += 1

            # filter the program if it is equal to the query relation
            temp = []
            for p in all_programs:
                if len(p) == 1 and p[0] == r:
                    continue
                temp.append(p)
            all_programs = temp

            if len(all_programs) > 0:
                non_zero_ctr += len(e2_list)

            all_uniq_programs = self.rank_programs(all_programs, r)

            num_programs.append(len(all_uniq_programs))
            # Now execute the program
            answers, not_executed_programs = self.execute_programs(e1, r, all_uniq_programs)
            # if len(not_executed_programs) > 0:
            #     import pdb
            #     pdb.set_trace()

            answers = self.rank_answers(answers)
            if len(answers) > 0:
                acc = self.get_accuracy(e2_list, [k[0] for k in answers])
                _10, _5, _3, _1, rr = self.get_hits([k[0] for k in answers], e2_list, query=(e1, r))
                hits_10 += _10
                hits_5 += _5
                hits_3 += _3
                hits_1 += _1
                mrr += rr
                if args.output_per_relation_scores:
                    if r not in per_relation_scores:
                        per_relation_scores[r] = {"hits_1": 0, "hits_3": 0, "hits_5": 0, "hits_10": 0, "mrr": 0}
                        per_relation_query_count[r] = 0
                    per_relation_scores[r]["hits_1"] += _1
                    per_relation_scores[r]["hits_3"] += _3
                    per_relation_scores[r]["hits_5"] += _5
                    per_relation_scores[r]["hits_10"] += _10
                    per_relation_scores[r]["mrr"] += rr
                    per_relation_query_count[r] += len(e2_list)
            else:
                acc = [0.0] * len(e2_list)
            all_acc += acc
            num_answers.append(len(answers))
            # put it back
            self.train_map[(e1, r)] = orig_train_e2_list
            for e2 in e2_list:
                self.train_map[(e2, r_inv)] = temp_map[(e2, r_inv)]

        if args.output_per_relation_scores:
            for r, r_scores in per_relation_scores.items():
                r_scores["hits_1"] /= per_relation_query_count[r]
                r_scores["hits_3"] /= per_relation_query_count[r]
                r_scores["hits_5"] /= per_relation_query_count[r]
                r_scores["hits_10"] /= per_relation_query_count[r]
                r_scores["mrr"] /= per_relation_query_count[r]
            out_file_name = os.path.join(args.output_dir, "per_relation_scores.json")
            fout = open(out_file_name, "w")
            logger.info("Writing per-relation scores to {}".format(out_file_name))
            fout.write(json.dumps(per_relation_scores, sort_keys=True, indent=4))
            fout.close()

        logger.info(
            "Out of {} queries, atleast one program was returned for {} queries".format(total_examples, non_zero_ctr))
        logger.info("Avg number of programs {:3.2f}".format(np.mean(num_programs)))
        logger.info("Avg number of answers after executing the programs: {}".format(np.mean(num_answers)))
        logger.info("Accuracy (Loose): {}".format(np.mean(all_acc)))
        logger.info("Hits@1 {}".format(hits_1 / total_examples))
        logger.info("Hits@3 {}".format(hits_3 / total_examples))
        logger.info("Hits@5 {}".format(hits_5 / total_examples))
        logger.info("Hits@10 {}".format(hits_10 / total_examples))
        logger.info("MRR {}".format(mrr / total_examples))
        logger.info("Avg number of nn, that do not have the query relation: {}".format(
            np.mean(self.all_zero_ctr)))
        logger.info("Avg num of returned nearest neighbors: {:2.4f}".format(np.mean(self.all_num_ret_nn)))
        logger.info("Avg number of programs that do not execute per query: {:2.4f}".format(
            np.mean(self.num_non_executable_programs)))
        if self.args.print_paths:
            for k, v in learnt_programs.items():
                logger.info("query: {}".format(k))
                logger.info("=====" * 2)
                for rel, _ in learnt_programs[k].items():
                    logger.info((rel, learnt_programs[k][rel]))
                logger.info("=====" * 2)
        if self.args.use_wandb:
            # Log all metrics
            wandb.log({'hits_1': hits_1 / total_examples, 'hits_3': hits_3 / total_examples,
                       'hits_5': hits_5 / total_examples, 'hits_10': hits_10 / total_examples,
                       'mrr': mrr / total_examples, 'total_examples': total_examples, 'non_zero_ctr': non_zero_ctr,
                       'all_zero_ctr': self.all_zero_ctr, 'avg_num_nn': np.mean(self.all_num_ret_nn),
                       'avg_num_prog': np.mean(num_programs), 'avg_num_ans': np.mean(num_answers),
                       'avg_num_failed_prog': np.mean(self.num_non_executable_programs), 'acc_loose': np.mean(all_acc)})

    def calc_precision_map(self, output_filenm=""):
        """
        Calculates precision of each path wrt a query relation, i.e. ratio of how many times, a path was successful when executed
        to how many times the path was executed.
        Note: In the current implementation, we compute precisions for the paths stored in the path_prior_map
        :return:
        """
        logger.info("Calculating precision map")
        success_map, total_map = {}, {}  # map from query r to a dict of path and ratio of success
        # not sure why I am getting RuntimeError: dictionary changed size during iteration.
        train_map = [((e1, r), e2_list) for ((e1, r), e2_list) in self.train_map.items()]
        for ((e1, r), e2_list) in tqdm(train_map):
            c = self.args.cluster_assignments[self.entity_vocab[e1]]
            if c not in success_map:
                success_map[c] = {}
            if c not in total_map:
                total_map[c] = {}
            if r not in success_map[c]:
                success_map[c][r] = {}
            if r not in total_map[c]:
                total_map[c][r] = {}
            paths_for_this_relation = self.args.path_prior_map_per_relation[c][r]
            for p_ctr, (path, _) in enumerate(paths_for_this_relation.items()):
                ans = self.execute_one_program(e1, path, depth=0, max_branch=100)
                if len(ans) == 0:
                    continue
                # execute the path get answer
                if path not in success_map[c][r]:
                    success_map[c][r][path] = 0
                if path not in total_map[c][r]:
                    total_map[c][r][path] = 0
                for a in ans:
                    if a in e2_list:
                        success_map[c][r][path] += 1
                    total_map[c][r][path] += 1

        precision_map = {}
        for c, _ in success_map.items():
            for r, _ in success_map[c].items():
                if c not in precision_map:
                    precision_map[c] = {}
                if r not in precision_map[c]:
                    precision_map[c][r] = {}
                for path, s_c in success_map[c][r].items():
                    precision_map[c][r][path] = s_c / total_map[c][r][path]

        if not output_filenm:
            dir_name = os.path.join(args.data_dir, "data", self.args.dataset_name, "linkage={}".format(self.args.linkage))
            output_filenm = os.path.join(dir_name, "precision_map.pkl")
        logger.info("Dumping precision map at {}".format(output_filenm))
        with open(output_filenm, "wb") as fout:
            pickle.dump(precision_map, fout)
        logger.info("Done...")

    def calc_prior_path_prob(self, output_filenm=""):
        """
        Calculate how probable a path is given a query relation, i.e P(path|query rel)
        For each entity in the graph, count paths that exists for each relation in the
        random subgraph.
        :return:
        """
        logger.info("Calculating prior map")
        programs_map = {}
        unique_cluster_ids = set()  # have to do this since the assigned cluster ids doesnt seems to be contiguous or start from 0 or end at K-1
        for c in self.args.cluster_assignments:
            unique_cluster_ids.add(c)
        for c in unique_cluster_ids:
            for _, ((e1, r), e2_list) in enumerate(tqdm((self.train_map.items()))):
                if self.args.cluster_assignments[self.entity_vocab[e1]] != c:
                    # if this entity does not belong to this cluster, don't consider.
                    continue
                if c not in programs_map:
                    programs_map[c] = {}
                if r not in programs_map[c]:
                    programs_map[c][r] = {}
                all_paths_around_e1 = self.all_paths[e1]
                nn_answers = e2_list
                for nn_ans in nn_answers:
                    programs = self.get_programs(e1, nn_ans, all_paths_around_e1)
                    for p in programs:
                        p = tuple(p)
                        if len(p) == 1:
                            if p[0] == r:  # don't store query relation
                                continue
                        if p not in programs_map[c][r]:
                            programs_map[c][r][p] = 0
                        programs_map[c][r][p] += 1
        for c, r in programs_map.items():
            for r, path_counts in programs_map[c].items():
                sum_path_counts = 0
                for p, p_c in path_counts.items():
                    sum_path_counts += p_c
                for p, p_c in path_counts.items():
                    programs_map[c][r][p] = p_c / sum_path_counts

        if not output_filenm:
            dir_name = os.path.join(args.data_dir, "data", self.args.dataset_name, "linkage={}".format(self.args.linkage))
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            output_filenm = os.path.join(dir_name, "path_prior_map.pkl")

        logger.info("Dumping path prior pickle at {}".format(output_filenm))
        with open(output_filenm, "wb") as fout:
            pickle.dump(programs_map, fout)


def main(args):
    dataset_name = args.dataset_name
    logger.info("==========={}============".format(dataset_name))
    data_dir = os.path.join(args.data_dir, "data", dataset_name)
    subgraph_dir = os.path.join(args.data_dir, "subgraphs", dataset_name)
    kg_file = os.path.join(data_dir, "full_graph.txt") if dataset_name == "nell" else os.path.join(data_dir,
                                                                                                   "graph.txt")

    if args.small:
        args.dev_file = os.path.join(data_dir, "dev.txt.small")
        args.test_file = os.path.join(data_dir, "test.txt")
    else:
        args.dev_file = os.path.join(data_dir, "dev.txt")
        args.test_file = os.path.join(data_dir, "test.txt") if not args.test_file_name \
            else os.path.join(data_dir, args.test_file_name)

    args.train_file = os.path.join(data_dir, "graph.txt") if dataset_name == "nell" else os.path.join(data_dir,
                                                                                                      "train.txt")

    if args.subgraph_file_name is "":
        args.subgraph_file_name = f"paths_{args.num_paths_to_collect}_{args.max_path_len}hop"
        if args.prevent_loops:
            args.subgraph_file_name += "_no_loops"
        args.subgraph_file_name += ".pkl"

    if os.path.exists(os.path.join(subgraph_dir, args.subgraph_file_name)):
        logger.info("Loading subgraph around entities:")
        with open(os.path.join(subgraph_dir, args.subgraph_file_name), "rb") as fin:
            all_paths = pickle.load(fin)
    else:
        logger.info("Sampling subgraph around entities:")
        unique_entities = get_unique_entities(kg_file)
        train_adj_list = create_adj_list(kg_file)
        all_paths = defaultdict(list)
        for ctr, e1 in enumerate(tqdm(unique_entities)):
            paths = get_paths(args, train_adj_list, e1, max_len=args.max_path_len)
            if paths is None:
                continue
            all_paths[e1] = paths
        os.makedirs(subgraph_dir, exist_ok=True)
        with open(os.path.join(subgraph_dir, args.subgraph_file_name), "wb") as fout:
            pickle.dump(all_paths, fout)

    entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab = create_vocab(kg_file)
    logger.info("Loading train map")
    train_map = load_data(kg_file)
    logger.info("Loading dev map")
    dev_map = load_data(args.dev_file)
    logger.info("Loading test map")
    test_map = load_data(args.test_file)
    eval_map = dev_map
    eval_file = args.dev_file
    if args.test:
        eval_map = test_map
        eval_file = args.test_file

    rel_ent_map = get_entities_group_by_relation(args.train_file)
    # Calculate nearest neighbors
    adj_mat = read_graph(kg_file, entity_vocab, rel_vocab)
    adj_mat = np.sqrt(adj_mat)
    l2norm = np.linalg.norm(adj_mat, axis=-1)
    l2norm[0] += np.finfo(np.float).eps  # to encounter zero values. These 2 indx are PAD / NULL
    l2norm[1] += np.finfo(np.float).eps
    adj_mat = adj_mat / l2norm.reshape(l2norm.shape[0], 1)

    # Lets put this to GPU
    adj_mat = torch.from_numpy(adj_mat)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    logger.info('Using device:'.format(device.__str__()))
    adj_mat = adj_mat.to(device)

    # get the unique entities in eval set, so that we can calculate similarity in advance.
    eval_entities = get_unique_entities(eval_file)
    eval_vocab, eval_rev_vocab = {}, {}
    query_ind = []

    e_ctr = 0
    for e in eval_entities:
        try:
            query_ind.append(entity_vocab[e])
        except KeyError:
            continue
        eval_vocab[e] = e_ctr
        eval_rev_vocab[e_ctr] = e
        e_ctr += 1

    logger.info("=========Config:============")
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    logger.info("Loading combined train/dev/test map for filtered eval")
    all_kg_map = load_data_all_triples(args.train_file, args.dev_file, os.path.join(data_dir, 'test.txt'))
    args.all_kg_map = all_kg_map

    prob_cbr_agent = ProbCBR(args, train_map, eval_map, entity_vocab, rev_entity_vocab, rel_vocab,
                             rev_rel_vocab, eval_vocab, eval_rev_vocab, all_paths, rel_ent_map)

    logger.info("Calculating distance matrix")
    query_ind = torch.LongTensor(query_ind).to(device)
    # Calculate similarity
    sim = prob_cbr_agent.calc_sim(adj_mat, query_ind)  # n X N (n== size of dev_entities, N: size of all entities)

    nearest_neighbor_1_hop = np.argsort(-sim.cpu(), axis=-1)
    prob_cbr_agent.set_nearest_neighbor_1_hop(nearest_neighbor_1_hop)

    # cluster entities

    if args.linkage > 0:
        if os.path.exists(os.path.join(data_dir, "linkage={}".format(args.linkage), "cluster_assignments.pkl")):
            logger.info("Clustering with linkage {} found, loading them....".format(args.linkage))
            fin = open(os.path.join(data_dir, "linkage={}".format(args.linkage), "cluster_assignments.pkl"), "rb")
            args.cluster_assignments = pickle.load(fin)
            fin.close()
        else:
            logger.info("Clustering entities with linkage = {}...".format(args.linkage))
            args.cluster_assignments = cluster_entities(adj_mat, args.linkage)
            logger.info("There are {} unique clusters".format(np.unique(args.cluster_assignments).shape[0]))
            dir_name = os.path.join(data_dir, "linkage={}".format(args.linkage))
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            logger.info("Dumping cluster assignments of entities at {}".format(dir_name))
            fout = open(os.path.join(dir_name, "cluster_assignments.pkl"), "wb")
            pickle.dump(args.cluster_assignments, fout)
            fout.close()

    path_prior_map_filenm = os.path.join(data_dir, "linkage={}".format(args.linkage), "path_prior_map.pkl")
    if not os.path.exists(path_prior_map_filenm):
        prob_cbr_agent.calc_prior_path_prob(output_filenm=path_prior_map_filenm)
    logger.info("Loading path prior weights")
    with open(path_prior_map_filenm, "rb") as fin:
        args.path_prior_map_per_relation = pickle.load(fin)

    linkage_bck = args.linkage
    args.linkage = 0.0

    bck_dir_name = os.path.join(data_dir, "linkage={}".format(args.linkage))
    if not os.path.exists(bck_dir_name):
        os.makedirs(bck_dir_name)

    cluster_assignments_bck = args.cluster_assignments
    args.cluster_assignments = np.zeros_like(args.cluster_assignments)
    path_prior_map_filenm = os.path.join(data_dir, "linkage={}".format(args.linkage), "path_prior_map.pkl")
    if not os.path.exists(path_prior_map_filenm):
        prob_cbr_agent.calc_prior_path_prob(output_filenm=path_prior_map_filenm)
    logger.info("Loading fall-back path prior weights")
    with open(path_prior_map_filenm, "rb") as fin:
        args.path_prior_map_per_relation_fallback = pickle.load(fin)
    args.linkage = linkage_bck
    args.cluster_assignments = cluster_assignments_bck

    precision_map_filenm = os.path.join(data_dir, "linkage={}".format(args.linkage), "precision_map.pkl")
    if not os.path.exists(precision_map_filenm):
        prob_cbr_agent.calc_precision_map(output_filenm=precision_map_filenm)
    logger.info("Loading precision map")
    with open(precision_map_filenm, "rb") as fin:
        args.precision_map = pickle.load(fin)

    linkage_bck = args.linkage
    args.linkage = 0.0
    path_prior_map_per_relation_bck = args.path_prior_map_per_relation
    args.path_prior_map_per_relation = args.path_prior_map_per_relation_fallback
    cluster_assignments_bck = args.cluster_assignments
    args.cluster_assignments = np.zeros_like(args.cluster_assignments)
    precision_map_filenm = os.path.join(data_dir, "linkage={}".format(args.linkage), "precision_map.pkl")
    if not os.path.exists(precision_map_filenm):
        prob_cbr_agent.calc_precision_map(output_filenm=precision_map_filenm)
    logger.info("Loading fall-back precision map")
    with open(precision_map_filenm, "rb") as fin:
        args.precision_map_fallback = pickle.load(fin)
    args.linkage = linkage_bck
    args.path_prior_map_per_relation = path_prior_map_per_relation_bck
    args.cluster_assignments = cluster_assignments_bck

    if not args.only_preprocess:
        prob_cbr_agent.do_symbolic_case_based_reasoning()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect subgraphs around entities")
    parser.add_argument("--dataset_name", type=str, default="nell")
    parser.add_argument("--data_dir", type=str, default="../prob_cbr_data/")
    parser.add_argument("--expt_dir", type=str, default="../prob_cbr_expts/")
    parser.add_argument("--subgraph_file_name", type=str, default="")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_file_name", type=str, default='',
                        help="Useful to switch between test files for FB122")
    parser.add_argument("--use_path_counts", type=int, choices=[0, 1], default=1,
                        help="Set to 1 if want to weight paths during ranking")
    parser.add_argument("--only_preprocess", action="store_true",
                        help="If on, only calculate prior and precision maps")
    # Clustering args
    parser.add_argument("--linkage", type=float, default=0.7,
                        help="Clustering threshold")
    # CBR args
    parser.add_argument("--k_adj", type=int, default=5,
                        help="Number of nearest neighbors to consider based on adjacency matrix")
    parser.add_argument("--max_num_programs", type=int, default=1000)
    # Output modifier args
    parser.add_argument("--name_of_run", type=str, default="unset")
    parser.add_argument("--output_per_relation_scores", action="store_true")
    parser.add_argument("--print_paths", action="store_true")
    parser.add_argument("--use_wandb", type=int, choices=[0, 1], default=0, help="Set to 1 if using W&B")
    # Path sampling args
    parser.add_argument("--num_paths_to_collect", type=int, default=1000)
    parser.add_argument("--max_path_len", type=int, default=3)
    parser.add_argument("--prevent_loops", type=int, choices=[0, 1], default=1)

    args = parser.parse_args()
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    if args.use_wandb:
        wandb.init(project='pr-cbr')

    if args.name_of_run == "unset":
        args.name_of_run = str(uuid.uuid4())[:8]
    args.output_dir = os.path.join(args.expt_dir, "outputs", args.dataset_name, args.name_of_run)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info(f"Output directory: {args.output_dir}")

    args.use_path_counts = (args.use_path_counts == 1)

    main(args)

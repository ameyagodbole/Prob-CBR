import argparse
from collections import Counter
import numpy as np
import os
from tqdm import tqdm, trange
from collections import defaultdict
import pickle
import torch
import uuid
from prob_cbr.data.data_utils import load_data_from_triples, get_unique_entities_from_triples, \
    read_graph_from_triples, get_entities_group_by_relation_from_triples, get_inv_relation, create_adj_list_from_triples
from prob_cbr.data.stream_utils import KBStream
from prob_cbr.data.get_paths import get_paths
from prob_cbr.clustering.grinch_with_deletes import GrinchWithDeletes
from typing import *
import logging
import json
import sys
import wandb
import copy

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
        self.query_c = None
        self.nearest_neighbor_1_hop = None

    def set_nearest_neighbor_1_hop(self, nearest_neighbor_1_hop):
        self.nearest_neighbor_1_hop = nearest_neighbor_1_hop

    @staticmethod
    def calc_sim(adj_mat: Type[torch.Tensor], query_entities: Type[torch.LongTensor]) -> Type[torch.LongTensor]:
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

    def get_programs(self, e: str, ans: str, all_paths_around_e: List[List[str]]):
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

    def rank_programs(self, list_programs: List[str], r: str) -> List[str]:
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
                path_and_scores.append((p, self.args.path_prior_map_per_relation[self.query_c][r][p] *
                                        self.args.precision_map[self.query_c][r][p]))
            except KeyError:
                # TODO: Fix key error
                if len(p) == 1 and p[0] == r:
                    continue  # ignore query relation
                else:
                    # use the fall back score
                    try:
                        c = 0
                        score = self.args.path_prior_map_per_relation_fallback[c][r][p] * \
                                self.args.precision_map_fallback[c][r][p]
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

    def execute_programs(self, e: str, r: str, path_list: List[List[str]], max_branch: Optional[int] = 1000) -> List[
        str]:

        def _fall_back(r, p):
            """
            When a cluster does not have a query relation (because it was not seen during counting)
            or if a path is not found, then fall back to no cluster statistics
            :param r:
            :param p:
            :return:
            """
            score = 0
            c = 0  # one cluster for all entity
            try:
                score = self.args.path_prior_map_per_relation_fallback[c][r][p] * \
                        self.args.precision_map_fallback[c][r][p]
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
            for a in ans:
                path = tuple(path)
                if self.args.use_path_counts:
                    try:
                        if path in self.args.path_prior_map_per_relation[self.query_c][r] and path in \
                                self.args.precision_map[self.query_c][r]:
                            temp.append((a,
                                         self.args.path_prior_map_per_relation[self.query_c][r][path] *
                                         self.args.precision_map[self.query_c][r][path],
                                         path))
                        else:
                            # logger.info("This path was not there in the cluster for the relation.")
                            score = _fall_back(r, path)
                            temp.append((a, score, path))
                    except KeyError:
                        # logger.info("Looks like the relation was not found in the cluster, have to fall back")
                        # fallback to the global scores
                        score = _fall_back(r, path)
                        temp.append((a, score, path))
                else:
                    temp.append((a, 1, path))
            ans = temp
            if ans == []:
                not_executed_paths.append(path)
                execution_fail_counter += 1
            else:
                executed_path_counter += 1
            all_answers += ans
            # if len(all_answers) == 0:
            #     all_answers = set(ans)
            # else:
            #     all_answers = all_answers.intersection(set(ans))
        self.num_non_executable_programs.append(execution_fail_counter)
        return all_answers, not_executed_paths

    def rank_answers(self, list_answers: List[str]) -> List[str]:
        """
        Different ways to re-rank answers
        """
        count_map = {}
        uniq_entities = set()
        for e, e_score, path in list_answers:
            if e not in count_map:
                count_map[e] = {}
            if path not in count_map:
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
        rank = 0
        for i, e_to_check in enumerate(predicted_answers):
            if e == e_to_check:
                return i + 1
        return -1

    def get_hits(self, list_answers: List[str], gold_answers: List[str], query: Tuple[str, str]) -> Tuple[
        float, float, float, float]:
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
        for _, ((e1, r), e2_list) in enumerate(tqdm((self.eval_map.items()))):
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
            r_inv = get_inv_relation(r, self.args.dataset_name)
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
                continue

            self.query_c = self.args.cluster_assignments[self.entity_vocab[e1]]
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
                if self.args.output_per_relation_scores:
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

        if self.args.output_per_relation_scores:
            for r, r_scores in per_relation_scores.items():
                r_scores["hits_1"] /= per_relation_query_count[r]
                r_scores["hits_3"] /= per_relation_query_count[r]
                r_scores["hits_5"] /= per_relation_query_count[r]
                r_scores["hits_10"] /= per_relation_query_count[r]
                r_scores["mrr"] /= per_relation_query_count[r]
            out_file_name = os.path.join(self.args.output_dir, "per_relation_scores.json")
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

    def calc_precision_map(self):
        """
        Calculates precision of each path wrt a query relation, i.e. ratio of how many times, a path was successful when executed
        to how many times the path was executed.
        Note: In the current implementation, we compute precisions for the paths stored in the path_prior_map
        :return:
        """
        logger.info("Calculating precision map")
        success_map, total_map = {}, {}  # map from query r to a dict of path and ratio of success
        success_map_fallback, total_map_fallback = {0: {}}, {0: {}}
        # not sure why I am getting RuntimeError: dictionary changed size during iteration.
        train_map_list = [((e1, r), e2_list) for ((e1, r), e2_list) in self.train_map.items()]
        for ((e1, r), e2_list) in tqdm(train_map_list):
            c = self.args.cluster_assignments[self.entity_vocab[e1]]
            if c not in success_map:
                success_map[c] = {}
            if c not in total_map:
                total_map[c] = {}
            if r not in success_map[c]:
                success_map[c][r] = {}
            if r not in total_map[c]:
                total_map[c][r] = {}
            if r not in success_map_fallback[0]:
                success_map_fallback[0][r] = {}
            if r not in total_map_fallback[0]:
                total_map_fallback[0][r] = {}
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
                if path not in success_map_fallback[0][r]:
                    success_map_fallback[0][r][path] = 0
                if path not in total_map_fallback[0][r]:
                    total_map_fallback[0][r][path] = 0
                for a in ans:
                    if a in e2_list:
                        success_map[c][r][path] += 1
                        success_map_fallback[0][r][path] += 1
                    total_map[c][r][path] += 1
                    total_map_fallback[0][r][path] += 1

        precision_map = {}
        for c, _ in success_map.items():
            for r, _ in success_map[c].items():
                if c not in precision_map:
                    precision_map[c] = {}
                if r not in precision_map[c]:
                    precision_map[c][r] = {}
                for path, s_c in success_map[c][r].items():
                    precision_map[c][r][path] = s_c / total_map[c][r][path]

        precision_map_fallback = {0: {}}
        for r, _ in success_map_fallback[0].items():
            if r not in precision_map_fallback[0]:
                precision_map_fallback[0][r] = {}
            for path, s_c in success_map_fallback[0][r].items():
                precision_map_fallback[0][r][path] = s_c / total_map_fallback[0][r][path]

        return precision_map, precision_map_fallback

    def calc_per_entity_precision_components(self, per_entity_path_prior_count, entity_set=None):
        """
        Calculates precision of each path wrt a query relation, i.e. ratio of how many times, a path was successful when executed
        to how many times the path was executed.
        Note: In the current implementation, we compute precisions for the paths stored in the path_prior_map
        :return:
        """
        logger.info("Calculating precision map at entity level")
        success_map, total_map = {}, {}  # map from query r to a dict of path and ratio of success
        if entity_set is None:
            entity_set = set(self.entity_vocab.keys())
        # # not sure why I am getting RuntimeError: dictionary changed size during iteration.
        train_map_list = [((e1, r), e2_list) for ((e1, r), e2_list) in self.train_map.items()]
        for ((e1, r), e2_list) in tqdm(train_map_list):
            if len(e2_list) == 0:
                del self.train_map[(e1, r)]
                continue
            if e1 not in entity_set:
                continue
            if e1 not in success_map:
                success_map[e1] = {}
            if e1 not in total_map:
                total_map[e1] = {}
            if r not in success_map[e1]:
                success_map[e1][r] = {}
            if r not in total_map[e1]:
                total_map[e1][r] = {}
            paths_for_this_relation = per_entity_path_prior_count[e1][r]
            for p_ctr, (path, _) in enumerate(paths_for_this_relation.items()):
                ans = self.execute_one_program(e1, path, depth=0, max_branch=100)
                if len(ans) == 0:
                    continue
                # execute the path get answer
                if path not in success_map[e1][r]:
                    success_map[e1][r][path] = 0
                if path not in total_map[e1][r]:
                    total_map[e1][r][path] = 0
                for a in ans:
                    if a in e2_list:
                        success_map[e1][r][path] += 1
                    total_map[e1][r][path] += 1

        return success_map, total_map

    def get_precision_map_entity2cluster(self, per_entity_success_map, per_entity_total_map, cluster_assignments,
                                         path_prior_map_per_relation):
        """
        Calculates precision of each path wrt a query relation, i.e. ratio of how many times, a path was successful when
         executed to how many times the path was executed.
        Note: In the current implementation, we compute precisions for the paths stored in the path_prior_map
        :return:
        """
        logger.info("Calculating precision map for cluster from entity level map")
        success_map, total_map = {}, {}  # map from query r to a dict of path and ratio of success
        train_map_list = [((e1, r), e2_list) for ((e1, r), e2_list) in self.train_map.items()]
        _skip_ctr = 0
        for ((e1, r), e2_list) in tqdm(train_map_list):
            if len(e2_list) == 0:
                del self.train_map[(e1, r)]
                continue
            c = cluster_assignments[self.entity_vocab[e1]]
            if c not in path_prior_map_per_relation or r not in path_prior_map_per_relation[c]:
                _skip_ctr += 1
                continue
            if c not in success_map:
                success_map[c] = {}
            if c not in total_map:
                total_map[c] = {}
            if r not in success_map[c]:
                success_map[c][r] = {}
            if r not in total_map[c]:
                total_map[c][r] = {}
            paths_for_this_relation = path_prior_map_per_relation[c][r]
            for p_ctr, (path, _) in enumerate(paths_for_this_relation.items()):
                if e1 in per_entity_success_map and r in per_entity_success_map[e1] and \
                        path in per_entity_success_map[e1][r]:
                    if path not in success_map[c][r]:
                        success_map[c][r][path] = 0
                    if path not in total_map[c][r]:
                        total_map[c][r][path] = 0
                    success_map[c][r][path] += per_entity_success_map[e1][r][path]
                    total_map[c][r][path] += per_entity_total_map[e1][r][path]
                else:
                    ans = self.execute_one_program(e1, path, depth=0, max_branch=100)
                    if len(ans) == 0:
                        continue
                    # execute the path get answer
                    if path not in success_map[c][r]:
                        success_map[c][r][path] = 0
                    if path not in total_map[c][r]:
                        total_map[c][r][path] = 0
                    if e1 not in per_entity_success_map:
                        per_entity_success_map[e1] = {}
                    if r not in per_entity_success_map[e1]:
                        per_entity_success_map[e1][r] = {}
                    if path not in per_entity_success_map[e1][r]:
                        per_entity_success_map[e1][r][path] = 0
                    if e1 not in per_entity_total_map:
                        per_entity_total_map[e1] = {}
                    if r not in per_entity_total_map[e1]:
                        per_entity_total_map[e1][r] = {}
                    if path not in per_entity_total_map[e1][r]:
                        per_entity_total_map[e1][r][path] = 0
                    for a in ans:
                        if a in e2_list:
                            per_entity_success_map[e1][r][path] += 1
                            success_map[c][r][path] += 1
                        per_entity_total_map[e1][r][path] += 1
                        total_map[c][r][path] += 1
        logger.info(f'[get_precision_map_entity2cluster] {_skip_ctr} skips')

        precision_map = {}
        for c, _ in success_map.items():
            for r, _ in success_map[c].items():
                if c not in precision_map:
                    precision_map[c] = {}
                if r not in precision_map[c]:
                    precision_map[c][r] = {}
                for path, s_c in success_map[c][r].items():
                    precision_map[c][r][path] = s_c / total_map[c][r][path]

        return precision_map, success_map, total_map

    def update_precision_map_entity2cluster(self, per_entity_success_map, per_entity_total_map,
                                            per_entity_success_map_updates, per_entity_total_map_updates,
                                            per_cluster_success_map, per_cluster_total_map,
                                            per_cluster_success_map_fallback, per_cluster_total_map_fallback,
                                            cluster_adds, cluster_dels,
                                            path_prior_map_per_relation, path_prior_map_per_relation_fallback):
        logger.info("Updating prior map for cluster from entity level map")
        # e2old_cluster = dict([(e, c) for c, elist in cluster_dels.items() for e in elist])
        e2new_cluster = dict([(e, c) for c, elist in cluster_adds.items() for e in elist])

        # First delete from old cluster
        for c_old, elist in cluster_dels.items():
            for e1 in elist:
                c_new = e2new_cluster.get(e1, -1)
                assert c_new != -1
                if e1 not in per_entity_success_map:
                    assert e1 not in per_entity_total_map
                else:
                    for r, path_counts in per_entity_success_map[e1].items():
                        for path, p_c in path_counts.items():
                            if r in per_cluster_success_map[c_old] and path in per_cluster_success_map[c_old][r]:
                                per_cluster_success_map[c_old][r][path] -= p_c
                                per_cluster_total_map[c_old][r][path] -= per_entity_total_map[e1][r][path]
                                assert per_cluster_success_map[c_old][r][path] >= 0
                                assert per_cluster_total_map[c_old][r][path] >= 0
                            per_cluster_success_map_fallback[0][r][path] -= p_c
                            per_cluster_total_map_fallback[0][r][path] -= per_entity_total_map[e1][r][path]
                            assert per_cluster_success_map_fallback[0][r][path] >= 0
                            assert per_cluster_total_map_fallback[0][r][path] >= 0

        train_map_list = [((e1, r), e2_list) for ((e1, r), e2_list) in self.train_map.items()]
        per_entity_success_map.update(per_entity_success_map_updates)
        per_entity_total_map.update(per_entity_total_map_updates)
        exec_ctr_k, reuse_ctr_k = 0, 0
        exec_ctr_f, reuse_ctr_f = 0, 0
        for ((e1, r), e2_list) in tqdm(train_map_list):
            if len(e2_list) == 0:
                del self.train_map[(e1, r)]
                continue
            c_new = e2new_cluster.get(e1, -1)
            if c_new == -1:
                continue
            # Add to new cluster
            if c_new not in per_cluster_success_map:
                per_cluster_success_map[c_new] = {}
            if c_new not in per_cluster_total_map:
                per_cluster_total_map[c_new] = {}
            if r not in per_cluster_success_map[c_new]:
                per_cluster_success_map[c_new][r] = {}
            if r not in per_cluster_total_map[c_new]:
                per_cluster_total_map[c_new][r] = {}

            if r not in per_cluster_success_map_fallback[0]:
                per_cluster_success_map_fallback[0][r] = {}
            if r not in per_cluster_total_map_fallback[0]:
                per_cluster_total_map_fallback[0][r] = {}

            paths_for_this_relation = path_prior_map_per_relation[c_new][r]
            for p_ctr, (path, _) in enumerate(paths_for_this_relation.items()):
                if e1 in per_entity_success_map and r in per_entity_success_map[e1] and \
                        path in per_entity_success_map[e1][r]:
                    if path not in per_cluster_success_map[c_new][r]:
                        per_cluster_success_map[c_new][r][path] = 0
                    if path not in per_cluster_total_map[c_new][r]:
                        per_cluster_total_map[c_new][r][path] = 0
                    per_cluster_success_map[c_new][r][path] += per_entity_success_map[e1][r][path]
                    per_cluster_total_map[c_new][r][path] += per_entity_total_map[e1][r][path]
                    reuse_ctr_k += 1
                else:
                    ans = self.execute_one_program(e1, path, depth=0, max_branch=100)
                    exec_ctr_k += 1
                    if len(ans) == 0:
                        continue
                    # execute the path get answer
                    if path not in per_cluster_success_map[c_new][r]:
                        per_cluster_success_map[c_new][r][path] = 0
                    if path not in per_cluster_total_map[c_new][r]:
                        per_cluster_total_map[c_new][r][path] = 0
                    if e1 not in per_entity_success_map:
                        per_entity_success_map[e1] = {}
                    if r not in per_entity_success_map[e1]:
                        per_entity_success_map[e1][r] = {}
                    if path not in per_entity_success_map[e1][r]:
                        per_entity_success_map[e1][r][path] = 0
                    if e1 not in per_entity_total_map:
                        per_entity_total_map[e1] = {}
                    if r not in per_entity_total_map[e1]:
                        per_entity_total_map[e1][r] = {}
                    if path not in per_entity_total_map[e1][r]:
                        per_entity_total_map[e1][r][path] = 0
                    for a in ans:
                        if a in e2_list:
                            per_entity_success_map[e1][r][path] += 1
                            per_cluster_success_map[c_new][r][path] += 1
                        per_entity_total_map[e1][r][path] += 1
                        per_cluster_total_map[c_new][r][path] += 1
            paths_for_this_relation = path_prior_map_per_relation_fallback[0][r]
            for p_ctr, (path, _) in enumerate(paths_for_this_relation.items()):
                if e1 in per_entity_success_map and r in per_entity_success_map[e1] and \
                        path in per_entity_success_map[e1][r]:
                    if path not in per_cluster_success_map_fallback[0][r]:
                        per_cluster_success_map_fallback[0][r][path] = 0
                    if path not in per_cluster_total_map_fallback[0][r]:
                        per_cluster_total_map_fallback[0][r][path] = 0
                    per_cluster_success_map_fallback[0][r][path] += per_entity_success_map[e1][r][path]
                    per_cluster_total_map_fallback[0][r][path] += per_entity_total_map[e1][r][path]
                    reuse_ctr_f += 1
                else:
                    ans = self.execute_one_program(e1, path, depth=0, max_branch=100)
                    exec_ctr_f += 1
                    if len(ans) == 0:
                        continue
                    # execute the path get answer
                    if path not in per_cluster_success_map_fallback[0][r]:
                        per_cluster_success_map_fallback[0][r][path] = 0
                    if path not in per_cluster_total_map_fallback[0][r]:
                        per_cluster_total_map_fallback[0][r][path] = 0
                    if e1 not in per_entity_success_map:
                        per_entity_success_map[e1] = {}
                    if r not in per_entity_success_map[e1]:
                        per_entity_success_map[e1][r] = {}
                    if path not in per_entity_success_map[e1][r]:
                        per_entity_success_map[e1][r][path] = 0
                    if e1 not in per_entity_total_map:
                        per_entity_total_map[e1] = {}
                    if r not in per_entity_total_map[e1]:
                        per_entity_total_map[e1][r] = {}
                    if path not in per_entity_total_map[e1][r]:
                        per_entity_total_map[e1][r][path] = 0
                    for a in ans:
                        if a in e2_list:
                            per_entity_success_map[e1][r][path] += 1
                            per_cluster_success_map_fallback[0][r][path] += 1
                        per_entity_total_map[e1][r][path] += 1
                        per_cluster_total_map_fallback[0][r][path] += 1
        logging.info(
            f"Update for cluster map required {exec_ctr_k} executions and {reuse_ctr_k} reuses of per_entity maps")
        logging.info(
            f"Update for fallback map required {exec_ctr_f} executions and {reuse_ctr_f} reuses of per_entity maps")

        precision_map = {}
        for c, _ in per_cluster_success_map.items():
            for r, _ in per_cluster_success_map[c].items():
                if c not in precision_map:
                    precision_map[c] = {}
                if r not in precision_map[c]:
                    precision_map[c][r] = {}
                for path, s_c in per_cluster_success_map[c][r].items():
                    if per_cluster_total_map[c][r][path] == 0:
                        precision_map[c][r][path] = 0
                    else:
                        precision_map[c][r][path] = s_c / per_cluster_total_map[c][r][path]

        precision_map_fallback = {0: {}}
        for r, _ in per_cluster_success_map_fallback[0].items():
            if r not in precision_map_fallback[0]:
                precision_map_fallback[0][r] = {}
            for path, s_c in per_cluster_success_map_fallback[0][r].items():
                if per_cluster_total_map_fallback[0][r][path] == 0:
                    precision_map_fallback[0][r][path] = 0
                else:
                    precision_map_fallback[0][r][path] = s_c / per_cluster_total_map_fallback[0][r][path]

        return precision_map, precision_map_fallback

    def calc_prior_path_prob(self):
        """
        Calculate how probable a path is given a query relation, i.e P(path|query rel)
        For each entity in the graph, count paths that exists for each relation in the
        random subgraph.
        :return:
        """
        logger.info("Calculating prior map")
        programs_map = {}
        programs_map_fallback = {0: {}}
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
                if r not in programs_map_fallback[0]:
                    programs_map_fallback[0][r] = {}
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
                        if p not in programs_map_fallback[0][r]:
                            programs_map_fallback[0][r][p] = 0
                        programs_map[c][r][p] += 1
                        programs_map_fallback[0][r][p] += 1
        for c, _ in programs_map.items():
            for r, path_counts in programs_map[c].items():
                sum_path_counts = 0
                for p, p_c in path_counts.items():
                    sum_path_counts += p_c
                for p, p_c in path_counts.items():
                    programs_map[c][r][p] = p_c / sum_path_counts

        for r, path_counts in programs_map_fallback[0].items():
            sum_path_counts = 0
            for p, p_c in path_counts.items():
                sum_path_counts += p_c
            for p, p_c in path_counts.items():
                programs_map_fallback[0][r][p] = p_c / sum_path_counts

        return programs_map, programs_map_fallback

    def calc_per_entity_prior_path_count(self, entity_set=None):
        """
        Calculate how probable a path is given a query relation, i.e P(path|query rel)
        For each entity in the graph, count paths that exists for each relation in the
        random subgraph.
        :return:
        """
        logger.info("Calculating prior map at entity level")
        per_entity_prior_map = {}
        if entity_set is None:
            entity_set = set(self.entity_vocab.keys())
        train_map_list = [((e1, r), e2_list) for ((e1, r), e2_list) in self.train_map.items()]
        for _, ((e1, r), e2_list) in enumerate(tqdm(train_map_list)):
            if e1 not in entity_set:
                continue
            if e1 not in per_entity_prior_map:
                per_entity_prior_map[e1] = {}
            if r not in per_entity_prior_map[e1]:
                per_entity_prior_map[e1][r] = {}
            all_paths_around_e1 = self.all_paths[e1]
            nn_answers = e2_list
            for nn_ans in nn_answers:
                programs = self.get_programs(e1, nn_ans, all_paths_around_e1)
                for p in programs:
                    p = tuple(p)
                    if len(p) == 1:
                        if p[0] == r:  # don't store query relation
                            continue
                    if p not in per_entity_prior_map[e1][r]:
                        per_entity_prior_map[e1][r][p] = 0
                    per_entity_prior_map[e1][r][p] += 1

        # Note the prior is un-normalized
        return per_entity_prior_map

    def get_prior_path_count_entity2cluster(self, per_entity_prior_map, cluster_assignments):
        """
        Calculate how probable a path is given a query relation, i.e P(path|query rel)
        For each entity in the graph, count paths that exists for each relation in the
        random subgraph.
        :return:
        """
        logger.info("Calculating prior map for cluster from entity level map")
        path_prior_map = {}
        path_prior_map_fallback = {0: {}}
        for e1, _ in per_entity_prior_map.items():
            c = cluster_assignments[self.entity_vocab[e1]]
            if c not in path_prior_map:
                path_prior_map[c] = {}
            for r, path_counts in per_entity_prior_map[e1].items():
                if r not in path_prior_map[c]:
                    path_prior_map[c][r] = {}
                if r not in path_prior_map_fallback[0]:
                    path_prior_map_fallback[0][r] = {}
                for p, p_c in path_counts.items():
                    if p not in path_prior_map[c][r]:
                        path_prior_map[c][r][p] = 0
                    if p not in path_prior_map_fallback[0][r]:
                        path_prior_map_fallback[0][r][p] = 0
                    path_prior_map[c][r][p] += p_c
                    path_prior_map_fallback[0][r][p] += p_c

        path_prior_map_normed = {}
        for c, _ in path_prior_map.items():
            for r, path_counts in path_prior_map[c].items():
                sum_path_counts = 0
                for p, p_c in path_counts.items():
                    sum_path_counts += p_c
                if c not in path_prior_map_normed:
                    path_prior_map_normed[c] = {}
                if r not in path_prior_map_normed[c]:
                    path_prior_map_normed[c][r] = {}
                for p, p_c in path_counts.items():
                    path_prior_map_normed[c][r][p] = p_c / sum_path_counts

        path_prior_map_normed_fallback = {0: {}}
        for r, path_counts in path_prior_map_fallback[0].items():
            if r not in path_prior_map_normed_fallback[0]:
                path_prior_map_normed_fallback[0][r] = {}
            sum_path_counts = 0
            for p, p_c in path_counts.items():
                sum_path_counts += p_c
            for p, p_c in path_counts.items():
                path_prior_map_fallback[0][r][p] = p_c / sum_path_counts

        return path_prior_map_normed, path_prior_map_normed_fallback, path_prior_map, path_prior_map_fallback

    def update_prior_path_count_entity2cluster(self, per_entity_prior_counts, per_entity_prior_path_count_updates,
                                               path_prior_map, path_prior_map_fallback,
                                               cluster_adds, cluster_dels):
        logger.info("Updating prior map for cluster from entity level map")

        # For points moving out of a cluster, delete their contribution to prior map of old cluster
        skip_counter = 0
        for c, e_changes in cluster_dels.items():
            if c not in path_prior_map:
                logger.debug(f"Unusual condition: Cluster {c} in cluster_dels but not in path_prior_map")
                continue
            for e1 in e_changes:
                if e1 not in per_entity_prior_counts:
                    skip_counter += 1
                    continue
                for r, path_counts in per_entity_prior_counts[e1].items():
                    for p, p_c in path_counts.items():
                        path_prior_map[c][r][p] -= p_c
                        assert path_prior_map[c][r][p] >= 0
                        assert path_prior_map_fallback[0][r][p] >= 0
                        if path_prior_map[c][r][p] == 0:
                            del path_prior_map[c][r][p]
                        if path_prior_map_fallback[0][r][p] == 0:
                            del path_prior_map_fallback[0][r][p]
        logging.info(f"Skipped {skip_counter} deletes")

        # For points moving into a cluster, add their contribution to prior map of new cluster
        skip_counter = 0
        for c, e_changes in cluster_adds.items():
            if c not in path_prior_map:
                path_prior_map[c] = {}
            for e1 in e_changes:
                if e1 in per_entity_prior_path_count_updates:
                    # Use new counts: Either bcoz new entity or neighborhood changed
                    e_count_map = per_entity_prior_path_count_updates[e1]
                elif e1 not in per_entity_prior_counts:
                    skip_counter += 1
                    continue
                else:
                    # Use old counts
                    e_count_map = per_entity_prior_counts[e1]
                for r, path_counts in e_count_map.items():
                    if r not in path_prior_map[c]:
                        path_prior_map[c][r] = {}
                    if r not in path_prior_map_fallback[0]:
                        path_prior_map_fallback[0][r] = {}
                    for p, p_c in path_counts.items():
                        if p not in path_prior_map[c][r]:
                            path_prior_map[c][r][p] = 0
                        if p not in path_prior_map_fallback[0][r]:
                            path_prior_map_fallback[0][r][p] = 0
                        path_prior_map[c][r][p] += p_c
                        path_prior_map_fallback[0][r][p] += p_c
        logging.info(f"Skipped {skip_counter} additions")

        path_prior_map_normed = {}
        for c, _ in path_prior_map.items():
            for r, path_counts in path_prior_map[c].items():
                sum_path_counts = 0
                for p, p_c in path_counts.items():
                    sum_path_counts += p_c
                if c not in path_prior_map_normed:
                    path_prior_map_normed[c] = {}
                if r not in path_prior_map_normed[c]:
                    path_prior_map_normed[c][r] = {}
                for p, p_c in path_counts.items():
                    path_prior_map_normed[c][r][p] = p_c / sum_path_counts

        path_prior_map_normed_fallback = {0: {}}
        for r, path_counts in path_prior_map_fallback[0].items():
            if r not in path_prior_map_normed_fallback[0]:
                path_prior_map_normed_fallback[0][r] = {}
            sum_path_counts = 0
            for p, p_c in path_counts.items():
                sum_path_counts += p_c
            for p, p_c in path_counts.items():
                path_prior_map_fallback[0][r][p] = p_c / sum_path_counts

        return path_prior_map_normed, path_prior_map_normed_fallback, path_prior_map, path_prior_map_fallback


def main_step(args, entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, adj_mat, train_map, dev_map, dev_entities,
              new_dev_map, new_dev_entities, test_map, test_entities, new_test_map, new_test_entities, all_paths,
              rel_ent_map):
    # Lets put adj_mat to GPU
    adj_mat = torch.from_numpy(adj_mat)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device.__str__()}')
    adj_mat = adj_mat.to(device)

    ######################################
    # Perform evaluation on full dev set #
    ######################################
    if not args.only_test:
        logger.info("Begin evaluation on full dev set ...")
        eval_map = dev_map
        # get the unique entities in eval set, so that we can calculate similarity in advance.
        eval_entities = dev_entities
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

        prob_cbr_agent = ProbCBR(args, train_map, eval_map, entity_vocab, rev_entity_vocab,
                                 rel_vocab, rev_rel_vocab, eval_vocab, eval_rev_vocab, all_paths,
                                 rel_ent_map)

        query_ind = torch.LongTensor(query_ind).to(device)
        # Calculate similarity
        sim = prob_cbr_agent.calc_sim(adj_mat,
                                      query_ind)  # n X N (n== size of dev_entities, N: size of all entities)

        nearest_neighbor_1_hop = np.argsort(-sim.cpu(), axis=-1)
        prob_cbr_agent.set_nearest_neighbor_1_hop(nearest_neighbor_1_hop)

        prob_cbr_agent.do_symbolic_case_based_reasoning()

    ######################################
    # Perform evaluation on new dev set #
    ######################################
    if not args.only_test and new_dev_map is not None:
        logger.info("Begin evaluation on new dev set ...")
        eval_map = new_dev_map
        # get the unique entities in eval set, so that we can calculate similarity in advance.
        eval_entities = new_dev_entities
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

        prob_cbr_agent = ProbCBR(args, train_map, eval_map, entity_vocab, rev_entity_vocab,
                                 rel_vocab, rev_rel_vocab, eval_vocab, eval_rev_vocab,
                                 all_paths, rel_ent_map)

        query_ind = torch.LongTensor(query_ind).to(device)
        # Calculate similarity
        sim = prob_cbr_agent.calc_sim(adj_mat,
                                      query_ind)  # n X N (n== size of dev_entities, N: size of all entities)

        nearest_neighbor_1_hop = np.argsort(-sim.cpu(), axis=-1)
        prob_cbr_agent.set_nearest_neighbor_1_hop(nearest_neighbor_1_hop)

        prob_cbr_agent.do_symbolic_case_based_reasoning()

    #######################################
    # Perform evaluation on full test set #
    #######################################
    if args.test:
        logger.info("Begin evaluation on full test set ...")
        eval_map = test_map
        # get the unique entities in eval set, so that we can calculate similarity in advance.
        eval_entities = test_entities
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

        prob_cbr_agent = ProbCBR(args, train_map, eval_map, entity_vocab, rev_entity_vocab, rel_vocab,
                                 rev_rel_vocab, eval_vocab, eval_rev_vocab, all_paths, rel_ent_map)

        query_ind = torch.LongTensor(query_ind).to(device)
        # Calculate similarity
        sim = prob_cbr_agent.calc_sim(adj_mat,
                                      query_ind)  # n X N (n== size of dev_entities, N: size of all entities)

        nearest_neighbor_1_hop = np.argsort(-sim.cpu(), axis=-1)
        prob_cbr_agent.set_nearest_neighbor_1_hop(nearest_neighbor_1_hop)

        prob_cbr_agent.do_symbolic_case_based_reasoning()

    ######################################
    # Perform evaluation on new test set #
    ######################################
    if args.test and new_test_map is not None:
        logger.info("Begin evaluation on new test set ...")
        eval_map = new_test_map
        # get the unique entities in eval set, so that we can calculate similarity in advance.
        eval_entities = new_test_entities
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

        prob_cbr_agent = ProbCBR(args, train_map, eval_map, entity_vocab, rev_entity_vocab,
                                 rel_vocab, rev_rel_vocab, eval_vocab, eval_rev_vocab, all_paths,
                                 rel_ent_map)

        query_ind = torch.LongTensor(query_ind).to(device)
        # Calculate similarity
        sim = prob_cbr_agent.calc_sim(adj_mat,
                                      query_ind)  # n X N (n== size of dev_entities, N: size of all entities)

        nearest_neighbor_1_hop = np.argsort(-sim.cpu(), axis=-1)
        prob_cbr_agent.set_nearest_neighbor_1_hop(nearest_neighbor_1_hop)

        prob_cbr_agent.do_symbolic_case_based_reasoning()


class CBRWrapper:

    def __init__(self, args, total_n_entity, total_n_relation):
        self.args = args

        # Create GRINCH clustering object
        self.clustering_model = GrinchWithDeletes(np.zeros((total_n_entity, total_n_relation)))

        self.seen_entities = set()
        self.entity_representation = np.zeros((total_n_entity, total_n_relation))
        self.all_paths = {}
        self.per_entity_prior_path_count = {}
        self.per_cluster_path_prior_count, self.per_cluster_path_prior_count_fallback = {}, {}

        self.per_entity_prec_success_counts, self.per_entity_prec_total_counts = {}, {}
        self.per_entity_prec_success_counts_fallback, self.per_entity_prec_total_counts_fallback = {}, {}
        self.per_cluster_prec_success_counts, self.per_cluster_prec_total_counts = {}, {}
        self.per_cluster_prec_success_counts_fallback, self.per_cluster_prec_total_counts_fallback = {}, {}
        self.per_cluster_precision_map, self.fallback_precision_map = {}, {}
        self.cluster_assignments = np.zeros(total_n_entity)

    def process_seed_kb(self, entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab,
                        known_true_triples, train_triples, valid_triples, test_triples):
        if self.args.just_preprocess and not (self.args.process_num == -1 or self.args.process_num == 0):
            # Important for later batches
            self.seen_entities = set(entity_vocab.keys())
            return

        self.args.output_dir = os.path.join(self.args.expt_dir, "outputs", self.args.dataset_name,
                                            self.args.name_of_run, 'stream_step_0')
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        logger.info(f"Output directory: {self.args.output_dir}")

        # 1. Load all maps
        logger.info("Loading train map")
        train_map = load_data_from_triples(train_triples)
        rel_ent_map = get_entities_group_by_relation_from_triples(train_triples)
        logger.info("Loading dev map")
        dev_map = load_data_from_triples(valid_triples)
        dev_entities = get_unique_entities_from_triples(valid_triples)
        logger.info("Loading test map")
        test_map = load_data_from_triples(test_triples)
        test_entities = get_unique_entities_from_triples(test_triples)
        logger.info("Loading combined train/dev/test map for filtered eval")
        all_kg_map = load_data_from_triples(known_true_triples)
        self.args.all_kg_map = all_kg_map

        logger.info("Dumping vocabs")
        with open(os.path.join(self.args.output_dir, "entity_vocab.pkl"), "wb") as fout:
            pickle.dump(entity_vocab, fout)
        with open(os.path.join(self.args.output_dir, "rel_vocab.pkl"), "wb") as fout:
            pickle.dump(rel_vocab, fout)

        # 2. Sample subgraph around entities
        logger.info("Load train adacency map")
        train_adj_map = create_adj_list_from_triples(train_triples)
        if self.args.warm_start:
            logger.info("[WARM_START] Load paths around graph entities")
            with open(os.path.join(self.args.output_dir, f'paths_{self.args.num_paths_to_collect}.pkl'), "rb") as fin:
                self.all_paths = pickle.load(fin)
        else:
            logger.info("Sample paths around graph entities")
            for ctr, e1 in enumerate(tqdm(entity_vocab.keys())):
                self.all_paths[e1] = get_paths(self.args, train_adj_map, e1, max_len=3)
            with open(os.path.join(self.args.output_dir, f'paths_{self.args.num_paths_to_collect}.pkl'), "wb") as fout:
                pickle.dump(self.all_paths, fout)
        self.args.all_paths = self.all_paths

        # 3. Obtain entity cluster assignments
        # Calculate adjacency matrix
        logger.info("Calculate adjacency matrix")
        adj_mat = read_graph_from_triples(train_triples, entity_vocab, rel_vocab)
        adj_mat = np.sqrt(adj_mat)
        l2norm = np.linalg.norm(adj_mat, axis=-1)
        l2norm = np.clip(l2norm, np.finfo(np.float).eps, None)
        adj_mat = adj_mat / l2norm.reshape(l2norm.shape[0], 1)

        self.seen_entities.update(entity_vocab.keys())
        self.entity_representation[:len(entity_vocab), :len(rel_vocab)] = adj_mat

        if not self.args.just_preprocess:
            logger.info("Cluster entities")
            for i in trange(adj_mat.shape[0]):
                # first arg is point id, second argument is the point vector.
                # if you leave second argument blank, it will take vector from points
                # passed in at constructor
                self.clustering_model.insert(i=i, i_vec=self.entity_representation[i])

            cluster_assignments = self.clustering_model.flat_clustering(threshold=self.args.cluster_threshold).astype(
                int)
            assert np.all(cluster_assignments[:len(entity_vocab)] != -1) and \
                   np.all(cluster_assignments[len(entity_vocab):] == -1)
            self.args.cluster_assignments = cluster_assignments[:len(entity_vocab)]
            self.cluster_assignments = cluster_assignments[:len(entity_vocab)].copy()
            cluster_population = Counter(self.args.cluster_assignments)
            logger.info(f"Found {len(cluster_population)} flat clusters")
            logger.info(f"Cluster stats :: Most common: {cluster_population.most_common(5)}")
            logger.info(f"Cluster stats :: Avg Size: {np.mean(list(cluster_population.values()))}")
            logger.info(f"Cluster stats :: Min Size: {np.min(list(cluster_population.values()))}")

            logger.info("Dumping cluster assignments")
            with open(os.path.join(self.args.output_dir, "cluster_assignments.pkl"), "wb") as fout:
                pickle.dump(self.cluster_assignments, fout)

        # 4. Create solver
        prob_cbr_agent = ProbCBR(args, train_map, {}, entity_vocab, rev_entity_vocab, rel_vocab,
                                 rev_rel_vocab, {}, {}, self.args.all_paths, rel_ent_map)

        # 5. Compute path prior map
        if self.args.warm_start:
            logger.info("[WARM_START] Load per entity prior map")
            with open(os.path.join(self.args.output_dir, f'per_entity_prior_path_count.pkl'), "rb") as fin:
                self.per_entity_prior_path_count = pickle.load(fin)
        else:
            self.per_entity_prior_path_count = prob_cbr_agent.calc_per_entity_prior_path_count()
            with open(os.path.join(self.args.output_dir, 'per_entity_prior_path_count.pkl'), "wb") as fout:
                pickle.dump(self.per_entity_prior_path_count, fout)

        if not self.args.just_preprocess:
            self.args.path_prior_map_per_relation, self.args.path_prior_map_per_relation_fallback, \
                self.per_cluster_path_prior_count, self.per_cluster_path_prior_count_fallback = \
                prob_cbr_agent.get_prior_path_count_entity2cluster(self.per_entity_prior_path_count,
                                                                   self.args.cluster_assignments)

            dir_name = os.path.join(self.args.output_dir, "t_{}".format(self.args.cluster_threshold))
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            logger.info("Dumping path prior map at {}".format(dir_name))
            with open(os.path.join(dir_name, "path_prior_map.pkl"), "wb") as fout:
                pickle.dump(self.args.path_prior_map_per_relation, fout)

            dir_name = os.path.join(self.args.output_dir, "K_1")
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            logger.info("Dumping fallback path prior map at {}".format(dir_name))
            with open(os.path.join(dir_name, "path_prior_map.pkl"), "wb") as fout:
                pickle.dump(self.args.path_prior_map_per_relation_fallback, fout)

        # 6. Compute path precision map
        if self.args.warm_start:
            logger.info("[WARM_START] Load per entity precision count maps")
            with open(os.path.join(self.args.output_dir, f'per_entity_prec_success_counts.pkl'), "rb") as fin:
                self.per_entity_prec_success_counts = pickle.load(fin)
            with open(os.path.join(self.args.output_dir, f'per_entity_prec_total_counts.pkl'), "rb") as fin:
                self.per_entity_prec_total_counts = pickle.load(fin)
        else:
            self.per_entity_prec_success_counts, self.per_entity_prec_total_counts = \
                prob_cbr_agent.calc_per_entity_precision_components(self.per_entity_prior_path_count)
            with open(os.path.join(self.args.output_dir, f'per_entity_prec_success_counts.pkl'), "wb") as fout:
                pickle.dump(self.per_entity_prec_success_counts, fout)
            with open(os.path.join(self.args.output_dir, f'per_entity_prec_total_counts.pkl'), "wb") as fout:
                pickle.dump(self.per_entity_prec_total_counts, fout)

        if not self.args.just_preprocess:
            self.args.precision_map, self.per_cluster_prec_success_counts, self.per_cluster_prec_total_counts = \
                prob_cbr_agent.get_precision_map_entity2cluster(self.per_entity_prec_success_counts,
                                                                self.per_entity_prec_total_counts,
                                                                self.args.cluster_assignments,
                                                                self.args.path_prior_map_per_relation)
            self.per_cluster_precision_map = self.args.precision_map

            self.per_entity_prec_success_counts_fallback, self.per_entity_prec_total_counts_fallback = \
                copy.deepcopy(self.per_entity_prec_success_counts), copy.deepcopy(self.per_entity_prec_total_counts)
            self.args.precision_map_fallback, self.per_cluster_prec_success_counts_fallback, \
                self.per_cluster_prec_total_counts_fallback = \
                prob_cbr_agent.get_precision_map_entity2cluster(self.per_entity_prec_success_counts_fallback,
                                                                self.per_entity_prec_total_counts_fallback,
                                                                np.zeros_like(self.args.cluster_assignments),
                                                                self.args.path_prior_map_per_relation_fallback)
            self.fallback_precision_map = self.args.precision_map_fallback

            dir_name = os.path.join(self.args.output_dir, "t_{}".format(self.args.cluster_threshold))
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            logger.info("Dumping path precision map at {}".format(dir_name))
            with open(os.path.join(dir_name, "precision_map.pkl"), "wb") as fout:
                pickle.dump(self.args.precision_map, fout)

            dir_name = os.path.join(self.args.output_dir, "K_1")
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            logger.info("Dumping path precision map at {}".format(dir_name))
            with open(os.path.join(dir_name, "precision_map.pkl"), "wb") as fout:
                pickle.dump(self.args.precision_map_fallback, fout)

        if not self.args.just_preprocess:
            main_step(self.args, entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, adj_mat,
                      train_map, dev_map, dev_entities, None, None, test_map, test_entities, None, None, self.all_paths,
                      rel_ent_map)

    def process_step(self, entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, known_true_triples,
                     all_train_triples, all_valid_triples, new_valid_triples, all_test_triples, new_test_triples,
                     stream_step):
        def _get_affected_entities(_e_query, _r_query, _ans, _all_paths_around_e):
            _affect_set = set()
            for _path in _all_paths_around_e:
                for _l, (_r, _e_dash) in enumerate(_path):
                    if _e_dash == _ans and not (_l == 0 and _r == _r_query) \
                            and not (_l > 0 and _r == _r_query and _path[_l - 1][1] == _e_query):
                        _affect_set.update([_x for (_, _x) in _path[:_l + 1]])
            return _affect_set

        def _get_cluster_changes(_old_cluster_assignments, _new_cluster_assignments, _rev_entity_vocab):
            _adds, _dels = {}, {}
            cluster_add_ctr, cluster_del_ctr = 0, 0
            for _idx in range(len(_old_cluster_assignments)):
                if _old_cluster_assignments[_idx] == _new_cluster_assignments[_idx]:
                    continue
                cluster_del_ctr += 1
                if _old_cluster_assignments[_idx] not in _dels:
                    _dels[_old_cluster_assignments[_idx]] = []
                _dels[_old_cluster_assignments[_idx]].append(_rev_entity_vocab[_idx])
                cluster_add_ctr += 1
                if _new_cluster_assignments[_idx] not in _adds:
                    _adds[_new_cluster_assignments[_idx]] = []
                _adds[_new_cluster_assignments[_idx]].append(_rev_entity_vocab[_idx])
            for _idx in range(len(_old_cluster_assignments), len(_new_cluster_assignments)):
                cluster_add_ctr += 1
                if _new_cluster_assignments[_idx] not in _adds:
                    _adds[_new_cluster_assignments[_idx]] = []
                _adds[_new_cluster_assignments[_idx]].append(_rev_entity_vocab[_idx])
            logger.info(f"{cluster_add_ctr} additions to clusters, {cluster_del_ctr} deletions to clusters")
            return _adds, _dels

        if self.args.just_preprocess and not (self.args.process_num == -1 or self.args.process_num == stream_step):
            # Important for later batches
            self.seen_entities = set(entity_vocab.keys())
            return

        self.args.output_dir = os.path.join(self.args.expt_dir, "outputs", self.args.dataset_name,
                                            self.args.name_of_run, f'stream_step_{stream_step}')
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        logger.info(f"Output directory: {self.args.output_dir}")

        # 1. Load all maps
        logger.info("Loading train map")
        train_map = load_data_from_triples(all_train_triples)
        rel_ent_map = get_entities_group_by_relation_from_triples(all_train_triples)
        logger.info("Loading dev map")
        dev_map = load_data_from_triples(all_valid_triples)
        dev_entities = get_unique_entities_from_triples(all_valid_triples)
        new_dev_map = load_data_from_triples(new_valid_triples)
        new_dev_entities = get_unique_entities_from_triples(new_valid_triples)
        logger.info("Loading test map")
        test_map = load_data_from_triples(all_test_triples)
        test_entities = get_unique_entities_from_triples(all_test_triples)
        new_test_map = load_data_from_triples(new_test_triples)
        new_test_entities = get_unique_entities_from_triples(new_test_triples)
        logger.info("Loading combined train/dev/test map for filtered eval")
        all_kg_map = load_data_from_triples(known_true_triples)
        self.args.all_kg_map = all_kg_map

        logger.info("Dumping vocabs")
        with open(os.path.join(self.args.output_dir, "entity_vocab.pkl"), "wb") as fout:
            pickle.dump(entity_vocab, fout)
        with open(os.path.join(self.args.output_dir, "rel_vocab.pkl"), "wb") as fout:
            pickle.dump(rel_vocab, fout)

        # 2 Sample subgraph around entities
        # 2.1 Identify NEW entities
        new_entities = set(entity_vocab.keys()).difference(self.seen_entities)
        train_adj_map = create_adj_list_from_triples(all_train_triples)

        if self.args.warm_start:
            all_paths_file_nm = os.path.join(self.args.output_dir, f'paths_{self.args.num_paths_to_collect}.pkl')
            logger.info(f"[WARM_START] Loading collected paths from {all_paths_file_nm}")
            with open(all_paths_file_nm, "rb") as fin:
                all_paths_updates = pickle.load(fin)
            self.all_paths.update(all_paths_updates)

            # Just compute AFFECTED entities set
            logger.info("Find AFFECTED entities around new entities")
            affected_neighbors = set()
            for e1 in tqdm(new_entities):
                for (r, nn_ans) in train_adj_map[e1]:
                    affect_ent = _get_affected_entities(e1, r, nn_ans, self.all_paths[e1])
                    affected_neighbors.update(affect_ent)
            affected_neighbors.difference_update(new_entities)
        else:
            # 2.2 Sample subgraph around NEW entities
            logger.info("Sample paths around NEW entities")
            for ctr, e1 in enumerate(tqdm(new_entities)):
                self.all_paths[e1] = get_paths(self.args, train_adj_map, e1, max_len=3)

            # 2.3 Find AFFECTED entities
            logger.info("Find AFFECTED entities around new entities")
            affected_neighbors = set()
            for e1 in tqdm(new_entities):
                for (r, nn_ans) in train_adj_map[e1]:
                    affect_ent = _get_affected_entities(e1, r, nn_ans, self.all_paths[e1])
                    affected_neighbors.update(affect_ent)
            affected_neighbors.difference_update(new_entities)

            # 2.4 Sample paths around AFFECTED entities
            logger.info(f"Resample paths around {len(affected_neighbors)} AFFECTED entities")
            for ctr, e1 in enumerate(tqdm(affected_neighbors)):
                self.all_paths[e1] = get_paths(self.args, train_adj_map, e1, max_len=3)

            all_paths_file_nm = os.path.join(self.args.output_dir, f'paths_{self.args.num_paths_to_collect}.pkl')
            logger.info(f"Dumping collected paths to {all_paths_file_nm}")
            with open(all_paths_file_nm, "wb") as fout:
                pickle.dump(self.all_paths, fout)

        self.args.all_paths = self.all_paths

        # 3 Obtain entity cluster assignments
        # Calculate adjacency matrix
        adj_mat = read_graph_from_triples(all_train_triples, entity_vocab, rel_vocab)
        adj_mat = np.sqrt(adj_mat)
        l2norm = np.linalg.norm(adj_mat, axis=-1)
        l2norm = np.clip(l2norm, np.finfo(np.float).eps, None)
        adj_mat = adj_mat / l2norm.reshape(l2norm.shape[0], 1)

        if not self.args.just_preprocess:
            # 3.1 Find MODIFIED entities whose repr changed
            modified_entities = list()
            modified_entity_idx = list()
            for idx, row in enumerate(adj_mat[:len(self.seen_entities)]):
                if not np.allclose(row, self.entity_representation[idx][:len(rel_vocab)]):
                    modified_entities.append(rev_entity_vocab[idx])
                    modified_entity_idx.append(idx)
            logger.info(f"Identified {len(modified_entity_idx)} MODIFIED entities")

            # 3.2 Update internal entity representations
            self.entity_representation[:len(entity_vocab), :len(rel_vocab)] = adj_mat

            # 3.3 Delete MODIFIED entities
            logger.info("Delete MODIFIED entities")
            for idx in tqdm(modified_entity_idx):
                self.clustering_model.delete_point(idx)

            # 3.4 Add back MODIFIED entities with new repr
            logger.info("Add back MODIFIED entities with new repr")
            for idx in tqdm(modified_entity_idx):
                self.clustering_model.insert(idx, self.entity_representation[idx])

            # 3.5 Add NEW entities
            logger.info("Add NEW entities")
            new_entity_idx = sorted([entity_vocab[ent] for ent in new_entities])
            assert new_entity_idx == np.arange(len(self.seen_entities), len(entity_vocab)).tolist()
            for idx in tqdm(new_entity_idx):
                self.clustering_model.insert(idx, self.entity_representation[idx])

            new_cluster_assignments = self.clustering_model.flat_clustering(threshold=self.args.cluster_threshold)
            new_cluster_assignments = new_cluster_assignments.astype(int)
            assert np.all(new_cluster_assignments[:len(entity_vocab)] != -1) and \
                   np.all(new_cluster_assignments[len(entity_vocab):] == -1)

            # 3.6 Record changes to clusters
            cluster_adds, cluster_dels = _get_cluster_changes(self.cluster_assignments,
                                                              new_cluster_assignments[:len(entity_vocab)],
                                                              rev_entity_vocab)

            # 3.7 Add AFFECTED entities as special cases to cluster adds and dels
            logger.info("Add AFFECTED entities as special cases to cluster adds and dels")
            ex_add_ctr, ex_del_ctr = 0, 0
            for e1 in affected_neighbors:
                if self.cluster_assignments[entity_vocab[e1]] not in cluster_dels:
                    cluster_dels[self.cluster_assignments[entity_vocab[e1]]] = [e1]
                    ex_del_ctr += 1
                elif e1 not in cluster_dels[self.cluster_assignments[entity_vocab[e1]]]:
                    cluster_dels[self.cluster_assignments[entity_vocab[e1]]].append(e1)
                    ex_del_ctr += 1
                if new_cluster_assignments[entity_vocab[e1]] not in cluster_adds:
                    cluster_adds[new_cluster_assignments[entity_vocab[e1]]] = [e1]
                    ex_add_ctr += 1
                elif e1 not in cluster_adds[new_cluster_assignments[entity_vocab[e1]]]:
                    cluster_adds[new_cluster_assignments[entity_vocab[e1]]].append(e1)
                    ex_add_ctr += 1
            logger.info(f"{ex_add_ctr} additional additions to clusters, {ex_del_ctr} additional deletions to clusters")

            self.args.cluster_assignments = new_cluster_assignments[:len(entity_vocab)]
            self.cluster_assignments = new_cluster_assignments[:len(entity_vocab)].copy()
            cluster_population = Counter(self.args.cluster_assignments)
            logger.info(f"Found {len(cluster_population)} flat clusters")
            logger.info(f"Cluster stats :: Most common: {cluster_population.most_common(5)}")
            logger.info(f"Cluster stats :: Avg Size: {np.mean(list(cluster_population.values()))}")
            logger.info(f"Cluster stats :: Min Size: {np.min(list(cluster_population.values()))}")

            logger.info("Dumping cluster assignments")
            with open(os.path.join(self.args.output_dir, "cluster_assignments.pkl"), "wb") as fout:
                pickle.dump(self.cluster_assignments, fout)

        # 4. Create solver
        prob_cbr_agent = ProbCBR(args, train_map, {}, entity_vocab, rev_entity_vocab, rel_vocab,
                                 rev_rel_vocab, {}, {}, self.args.all_paths, rel_ent_map)

        # 5. Compute path prior map
        # 5.1 Compute path prior map for NEW and AFFECTED entities
        if self.args.warm_start:
            logger.info("[WARM_START] Load per entity prior count updates")
            with open(os.path.join(self.args.output_dir, f'per_entity_prior_path_count_updates.pkl'), "rb") as fin:
                per_entity_prior_path_count_updates = pickle.load(fin)
        else:
            per_entity_prior_path_count_updates = \
                prob_cbr_agent.calc_per_entity_prior_path_count(
                    entity_set=new_entities.union(affected_neighbors))
            with open(os.path.join(self.args.output_dir, 'per_entity_prior_path_count_updates.pkl'), "wb") as fout:
                pickle.dump(per_entity_prior_path_count_updates, fout)

        self.per_entity_prior_path_count.update(per_entity_prior_path_count_updates)

        # 5.2 Compute path prior map new clusters
        # self.args.path_prior_map_per_relation, self.args.path_prior_map_per_relation_fallback, \
        #     self.per_cluster_path_prior_count, self.per_cluster_path_prior_count_fallback = \
        #     prob_cbr_agent.update_prior_path_count_entity2cluster(self.per_entity_prior_path_count,
        #                                                                     per_entity_prior_path_count_updates,
        #                                                                     self.per_cluster_path_prior_count,
        #                                                                     self.per_cluster_path_prior_count_fallback,
        #                                                                     cluster_adds, cluster_dels)
        if not self.args.just_preprocess:
            self.args.path_prior_map_per_relation, self.args.path_prior_map_per_relation_fallback, \
                self.per_cluster_path_prior_count, self.per_cluster_path_prior_count_fallback = \
                prob_cbr_agent.get_prior_path_count_entity2cluster(self.per_entity_prior_path_count,
                                                                   self.args.cluster_assignments)

            # dir_name = os.path.join(self.args.output_dir, "K_{}".format(self.args.num_clusters))
            dir_name = os.path.join(self.args.output_dir, "t_{}".format(self.args.cluster_threshold))
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            logger.info("Dumping path prior map at {}".format(dir_name))
            with open(os.path.join(dir_name, "path_prior_map.pkl"), "wb") as fout:
                pickle.dump(self.args.path_prior_map_per_relation, fout)

            dir_name = os.path.join(self.args.output_dir, "K_1")
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            logger.info("Dumping fallback path prior map at {}".format(dir_name))
            with open(os.path.join(dir_name, "path_prior_map.pkl"), "wb") as fout:
                pickle.dump(self.args.path_prior_map_per_relation_fallback, fout)

        # 6. Compute path precision map
        # 6.1 Compute path precision map for NEW and AFFECTED entities
        if self.args.warm_start:
            logger.info("[WARM_START] Load per entity precision count updates")
            with open(os.path.join(self.args.output_dir, f'per_entity_prec_success_counts_updates.pkl'), "rb") as fin:
                per_entity_prec_success_counts_updates = pickle.load(fin)
            with open(os.path.join(self.args.output_dir, f'per_entity_prec_total_counts_updates.pkl'), "rb") as fin:
                per_entity_prec_total_counts_updates = pickle.load(fin)
        else:
            per_entity_prec_success_counts_updates, per_entity_prec_total_counts_updates = \
                prob_cbr_agent.calc_per_entity_precision_components(self.per_entity_prior_path_count,
                                                                    new_entities.union(affected_neighbors))
            with open(os.path.join(self.args.output_dir, f'per_entity_prec_success_counts_updates.pkl'), "wb") as fout:
                pickle.dump(per_entity_prec_success_counts_updates, fout)
            with open(os.path.join(self.args.output_dir, f'per_entity_prec_total_counts_updates.pkl'), "wb") as fout:
                pickle.dump(per_entity_prec_total_counts_updates, fout)

        self.per_entity_prec_success_counts.update(per_entity_prec_success_counts_updates)
        self.per_entity_prec_total_counts.update(per_entity_prec_total_counts_updates)
        self.per_entity_prec_success_counts_fallback.update(per_entity_prec_success_counts_updates)
        self.per_entity_prec_total_counts_fallback.update(per_entity_prec_total_counts_updates)

        if not self.args.just_preprocess:
            # 6.2 Compute path precision map new clusters
            self.args.precision_map, self.per_cluster_prec_success_counts, self.per_cluster_prec_total_counts = \
                prob_cbr_agent.get_precision_map_entity2cluster(self.per_entity_prec_success_counts,
                                                                self.per_entity_prec_total_counts,
                                                                self.args.cluster_assignments,
                                                                self.args.path_prior_map_per_relation)
            self.per_cluster_precision_map = self.args.precision_map

            self.args.precision_map_fallback, self.per_cluster_prec_success_counts_fallback, \
                self.per_cluster_prec_total_counts_fallback = \
                prob_cbr_agent.get_precision_map_entity2cluster(self.per_entity_prec_success_counts_fallback,
                                                                self.per_entity_prec_total_counts_fallback,
                                                                np.zeros_like(self.args.cluster_assignments),
                                                                self.args.path_prior_map_per_relation_fallback)
            self.fallback_precision_map = self.args.precision_map_fallback

            # self.args.precision_map, self.args.precision_map_fallback = \
            #     prob_cbr_agent.update_precision_map_entity2cluster(self.per_entity_prec_success_counts,
            #                                                                  self.per_entity_prec_total_counts,
            #                                                                  per_entity_prec_success_counts_updates,
            #                                                                  per_entity_prec_total_counts_updates,
            #                                                                  self.per_cluster_prec_success_counts,
            #                                                                  self.per_cluster_prec_total_counts,
            #                                                                  self.per_cluster_prec_success_counts_fallback,
            #                                                                  self.per_cluster_prec_total_counts_fallback,
            #                                                                  cluster_adds, cluster_dels,
            #                                                                  self.args.path_prior_map_per_relation,
            #                                                                  self.args.path_prior_map_per_relation_fallback)
            # self.per_cluster_precision_map = self.args.precision_map
            # self.fallback_precision_map = self.args.precision_map_fallback

            # dir_name = os.path.join(self.args.output_dir, "K_{}".format(self.args.num_clusters))
            dir_name = os.path.join(self.args.output_dir, "t_{}".format(self.args.cluster_threshold))
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            logger.info("Dumping path precision map at {}".format(dir_name))
            with open(os.path.join(dir_name, "precision_map.pkl"), "wb") as fout:
                pickle.dump(self.args.precision_map, fout)

            dir_name = os.path.join(self.args.output_dir, "K_1")
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            logger.info("Dumping path precision map at {}".format(dir_name))
            with open(os.path.join(dir_name, "precision_map.pkl"), "wb") as fout:
                pickle.dump(self.args.precision_map_fallback, fout)

        self.seen_entities.update(entity_vocab.keys())

        if not self.args.just_preprocess:
            main_step(self.args, entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, adj_mat,
                      train_map, dev_map, dev_entities, new_dev_map, new_dev_entities, test_map, test_entities,
                      new_test_map, new_test_entities, self.all_paths, rel_ent_map)


def main(args):
    dataset_name = args.dataset_name
    logger.info("==========={}============".format(dataset_name))
    data_dir = os.path.join(args.data_dir, "data", dataset_name)

    args.output_dir = os.path.join(args.expt_dir, "outputs", args.dataset_name, args.name_of_run)
    logger.info(f"Output directory: {args.output_dir}")

    logger.info("=========Config:============")
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    stream_obj = KBStream(args.dataset_name, data_dir, test_file_name=args.test_file_name,
                          stream_init_proportion=args.stream_init_proportion, n_stream_updates=args.n_stream_updates,
                          seed=args.stream_seed)
    max_num_entities, max_num_relations = stream_obj.get_max_num_entities(), stream_obj.get_max_num_relations()
    online_cbreasoner = CBRWrapper(args, max_num_entities, max_num_relations)

    entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, \
    known_true_triples, train_triples, valid_triples, test_triples = stream_obj.get_init_kb()

    online_cbreasoner.process_seed_kb(entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab,
                                      known_true_triples, train_triples, valid_triples, test_triples)

    for batch_ctr, kb_batch_update in enumerate(stream_obj.batch_generator(), start=1):
        if args.run_init_only:
            continue

        entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, known_true_triples, all_train_triples, \
        all_valid_triples, new_valid_triples, all_test_triples, new_test_triples = kb_batch_update

        online_cbreasoner.process_step(entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab,
                                       known_true_triples, all_train_triples, all_valid_triples, new_valid_triples,
                                       all_test_triples, new_test_triples, stream_step=batch_ctr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect subgraphs around entities")
    parser.add_argument("--dataset_name", type=str, default="nell")
    parser.add_argument("--data_dir", type=str, default="../prob_cbr_data/")
    parser.add_argument("--expt_dir", type=str, default="../prob_cbr_expts/")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--only_test", action="store_true")
    parser.add_argument("--test_file_name", type=str, default='')
    parser.add_argument("--use_path_counts", type=int, choices=[0, 1], default=1,
                        help="Set to 1 if want to weight paths during ranking")
    # Clustering args
    parser.add_argument("--cluster_threshold", type=float, default=0.9)
    # CBR args
    parser.add_argument("--k_adj", type=int, default=5,
                        help="Number of nearest neighbors to consider based on adjacency matrix")
    parser.add_argument("--max_num_programs", type=int, default=1000)
    # Output modifier args
    parser.add_argument("--name_of_run", type=str, default="unset")
    parser.add_argument("--print_paths", action="store_true")
    parser.add_argument("--output_per_relation_scores", action="store_true")
    parser.add_argument("--use_wandb", type=int, choices=[0, 1], default=0, help="Set to 1 if using W&B")
    # Path sampling args
    parser.add_argument("--num_paths_to_collect", type=int, default=1000)
    # Stream settings
    parser.add_argument('--stream_init_proportion', type=float, default=0.5)
    parser.add_argument('--n_stream_updates', type=int, default=10)
    parser.add_argument('--stream_seed', type=int, default=42)
    # Allow for preprocessing
    parser.add_argument('--just_preprocess', action='store_true')
    parser.add_argument('--process_num', type=int, default=-1)
    parser.add_argument('--overall_seed', type=int, default=4242)
    parser.add_argument('--warm_start', action='store_true')
    # Faster experimentation while debugging
    parser.add_argument('--run_init_only', action='store_true')

    args = parser.parse_args()
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    if args.use_wandb:
        wandb.init(project='case-based-reasoning')

    if args.name_of_run == "unset":
        args.name_of_run = str(uuid.uuid4())[:8]

    if args.only_test and not args.test:
        logger.info("--only_test is True and --test is False. Setting --test to True")
        args.test = True

    np.random.seed(args.overall_seed)

    args.use_path_counts = (args.use_path_counts == 1)

    main(args)

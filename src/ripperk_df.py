import pandas as pd
import numpy as np
import math

class ripperk(object):
    def __init__(self, prun_ratio, dl_threshold, k=2):
        self.prun_ratio = prun_ratio
        self.dl_threshold = dl_threshold
        self.k = k

    def fit(self, df, label):
        self.rulesets = {}

        self._get_conditions(df)

        items =  list(label.value_counts().sort_values(ascending=False).index)

        while len(items) > 1:
            # get cls from end to start, from small to big
            item = items.pop()
            pos = df[label==item]
            neg = df[label!=item]

            ruleset = self.irep(pos, neg)

            for _ in range(self.k):
                ruleset = self.optimize(pos, neg, ruleset)

            df = self.remove_cases(df, ruleset)

            self.rulesets[item] = ruleset


    def predict(self, df):
        pass

    def irep(self, pos, neg):
        rule_set = []

        min_dl = self.init_dl

        while pos:
            pos_chunk = int(self.prun_ratio * pos.shape[0])
            neg_chunk = int(self.prun_ratio * neg.shape[0])

            pos_grow = pos.iloc[:pos_chunk, :]
            neg_grow = neg.iloc[:neg_chunk, :]
            rule = self.grow_rule(pos_grow, neg_grow)

            if self.prun_ratio > 0:
                pos_prun = pos.iloc[pos_chunk:, :]
                neg_prun = neg.iloc[neg_chunk:, :]
                rule = self.prun(pos_prun, neg_prun, rule)

            rule_dl = self.dl(rule)
            if min_dl + self.dl_threshold < rule_dl:
                return rule_set
            else:
                rule_set.append(rule)
                if rule_dl < min_dl:
                    min_dl = rule_dl

                pos = self.remove_cases(pos, [rule])
                neg = self.remove_cases(neg, [rule])
        return rule_set

    def grow_rule(self, pos, neg, rule=None, rules=None):
        pass

    def prun(self, pos, neg, rule, ruleset=None):
        pass

    def dl(self, rule):
        """
        Finds the description length for a rule.

        Key arguments:
        rule -- the rule.
        """
        k = len(rule.keys())
        p = k / float(self.init_dl)

        p1 = float(k) * math.log(1 / p, 2)
        p2 = float(self.init_dl - k) * math.log(1 / float(1 - p), 2)

        return int(0.5 * (math.log(k, 2) + p1 + p2))

    def _get_conditions(self, df):
        init_dl = 0

        s = df.dtypes
        discrete_cols = list(s.index[s=='object'])
        continuous_cols = [i for i in df.columns if i not in discrete_cols]

        conditions = []

        for c in discrete_cols:
            for v in df[c].unique():
                conditions.append((c, ('==', v)))
                conditions.append((c, ('!=', v)))
                init_dl += 2

        for c in continuous_cols:
            for v in df[c].unique():
                conditions.append((c, ('>=', v)))
                conditions.append((c, ('<=', v)))
                init_dl += 2

        self.conditions = conditions
        self.init_dl = init_dl

    def bindings(self, df, ruleset):
        l_t = df.iloc[:, :1].astype(bool)
        l_t[l_t==True] = False

        for rule in ruleset:
            l = df.iloc[:, :1].astype(bool)
            l[l==False] = True
            for condition in rule:
                if condition[1][0] == '==':
                    l &= df[condition[0]] == condition[1][1]
                elif condition[1][0] == '!=':
                    l &= df[condition[0]] != condition[1][1]
                elif condition[0] == '>=':
                    l &= df[condition[0]] >= condition[1][1]
                elif condition[1][0] == '<=':
                    l &= df[condition[0]] <= condition[1][1]
            l_t |= l
        return l_t

    def remove_cases(self, df, ruleset):
        l_t = self.bindings(df, ruleset)
        df = df[~l_t]
        return df


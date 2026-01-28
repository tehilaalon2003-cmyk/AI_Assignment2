import math
from collections import Counter

# Entropy calculation for the class labels
#H(Y) = - sum_c P(c)log2(P(c))
def _entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    #Count how many times each label appears.
    counts = Counter(labels)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def _majority(labels):
    counts = Counter(labels)
    max_c = max(counts.values())
    winners = sorted([lab for lab, c in counts.items() if c == max_c])
    return winners[0]

# Extracts feature value domains from the training set only (in appearance order).
def _domains_in_train_order(X, features):
    domains = {f: [] for f in features}
    seen = {f: set() for f in features}
    for row in X:
        for f in features:
            v = row[f]
            if v not in seen[f]:
                seen[f].add(v)
                domains[f].append(v)
    return domains

# Calculate Information Gain for a feature
#IG(Y, feature) = H(Y) - sum_v P(v)*H(Y|v)
def _info_gain(X, y, feature, domain):
    # Calculate information gain of splitting on 'feature'
    base = _entropy(y)
    n = len(y)
    # Conditional entropy
    cond = 0.0
    for v in domain:
        y_v = [lbl for row, lbl in zip(X, y) if row[feature] == v]
        if len(y_v) == 0:
            continue
        cond += (len(y_v) / n) * _entropy(y_v)
    return base - cond


#The training of the Decision Tree using ID3 algorithm
def train_id3(X, y, features):
    domains = _domains_in_train_order(X, features)

    # Recursive build function
    def build(X_sub, y_sub, feats_left, default_label):
        #no examples-default
        if len(y_sub) == 0:
            return {"leaf": True, "class": default_label}

        #pure-leaf
        if all(lbl == y_sub[0] for lbl in y_sub):
            return {"leaf": True, "class": y_sub[0]}

        #no features-majority
        maj = _majority(y_sub)
        if len(feats_left) == 0:
            return {"leaf": True, "class": maj}

        #choose best feature by IG
        best = None
        best_ig = float("-inf")
        for f in feats_left:
            ig = _info_gain(X_sub, y_sub, f, domains[f])
            if ig > best_ig or (abs(ig - best_ig) < 1e-12 and (best is None or f < best)):
                best_ig = ig
                best = f

        node = {"leaf": False, "attr": best, "children": {}}
        next_feats = [f for f in feats_left if f != best]

        #split by each value
        for v in domains[best]:
            X_v = [row for row in X_sub if row[best] == v]
            y_v = [lbl for row, lbl in zip(X_sub, y_sub) if row[best] == v]
            node["children"][v] = build(X_v, y_v, next_feats, maj)

        return node

    overall_default = _majority(y)
    return build(X, y, features, overall_default)

#print the decision tree to the file
def export_tree_to_file(tree, path="output_tree.txt"):
    lines = []

    def dfs(node, level, is_root):
        if node["leaf"]:
            return

        attr = node["attr"]
        for value, child in node["children"].items():
            if is_root:
                if child["leaf"]:
                    lines.append(f"{attr} = {value}: {child['class']}")
                else:
                    lines.append(f"{attr} = {value}")
                    dfs(child, level + 1, False)
            else:
                prefix = ("\t" * level) + "| "
                if child["leaf"]:
                    lines.append(f"{prefix}{attr} = {value}: {child['class']}")
                else:
                    lines.append(f"{prefix}{attr} = {value}")
                    dfs(child, level + 1, False)

    dfs(tree, level=1, is_root=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

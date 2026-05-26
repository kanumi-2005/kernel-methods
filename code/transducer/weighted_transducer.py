import networkx as nx
import matplotlib.pyplot as plt


class WeightedTransducer:
    def __init__(self):
        self.states = set()
        self.start_state = None
        self.final_states = set()

        # (src, dst, input, output, weight)
        self.transitions = []

    def add_state(self, state, start=False, final=False):
        self.states.add(state)

        if start:
            self.start_state = state

        if final:
            self.final_states.add(state)

    def add_transition(self, src, dst, inp, out, weight):
        self.transitions.append((src, dst, inp, out, weight))

    def draw(self, title="Weighted Transducer"):
        G = nx.MultiDiGraph()

        for s in self.states:
            G.add_node(s)

        for src, dst, inp, out, w in self.transitions:
            label = f"{inp}:{out}/{w}"
            G.add_edge(src, dst, label=label)

        pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(10, 6))

        # node
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=1800,
            node_color="white",
            edgecolors="black"
        )

        # final state
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(self.final_states),
            node_size=2200,
            node_color="white",
            edgecolors="black",
            linewidths=2
        )

        # labels
        nx.draw_networkx_labels(G, pos, font_size=12)

        # edges
        nx.draw_networkx_edges(
            G,
            pos,
            arrows=True,
            connectionstyle="arc3,rad=0.1"
        )

        edge_labels = {
            (u, v, k): d["label"]
            for u, v, k, d in G.edges(keys=True, data=True)
        }

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=10
        )

        plt.title(title)
        plt.axis("off")
        plt.show()


def compose_transducers(T1, T2):
    """
    Gộp (compose) hai weighted transducer:
        output(T1) == input(T2)
    """

    T = WeightedTransducer()

    # state mới = (q1, q2)
    for q1 in T1.states:
        for q2 in T2.states:

            new_state = (q1, q2)

            start = (
                q1 == T1.start_state and
                q2 == T2.start_state
            )

            final = (
                q1 in T1.final_states and
                q2 in T2.final_states
            )

            T.add_state(new_state, start=start, final=final)

    # transition composition
    for (s1, d1, i1, o1, w1) in T1.transitions:
        for (s2, d2, i2, o2, w2) in T2.transitions:

            # output T1 phải match input T2
            if o1 == i2:

                src = (s1, s2)
                dst = (d1, d2)

                inp = i1
                out = o2

                # nhân trọng số
                weight = round(w1 * w2, 3)

                T.add_transition(
                    src,
                    dst,
                    inp,
                    out,
                    weight
                )

    return T


# =========================
# Transducer 1
# =========================

T1 = WeightedTransducer()

T1.add_state(0, start=True)
T1.add_state(1)
T1.add_state(2)
T1.add_state(3, final=True)

T1.add_transition(0, 1, "a", "b", 0.1)
T1.add_transition(1, 0, "a", "b", 0.2)

T1.add_transition(1, 3, "b", "b", 0.4)
T1.add_transition(2, 1, "b", "b", 0.3)

T1.add_transition(2, 3, "a", "b", 0.5)
T1.add_transition(3, 3, "a", "a", 0.6)

# =========================
# Transducer 2
# =========================

T2 = WeightedTransducer()

T2.add_state(0, start=True)
T2.add_state(1)
T2.add_state(2)
T2.add_state(3, final=True)

T2.add_transition(0, 1, "b", "b", 0.1)

T2.add_transition(1, 2, "a", "b", 0.3)
T2.add_transition(1, 3, "a", "b", 0.4)

T2.add_transition(1, 1, "b", "a", 0.2)

T2.add_transition(2, 3, "b", "a", 0.5)

# =========================
# Compose
# =========================

TC = compose_transducers(T1, T2)

# =========================
# Draw
# =========================

T1.draw("Transducer T1")
T2.draw("Transducer T2")
TC.draw("Composition T1 o T2")
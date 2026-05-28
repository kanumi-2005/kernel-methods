import networkx as nx
import matplotlib.pyplot as plt

EPS = "ε"

class WeightedTransducer:
    def __init__(self):
        self.states = set()
        self.start_state = None
        self.final_states = set()
        # Mỗi phần tử: (src, dst, input, output, weight)
        self.transitions = []

    def add_state(self, state, start=False, final=False):
        self.states.add(state)
        if start:
            self.start_state = state
        if final:
            self.final_states.add(state)

    def add_transition(self, src, dst, inp, out, weight):
        self.transitions.append((src, dst, inp, out, weight))

    def trim(self):
        """
        Loại bỏ các trạng thái không thể dẫn đến trạng thái kết thúc (co-accessible)
        và các trạng thái không đến được từ trạng thái bắt đầu (reachable).
        Phục vụ minh họa cho Hình 6.5.
        """
        if self.start_state is None:
            return

        # 1. Tìm các trạng thái đến được từ điểm xuất phát (Forward BFS)
        adj_forward = {s: [] for s in self.states}
        adj_backward = {s: [] for s in self.states}
        for src, dst, inp, out, w in self.transitions:
            adj_forward[src].append(dst)
            adj_backward[dst].append(src)

        reachable = set()
        queue = [self.start_state]
        reachable.add(self.start_state)
        while queue:
            curr = queue.pop(0)
            for nxt in adj_forward[curr]:
                if nxt not in reachable:
                    reachable.add(nxt)
                    queue.append(nxt)

        # 2. Tìm các trạng thái có thể đi tới trạng thái kết thúc (Backward BFS)
        co_accessible = set()
        queue = list(self.final_states)
        for s in self.final_states:
            co_accessible.add(s)

        while queue:
            curr = queue.pop(0)
            for prv in adj_backward[curr]:
                if prv not in co_accessible:
                    co_accessible.add(prv)
                    queue.append(prv)

        # Trạng thái hợp lệ phải thỏa mãn cả 2 điều kiện
        keep_states = reachable.intersection(co_accessible)

        # Cập nhật lại Transducer
        self.states = keep_states
        self.final_states = self.final_states.intersection(keep_states)
        self.transitions = [
            t for t in self.transitions if t[0] in keep_states and t[1] in keep_states
        ]
        if self.start_state not in keep_states:
            self.start_state = None

    def draw(self, title="Weighted Transducer"):
        G = nx.MultiDiGraph()
        for s in self.states:
            G.add_node(s)

        for src, dst, inp, out, w in self.transitions:
            # Chuyển tuple trạng thái thành string để hiển thị đẹp mắt
            s_label = str(src) if not isinstance(src, tuple) else f"{src[0]},{src[1]}"
            d_label = str(dst) if not isinstance(dst, tuple) else f"{dst[0]},{dst[1]}"
            label = f"{inp}:{out}/{w}"
            G.add_edge(s_label, d_label, label=label)

        # Chuyển đổi tên các danh sách trạng thái để khớp với nhãn đồ thị
        def to_str(s): return str(s) if not isinstance(s, tuple) else f"{s[0]},{s[1]}"
        
        pos = nx.spring_layout(G, seed=42, k=2)
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        normal_states = [to_str(s) for s in self.states - self.final_states - {self.start_state}]
        if normal_states:
            nx.draw_networkx_nodes(G, pos, nodelist=normal_states, node_size=2000, node_color="white", edgecolors="black", linewidths=2)

        if self.start_state is not None:
            nx.draw_networkx_nodes(G, pos, nodelist=[to_str(self.start_state)], node_size=2000, node_color="lightgray", edgecolors="black", linewidths=3)

        final_labels = [to_str(s) for s in self.final_states]
        if final_labels:
            nx.draw_networkx_nodes(G, pos, nodelist=final_labels, node_size=2000, node_color="white", edgecolors="black", linewidths=3)
            nx.draw_networkx_nodes(G, pos, nodelist=final_labels, node_size=2500, node_color="none", edgecolors="black", linewidths=1.5)

        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

        for i, (src, dst, data) in enumerate(G.edges(data=True)):
            rad = 0.1 * ((i % 3) + 1)
            nx.draw_networkx_edges(G, pos, edgelist=[(src, dst)], arrows=True, arrowsize=20, width=1.5, connectionstyle=f"arc3,rad={rad}")

        edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, rotate=False)

        plt.title(title, fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


# =====================================================
# THUẬT TOÁN HỢP SỬ DỤNG BỘ LỌC EPSILON (Hình 6.6)
# =====================================================
def compose_transducers(T1, T2):
    """
    Hợp hai WFST sử dụng bộ lọc trạng thái epsilon (0, 1, 2) 
    để loại bỏ các đường đi trùng lặp và tính đúng trọng số.
    Trạng thái mới có dạng: (trạng_thái_T1, trạng_thái_T2, trạng_thái_bộ_lọc)
    """
    # Gom nhóm transition theo trạng thái nguồn để tăng tốc tìm kiếm
    t1_by_src = {}
    for src, dst, inp, out, w in T1.transitions:
        t1_by_src.setdefault(src, []).append((dst, inp, out, w))

    t2_by_src = {}
    for src, dst, inp, out, w in T2.transitions:
        t2_by_src.setdefault(src, []).append((dst, inp, out, w))

    T = WeightedTransducer()
    
    # Trạng thái bắt đầu kết hợp với bộ lọc ở trạng thái 0
    start_state = (T1.start_state, T2.start_state, 0)
    T.add_state(start_state, start=True)
    
    queue = [start_state]
    visited = {start_state}
    
    while queue:
        curr = queue.pop(0)
        q1, q2, qf = curr
        
        transitions1 = t1_by_src.get(q1, [])
        transitions2 = t2_by_src.get(q2, [])
        
        # 1. KHỚP THÔNG THƯỜNG (Normal Match): Output(T1) == Input(T2) và cả hai đều khác EPS
        for d1, i1, o1, w1 in transitions1:
            if o1 != EPS:
                for d2, i2, o2, w2 in transitions2:
                    if o1 == i2 and i2 != EPS:
                        next_state = (d1, d2, 0) # Khớp thường đưa bộ lọc về 0
                        if next_state not in visited:
                            visited.add(next_state)
                            queue.append(next_state)
                        T.add_state(next_state)
                        T.add_transition(curr, next_state, i1, o2, round(w1 * w2, 3))

        # 2. KHỚP EPSILON ĐỒNG THỜI (Simultaneous Epsilon): Cả hai cùng đứng tại chỗ giải quyết epsilon
        # Phục vụ vòng lặp tự thân (self-loop) tại trạng thái 0 của bộ lọc Hình 6.6b
        if qf == 0:
            for d1, i1, o1, w1 in transitions1:
                if o1 == EPS:
                    for d2, i2, o2, w2 in transitions2:
                        if i2 == EPS:
                            next_state = (d1, d2, 0)
                            if next_state not in visited:
                                visited.add(next_state)
                                queue.append(next_state)
                            T.add_state(next_state)
                            T.add_transition(curr, next_state, i1, o2, round(w1 * w2, 3))

        # 3. CHỈ T1 DI CHUYỂN EPSILON (T1 Epsilon Move): Output(T1) == EPS, T2 đứng im.
        # Điều kiện bộ lọc: qf phải là 0 hoặc 2. Trạng thái tiếp theo sẽ chuyển thành 2.
        if qf == 0 or qf == 2:
            for d1, i1, o1, w1 in transitions1:
                if o1 == EPS:
                    next_state = (d1, q2, 2)
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append(next_state)
                    T.add_state(next_state)
                    T.add_transition(curr, next_state, i1, EPS, round(w1, 3))

        # 4. CHỈ T2 DI CHUYỂN EPSILON (T2 Epsilon Move): Input(T2) == EPS, T1 đứng im.
        # Điều kiện bộ lọc: qf phải là 0 hoặc 1. Trạng thái tiếp theo sẽ chuyển thành 1.
        if qf == 0 or qf == 1:
            for d2, i2, o2, w2 in transitions2:
                if i2 == EPS:
                    next_state = (q1, d2, 1)
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append(next_state)
                    T.add_state(next_state)
                    T.add_transition(curr, next_state, EPS, o2, round(w2, 3))

    # Xác định các trạng thái kết thúc (Cả hai thành phần đều là trạng thái kết thúc)
    for state in T.states:
        q1, q2, qf = state
        if q1 in T1.final_states and q2 in T2.final_states:
            T.final_states.add(state)
            
    return T

if __name__ == "__main__":

    # Thiết lập T1 từ Hình 6.6
    T1 = WeightedTransducer()
    for s in range(5):
        T1.add_state(s, start=(s == 0), final=(s == 4))

    T1.add_transition(0, 1, "a", "a", 1.0)
    T1.add_transition(1, 2, "b", EPS, 1.0)  # b:ε
    T1.add_transition(2, 3, "c", EPS, 1.0)  # c:ε
    T1.add_transition(3, 4, "d", "d", 1.0)

    # Thiết lập T2 từ Hình 6.6
    T2 = WeightedTransducer()
    for s in range(4):
        T2.add_state(s, start=(s == 0), final=(s == 3))

    T2.add_transition(0, 1, "a", "d", 1.0)
    T2.add_transition(1, 2, EPS, "e", 1.0)  # ε:e
    T2.add_transition(2, 3, "d", "a", 1.0)

    # Tiến hành hợp
    T1.draw("T1")
    T2.draw("T2")

    T_composed_66 = compose_transducers(T1, T2)

    print("--- KẾT QUẢ HÌNH 6.6 ---")
    print(f"Số lượng liên kết sinh ra (Đã qua bộ lọc): {len(T_composed_66.transitions)}")

    T_composed_66.draw("Composition Kết Quả Hình 6.6 (Đã lọc ε)")
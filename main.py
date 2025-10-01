# fsm_generator_full.py
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QComboBox, QFileDialog, QMessageBox, QGroupBox, QSizePolicy, QFormLayout, QSpacerItem
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from collections import defaultdict, deque
import graphviz
import copy
import sys, os

# Ensure Graphviz path (adjust if needed)
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# ---------------- FSM Classes ----------------
class DFA:
    def __init__(self, states, alphabet, transitions, start, finals):
        self.states = set(states)
        self.alphabet = set(alphabet)
        self.transitions = {s: dict(transitions.get(s, {})) for s in self.states}
        self.start = start
        self.finals = set(finals)

    def to_graphviz(self):
        dot = graphviz.Digraph(format='png')
        dot.attr(rankdir='LR')
        dot.node('', shape='none')
        dot.edge('', self.start)
        for s in sorted(self.states):
            shape = 'doublecircle' if s in self.finals else 'circle'
            dot.node(s, shape=shape)
        edges = defaultdict(list)
        for s, trans in self.transitions.items():
            for a, t in trans.items():
                edges[(s, t)].append(str(a))
        for (s, t), labels in edges.items():
            dot.edge(s, t, label=','.join(labels))
        return dot

class NFA:
    def __init__(self, states, alphabet, transitions, start, finals):
        self.states = set(states)
        self.alphabet = set(alphabet)
        self.transitions = {s: {k:set(v) for k,v in transitions.get(s, {}).items()} for s in self.states}
        self.start = start
        self.finals = set(finals)

    def to_graphviz(self):
        dot = graphviz.Digraph(format='png')
        dot.attr(rankdir='LR')
        dot.node('', shape='none')
        dot.edge('', self.start)
        for s, trans in self.transitions.items():
            shape = 'doublecircle' if s in self.finals else 'circle'
            dot.node(s, shape=shape)
            for a, targets in trans.items():
                for t in targets:
                    dot.edge(s, t, label=str(a))
        return dot

class ENFA(NFA):
    EPS = 'ε'
    def add_epsilon(self, s, t):
        self.transitions.setdefault(s, {}).setdefault(self.EPS, set()).add(t)

class Moore:
    def __init__(self, states, alphabet, transitions, start, outputs):
        self.states = set(states)
        self.alphabet = set(alphabet)
        self.transitions = {s: dict(transitions.get(s, {})) for s in self.states}
        self.start = start
        self.outputs = outputs

    def to_graphviz(self):
        dot = graphviz.Digraph(format='png')
        dot.attr(rankdir='LR')
        dot.node('', shape='none')
        dot.edge('', self.start)
        for s in sorted(self.states):
            dot.node(s, label=f"{s}/{self.outputs.get(s,'')}", shape='circle')
        for s, trans in self.transitions.items():
            for a, t in trans.items():
                dot.edge(s, t, label=str(a))
        return dot

class Mealy:
    def __init__(self, states, alphabet, transitions, start):
        self.states = set(states)
        self.alphabet = set(alphabet)
        self.transitions = {s: dict(transitions.get(s, {})) for s in self.states}
        self.start = start

    def to_graphviz(self):
        dot = graphviz.Digraph(format='png')
        dot.attr(rankdir='LR')
        dot.node('', shape='none')
        dot.edge('', self.start)
        for s in sorted(self.states):
            dot.node(s, shape='circle')
        for s, trans in self.transitions.items():
            for a, (t, out) in trans.items():
                dot.edge(s, t, label=f"{a}/{out}")
        return dot

# ---------------- Conversion Helpers ----------------
def nfa_to_dfa(nfa):
    start_set = frozenset([nfa.start])
    queue = deque([start_set])
    seen = {start_set}
    trans = {}
    finals = set()
    while queue:
        S = queue.popleft()
        trans[S] = {}
        for a in nfa.alphabet:
            dest = set()
            for q in S:
                dest |= nfa.transitions.get(q, {}).get(a, set())
            dest_f = frozenset(dest)
            trans[S][a] = dest_f
            if dest_f and dest_f not in seen:
                seen.add(dest_f)
                queue.append(dest_f)
    mapping = {s: f"q{i}" for i, s in enumerate(seen)}
    d_states = set(mapping.values())
    d_trans = {}
    for sset, dname in mapping.items():
        d_trans[dname] = {}
        for a, dest in trans[sset].items():
            d_trans[dname][a] = mapping.get(dest)
        if set(sset) & nfa.finals:
            finals.add(dname)
    d_start = mapping[start_set]
    return DFA(d_states, nfa.alphabet, d_trans, d_start, finals)

# ---------------- Minimization Algorithms ----------------
def table_filling_minimize(dfa):
    states = list(dfa.states)
    n = len(states)
    idx = {s:i for i,s in enumerate(states)}
    marked = [[False]*n for _ in range(n)]
    # Step 1: Mark pairs with one final and one non-final
    for i in range(n):
        for j in range(i+1,n):
            if (states[i] in dfa.finals) != (states[j] in dfa.finals):
                marked[i][j] = True
    changed = True
    # Step 2: Iteratively mark pairs
    while changed:
        changed = False
        for i in range(n):
            for j in range(i+1,n):
                if marked[i][j]: continue
                for a in dfa.alphabet:
                    ti = dfa.transitions.get(states[i], {}).get(a)
                    tj = dfa.transitions.get(states[j], {}).get(a)
                    if ti is None or tj is None: continue
                    ii,jj = idx[ti], idx[tj]
                    x,y = (ii,jj) if ii<jj else (jj,ii)
                    if marked[x][y]:
                        marked[i][j] = True
                        changed = True
                        break
    # Step 3: Merge states
    parent = {}
    for i in range(n):
        for j in range(i+1,n):
            if not marked[i][j]:
                parent[states[j]] = states[i]
    rep = {}
    for s in states:
        r = s
        while r in parent: r=parent[r]
        rep[s]=r
    new_states = set(rep.values())
    new_trans = {s:{} for s in new_states}
    for s in dfa.states:
        for a,t in dfa.transitions.get(s, {}).items():
            new_trans[rep[s]][a] = rep[t]
    new_start = rep[dfa.start]
    new_finals = {rep[s] for s in dfa.finals}
    # For visualization
    table = marked
    merge = rep
    return DFA(new_states, dfa.alphabet, new_trans, new_start, new_finals), table, merge, states

def hopcroft_minimize(dfa):
    P = [dfa.finals, dfa.states - dfa.finals]
    W = [dfa.finals.copy(), (dfa.states - dfa.finals).copy()]
    P_steps = [ [set(g) for g in P] ]
    while W:
        A = W.pop()
        for c in dfa.alphabet:
            X = set(s for s in dfa.states if dfa.transitions.get(s, {}).get(c) in A)
            for Y in P[:]:
                inter = X & Y
                diff = Y - X
                if inter and diff:
                    P.remove(Y)
                    P.extend([inter,diff])
                    if Y in W:
                        W.remove(Y)
                        W.extend([inter,diff])
                    else:
                        W.append(inter if len(inter)<=len(diff) else diff)
        P_steps.append([set(g) for g in P])
    rep = {}
    for block in P:
        r = sorted(block)[0]
        for s in block:
            rep[s]=r
    new_states = set(rep.values())
    new_trans = {s:{} for s in new_states}
    for s in dfa.states:
        for a,t in dfa.transitions.get(s, {}).items():
            new_trans[rep[s]][a]=rep[t]
    new_start = rep[dfa.start]
    new_finals = {rep[s] for s in dfa.finals}
    return DFA(new_states,dfa.alphabet,new_trans,new_start,new_finals), P_steps

def brzozowski_minimize_steps(dfa):
    # 1. Reverse DFA to NFA
    rev_trans = defaultdict(lambda: defaultdict(set))
    for s,tmap in dfa.transitions.items():
        for a,t in tmap.items():
            rev_trans[t][a].add(s)
    nfa_rev = NFA(dfa.states, dfa.alphabet, rev_trans, list(dfa.finals)[0], [dfa.start])
    # 2. Determinize
    dfa1 = nfa_to_dfa(nfa_rev)
    # 3. Reverse again
    rev_trans2 = defaultdict(lambda: defaultdict(set))
    for s,tmap in dfa1.transitions.items():
        for a,t in tmap.items():
            rev_trans2[t][a].add(s)
    nfa_rev2 = NFA(dfa1.states, dfa1.alphabet, rev_trans2, list(dfa1.finals)[0], [dfa1.start])
    # 4. Determinize
    dfa2 = nfa_to_dfa(nfa_rev2)
    return [
        ("Reverse (DFA→NFA)", nfa_rev),
        ("Determinize (NFA→DFA)", dfa1),
        ("Reverse (DFA→NFA)", nfa_rev2),
        ("Determinize (NFA→DFA)", dfa2)
    ]

# ---------------- GUI ----------------
class FSMWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FSM Generator")
        self.resize(1300, 800)
        self.fsm = None

        # --- Left Panel: Input & Controls (sidebar style) ---
        left = QGroupBox()
        left.setFixedWidth(270)  # Sidebar width
        left_layout = QVBoxLayout(left)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        # Move label and input to vertical arrangement
        label_desc = QLabel("Problem Description:")
        self.desc_input = QTextEdit(placeholderText="Write problem/language description here...")
        self.desc_input.setFixedHeight(50)
        form.addRow(label_desc)
        form.addRow(self.desc_input)

        self.target_combo = QComboBox()
        self.target_combo.addItems(["DFA", "NFA", "E-NFA", "Moore", "Mealy"])
        form.addRow(QLabel("Generate as:"), self.target_combo)

        left_layout.addLayout(form)

        # --- Control Buttons (vertical, clean) ---
        ctrl_btn_layout = QVBoxLayout()
        ctrl_btn_layout.setSpacing(8)
        self.btn_generate = QPushButton("Generate")
        self.btn_generate.clicked.connect(self.on_generate)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_save = QPushButton("Save Diagram")
        self.btn_save.clicked.connect(self.on_save)
        ctrl_btn_layout.addWidget(self.btn_generate)
        ctrl_btn_layout.addWidget(self.btn_reset)
        ctrl_btn_layout.addWidget(self.btn_save)
        ctrl_btn_layout.addStretch(1)
        left_layout.addLayout(ctrl_btn_layout)

        # --- Conversion/Minimization Buttons (vertical, clean) ---
        conv_btn_layout = QVBoxLayout()
        conv_btn_layout.setSpacing(8)
        self.btn_tablefill = QPushButton("Minimize DFA (Table Filling)")
        self.btn_tablefill.clicked.connect(self.on_tablefill)
        self.btn_hopcroft = QPushButton("Minimize DFA (Hopcroft)")
        self.btn_hopcroft.clicked.connect(self.on_hopcroft)
        self.btn_brzozowski = QPushButton("Minimize DFA (Brzozowski)")
        self.btn_brzozowski.clicked.connect(self.on_brzozowski)
        self.btn_dfa_to_nfa = QPushButton("DFA → NFA")
        self.btn_dfa_to_nfa.clicked.connect(self.on_dfa_to_nfa)  # <-- this needs the method below
        self.btn_nfa_to_dfa = QPushButton("NFA → DFA")
        self.btn_nfa_to_dfa.clicked.connect(self.on_nfa_to_dfa)
        self.btn_enfa_to_nfa = QPushButton("E-NFA → NFA")
        self.btn_enfa_to_nfa.clicked.connect(self.on_enfa_to_nfa)
        # Add Moore <-> Mealy conversion buttons
        self.btn_moore_to_mealy = QPushButton("Moore → Mealy")
        self.btn_moore_to_mealy.clicked.connect(self.on_moore_to_mealy)
        self.btn_mealy_to_moore = QPushButton("Mealy → Moore")
        self.btn_mealy_to_moore.clicked.connect(self.on_mealy_to_moore)
        conv_btn_layout.addWidget(self.btn_tablefill)
        conv_btn_layout.addWidget(self.btn_hopcroft)
        conv_btn_layout.addWidget(self.btn_brzozowski)
        conv_btn_layout.addWidget(self.btn_dfa_to_nfa)
        conv_btn_layout.addWidget(self.btn_nfa_to_dfa)
        conv_btn_layout.addWidget(self.btn_enfa_to_nfa)
        conv_btn_layout.addWidget(self.btn_moore_to_mealy)
        conv_btn_layout.addWidget(self.btn_mealy_to_moore)
        conv_btn_layout.addStretch(1)
        left_layout.addSpacing(10)
        left_layout.addLayout(conv_btn_layout)
        left_layout.addStretch(2)

        # --- Right Panel: Output / Results ---
        right = QGroupBox()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # --- Top: Tuples and Transition Table side by side ---
        top_hbox = QHBoxLayout()
        # Tuples
        tuple_vbox = QVBoxLayout()
        label_tuple = QLabel("Tuples")
        label_tuple.setStyleSheet("font-weight: bold;")
        self.tuple_display = QTextEdit(readOnly=True)
        self.tuple_display.setMinimumHeight(90)
        self.tuple_display.setMaximumHeight(120)
        self.tuple_display.setStyleSheet("background: #fafbfc;")
        tuple_vbox.addWidget(label_tuple)
        tuple_vbox.addWidget(self.tuple_display)
        top_hbox.addLayout(tuple_vbox, 1)

        # Transition Table
        table_vbox = QVBoxLayout()
        label_table = QLabel("Transition Table")
        label_table.setStyleSheet("font-weight: bold;")
        self.table_display = QTextEdit(readOnly=True)
        self.table_display.setMinimumHeight(90)
        self.table_display.setMaximumHeight(120)
        self.table_display.setStyleSheet("background: #fafbfc;")
        table_vbox.addWidget(label_table)
        table_vbox.addWidget(self.table_display)
        top_hbox.addLayout(table_vbox, 1)

        right_layout.addLayout(top_hbox)

        # --- FSM Diagram Centered ---
        label_diag = QLabel("FSM Diagram")
        label_diag.setStyleSheet("font-weight: bold;")
        label_diag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diagram_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.diagram_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.diagram_label.setStyleSheet("background: #fafbfc; border: 1px solid #ddd;")
        right_layout.addWidget(label_diag)
        right_layout.addWidget(self.diagram_label, 4)

        # --- Results for Minimization (Table Filling, Hopcroft, Brzozowski) ---
        self.result_label = QLabel("Minimization Results")
        self.result_label.setStyleSheet("font-weight: bold;")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_display = QTextEdit(readOnly=True)
        self.result_display.setMinimumHeight(120)
        self.result_display.setStyleSheet("background: #f7f7f7;")
        right_layout.addWidget(self.result_label)
        right_layout.addWidget(self.result_display)

        # --- Main Layout ---
        layout = QHBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(left, 0)   # Sidebar: fixed width, no stretch
        layout.addWidget(right, 1)  # Output: takes all remaining space
        self.setLayout(layout)

    # --- FSM Display ---
    def display_fsm(self, fsm):
        # Display tuples
        tup_text = f"States: {fsm.states}\nAlphabet: {fsm.alphabet}\nStart: {fsm.start}\n"
        if hasattr(fsm, "finals"): tup_text += f"Finals: {fsm.finals}\n"
        if hasattr(fsm, "outputs"): tup_text += f"Outputs: {getattr(fsm, 'outputs')}\n"
        tup_text += "Transitions:\n"
        for s, t in getattr(fsm, "transitions").items():
            tup_text += f"  {s}: {t}\n"
        self.tuple_display.setText(tup_text)

        # Display transition table
        table_text = ""
        for s in sorted(fsm.states):
            row = [s] + [str(getattr(fsm, "transitions")[s].get(a, "-")) for a in sorted(fsm.alphabet)]
            table_text += "\t".join(row) + "\n"
        self.table_display.setText(table_text)

        # Render diagram
        dot = fsm.to_graphviz()
        dot.render("fsm_diagram", format="png", cleanup=True)
        self.diagram_label.setPixmap(QPixmap("fsm_diagram.png").scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio))

        # Clear minimization result area
        self.result_display.clear()

    # --- FSM Generation ---
    def on_generate(self):
        desc = self.desc_input.toPlainText().strip()
        if not desc:
            QMessageBox.warning(self, "Error", "Please input problem description!")
            return
        target = self.target_combo.currentText()

        # Restrict alphabet to {0,1} or {a,b}
        if "0" in desc or "1" in desc:
            alpha = ["0", "1"]
            # Example: DFA for strings ending with '01'
            states = ["q0", "q1", "q2"]
            trans = {"q0": {"0": "q0", "1": "q1"},
                     "q1": {"0": "q2", "1": "q1"},
                     "q2": {"0": "q0", "1": "q1"}}
            start = "q0"
            finals = ["q2"]
        else:
            alpha = ["a", "b"]
            # Example: DFA for strings ending with 'a'
            states = ["q0", "q1"]
            trans = {"q0": {"a": "q1", "b": "q0"}, "q1": {"a": "q1", "b": "q0"}}
            start = "q0"
            finals = ["q1"]

        fsm = DFA(states, alpha, trans, start, finals)

        # Convert to target type if needed
        if target == "NFA":
            fsm = NFA(fsm.states, fsm.alphabet, {s: {a: {t} for a, t in trans.items()} for s, trans in fsm.transitions.items()}, fsm.start, fsm.finals)
        elif target == "E-NFA":
            fsm = ENFA(fsm.states, fsm.alphabet, {s: {a: {t} for a, t in trans.items()} for s, trans in fsm.transitions.items()}, fsm.start, fsm.finals)
        elif target == "Moore":
            outputs = {s: 1 if s in fsm.finals else 0 for s in fsm.states}
            fsm = Moore(fsm.states, fsm.alphabet, fsm.transitions, fsm.start, outputs)
        elif target == "Mealy":
            fsm = Mealy(fsm.states, fsm.alphabet, {s: {a: (t, 1 if t in fsm.finals else 0) for a, t in fsm.transitions[s].items()} for s in fsm.states}, fsm.start)

        self.fsm = fsm
        self.display_fsm(fsm)

    def on_reset(self):
        self.desc_input.clear()
        self.tuple_display.clear()
        self.table_display.clear()
        self.diagram_label.clear()
        self.fsm = None

    def on_save(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Diagram", "fsm_diagram.png", "PNG Files (*.png)")
        if fname and self.diagram_label.pixmap():
            self.diagram_label.pixmap().save(fname)

    # --- Minimization and Conversion Actions ---
    def on_tablefill(self):
        if isinstance(self.fsm, DFA):
            self.tuple_display.clear()
            self.table_display.clear()
            self.diagram_label.clear()
            minimized, table, merge, state_order = table_filling_minimize(self.fsm)
            self.fsm = minimized
            self.display_fsm(self.fsm)
            # Show Table Filling Table
            table_str = "Table Filling Table (X=marked):\n"
            n = len(state_order)
            table_str += "\t" + "\t".join(state_order) + "\n"
            for i in range(n):
                row = [state_order[i]]
                for j in range(n):
                    if j <= i:
                        row.append("-")
                    else:
                        row.append("X" if table[i][j] else "")
                table_str += "\t".join(row) + "\n"
            # Show merge result
            merge_str = "Merged States:\n"
            for s in state_order:
                merge_str += f"{s} → {merge[s]}\n"
            self.result_display.setPlainText(
                table_str + "\n" + merge_str +
                "\nNew DFA:\n" +
                f"States: {minimized.states}\nAlphabet: {minimized.alphabet}\nStart: {minimized.start}\n" +
                f"Finals: {minimized.finals}\nTransitions:\n" +
                "\n".join([f"  {s}: {t}" for s, t in minimized.transitions.items()])
            )
        else:
            QMessageBox.warning(self, "Error", "Table Filling minimization is only for DFA.")

    def on_hopcroft(self):
        if isinstance(self.fsm, DFA):
            self.tuple_display.clear()
            self.table_display.clear()
            self.diagram_label.clear()
            minimized, P_steps = hopcroft_minimize(self.fsm)
            self.fsm = minimized
            self.display_fsm(self.fsm)
            # Show partition steps
            part_str = "Hopcroft Partition Steps:\n"
            for i, P in enumerate(P_steps):
                part_str += f"Step {i}: " + ", ".join([f"P{i+1}={sorted(list(g))}" for i,g in enumerate(P)]) + "\n"
            part_str += "\nNew DFA:\n"
            part_str += f"States: {minimized.states}\nAlphabet: {minimized.alphabet}\nStart: {minimized.start}\n"
            part_str += f"Finals: {minimized.finals}\nTransitions:\n"
            part_str += "\n".join([f"  {s}: {t}" for s, t in minimized.transitions.items()])
            self.result_display.setPlainText(part_str)
        else:
            QMessageBox.warning(self, "Error", "Hopcroft minimization is only for DFA.")

    def on_brzozowski(self):
        if isinstance(self.fsm, DFA):
            self.tuple_display.clear()
            self.table_display.clear()
            self.diagram_label.clear()
            steps = brzozowski_minimize_steps(self.fsm)
            result_str = ""
            for i, (desc, fsm) in enumerate(steps):
                # Table
                table_text = ""
                for s in sorted(fsm.states):
                    row = [s] + [str(getattr(fsm,"transitions")[s].get(a,"-")) for a in sorted(fsm.alphabet)]
                    table_text += "\t".join(row)+"\n"
                # Diagram
                dot = fsm.to_graphviz()
                fname = f"brzozowski_step_{i}.png"
                dot.render(f"brzozowski_step_{i}", format="png", cleanup=True)
                result_str += f"{desc}:\nStates: {fsm.states}\nAlphabet: {fsm.alphabet}\nStart: {fsm.start}\n"
                if hasattr(fsm, "finals"):
                    result_str += f"Finals: {fsm.finals}\n"
                result_str += f"Transitions:\n{table_text}\n"
                result_str += f"Diagram saved as {fname}\n\n"
            # Show last DFA as current
            self.fsm = steps[-1][1]
            self.display_fsm(self.fsm)
            self.result_display.setPlainText(result_str)
        else:
            QMessageBox.warning(self, "Error", "Brzozowski minimization is only for DFA.")

    # --- DFA to NFA conversion ---
    def on_dfa_to_nfa(self):
        if isinstance(self.fsm, DFA):
            nfa_trans = {s: {a: {t} for a, t in trans.items()}
                         for s, trans in self.fsm.transitions.items()}
            self.fsm = NFA(self.fsm.states, self.fsm.alphabet, nfa_trans, self.fsm.start, self.fsm.finals)
            self.display_fsm(self.fsm)
            self.result_display.setPlainText("Converted DFA to NFA.")
        else:
            QMessageBox.warning(self, "Error", "Current FSM is not a DFA.")

    # --- NFA to DFA conversion ---
    def on_nfa_to_dfa(self):
        if isinstance(self.fsm, (NFA, ENFA)):
            self.fsm = nfa_to_dfa(self.fsm)
            self.display_fsm(self.fsm)
            self.result_display.setPlainText("Converted NFA/E-NFA to DFA.")
        else:
            QMessageBox.warning(self, "Error", "Current FSM is not an NFA or E-NFA.")

    # --- Mealy to Moore conversion ---
    def on_mealy_to_moore(self):
        if isinstance(self.fsm, Mealy):
            # Each (state, output) pair becomes a new Moore state
            new_states = set()
            new_outputs = {}
            new_trans = {}
            state_map = {}
            for s in self.fsm.states:
                for a, (t, out) in self.fsm.transitions[s].items():
                    state_map[(s, out)] = f"{s}_{out}"
                    state_map[(t, out)] = f"{t}_{out}"
            for (s, out) in state_map:
                new_states.add(state_map[(s, out)])
                new_outputs[state_map[(s, out)]] = out
            for (s, out) in state_map:
                new_trans[state_map[(s, out)]] = {}
                for a, (t, out2) in self.fsm.transitions.get(s, {}).items():
                    new_trans[state_map[(s, out)]][a] = state_map[(t, out2)]
            # Pick start state with output from original start
            start_out = None
            for a, (t, out) in self.fsm.transitions.get(self.fsm.start, {}).items():
                start_out = out
                break
            new_start = state_map.get((self.fsm.start, start_out), list(new_states)[0])
            moore = Moore(new_states, self.fsm.alphabet, new_trans, new_start, new_outputs)
            self.fsm = moore
            self.display_fsm(self.fsm)
            self.result_display.setPlainText("Converted Mealy machine to Moore machine.")
        else:
            QMessageBox.warning(self, "Error", "Current FSM is not a Mealy machine.")

    # --- E-NFA to NFA conversion ---
    def on_enfa_to_nfa(self):
        if isinstance(self.fsm, ENFA):
            nfa_trans = {}
            for s in self.fsm.states:
                nfa_trans[s] = {}
                # Compute epsilon closure
                closure = set([s])
                stack = [s]
                while stack:
                    curr = stack.pop()
                    for t in self.fsm.transitions.get(curr, {}).get(ENFA.EPS, set()):
                        if t not in closure:
                            closure.add(t)
                            stack.append(t)
                # For each symbol, collect all reachable states via closure
                for a in self.fsm.alphabet:
                    dest = set()
                    for q in closure:
                        dest |= self.fsm.transitions.get(q, {}).get(a, set())
                    nfa_trans[s][a] = dest
            self.fsm = NFA(self.fsm.states, self.fsm.alphabet, nfa_trans, self.fsm.start, self.fsm.finals)
            self.display_fsm(self.fsm)
            self.result_display.setPlainText("Converted E-NFA to NFA (epsilon transitions removed).")
        else:
            QMessageBox.warning(self, "Error", "Current FSM is not an E-NFA.")

# ---------------- Main ----------------
if __name__=="__main__":
    app = QApplication(sys.argv)
    window = FSMWindow()
    window.show()
    sys.exit(app.exec())

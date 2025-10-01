# fsm_generator_full.py
import sys, os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QComboBox, QFileDialog, QMessageBox, QGroupBox, QSizePolicy
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from collections import defaultdict, deque
import graphviz
import copy

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
    for i in range(n):
        for j in range(i+1,n):
            if (states[i] in dfa.finals) != (states[j] in dfa.finals):
                marked[i][j] = True
    changed = True
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
    return DFA(new_states, dfa.alphabet, new_trans, new_start, new_finals)

def hopcroft_minimize(dfa):
    P = [dfa.finals, dfa.states - dfa.finals]
    W = [dfa.finals.copy(), (dfa.states - dfa.finals).copy()]
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
    return DFA(new_states,dfa.alphabet,new_trans,new_start,new_finals)

def brzozowski_minimize(dfa):
    # Reverse DFA to NFA
    rev_trans = defaultdict(lambda: defaultdict(set))
    for s,tmap in dfa.transitions.items():
        for a,t in tmap.items():
            rev_trans[t][a].add(s)
    nfa_rev = NFA(dfa.states, dfa.alphabet, rev_trans, list(dfa.finals)[0], [dfa.start])
    dfa1 = nfa_to_dfa(nfa_rev)
    # Reverse again
    rev_trans2 = defaultdict(lambda: defaultdict(set))
    for s,tmap in dfa1.transitions.items():
        for a,t in tmap.items():
            rev_trans2[t][a].add(s)
    nfa_rev2 = NFA(dfa1.states, dfa1.alphabet, rev_trans2, list(dfa1.finals)[0], [dfa1.start])
    return nfa_to_dfa(nfa_rev2)

# ---------------- GUI ----------------
class FSMWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FSM Generator")
        self.resize(1300, 800)
        self.fsm = None

        # Left panel
        left = QGroupBox("Input / Controls")
        left_layout = QVBoxLayout()
        left.setLayout(left_layout)

        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText("Write problem/language description here...")
        left_layout.addWidget(QLabel("Problem Description"))
        left_layout.addWidget(self.desc_input)

        self.target_combo = QComboBox()
        self.target_combo.addItems(["DFA","NFA","E-NFA","Moore","Mealy"])
        left_layout.addWidget(QLabel("Generate as"))
        left_layout.addWidget(self.target_combo)

        row1 = QHBoxLayout()
        self.btn_generate = QPushButton("Generate")
        self.btn_generate.clicked.connect(self.on_generate)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_save = QPushButton("Save Diagram")
        self.btn_save.clicked.connect(self.on_save)
        row1.addWidget(self.btn_generate)
        row1.addWidget(self.btn_reset)
        row1.addWidget(self.btn_save)
        left_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_tablefill = QPushButton("Minimize DFA (Table Filling)")
        self.btn_tablefill.clicked.connect(self.on_tablefill)
        self.btn_hopcroft = QPushButton("Minimize DFA (Hopcroft)")
        self.btn_hopcroft.clicked.connect(self.on_hopcroft)
        self.btn_brzozowski = QPushButton("Minimize DFA (Brzozowski)")
        self.btn_brzozowski.clicked.connect(self.on_brzozowski)
        self.btn_dfa_to_nfa = QPushButton("DFA → NFA")
        self.btn_dfa_to_nfa.clicked.connect(self.on_dfa_to_nfa)

        self.btn_nfa_to_dfa = QPushButton("NFA → DFA")
        self.btn_nfa_to_dfa.clicked.connect(self.on_nfa_to_dfa)

        self.btn_enfa_to_nfa = QPushButton("E-NFA → NFA")
        self.btn_enfa_to_nfa.clicked.connect(self.on_enfa_to_nfa)

        row2.addWidget(self.btn_tablefill)
        row2.addWidget(self.btn_hopcroft)
        row2.addWidget(self.btn_brzozowski)
        row2.addWidget(self.btn_dfa_to_nfa)
        row2.addWidget(self.btn_nfa_to_dfa)
        row2.addWidget(self.btn_enfa_to_nfa)

        left_layout.addLayout(row2)

        # Right panel
        right = QGroupBox("Output / Results")
        right_layout = QVBoxLayout()
        right.setLayout(right_layout)

        self.tuple_display = QTextEdit()
        self.tuple_display.setReadOnly(True)
        right_layout.addWidget(QLabel("Tuples"))
        right_layout.addWidget(self.tuple_display)

        self.table_display = QTextEdit()
        self.table_display.setReadOnly(True)
        right_layout.addWidget(QLabel("Transition Table"))
        right_layout.addWidget(self.table_display)

        self.diagram_label = QLabel()
        self.diagram_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diagram_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(QLabel("FSM Diagram"))
        right_layout.addWidget(self.diagram_label)

        # Main layout
        layout = QHBoxLayout()
        layout.addWidget(left, 1)
        layout.addWidget(right, 2)
        self.setLayout(layout)

    # ---------------- GUI Actions ----------------
    def display_fsm(self,fsm):
        # Display tuples
        tup_text = f"States: {fsm.states}\nAlphabet: {fsm.alphabet}\nStart: {fsm.start}\n"
        if hasattr(fsm,"finals"): tup_text += f"Finals: {fsm.finals}\n"
        if hasattr(fsm,"outputs"): tup_text += f"Outputs: {getattr(fsm,'outputs')}\n"
        tup_text += f"Transitions:\n"
        for s,t in getattr(fsm,"transitions").items():
            tup_text += f"  {s}: {t}\n"
        self.tuple_display.setText(tup_text)

        # Display transition table
        table_text = ""
        for s in sorted(fsm.states):
            row = [s]+[str(getattr(fsm,"transitions")[s].get(a,"-")) for a in sorted(fsm.alphabet)]
            table_text += "\t".join(row)+"\n"
        self.table_display.setText(table_text)

        # Render diagram
        dot = fsm.to_graphviz()
        dot.render("fsm_diagram", format="png", cleanup=True)
        self.diagram_label.setPixmap(QPixmap("fsm_diagram.png").scaled(600,400,Qt.AspectRatioMode.KeepAspectRatio))

    def on_generate(self):
        desc = self.desc_input.toPlainText().strip()
        if not desc:
            QMessageBox.warning(self,"Error","Please input problem description!")
            return
        target = self.target_combo.currentText()

        # --- Automatic FSM generation (example simplified) ---
        if "01" in desc:
            states = ["q0","q1","q2"]
            alpha = ["0","1"]
            trans = {"q0":{"0":"q0","1":"q1"},
                     "q1":{"0":"q2","1":"q1"},
                     "q2":{"0":"q0","1":"q1"}}
            start = "q0"
            finals = ["q2"]
            fsm = DFA(states, alpha, trans, start, finals)
        else:
            states = ["q0","q1"]
            alpha = ["a","b"]
            trans = {"q0":{"a":"q1","b":"q0"}, "q1":{"a":"q1","b":"q0"}}
            start = "q0"
            finals = ["q1"]
            fsm = DFA(states, alpha, trans, start, finals)

        # Convert to target type if needed
        if target=="NFA":
            fsm = NFA(fsm.states,fsm.alphabet,{s:{a:{t} for a,t in trans.items()} for s,trans in fsm.transitions.items()},fsm.start,fsm.finals)
        elif target=="E-NFA":
            fsm = ENFA(fsm.states,fsm.alphabet,{s:{a:{t} for a,t in trans.items()} for s,trans in fsm.transitions.items()},fsm.start,fsm.finals)
        elif target=="Moore":
            outputs = {s:1 if s in fsm.finals else 0 for s in fsm.states}
            fsm = Moore(fsm.states,fsm.alphabet,fsm.transitions,fsm.start,outputs)
        elif target=="Mealy":
            fsm = Mealy(fsm.states,fsm.alphabet,{s:{a:(t,1 if t in fsm.finals else 0) for a,t in fsm.transitions[s].items()} for s in fsm.states},fsm.start)

        self.fsm = fsm
        self.display_fsm(fsm)

    def on_reset(self):
        self.desc_input.clear()
        self.tuple_display.clear()
        self.table_display.clear()
        self.diagram_label.clear()
        self.fsm = None

    def on_save(self):
        fname,_ = QFileDialog.getSaveFileName(self,"Save Diagram","fsm_diagram.png","PNG Files (*.png)")
        if fname and self.diagram_label.pixmap():
            self.diagram_label.pixmap().save(fname)

    def on_tablefill(self):
        if isinstance(self.fsm,DFA):
            self.fsm = table_filling_minimize(self.fsm)
            self.display_fsm(self.fsm)

    def on_hopcroft(self):
        if isinstance(self.fsm,DFA):
            self.fsm = hopcroft_minimize(self.fsm)
            self.display_fsm(self.fsm)

    def on_brzozowski(self):
        if isinstance(self.fsm,DFA):
            self.fsm = brzozowski_minimize(self.fsm)
            self.display_fsm(self.fsm)

    def on_convert(self):
        if isinstance(self.fsm,(NFA,ENFA)):
            self.fsm = nfa_to_dfa(self.fsm)
            self.display_fsm(self.fsm)
    
    

# ---------------- Main ----------------
if __name__=="__main__":
    app = QApplication(sys.argv)
    window = FSMWindow()
    window.show()
    sys.exit(app.exec())

# FSM-GENERATOR

A Python tool for generating **Finite State Machines (FSMs)**, their **visual diagrams**, and **minimized DFAs** using Hopcroft’s and Brzozowski’s algorithms.

---

## 📘 Table of Contents
1. [Introduction](#introduction)  
2. [Features](#features)  
3. [Requirements](#requirements)  
4. [Installation](#installation)  
5. [Usage](#usage)  
   - [Command Line](#command-line)  
   - [Input Format](#input-format)  
   - [Outputs](#outputs)  
6. [Architecture & Design](#architecture--design)  
7. [Algorithms & Techniques](#algorithms--techniques)  
8. [Examples](#examples)  
9. [Contributing](#contributing)  
10. [License](#license)  
11. [References](#references)

---

## 🧩 Introduction

**FSM-GENERATOR** is a Python-based tool designed to:
- Create finite state machines (FSMs)
- Generate visual representations (diagrams)
- Minimize deterministic finite automata (DFAs)

It supports multiple minimization algorithms and provides step-by-step visual outputs for better understanding.

---

## ⚙️ Features

- Parse and represent FSMs from text or data files  
- Generate FSM and DFA diagrams  
- Apply **Hopcroft’s** and **Brzozowski’s** minimization algorithms  
- Show intermediate steps (e.g., `brzozowski_step_0.png`, etc.)  
- Export minimized DFA diagrams  

Example output files include:
```

fsm_diagram.png
minimized_dfa_diagram.png
hopcroft_minimized_dfa_diagram.png
brzozowski_step_0.png ... brzozowski_step_n.png

````

---

## 🧠 Requirements

- **Python 3.x**  
- The following Python libraries (based on imports in `main.py`):
  ```bash
  pip install graphviz pydot networkx matplotlib
````

*(Install only what’s needed — check `main.py` imports for accuracy.)*

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/erica-monacillo/FSM-GENERATOR.git
cd FSM-GENERATOR

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> If `requirements.txt` is missing, manually install dependencies as shown above.

---

## 🚀 Usage

### Command Line

Run the tool using:

```bash
python main.py
```

If `--help` is implemented, you can check available options with:

```bash
python main.py --help
```

---

### Input Format

The program accepts FSM specifications containing:

* States
* Input symbols (alphabet)
* Transition table
* Start state
* Accepting (final) states

Example (JSON-style conceptual format):

```json
{
  "states": ["q0", "q1", "q2"],
  "alphabet": ["0", "1"],
  "start_state": "q0",
  "accept_states": ["q2"],
  "transitions": {
    "q0": {"0": "q1", "1": "q0"},
    "q1": {"0": "q2", "1": "q1"},
    "q2": {"0": "q2", "1": "q2"}
  }
}
```

*(Check `main.py` for the exact input format expected.)*

---

### Outputs

After running, the tool will generate:

* **FSM diagram** – visualizes the original automaton
* **Minimized DFA diagram** – shows the reduced version
* **Intermediate steps** (Brzozowski) – multiple images illustrating the process

Example output files:

```
fsm_diagram.png
minimized_dfa_diagram.png
hopcroft_minimized_dfa_diagram.png
brzozowski_step_0.png
brzozowski_step_1.png
...
```

---

## 🏗️ Architecture & Design

The project structure generally includes:

* `main.py` — main program entry point
* **Parser Module** — reads FSM input data
* **FSM Module** — defines states, transitions, and structures
* **Minimizer Module** — implements minimization algorithms
* **Diagram Module** — generates visual diagrams using Graphviz or NetworkX

**Workflow:**

1. Load FSM data
2. Draw original FSM
3. Minimize using Hopcroft or Brzozowski
4. Generate minimized DFA diagram
5. (Optional) Save intermediate steps

---

## 🔢 Algorithms & Techniques

### 🧮 Hopcroft’s Algorithm

* Efficient DFA minimization method
* Works by refining state partitions until no further splits occur
* Time complexity: **O(n log n)**
* Output diagram: `hopcroft_minimized_dfa_diagram.png`

### 🔁 Brzozowski’s Algorithm

* Conceptually simple but slower
* Steps:

  1. Reverse automaton
  2. Determinize
  3. Reverse again
  4. Determinize again
* Each step can be visualized (`brzozowski_step_*.png`)

---

## 🧾 Example

For a small FSM:

| State | Input | Next State |
| ----- | ----- | ---------- |
| S0    | 0     | S1         |
| S0    | 1     | S0         |
| S1    | 0     | S0         |
| S1    | 1     | S1         |

Start State: **S0**
Final State: **S1**

Running the generator would produce:

* `fsm_diagram.png` – original FSM
* `minimized_dfa_diagram.png` – minimized DFA
* Step-by-step Brzozowski images (if enabled)

---

## 🤝 Contributing

Contributions are welcome!

1. **Fork** the repository
2. **Create** a new branch
3. **Commit** your changes
4. **Push** to your fork
5. **Submit** a Pull Request

You can contribute by:

* Adding more minimization algorithms
* Supporting more input formats (JSON, YAML, DOT)
* Improving UI or CLI options
* Enhancing visualization features

---

## 🪪 License

Currently, this repository **does not specify a license**.
It’s recommended to add one (e.g. MIT, Apache 2.0, or GPL) for open-source clarity.

---

## 📚 References

* Hopcroft, John. “An n log n Algorithm for Minimizing States in a Finite Automaton.” (1971)
* Brzozowski, Janusz. “Canonical Regular Expressions and Minimal State Graphs for Definite Events.” (1962)
* [Graphviz Documentation](https://graphviz.org/documentation/)
* [NetworkX Documentation](https://networkx.org/documentation/stable/)
* [Finite State Machine (Wikipedia)](https://en.wikipedia.org/wiki/Finite-state_machine)

---

**Author:** [@erica-monacillo](https://github.com/erica-monacillo)
**Repository:** [FSM-GENERATOR](https://github.com/erica-monacillo/FSM-GENERATOR.git)

```

---

Would you like me to add a **“Project Screenshot”** section (showing the FSM diagrams from your repo) before the Features part? That makes the README look more attractive on GitHub.
```

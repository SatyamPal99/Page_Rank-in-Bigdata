# CSL7110 Assignment 4 — Clustering, Web Search & PageRank

**Course:** CSL7110 — Big Data Analytics  
---

## Repository Structure

```
.
├── Part1_and_2_Assignment4.ipynb   # Part 1: Clustering & Part 2: Web Search
├── Part3_Assignment4.py            # Part 3: PageRank on Spark
├── M25CSA026_CSL7110_Assignment4.pdf  # Assignment report
└── README.md
```

---

## Part 1: Clustering (Notebook)

### Overview
Implements two clustering center-selection algorithms on the **UCI Spambase dataset**:
- **Farthest-First Traversal** (`kcenter`) for k-center clustering
- **k-means++** initialization (`kmeansPP`)
- **k-means objective** evaluation (`kmeansObj`)

### Dataset
Download the UCI Spambase dataset (`spambase.data`)
The last column (class label) is automatically ignored — only the first 57 feature columns are used.

### Dependencies
```bash
pip install pyspark numpy matplotlib
```

### Running the Notebook
1. Open `Part1_and_2_Assignment4.ipynb` in Jupyter or Google Colab.
2. Update the `filename` variable in the notebook to point to your local `spambase.data` path:
   ```python
   filename = "/path/to/spambase.data"
   ```
3. Set the desired values of `k` and `k1` (defaults: `k=10`, `k1=50`):
   ```python
   k = 10
   k1 = 50
   results = runClusteringExperiment(filename, k, k1, seed=42)
   ```
4. Run all cells sequentially.

### Command-Line Usage (as `.py`)
```bash
python part1_clustering.py spambase.data 10 50
# Arguments: <filename> <k> <k1>
```

### Key Functions
| Function | Description |
|---|---|
| `readVectorsSeq(filename)` | Reads dataset; returns list of Spark dense vectors (ignores label column) |
| `kcenter(P, k)` | Farthest-First Traversal center selection — O(|P| × k) |
| `kmeansPP(P, k)` | k-means++ probabilistic center selection — O(|P| × k) |
| `kmeansObj(P, C)` | Average squared distance from points to nearest center |
| `runClusteringExperiment(filename, k, k1)` | Runs all 3 required experiments and prints results |

### Sample Output (k=10, k1=50)
```
Number of points = 4601
Dimension of each point = 57
1. Running time of kcenter(P, 10) = 2.10 seconds
2. kmeansObj(P, kmeansPP(P, 10)) = 31250.987038
3. kmeansObj(P, kmeansPP(kcenter(P, 50), 10)) = 99261.564855
```

---

## Part 2: Web Search (Notebook)

### Overview
Builds an **inverted index**-based search engine over a collection of webpages, supporting positional word queries.

### Dataset
Download the web search dataset (webpages + `actions.txt`) from the course portal.  
Directory structure expected:
```
Q2- webSearch/
├── webpages/
│   ├── stack_cprogramming
│   ├── stack_datastructure_wiki
│   └── ...
└── actions.txt
```

### Running
1. Open `Part1_and_2_Assignment4.ipynb` and navigate to the **Part 2** section.
2. Update the paths:
   ```python
   base_path = "/path/to/webpages"
   actions_file = "/path/to/actions.txt"
   ```
3. Run the Part 2 cells.

### Supported Actions (in `actions.txt`)
```
addPage <pageName>
queryFindPagesWhichContainWord <word>
queryFindPositionsOfWordInAPage <word> <pageName>
```

### Preprocessing Rules Applied
- All words lowercased
- Punctuation replaced with spaces
- Connector words excluded from the index (but counted in position numbering)
- Singular/plural normalization: `stack`/`stacks`, `structure`/`structures`, `application`/`applications`

---

## Part 3: PageRank on Spark

### Overview
Implements the **PageRank algorithm** using **PySpark RDDs** on a directed graph, with a damping factor of `β = 0.8` over 40 iterations.

### Dataset
Two graph files are required (edge list format — one `source destination` pair per line):
- `small.txt` — 100-node graph
- `whole.txt` — 1000-node graph

Upload these to your Google Drive (or update paths for local Spark).

### Dependencies
```bash
pip install pyspark
```

### Running on Google Colab
1. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Update the file paths in `Part3_Assignment4.py`:
   ```python
   small_file = "/content/drive/MyDrive/small.txt"
   whole_file  = "/content/drive/MyDrive/whole.txt"
   ```
3. Run the script:
   ```bash
   python Part3_Assignment4.py
   ```

### Running Locally
If running PySpark locally, replace the file paths with your local paths:
```python
small_file = "/path/to/small.txt"
whole_file  = "/path/to/whole.txt"
```
Then run:
```bash
python Part3_Assignment4.py
```

### Key Functions
| Function | Description |
|---|---|
| `read_graph_edges(file_path)` | Reads edge list; removes duplicate directed edges |
| `build_graph(edges_rdd)` | Builds adjacency lists and computes total node count |
| `pagerank_spark(file_path, beta, num_iterations)` | Full PageRank pipeline; returns `(node, rank)` RDD |

### Algorithm Parameters
| Parameter | Value |
|---|---|
| Damping factor `β` | 0.8 |
| Iterations | 40 |
| Initial rank | 1/n (uniform) |

### Sample Output
```
Number of nodes in small graph: 100
Top 10 nodes in small graph:
53  0.03573
14  0.03417
40  0.03363
...

Number of nodes in whole graph: 1000
Sum of PageRank scores = 1.0

Top 5 nodes (highest PageRank):
263  0.002020
537  0.001943
...

Bottom 5 nodes (lowest PageRank):
558  0.000329
93   0.000351
...
```

> **Correctness check:** The sum of all PageRank scores equals `1.0`, confirming proper normalization.

---

## Results Summary

### Part 1 — Clustering

| k1 | `kmeansObj` (Direct k-means++) | `kmeansObj` (Coreset-based) |
|---|---|---|
| 20  | 31250.99 | 72138.31  |
| 50  | 31250.99 | 99261.56  |
| 100 | 31250.99 | 745044.24 |

Direct k-means++ consistently outperforms the coreset-based approach for this dataset.

### Part 3 — PageRank (Whole Graph, 1000 nodes)

| Rank | Node | PageRank Score |
|---|---|---|
| 1st | 263 | 0.002020 |
| 2nd | 537 | 0.001943 |
| 3rd | 965 | 0.001925 |

---

## Notes
- Part 3 was developed and tested on **Google Colab** using PySpark.
- The notebook (Parts 1 & 2) uses `pyspark.mllib.linalg.Vectors` for point representation; `Vectors.sqdist()` is unavailable in local Spark environments, so squared Euclidean distance is computed manually.
- Ensure `k < k1` when running clustering experiments.

# INFOMDWR – Assignment 2: Data Integration & Preparation

## Introduction
This repository contains the solutions for **Assignment 2** of the course *INFOMDWR – Data Integration & Preparation*.  
The assignment focuses on tasks related to **data preparation**, including:

- Data profiling
- Entity resolution
- Correlation analysis and handling missing values

Datasets used in this assignment are available online (links are provided in the assignment description).

⚠️ **Important note:** According to the assignment instructions, the use of AI tools such as ChatGPT for solving the assignment is not allowed.  
This repository only documents and organises the work.

---

## Task Overview

### **Task 1: Profiling relational data**
- Read the paper on profiling relational data.  
- Select a set of **at least 10 different summary statistics** to characterise a dataset (e.g. from the *road safety* dataset).  
- Implement Python code to compute these statistics.  
- Provide explanations of the importance of each statistic in understanding dataset characteristics.

> Note: Using the same statistic across multiple columns counts only once.

---

### **Task 2: Entity resolution**

#### Part 1: Record pairwise comparison (DBLP–ACM)
- Compare every record in **ACM.csv** with every record in **DBLP2.csv**.  
- Apply the following similarity measures:
  - **Title:** Levenshtein similarity (based on edit distance).
  - **Authors:** Jaro similarity.
  - **Venue:** Modified affine similarity (scaled to [0, 1]).
  - **Year:** Exact match (1 for match, 0 for mismatch).
- Combine the scores using the provided formula.  
- Report pairs with **rec_sim > 0.7** as duplicates.  
- Evaluate precision against the ground truth in **DBLP-ACM_perfectMapping.csv**.  
- Record and discuss the **running time** and potential improvements.

#### Part 2: Locality Sensitive Hashing (LSH)
- Concatenate attributes into a single string per record.  
- Normalise (lowercase, remove multiple spaces).  
- Combine all records from both datasets.  
- Apply the **LSH-based approach** from Lab 5:
  - Shingling
  - Minhash signatures
  - Candidate generation with LSH
- Compare top candidates to the ground truth and compute precision.  
- Record running time.  
- **Compare** precision and running times from Part 1 and Part 2.

---

### **Task 3: Data preparation (Pima Indians Diabetes Database)**
- Compute correlations between columns (excluding the outcome column).  
- Handle disguised missing values:
  - Replace `0` values in **BloodPressure**, **SkinThickness**, and **BMI** with `null`.  
  - Retain records (do not drop).  
- Fill missing values with the **mean values per class label**.  
- Recompute correlations.  
- Compare and comment on the differences between the initial and updated correlations.

---

## Submission Instructions
The submission package must be a `.zip` file containing:

1. **Report (PDF):**  
   - Answers and explanations for all tasks.  
   - Discussion of methods and most important results.

2. **Python Notebook (.ipynb):**  
   - Code implementation for all tasks.  
   - Markdown cells indicating the task/question being solved.  
   - Comments in the code for clarity.

> Submit on **Blackboard**.  
> Only **one submission per group** is required.  
> Multiple submissions are allowed, but **only the last submission before the deadline** will be graded.

---

## Deadline
📅 **07 October 2025, 09:00 AM** (as specified on the course website)

---

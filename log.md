- When using the `chroma` vector db, asking for "condition and design matrices" search string specifically, it retrieved the *HedSummaryGuide* document instead of *HedConditionsAndDesignMatrices*. 
- There seems to be a context length issue
```
VectorDB returns doc_ids:  [['4d4989eb', '22a4f1b0', '9e4cab43', 'c5088cd3', '29810f2f', 'b261690f', '3e0e00f8', '369bafba']]
Skip doc_id 4d4989eb as it is too long to fit in the context.
Skip doc_id 22a4f1b0 as it is too long to fit in the context.
Adding content of doc 9e4cab43 to context.
Skip doc_id c5088cd3 as it is too long to fit in the context.
```
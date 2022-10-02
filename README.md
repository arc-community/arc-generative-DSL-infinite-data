# ARC Generative DSL
Slowly building a collection of infinite riddle generators for benchmarking data-hungry methods.


## Implemented Puzzles


### Dungeon Generation

https://github.com/arc-community/arc/wiki/Riddle_Evaluation_09c534e7

https://volotat.github.io/ARC-Game/?task=evaluation%2F09c534e7.json

Here we generate the dataset from puzzle "09c534e7" which looks a lot like a 2D rogue-like dungeon.
Here is the notebook for generating a dataset from the script: "gen_dungeon_dataset.ipynb"

<img src="figures/dungeon_generation_figure.png" alt="dungeon image" width=100% >



### Sort-of-ARC Dataset

https://openreview.net/pdf?id=rCzfIruU5x5


We generate a dataset using this script in the notebook: "gen_sortOfARC_dataset.ipynb"

<img src="figures/sortOfARC_generation_sample.PNG" alt="sortOfARC_image" width=100% >
<img src="figures/sortOfARC_generation_sample2.PNG" alt="sortOfARC_image" width=100% >








<br>
<br>
<br>
<br>

## TODO List

1. Group the helper functions (for reusability in a programming language style manner) - hence giving the name generative DSL.

2. Extend "sort of ARC" to more simultaneous rules and using different {board size, number of objects, size of objects, number of transformations}

3. More efficient rejection sampling for tensor dataset generation instead of iteration (let me know if this is urgent/ one of the functions is not fast enough)

4. Puzzle TODO list
- Easier next steps
  - 'A2FD1CF0' (https://volotat.github.io/ARC-Game/?task=training%2Fa2fd1cf0.json)
  - '137EAA0F' (https://volotat.github.io/ARC-Game/?task=training%2F137eaa0f.json) 
  - '321B1FC6' (https://volotat.github.io/ARC-Game/?task=training%2F321b1fc6.json)
  - '27F8CE4F' (https://volotat.github.io/ARC-Game/?task=evaluation%2F27f8ce4f.json)
- Parapraxis desired
  - '29700607' (https://volotat.github.io/ARC-Game/?task=evaluation%2F29700607.json)
  - '15663BA9' (https://volotat.github.io/ARC-Game/?task=evaluation%2F15663ba9.json)
  - '1ACC24AF' (https://volotat.github.io/ARC-Game/?task=evaluation%2F1acc24af.json)
- Not-so-easy 
  - '009D5C81' (Andreas already did?) (https://volotat.github.io/ARC-Game/?task=evaluation%2F009d5c81.json)
  - '150DEFF5' (https://volotat.github.io/ARC-Game/?task=training%2F150deff5.json)
  
  
  
  
  
  
  




  
# 12.2022

# Common-Gen
Commonsense Generation model, which is capable of generating human-like coherent texts from a given set of constraints (our case - words).

Data availability is often a determining factor in NLP algorithms. A larger dataset would provide the model with additional training examples and allow it to better represent the distribution of the data. In light of this, we understand the importance of automatically generating new examples.
One way to generate new examples is to use a Commonsense Generation model, which is capable of generating human-like coherent texts from a given set of constraints

In this project I am going to propose an algorithm for common sense generation which will work in two steps:
In the first step, a parsing algorithm will arrange the constraints (the words) in a reasonable order. The purpose of this step is to simplify and logically organize the next step model since we will "force" the order of the words in the next step.

In the second step I will add and pad mask tokens between the words and will ask a generator to complete the masked spans, e.g. {apple, pick, eat} → {pick, apple, eat} → [mask] pick [mask] apple [mask] eat [mask] →  A kid picked an apple from the tree and ate it. Depending on the need, we would like to be able to generate one word, two words, or more for each mask, and sometimes nothing at all. Furthermore, we would like the model to be able to inflect the roots of words he receives.

An end-to-end implementation of the problem is provided in this project.

Here is an example of the model's output

<img src="https://user-images.githubusercontent.com/111754948/205606745-e3ffb1c5-d25e-4cf0-838a-e1804db565fa.png" width=50% height=50%>





